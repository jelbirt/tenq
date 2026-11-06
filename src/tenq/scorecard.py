"""The scorecard: each daily recommendation judged by what the stock did next.

    python -m tenq.scorecard            # print the scorecard from the records
    python -m tenq.scorecard --save     # and add newly finished calls to the ledger

A call is entered at the open of the first session that starts after it was
made (`made_at` in the day's record, which can be after that day's close) and
left at the close of the 20th session, counting the entry session. The stock's
return over that span less QQQ's over the same span is the excess return; the
call's score is its weight (config/score.yaml) times that excess. Both prices
come from one fresh fetch, so dividends rescaling past prices cannot skew it.

A finished call is written once to the ledger, runs/daily/scores/<version>.json,
and never scored again: a later price revision or config change cannot rewrite
the past (a new config version starts a new ledger).

The scorecard compares the two methods on the calls both made, with a range
from a block bootstrap by week, and says "too early to tell" until enough weeks
are scored. Beside them: what holding, buying everything, buying last month's
winners and a shuffled version of each method's own calls would have scored.

Only full daily records count: `YYYY-MM-DD.json`, kind daily, no check id,
complete. Check runs and consistency runs are never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock
from pathlib import Path
from typing import Any

import holidays
import yaml

from tenq.config import REPO_ROOT
from tenq.prices import BENCHMARK, MARKET_TZ, Bar, Prices, YahooPrices, fetch

log = logging.getLogger(__name__)

DEFAULT_CONFIG = REPO_ROOT / "config" / "score.yaml"
SCORES_DIR = "scores"
DAY_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
SESSION_OPENS = clock(9, 30)
CALLS = ("Buy", "Hold", "Sell", "Short")
METHODS = {"jev": "Jev", "opus": "Opus"}
MIN_RANKED = 5  # fewer stocks than this on a day: no rank correlation for that day
TOP_LEVEL = {
    "version",
    "sessions",
    "weights",
    "sell_weights_shown",
    "weeks_first_verdict",
    "range",
    "draws",
    "seed",
    "no_price_after",
}

PriceFetch = Callable[[list[str], str], Prices]


@dataclass(frozen=True)
class ScoreConfig:
    version: str
    sha256: str
    sessions: int
    weights: dict[str, float]
    sell_weights_shown: tuple[float, ...]
    weeks_first_verdict: int
    range: float
    draws: int
    seed: int
    no_price_after: int


def _int(raw: Any, name: str, least: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < least:
        raise ValueError(f"score {name} must be a whole number of at least {least}")
    return raw


def _float(raw: Any, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        raise ValueError(f"score {name} must be a number")
    return float(raw)


def load_config(path: Path = DEFAULT_CONFIG) -> ScoreConfig:
    data = path.read_bytes()
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ValueError(f"score file unreadable ({type(exc).__name__})") from exc
    if not isinstance(raw, dict) or set(raw) != TOP_LEVEL:
        raise ValueError(f"score file must hold exactly {sorted(TOP_LEVEL)}")
    if not isinstance(raw["version"], str) or not raw["version"]:
        raise ValueError("score version must be a string")
    weights = raw["weights"]
    if not isinstance(weights, dict) or set(weights) != set(CALLS):
        raise ValueError(f"score weights must be exactly {list(CALLS)}")
    w = {k: _float(v, f"weights.{k}") for k, v in weights.items()}
    if not (w["Buy"] > 0 and w["Hold"] == 0 and w["Short"] <= w["Sell"] <= 0):
        raise ValueError("score weights must be Buy > 0, Hold 0, Short <= Sell <= 0")
    shown = raw["sell_weights_shown"]
    if not isinstance(shown, list) or not shown:
        raise ValueError("score sell_weights_shown must be a list")
    level = _float(raw["range"], "range")
    if not 0.5 <= level < 1:
        raise ValueError("score range must be from 0.5 to under 1")
    return ScoreConfig(
        version=raw["version"],
        sha256=hashlib.sha256(data).hexdigest(),
        sessions=_int(raw["sessions"], "sessions", 1),
        weights=w,
        sell_weights_shown=tuple(_float(x, "sell_weights_shown") for x in shown),
        weeks_first_verdict=_int(raw["weeks_first_verdict"], "weeks_first_verdict", 1),
        range=level,
        draws=_int(raw["draws"], "draws", 100),
        seed=_int(raw["seed"], "seed", 0),
        no_price_after=_int(raw["no_price_after"], "no_price_after", 1),
    )


# --- the market calendar ---------------------------------------------------


def _closed(year: int) -> Any:
    return holidays.financial_holidays("NYSE", years=range(year - 1, year + 2))


def is_session(day: date) -> bool:
    return day.weekday() < 5 and day not in _closed(day.year)


def next_session(day: date) -> date:
    day += timedelta(days=1)
    while not is_session(day):
        day += timedelta(days=1)
    return day


def last_session_of_week(day: date) -> bool:
    """True when no later session falls in `day`'s week (Monday to Friday)."""
    return next_session(day).isocalendar()[:2] != day.isocalendar()[:2]


def _opens_at(day: date) -> datetime:
    return datetime.combine(day, SESSION_OPENS, tzinfo=MARKET_TZ)


def planned_entry(made_at: datetime) -> date:
    """The first session that opens after the call was made, by the calendar."""
    day = made_at.astimezone(MARKET_TZ).date()
    if not (is_session(day) and _opens_at(day) > made_at):
        day = next_session(day)
    return day


def planned_exit(entry: date, sessions: int) -> date:
    day = entry
    for _ in range(sessions - 1):
        day = next_session(day)
    return day


# --- the calls and their outcomes -------------------------------------------


@dataclass(frozen=True)
class Made:
    """One stock on one day: both methods' calls and what code knew then."""

    day: date
    made_at: datetime
    ticker: str
    jev: str | None
    jev_score: float | None  # the rules' score behind Jev's call
    opus: str | None
    opus_confidence: float | None
    momentum: float | None  # the stock's excess return over QQQ in the 20 sessions before
    volatility: float | None  # its annualised volatility over those sessions
    versions: dict[str, str]


def _number(pack: Any, wanted: str) -> float | None:
    for n in (pack or {}).get("numbers") or []:
        if n.get("id") == wanted:
            v = n.get("value")
            return float(v) if isinstance(v, int | float) and math.isfinite(v) else None
    return None


def _call(raw: Any) -> str | None:
    return raw if raw in CALLS else None


def calls_from(record: dict[str, Any]) -> list[Made]:
    day = date.fromisoformat(record["day"])
    made_at = datetime.fromisoformat(record["made_at"])
    versions = {k: str(v) for k, v in (record.get("versions") or {}).items()}
    out = []
    for s in record.get("stocks") or []:
        call = s.get("call") or {}
        opus = s.get("opus") or {}
        jev, own = _call(call.get("call")), _call(opus.get("call"))
        if jev is None and own is None:
            continue
        t = s["ticker"]
        out.append(
            Made(
                day=day,
                made_at=made_at,
                ticker=t,
                jev=jev,
                jev_score=call.get("score") if jev else None,
                opus=own,
                opus_confidence=opus.get("confidence") if own else None,
                momentum=_number(s.get("pack"), f"px:{t}:excess_20d"),
                volatility=_number(s.get("pack"), f"px:{t}:vol_20d"),
                versions=versions,
            )
        )
    return out


def daily_records(runs: Path) -> list[dict[str, Any]]:
    """Every full, finished daily record, oldest first. Check runs never count."""
    out = []
    for path in sorted((runs / "daily").glob("*.json")):
        if not DAY_FILE.match(path.name):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("%s unreadable (%s): not scored", path.name, type(exc).__name__)
            continue
        if (
            isinstance(record, dict)
            and record.get("kind") == "daily"
            and record.get("check_id") is None
            and record.get("complete") is True
        ):
            out.append(record)
    return out


@dataclass(frozen=True)
class Outcome:
    status: str  # "scored", "unscored" or "open"
    reason: str
    entry_day: date | None
    exit_day: date | None
    excess: float | None


def _key(day: date | str, ticker: str) -> str:
    return f"{day}:{ticker}"


def outcome(made: Made, bars: dict[str, tuple[Bar, ...]], cfg: ScoreConfig) -> Outcome:
    """Where one call stands: scored, open (not 20 sessions yet), or unscored."""
    bench = bars.get(BENCHMARK) or ()
    first = next((i for i, b in enumerate(bench) if _opens_at(b.day) > made.made_at), None)
    if first is None:
        return Outcome("open", "not entered yet", None, None, None)
    last = first + cfg.sessions - 1
    entry_day = bench[first].day
    if last >= len(bench):
        return Outcome("open", "not 20 sessions yet", entry_day, None, None)
    exit_day = bench[last].day
    mine = {b.day: b for b in bars.get(made.ticker) or ()}
    s_in, s_out = mine.get(entry_day), mine.get(exit_day)
    if s_in is None or s_out is None:
        if len(bench) - 1 >= last + cfg.no_price_after:
            return Outcome("unscored", "no price", entry_day, exit_day, None)
        return Outcome("open", "waiting for the stock's price", entry_day, exit_day, None)
    q_in, q_out = bench[first], bench[last]
    if s_in.open is None or q_in.open is None:
        return Outcome("unscored", "no open price", entry_day, exit_day, None)
    excess = (s_out.close / s_in.open - 1) - (q_out.close / q_in.open - 1)
    return Outcome("scored", "", entry_day, exit_day, excess)


def _row(made: Made, out: Outcome, cfg: ScoreConfig) -> dict[str, Any]:
    row = asdict(made)
    row |= {
        "day": made.day.isoformat(),
        "made_at": made.made_at.isoformat(),
        "status": out.status,
        "reason": out.reason,
        "entry_day": out.entry_day.isoformat() if out.entry_day else None,
        "exit_day": out.exit_day.isoformat() if out.exit_day else None,
        "excess": out.excess,
        "score_version": cfg.version,
    }
    return row


def ledger_path(runs: Path, cfg: ScoreConfig) -> Path:
    return runs / "daily" / SCORES_DIR / f"{cfg.version}.json"


def read_ledger(path: Path, cfg: ScoreConfig) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != cfg.version or data.get("sha256") != cfg.sha256:
        raise ValueError(f"{path.name} was written under other score settings")
    rows = data.get("rows")
    if not isinstance(rows, dict):
        raise ValueError(f"{path.name} has no rows")
    return rows


def write_ledger(path: Path, rows: dict[str, dict[str, Any]], cfg: ScoreConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    body = {"version": cfg.version, "sha256": cfg.sha256, "rows": dict(sorted(rows.items()))}
    tmp.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def period_for(oldest: date, today: date) -> str:
    """A Yahoo period long enough to reach back to `oldest`'s entry."""
    days = (today - oldest).days
    for span, period in ((80, "6mo"), (170, "1y"), (350, "2y"), (1080, "5y")):
        if days < span:
            return period
    return "max"


def yahoo(tickers: list[str], period: str) -> Prices:
    return fetch(tickers, YahooPrices(), period=period)


def update(
    runs: Path,
    cfg: ScoreConfig,
    today: date,
    prices: PriceFetch = yahoo,
    *,
    save: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """(finished rows, open rows, rows newly finished). Finished rows come from
    the ledger, plus any call that has finished since, which `save` adds to it."""
    path = ledger_path(runs, cfg)
    ledger = read_ledger(path, cfg)
    pending = [m for r in daily_records(runs) for m in calls_from(r)]
    pending = [m for m in pending if _key(m.day, m.ticker) not in ledger]
    opened: list[dict[str, Any]] = []
    new = 0
    due = [m for m in pending if planned_exit(planned_entry(m.made_at), cfg.sessions) <= today]
    bars: dict[str, tuple[Bar, ...]] = {}
    if due:
        tickers = sorted({m.ticker for m in due})
        got = prices(tickers, period_for(min(m.day for m in due), today))
        bars = got.bars
    for m in pending:
        out = outcome(m, bars, cfg) if m in due else Outcome("open", "", None, None, None)
        row = _row(m, out, cfg)
        if out.status == "open":
            entry = planned_entry(m.made_at)
            row["exit_day"] = row["exit_day"] or planned_exit(entry, cfg.sessions).isoformat()
            opened.append(row)
        else:
            ledger[_key(m.day, m.ticker)] = row
            new += 1
    if new and save:
        write_ledger(path, ledger, cfg)
    return list(ledger.values()), opened, new


# --- the numbers -------------------------------------------------------------


def score(call: str | None, excess: float, weights: dict[str, float]) -> float | None:
    return None if call is None else weights[call] * excess


def _weights(cfg: ScoreConfig, sell: float | None = None) -> dict[str, float]:
    return cfg.weights if sell is None else cfg.weights | {"Sell": -sell}


def _mean(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(_ranks(xs), _ranks(ys))


def _signed(row: dict[str, Any], method: str, cfg: ScoreConfig) -> float | None:
    """How strongly a method leaned, for ranking: Jev's rules score, or Opus's
    call weight times its confidence."""
    if method == "jev":
        v = row.get("jev_score")
        return float(v) if isinstance(v, int | float) else None
    call, conf = row.get("opus"), row.get("opus_confidence")
    if call is None or not isinstance(conf, int | float):
        return None
    return cfg.weights[call] * float(conf)


def rank_ic(rows: Sequence[dict[str, Any]], method: str, cfg: ScoreConfig) -> float | None:
    """Mean over days of the rank correlation between how a method leaned and
    the excess return that followed: does it order the stocks well?"""
    by_day: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        v = _signed(r, method, cfg)
        if v is not None:
            by_day.setdefault(r["day"], []).append((v, r["excess"]))
    ics = []
    for pairs in by_day.values():
        if len(pairs) >= MIN_RANKED:
            ic = spearman([a for a, _ in pairs], [b for _, b in pairs])
            if ic is not None:
                ics.append(ic)
    return _mean(ics)


def method_numbers(rows: Sequence[dict[str, Any]], method: str, cfg: ScoreConfig) -> dict[str, Any]:
    mine = [r for r in rows if r.get(method)]
    scores = [score(r[method], r["excess"], cfg.weights) or 0.0 for r in mine]
    bets = [s for r, s in zip(mine, scores, strict=True) if r[method] != "Hold"]
    scaled = [
        s / (r["volatility"] * math.sqrt(cfg.sessions / 252))
        for r, s in zip(mine, scores, strict=True)
        if isinstance(r.get("volatility"), int | float) and r["volatility"] > 0
    ]
    by_sell = {
        f"{w:g}": _mean([score(r[method], r["excess"], _weights(cfg, w)) or 0.0 for r in mine])
        for w in cfg.sell_weights_shown
    }
    # each day's calls given to that day's stocks at random, on average
    shuffled = []
    by_day: dict[str, list[dict[str, Any]]] = {}
    for r in mine:
        by_day.setdefault(r["day"], []).append(r)
    for day_rows in by_day.values():
        w = statistics.fmean(cfg.weights[r[method]] for r in day_rows)
        shuffled += [w * statistics.fmean(r["excess"] for r in day_rows)] * len(day_rows)
    return {
        "calls": len(mine),
        "bets": len(bets),
        "mean": _mean(scores),
        "hit_rate": (sum(1 for s in bets if s > 0) / len(bets)) if bets else None,
        "mean_scaled": _mean(scaled),
        "by_sell_weight": by_sell,
        "rank_ic": rank_ic(mine, method, cfg),
        "shuffled": _mean(shuffled),
        "counts": {c: sum(1 for r in mine if r[method] == c) for c in CALLS},
    }


def _week(day: str) -> str:
    y, w, _ = date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def paired(rows: Sequence[dict[str, Any]], cfg: ScoreConfig) -> dict[str, Any]:
    """Jev less Opus on the stocks and days both made a call, with a range from
    resampling whole weeks (calls in the same week move together)."""
    weeks: dict[str, list[float]] = {}
    for r in rows:
        if r.get("jev") and r.get("opus"):
            d = cfg.weights[r["jev"]] * r["excess"] - cfg.weights[r["opus"]] * r["excess"]
            weeks.setdefault(_week(r["day"]), []).append(d)
    diffs = [d for ds in weeks.values() for d in ds]
    out: dict[str, Any] = {"pairs": len(diffs), "weeks": len(weeks), "mean": _mean(diffs)}
    out["low"] = out["high"] = None
    if len(weeks) >= 2:
        rng = random.Random(cfg.seed)
        blocks = list(weeks.values())
        means = []
        for _ in range(cfg.draws):
            picked = [d for _ in blocks for d in rng.choice(blocks)]
            means.append(statistics.fmean(picked))
        means.sort()
        tail = (1 - cfg.range) / 2
        out["low"] = means[int(tail * (len(means) - 1))]
        out["high"] = means[int((1 - tail) * (len(means) - 1))]
    if len(weeks) < cfg.weeks_first_verdict:
        out["verdict"] = (
            f"too early to tell ({len(weeks)} of {cfg.weeks_first_verdict} weeks scored)"
        )
    elif out["low"] is not None and out["low"] > 0:
        out["verdict"] = "Jev ahead"
    elif out["high"] is not None and out["high"] < 0:
        out["verdict"] = "Opus ahead"
    else:
        out["verdict"] = "no clear difference yet"
    return out


def disagreements(rows: Sequence[dict[str, Any]], cfg: ScoreConfig) -> dict[str, Any]:
    differ = [r for r in rows if r.get("jev") and r.get("opus") and r["jev"] != r["opus"]]
    return {
        "count": len(differ),
        "jev": _mean([cfg.weights[r["jev"]] * r["excess"] for r in differ]),
        "opus": _mean([cfg.weights[r["opus"]] * r["excess"] for r in differ]),
    }


def baselines(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Simple rules every method must beat, on the same stocks and days."""
    excess = [r["excess"] for r in rows]
    momentum = [
        r["excess"] if r["momentum"] > 0 else 0.0
        for r in rows
        if isinstance(r.get("momentum"), int | float)
    ]
    return {
        "always_hold": 0.0 if rows else None,
        "always_buy": _mean(excess),
        "last_months_winners": _mean(momentum),
        "beat_qqq": (sum(1 for x in excess if x > 0) / len(excess)) if excess else None,
    }


def _costs(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    jev_cost = 0.0
    calls = whys = 0
    days = 0
    for r in records:
        days += 1
        jev_cost += float(r.get("jev_cost") or 0.0)
        for s in r.get("stocks") or []:
            calls += 1 if (s.get("opus") or {}).get("agent") else 0
            whys += 1 if (s.get("why") or {}).get("agent") else 0
    return {"days": days, "jev_usd": jev_cost, "opus_calls": calls, "opus_whys": whys}


def scorecard(
    runs: Path,
    today: date,
    cfg: ScoreConfig | None = None,
    prices: PriceFetch = yahoo,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """The scorecard as of `today`, JSON-safe. With `save`, newly finished calls
    go into the ledger first."""
    cfg = cfg or load_config()
    done, opened, new = update(runs, cfg, today, prices, save=save)
    scored = [r for r in done if r["status"] == "scored"]
    week_records = [
        r
        for r in daily_records(runs)
        if date.fromisoformat(r["day"]).isocalendar()[:2] == today.isocalendar()[:2]
    ]
    versions: dict[str, int] = {}
    for r in scored:
        v = (r.get("versions") or {}).get("call_prompt", "?")
        versions[v] = versions.get(v, 0) + 1
    return {
        "as_of": today.isoformat(),
        "version": cfg.version,
        "sha256": cfg.sha256,
        "scored": len(scored),
        "unscored": [
            {"day": r["day"], "ticker": r["ticker"], "reason": r["reason"]}
            for r in done
            if r["status"] == "unscored"
        ],
        "newly_scored": new,
        "open": len(opened),
        "next_due": min((r["exit_day"] for r in opened), default=None),
        "methods": {m: method_numbers(scored, m, cfg) for m in METHODS},
        "paired": paired(scored, cfg),
        "disagreements": disagreements(scored, cfg),
        "baselines": baselines(scored),
        "call_prompts": versions,
        "this_week": _costs(week_records),
    }


# --- words ---------------------------------------------------------------------


def _pct(x: Any) -> str:
    return "n/a" if not isinstance(x, int | float) else f"{x * 100:+.2f}%"


def _share(x: Any) -> str:
    return "n/a" if not isinstance(x, int | float) else f"{x:.0%}"


def _ic(x: Any) -> str:
    return "n/a" if not isinstance(x, int | float) else f"{x:+.2f}"


def lines(card: dict[str, Any]) -> list[str]:
    """The scorecard in plain words; the verdict comes first."""
    p = card["paired"]
    out = [f"Verdict, Jev against Opus: {p['verdict']}."]
    if p["pairs"]:
        rng = (
            f", likely range {_pct(p['low'])} to {_pct(p['high'])}" if p["low"] is not None else ""
        )
        out.append(
            f"On the {p['pairs']} calls both made over {p['weeks']} week(s), Jev scored "
            f"{_pct(p['mean'])} a call more than Opus{rng}."
        )
    if card["open"]:
        out.append(
            f"{card['open']} call(s) still open; the next are scored after the close on "
            f"{card['next_due']}."
        )
    if not card["scored"]:
        out.append("No call has finished its 20 sessions yet.")
    else:
        out.append(f"Scored so far: {card['scored']} stock-days.")
        for m, name in METHODS.items():
            n = card["methods"][m]
            if not n["calls"]:
                continue
            sell = ", ".join(f"{w} {_pct(v)}" for w, v in n["by_sell_weight"].items())
            out.append(
                f"{name}: mean {_pct(n['mean'])} a call over {n['calls']} "
                f"(right on {_share(n['hit_rate'])} of {n['bets']} non-Hold calls); "
                f"ranking {_ic(n['rank_ic'])}; shuffled {_pct(n['shuffled'])}; "
                f"per unit of risk {_ic(n['mean_scaled'])}; with Sell weighted {sell}."
            )
        b = card["baselines"]
        out.append(
            f"Baselines: always Hold {_pct(b['always_hold'])}, always Buy "
            f"{_pct(b['always_buy'])}, buy last month's winners "
            f"{_pct(b['last_months_winners'])}; {_share(b['beat_qqq'])} of stocks beat QQQ."
        )
        d = card["disagreements"]
        if d["count"]:
            out.append(
                f"Where they disagreed ({d['count']}): Jev {_pct(d['jev'])}, "
                f"Opus {_pct(d['opus'])} a call."
            )
        if len(card["call_prompts"]) > 1:
            mix = ", ".join(f"{v} {n}" for v, n in sorted(card["call_prompts"].items()))
            out.append(f"Opus prompts among the scored calls: {mix}.")
    for u in card["unscored"]:
        out.append(f"Not scored: {u['ticker']} from {u['day']} ({u['reason']}).")
    w = card["this_week"]
    out.append(
        f"This week's cost: Jev ${w['jev_usd']:.2f} over {w['days']} day(s); Opus "
        f"{w['opus_calls'] + w['opus_whys']} Claude runs on the subscription "
        f"({w['opus_calls']} calls, {w['opus_whys']} whys)."
    )
    return out


def weekly(runs: Path, day: date, prices: PriceFetch = yahoo) -> dict[str, Any] | None:
    """The scorecard for the daily record of `day` when it is the week's last
    session, else None. Never raises: a failure is the scorecard's error."""
    if not last_session_of_week(day):
        return None
    try:
        card = scorecard(runs, day, prices=prices)
        return card | {"lines": lines(card)}
    except Exception as exc:  # the scorecard must never stop the daily email
        log.warning("scorecard failed (%s)", type(exc).__name__)
        return {"error": f"the scorecard could not be made ({type(exc).__name__})"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print the scorecard of the daily calls")
    ap.add_argument("--runs", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--save", action="store_true", help="add finished calls to the ledger")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    today = datetime.now(UTC).astimezone(MARKET_TZ).date()
    card = scorecard(args.runs, today, save=args.save)
    for line in lines(card):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
