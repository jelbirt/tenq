"""The scorecard: each daily recommendation judged by what the stock did next.

    python -m tenq.scorecard            # print the scorecard from the records

A call is entered at the open of the first session that starts after it was
made (`made_at` in the day's record, which can be after that day's close) and
left at the close of the 20th session, counting the entry session. The stock's
return over that span less QQQ's over the same span is the excess return; the
call's score is its weight (config/score.yaml) times that excess. Both prices
come from one fresh fetch, so dividends rescaling past prices cannot skew it.

A finished call is written once to the ledger, runs/daily/scores/<version>.json,
with the excess split into its 20 daily moves (they add up to it exactly), and
never scored again: a later price revision or config change cannot rewrite the
past (a new config version starts a new ledger).

Jev against Opus, on the calls both made: who is ahead so far, every week,
and how big a gap luck alone could give at this point. The luck range adds up
the daily moves of every paired call by calendar day, so overlapping 20-session
windows, calls that stick for weeks and stocks moving together on the same day
are all counted once (Hodrick 1992; Britten-Jones, Neuberger and Nolte 2011),
with a short correction for any carry-over from one day to the next (Newey and
West 1987). "Settled" needs 13 weeks and a gap of `settled_bar` standard errors,
wider than one 90% check because the card is read every week (Armitage,
McPherson and Rowe 1969). Beside them: what holding, buying everything, buying
last month's winners and a shuffled version of each method's own calls would
have scored.

Only full daily records count: `YYYY-MM-DD.json`, kind daily, no check id,
complete. Check runs and consistency runs are never scored.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import logging
import math
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
    "settled_bar",
    "carry_days",
    "no_price_after",
    "no_download_after",
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
    settled_bar: float
    carry_days: int
    no_price_after: int
    no_download_after: int


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
    bar = _float(raw["settled_bar"], "settled_bar")
    if not 1.0 <= bar <= 5.0:
        raise ValueError("score settled_bar must be from 1 to 5 standard errors")
    return ScoreConfig(
        version=raw["version"],
        sha256=hashlib.sha256(data).hexdigest(),
        sessions=_int(raw["sessions"], "sessions", 1),
        weights=w,
        sell_weights_shown=tuple(_float(x, "sell_weights_shown") for x in shown),
        weeks_first_verdict=_int(raw["weeks_first_verdict"], "weeks_first_verdict", 1),
        settled_bar=bar,
        carry_days=_int(raw["carry_days"], "carry_days", 0),
        no_price_after=_int(raw["no_price_after"], "no_price_after", 1),
        no_download_after=_int(raw["no_download_after"], "no_download_after", 1),
    )


# --- the market calendar ---------------------------------------------------


@functools.lru_cache(maxsize=16)
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
    opus_lean: float | None  # -1 to +1 from call-v11 on; None before
    momentum: float | None  # the stock's excess return over QQQ in the 20 sessions before
    volatility: float | None  # its annualised volatility over those sessions
    versions: dict[str, str]


def _number(pack: Any, wanted: str) -> float | None:
    for n in (pack or {}).get("numbers") or []:
        if n.get("id") == wanted:
            v = n.get("value")
            return float(v) if isinstance(v, int | float) and math.isfinite(v) else None
    return None


def _finite(raw: Any) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw) if math.isfinite(raw) else None


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
                opus_lean=_finite(opus.get("lean")) if own else None,
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
    moves: tuple[float, ...] | None = None  # the excess, day by day; they add up to it
    move_days: tuple[date, ...] | None = None  # the QQQ session each move belongs to


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
    past = len(bench) - 1 - last  # sessions QQQ has gone on past the exit
    got = bars.get(made.ticker) or ()
    if not got:  # nothing downloaded: a passing glitch looks the same as a delisting
        if past >= cfg.no_download_after:
            return Outcome("unscored", "no price downloaded", entry_day, exit_day, None)
        return Outcome("open", "waiting for the stock's price", entry_day, exit_day, None)
    mine = {b.day: b for b in got}
    s_in, s_out = mine.get(entry_day), mine.get(exit_day)
    q_in, q_out = bench[first], bench[last]
    if s_in is None or s_out is None or s_in.open is None or q_in.open is None:
        if past >= cfg.no_price_after:
            reason = "no price" if s_in is None or s_out is None else "no open price"
            return Outcome("unscored", reason, entry_day, exit_day, None)
        return Outcome("open", "waiting for the stock's price", entry_day, exit_day, None)
    excess = (s_out.close / s_in.open - 1) - (q_out.close / q_in.open - 1)
    moves = _moves(bench[first : last + 1], mine, s_in.open, q_in.open)
    if not math.isfinite(excess) or not all(math.isfinite(m) for m in moves):
        return Outcome("unscored", "bad price", entry_day, exit_day, None)
    held = tuple(b.day for b in bench[first : last + 1])
    return Outcome("scored", "", entry_day, exit_day, excess, moves, held)


def _moves(
    held: Sequence[Bar], mine: dict[date, Bar], s_open: float, q_open: float
) -> tuple[float, ...]:
    """The excess split by session: each day's change in the stock less QQQ's,
    both over the entry open, so they add up to the excess exactly. A day the
    stock has no bar moves 0 and the next bar carries the change."""
    out = []
    s_prev, q_prev = s_open, q_open
    for b in held:
        s = mine.get(b.day)
        s_now = s.close if s is not None else s_prev
        out.append((s_now - s_prev) / s_open - (b.close - q_prev) / q_open)
        s_prev, q_prev = s_now, b.close
    return tuple(out)


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
        "moves": list(out.moves) if out.moves is not None else None,
        "move_days": [d.isoformat() for d in out.move_days] if out.move_days else None,
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


def _signed(row: dict[str, Any], method: str, cfg: ScoreConfig, lean: bool) -> float | None:
    """How strongly a method leaned, for ranking: Jev's rules score; Opus's own
    lean when `lean`, otherwise its call weight times its confidence."""
    if method == "jev":
        v = row.get("jev_score")
        return float(v) if isinstance(v, int | float) else None
    if lean:
        return _finite(row.get("opus_lean"))
    call, conf = row.get("opus"), row.get("opus_confidence")
    if call is None or not isinstance(conf, int | float):
        return None
    return cfg.weights[call] * float(conf)


def ranking(rows: Sequence[dict[str, Any]], method: str, cfg: ScoreConfig) -> dict[str, Any]:
    """Mean over days of the rank correlation between how a method leaned and
    the excess return that followed: does it order the stocks well? On a day
    where at least MIN_RANKED Opus calls carry a lean, Opus is ranked by lean
    over those calls only; otherwise by call weight times confidence over all
    its calls, so one day never mixes the two. Also says on how many of the
    ranked days the lean was used."""
    days: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        days.setdefault(r["day"], []).append(r)
    ics = []
    lean_days = 0
    for day_rows in days.values():
        leaned = [r for r in day_rows if r.get("opus") and _finite(r.get("opus_lean")) is not None]
        lean = method == "opus" and len(leaned) >= MIN_RANKED
        pairs = [
            (v, r["excess"])
            for r in (leaned if lean else day_rows)
            if (v := _signed(r, method, cfg, lean)) is not None
        ]
        if len(pairs) >= MIN_RANKED:
            ic = spearman([a for a, _ in pairs], [b for _, b in pairs])
            if ic is not None:
                ics.append(ic)
                lean_days += lean
    return {"ic": _mean(ics), "lean_days": lean_days, "days": len(ics)}


def rank_ic(rows: Sequence[dict[str, Any]], method: str, cfg: ScoreConfig) -> float | None:
    """The mean daily rank correlation alone; see `ranking`."""
    ic: float | None = ranking(rows, method, cfg)["ic"]
    return ic


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
        "rank_ic": (rk := ranking(mine, method, cfg))["ic"],
        "ranked_days": rk["days"],
        "lean_days": rk["lean_days"],
        "shuffled": _mean(shuffled),
        "counts": {c: sum(1 for r in mine if r[method] == c) for c in CALLS},
    }


# 0.95 quantiles of Student's t by degrees of freedom: a 90% two-sided range.
T95 = (
    (1, 6.314), (2, 2.920), (3, 2.353), (4, 2.132), (5, 2.015), (6, 1.943), (8, 1.860),
    (10, 1.812), (12, 1.782), (15, 1.753), (20, 1.725), (30, 1.697), (60, 1.671),
    (120, 1.658),
)  # fmt: skip
Z95 = 1.645


def t95(df: int) -> float:
    """Student's t for a 90% range, straight-line between table rows in 1/df."""
    if df < 1:
        raise ValueError("t95 needs at least 1 degree of freedom")
    rows = (*T95, (math.inf, Z95))
    for (d0, t0), (d1, t1) in itertools.pairwise(rows):
        if d0 <= df <= d1:
            if df == d0:
                return t0
            x, x0, x1 = 1 / df, 1 / d0, (0.0 if d1 == math.inf else 1 / d1)
            return t1 + (t0 - t1) * (x - x1) / (x0 - x1)
    return Z95


def _week(day: str) -> str:
    y, w, _ = date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def _by_day(pairs: Sequence[tuple[float, dict[str, Any]]]) -> list[float]:
    """The Jev less Opus gap of every paired call, added up by calendar day:
    a call's weight gap times each of its daily moves, on the QQQ session the
    move was measured on (stored with it, so a later calendar never moves a
    written row). A row without usable moves puts its whole excess on its exit
    day."""
    return list(day_totals(pairs).values())


def day_totals(pairs: Sequence[tuple[float, dict[str, Any]]]) -> dict[date, float]:
    total: dict[date, float] = {}
    for dw, r in pairs:
        moves, days = r.get("moves"), r.get("move_days")
        if (
            isinstance(moves, list)
            and isinstance(days, list)
            and moves
            and len(days) == len(moves)
            and math.isclose(sum(moves), r["excess"], rel_tol=0, abs_tol=1e-9)
        ):
            for d, m in zip(days, moves, strict=True):
                key = date.fromisoformat(d)
                total[key] = total.get(key, 0.0) + dw * m
        else:
            key = date.fromisoformat(r.get("exit_day") or r["day"])
            total[key] = total.get(key, 0.0) + dw * r["excess"]
    return dict(sorted(total.items()))


def _standard_error(by_day: Sequence[float], pairs: int, carry: int) -> float | None:
    """The standard error of the mean gap per call, from the day totals: their
    spread, plus `carry` days of carry-over with Bartlett weights so it can
    never go below zero, scaled by m/(m-1) for a short sample."""
    m = len(by_day)
    if m < 2 or not pairs:
        return None
    bar = statistics.fmean(by_day)
    dev = [x - bar for x in by_day]

    def lag(j: int) -> float:
        return sum(dev[t] * dev[t - j] for t in range(j, m))

    spread = lag(0) + 2 * sum(
        (1 - j / (carry + 1)) * lag(j) for j in range(1, min(carry, m - 1) + 1)
    )
    return math.sqrt(max(spread, 0.0) * m / (m - 1)) / pairs


def paired(rows: Sequence[dict[str, Any]], cfg: ScoreConfig) -> dict[str, Any]:
    """Jev less Opus on the calls both made: the gap so far, who leads, the
    gap luck alone could give (a 90% range around zero) and, once enough weeks
    are scored, whether the lead is settled. A gap that rounds to 0.00% on the
    card has no leader."""
    pairs = [
        (cfg.weights[r["jev"]] - cfg.weights[r["opus"]], r)
        for r in rows
        if r.get("jev") and r.get("opus")
    ]
    weeks = len({_week(r["day"]) for _, r in pairs})
    mean = _mean([dw * r["excess"] for dw, r in pairs])
    by_day = _by_day(pairs)
    se = _standard_error(by_day, len(pairs), cfg.carry_days)
    level = round(abs(mean or 0.0) * 100, 2) == 0
    out: dict[str, Any] = {
        "pairs": len(pairs),
        "differ": sum(1 for dw, _ in pairs if dw),
        "weeks": weeks,
        "days": len(by_day),
        "mean": mean,
        "se": se,
        "luck": None if not se else t95(len(by_day) - 1) * se,
        "bar": None if not se else cfg.settled_bar * se,
        "leader": None if level else ("Jev" if (mean or 0.0) > 0 else "Opus"),
        "settled": None,
    }
    enough = weeks >= cfg.weeks_first_verdict
    if out["leader"] and se and enough and abs(mean or 0.0) >= cfg.settled_bar * se:
        out["settled"] = out["leader"]
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
        "weeks_first_verdict": cfg.weeks_first_verdict,
        "scored": len(scored),
        "unscored": [
            {"day": r["day"], "ticker": r["ticker"], "reason": r["reason"]}
            for r in done
            if r["status"] == "unscored"
        ],
        "newly_scored": new,
        "open": len(opened),
        "next_due": min(
            (r["exit_day"] for r in opened if r["exit_day"] > str(today)), default=None
        ),
        "waiting": sum(1 for r in opened if r["exit_day"] <= str(today)),
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


def _lead_lines(p: dict[str, Any], first: int) -> list[str]:
    """Who is ahead so far (always shown, luck or not), then how settled it is."""
    if not p["pairs"]:
        return ["Ahead so far: no recommendation both made has finished yet."]
    gap = abs(p["mean"] or 0.0)
    if p["leader"]:
        lead = f"{p['leader']}, by {gap * 100:.2f}% a recommendation"
    elif p.get("differ"):
        lead = "neither, level (under 0.01% apart)"
    else:
        lead = "neither, level"
    out = [f"Ahead so far: {lead} over {p['pairs']} pair(s) in {p['weeks']} week(s)."]
    if p["settled"]:
        out.append(
            f"Settled: {p['settled']} ahead (a gap this size is unlikely to be luck, "
            "even counting every weekly check)."
        )
        return out
    early = f" (too early: {p['weeks']} of {first} weeks scored)" if p["weeks"] < first else ""
    if not p.get("differ"):
        out.append("Not settled: the two have not disagreed on a finished call yet.")
    elif p["luck"] is None:
        out.append(f"Not settled{early}: too few days scored to size luck yet.")
    else:
        luck = p["luck"] * 100
        out.append(
            f"Not settled{early}: 9 times in 10, luck alone gives a gap between "
            f"-{luck:.2f}% and +{luck:.2f}% at this point."
        )
        bar = p.get("bar")
        if p["leader"] and bar and early and gap >= bar:
            out.append(
                f"The gap already clears the settle bar of {bar * 100:.2f}%; a lead can "
                f"settle only once {first} weeks are scored."
            )
        elif p["leader"] and bar and gap > p["luck"]:
            wait = f", and only once {first} weeks are scored" if early else ""
            out.append(
                f"The gap is outside that range, but since the card is read every week "
                f"a lead settles only at {bar * 100:.2f}%{wait}."
            )
    return out


def lines(card: dict[str, Any]) -> list[str]:
    """The scorecard in plain words; who leads comes first."""
    p = card["paired"]
    out = _lead_lines(p, card.get("weeks_first_verdict") or 13)
    if card["open"]:
        due = f"; the next finish on {card['next_due']}" if card["next_due"] else ""
        wait = card.get("waiting") or 0
        held = f"; {wait} finished but waiting for a price" if wait else ""
        out.append(
            f"{card['open']} call(s) still open{due}{held}. Finished calls are scored "
            "in the next weekly card."
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
            by_lean = (
                f" (by lean on {n.get('lean_days', 0)} of {n['ranked_days']} days)"
                if m == "opus" and n.get("ranked_days")
                else ""
            )
            out.append(
                f"{name}: mean {_pct(n['mean'])} a call over {n['calls']} "
                f"(right on {_share(n['hit_rate'])} of {n['bets']} non-Hold calls); "
                f"ranking {_ic(n['rank_ic'])}{by_lean}; shuffled {_pct(n['shuffled'])}; "
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
    groups: dict[tuple[str, str], list[str]] = {}
    for u in card["unscored"]:
        groups.setdefault((u["ticker"], u["reason"]), []).append(u["day"])
    for (ticker, reason), days in sorted(groups.items()):
        span = days[0] if len(days) == 1 else f"{min(days)} to {max(days)}"
        out.append(f"Not scored: {ticker}, {len(days)} call(s) from {span} ({reason}).")
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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    today = datetime.now(UTC).astimezone(MARKET_TZ).date()
    card = scorecard(args.runs, today, save=False)  # only the daily workflow writes the ledger
    for line in lines(card):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
