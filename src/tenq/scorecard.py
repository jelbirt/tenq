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

The screen gets two checks of its own, on days it chose the stocks (not on a
watchlist day or a day it failed). Both set each day's stocks only against the
same day's, then average over the days, so a busy day (many stocks pass, all
moving a lot) cannot pass for skill. Check 1, do the stocks that pass it move
more: every screened stock gets the same 20-session excess as a call made that
day, kept in a second ledger, runs/daily/scores/<version>-screen.json, and the
size of the move (up or down) of the stocks that passed is set against that of
the stocks that did not (the random draw and those Hold by screen). Its luck
range comes from shuffling: each stock's run of screen results is handed to
another stock at random and the gap measured again, many times. A stock keeps
its own moves (neighbouring days share 19 of their 20 sessions) and each run
of results keeps its own shape, so only the link between the two is broken.
Check 2, are the calls better on the screen's picks: each method's mean score
on the picks against its mean on the random draw, with the same daily-move
luck range as Jev against Opus; a day with no random draw sits out. Stocks that
could not be screened belong to neither side.

Only full daily records count: `YYYY-MM-DD.json`, kind daily, no check id,
complete. Check runs and consistency runs are never scored. One kind of call
is left out, and the card names each one: an Opus Short asked under a call
prompt before call-v10, which defined Short as "will fall outright" rather
than "will lag QQQ by a wide margin", so it was never a bet on the gap this
scores.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
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
# The screen's result key for each group, in the order a ticker is looked up.
# Check 1 sets the stocks that passed against those that did not; check 2 sets
# the calls on the picks against the calls on the random draw.
SCREEN_GROUPS = {"chosen": "picked", "over_cap": "over_cap", "filled": "random", "held": "held"}
PASSED = ("picked", "over_cap")
NOT_PASSED = ("random", "held")
SHUFFLES = 1000  # check 1's luck range: the screen's results reshuffled this many times
MIN_RANKED = 5  # fewer stocks than this on a day: no rank correlation for that day
# Opus is ranked by lean on a day only when this share of its calls kept one: a
# lean is lost mostly when it points against its call, a borderline stock that
# belongs mid-ranking, so ranking the rest alone would flatter Opus
LEAN_SHARE = 0.8
# Opus's call prompts before this version defined Short as "will fall outright",
# not "will lag QQQ by a wide margin" (PR #24), so those Shorts are left out. Only
# Short changed meaning; Jev's rules kept the same settings (rules-v7 to v8).
SHORT_LAGS_QQQ_FROM = 10
CALL_PROMPT = re.compile(r"call-v(\d+)")
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
    group: str | None = None  # the screen's group that day (SCREEN_GROUPS); None: no screen


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


def short_meant_fall(record: dict[str, Any]) -> bool:
    """Whether an Opus Short in this record was asked to mean "will fall
    outright". A prompt version that cannot be read is taken as current."""
    found = CALL_PROMPT.fullmatch(str((record.get("versions") or {}).get("call_prompt", "")))
    return found is not None and int(found.group(1)) < SHORT_LAGS_QQQ_FROM


def left_out(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The Opus calls in this record that are never scored, and why."""
    if not short_meant_fall(record):
        return []
    return [
        {
            "day": record["day"],
            "ticker": s["ticker"],
            "method": "opus",
            "call": "Short",
            "prompt": str(record["versions"]["call_prompt"]),
            "jev": _call((s.get("call") or {}).get("call")),  # still scored
        }
        for s in record.get("stocks") or []
        if _call((s.get("opus") or {}).get("call")) == "Short"
    ]


def screen_groups(record: dict[str, Any]) -> dict[str, str]:
    """Each screened stock's group, on a day the screen chose the stocks. Empty
    on a watchlist day (before the screen, or a day it failed). A stock that
    could not be screened has no group."""
    screen = record.get("screen")
    if not isinstance(screen, dict) or screen.get("used") != "screen":
        return {}
    out: dict[str, str] = {}
    for key, group in SCREEN_GROUPS.items():
        listed = screen.get(key)
        for x in listed if isinstance(listed, list) else []:
            t = x.get("ticker") if isinstance(x, dict) else x
            if isinstance(t, str) and t not in out:
                out[t] = group
    return out


def screened_from(record: dict[str, Any]) -> list[Made]:
    """Every screened stock of the day, called or not, for check 1."""
    groups = screen_groups(record)
    if not groups:
        return []
    day = date.fromisoformat(record["day"])
    made_at = datetime.fromisoformat(record["made_at"])
    versions = {"screen": str(record["screen"].get("version"))}
    return [
        Made(day, made_at, t, None, None, None, None, None, None, None, versions, group)
        for t, group in sorted(groups.items())
    ]


def calls_from(record: dict[str, Any]) -> list[Made]:
    day = date.fromisoformat(record["day"])
    made_at = datetime.fromisoformat(record["made_at"])
    versions = {k: str(v) for k, v in (record.get("versions") or {}).items()}
    old_short = short_meant_fall(record)
    groups = screen_groups(record)
    if groups:  # so the screen checks can judge each screen version on its own
        versions["screen"] = str(record["screen"].get("version"))
    out = []
    for s in record.get("stocks") or []:
        call = s.get("call") or {}
        opus = s.get("opus") or {}
        jev, own = _call(call.get("call")), _call(opus.get("call"))
        if own == "Short" and old_short:
            own = None  # see left_out: the card names it
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
                group=groups.get(t),
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


def write_ledger(
    path: Path, rows: dict[str, dict[str, Any]], cfg: ScoreConfig, *, compact: bool = False
) -> None:
    """The whole ledger, written by swapping in a new file. `compact` puts each
    row on one line, for the screen's ledger of over a hundred rows a day."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    ordered = dict(sorted(rows.items()))
    if compact:
        head = json.dumps({"version": cfg.version, "sha256": cfg.sha256})[:-1]
        each = [f" {json.dumps(k)}: {json.dumps(v)}" for k, v in ordered.items()]
        text = head + ', "rows": {\n' + ",\n".join(each) + "\n}}"
    else:
        body = {"version": cfg.version, "sha256": cfg.sha256, "rows": ordered}
        text = json.dumps(body, indent=1)
    tmp.write_text(text + "\n", encoding="utf-8")
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
    made = [m for r in daily_records(runs) for m in calls_from(r)]
    done, opened, new = _settle(ledger_path(runs, cfg), made, cfg, today, prices, save=save)
    return done, opened, len(new)


Row = dict[str, Any]
# what check 1 keeps of each screened stock: no calls, no daily moves
SCREEN_ROW = (
    "day", "made_at", "ticker", "group", "versions", "status", "reason",
    "entry_day", "exit_day", "excess", "score_version",
)  # fmt: skip


def screen_ledger_path(runs: Path, cfg: ScoreConfig) -> Path:
    return runs / "daily" / SCORES_DIR / f"{cfg.version}-screen.json"


def screen_update(
    runs: Path,
    cfg: ScoreConfig,
    today: date,
    prices: PriceFetch = yahoo,
    *,
    save: bool = True,
) -> tuple[list[Row], list[Row], list[Row]]:
    """Check 1's rows, as `update` but for every screened stock of every
    screened day: (finished rows, open rows, the rows newly finished)."""
    made = [m for r in daily_records(runs) for m in screened_from(r)]
    path = screen_ledger_path(runs, cfg)
    return _settle(path, made, cfg, today, prices, save=save, keep=SCREEN_ROW, compact=True)


def _settle(
    path: Path,
    made: list[Made],
    cfg: ScoreConfig,
    today: date,
    prices: PriceFetch,
    *,
    save: bool,
    keep: Sequence[str] | None = None,
    compact: bool = False,
) -> tuple[list[Row], list[Row], list[Row]]:
    """One ledger brought up to `today`: prices are fetched only for what is
    due, and a finished row is written once (with `save`). `keep` names the
    fields a row keeps; None keeps them all."""
    ledger = read_ledger(path, cfg)
    pending = [m for m in made if _key(m.day, m.ticker) not in ledger]
    exits = [planned_exit(planned_entry(m.made_at), cfg.sessions) for m in pending]
    due = [m for m, x in zip(pending, exits, strict=True) if x <= today]
    bars: dict[str, tuple[Bar, ...]] = {}
    if due:
        tickers = sorted({m.ticker for m in due})
        bars = prices(tickers, period_for(min(m.day for m in due), today)).bars
    opened: list[Row] = []
    new: list[Row] = []
    for m, x in zip(pending, exits, strict=True):
        out = outcome(m, bars, cfg) if x <= today else Outcome("open", "", None, None, None)
        row = _row(m, out, cfg)
        if keep is not None:
            row = {k: row[k] for k in keep}
        if out.status == "open":
            row["exit_day"] = row["exit_day"] or x.isoformat()
            opened.append(row)
        else:
            ledger[_key(m.day, m.ticker)] = row
            new.append(row)
    if new and save:
        write_ledger(path, ledger, cfg, compact=compact)
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
    where at least MIN_RANKED Opus calls, and at least LEAN_SHARE of them,
    carry a lean, Opus is ranked by lean over those calls only; otherwise by
    call weight times confidence over all its calls, so one day never mixes the
    two. Also says on how many of the ranked days the lean was used."""
    days: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        days.setdefault(r["day"], []).append(r)
    ics = []
    lean_days = 0
    for day_rows in days.values():
        made = [r for r in day_rows if r.get("opus")]
        leaned = [r for r in made if _finite(r.get("opus_lean")) is not None]
        need = max(MIN_RANKED, math.ceil(LEAN_SHARE * len(made)))
        lean = method == "opus" and len(leaned) >= need
        ic = _day_ic(leaned if lean else day_rows, method, cfg, lean)
        if ic is None and lean:
            # leans all tied (or too few distinct): the day is still ranked, by
            # call weight times confidence over all its calls
            lean = False
            ic = _day_ic(day_rows, method, cfg, lean)
        if ic is not None:
            ics.append(ic)
            lean_days += lean
    return {"ic": _mean(ics), "lean_days": lean_days, "days": len(ics)}


def _day_ic(
    rows: Sequence[dict[str, Any]], method: str, cfg: ScoreConfig, lean: bool
) -> float | None:
    pairs = [(v, r["excess"]) for r in rows if (v := _signed(r, method, cfg, lean)) is not None]
    if len(pairs) < MIN_RANKED:
        return None
    return spearman([a for a, _ in pairs], [b for _, b in pairs])


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


def standard_error(by_day: Sequence[float], pairs: int, carry: int) -> float | None:
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
    se = standard_error(by_day, len(pairs), cfg.carry_days)
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


def _verdict(gap: float, se: float | None, luck: float | None, weeks: int, cfg: ScoreConfig) -> Row:
    """Whether a gap between two groups is settled, as for Jev against Opus:
    `luck` is the gap luck alone stays within 9 times in 10, `se` its standard
    error. A gap that rounds to 0.00% on the card never settles."""
    level = round(abs(gap) * 100, 2) == 0
    bar = cfg.settled_bar * se if se else None
    settled = bool(bar and not level and weeks >= cfg.weeks_first_verdict and abs(gap) >= bar)
    return {"gap": gap, "weeks": weeks, "se": se, "luck": luck, "bar": bar, "settled": settled}


def _weeks(days: Iterable[str]) -> int:
    return len({_week(d) for d in days})


def _size_gap(
    sizes: Sequence[Sequence[tuple[int, float]]],
    passed: Sequence[Sequence[bool | None]],
    order: Sequence[int],
) -> tuple[float, float, int, int, list[int]] | None:
    """Check 1 on one way of handing out the screen's results: stock s takes
    the results of stock order[s]. Each day, the mean size of the stocks that
    passed less that of those that did not; (mean passed, mean not passed,
    stock-days passed, stock-days not passed, days used), averaged over the
    days that had both. None if no day had both."""
    yes_means: list[float] = []
    no_means: list[float] = []
    n_yes = n_no = 0
    used: list[int] = []
    for d, day in enumerate(sizes):
        yes = no = 0.0
        cy = cn = 0
        for s, x in day:
            got = passed[order[s]][d]
            if got is None:
                continue
            if got:
                yes, cy = yes + x, cy + 1
            else:
                no, cn = no + x, cn + 1
        if cy and cn:
            yes_means.append(yes / cy)
            no_means.append(no / cn)
            n_yes, n_no = n_yes + cy, n_no + cn
            used.append(d)
    if not used:
        return None
    return statistics.fmean(yes_means), statistics.fmean(no_means), n_yes, n_no, used


def moved(rows: Sequence[Row], cfg: ScoreConfig, shuffles: int = SHUFFLES) -> Row:
    """Check 1: the size of the move against QQQ, up or down, of the stocks
    that passed the screen against those that did not, each day's stocks set
    only against that day's, averaged over the days. The luck range is the gap
    9 shuffles in 10 stay within, when each stock's run of screen results is
    handed to another stock at random (seeded, so a card is the same when made
    again); the standard error is the shuffled gaps' spread around zero."""
    scored = [r for r in rows if r["status"] == "scored" and r.get("group") in PASSED + NOT_PASSED]
    tickers = sorted({r["ticker"] for r in scored})
    days = sorted({r["day"] for r in scored})
    at = {t: i for i, t in enumerate(tickers)}
    on = {d: i for i, d in enumerate(days)}
    passed: list[list[bool | None]] = [[None] * len(days) for _ in tickers]
    sizes: list[list[tuple[int, float]]] = [[] for _ in days]
    for r in scored:
        s, d = at[r["ticker"]], on[r["day"]]
        passed[s][d] = r["group"] in PASSED
        sizes[d].append((s, abs(r["excess"])))
    out: Row = {
        "passed": 0,
        "not_passed": 0,
        "days": 0,
        "passed_mean": None,
        "not_passed_mean": None,
    }
    real = _size_gap(sizes, passed, range(len(tickers)))
    if real is None:
        return out
    yes, no, n_yes, n_no, used = real
    out |= {"passed": n_yes, "not_passed": n_no, "days": len(used)}
    out |= {"passed_mean": yes, "not_passed_mean": no}
    rng = random.Random(f"{cfg.version}:check-1")
    gaps: list[float] = []
    order = list(range(len(tickers)))
    for _ in range(shuffles):
        rng.shuffle(order)
        got = _size_gap(sizes, passed, order)
        if got is not None:
            gaps.append(got[0] - got[1])
    se = math.sqrt(statistics.fmean(g * g for g in gaps)) if gaps else None
    luck = None
    if se:
        wide = sorted(abs(g) for g in gaps)
        luck = wide[math.ceil(0.9 * len(wide)) - 1]
    return out | _verdict(yes - no, se, luck, _weeks(days[d] for d in used), cfg)


def on_picks(rows: Sequence[Row], cfg: ScoreConfig) -> dict[str, Row]:
    """Check 2: each method's mean score a call on the screen's picks against
    its mean on the random draw, each day's picks set only against that day's
    draw, averaged over the days that had both (a day with no draw, or no
    picks, sits out).
    The luck range comes from the daily moves, as for Jev against Opus."""
    w = cfg.weights
    out: dict[str, Row] = {}
    for m in METHODS:
        days: dict[str, tuple[list[Row], list[Row]]] = {}
        for r in rows:
            if (
                r["status"] != "scored"
                or not r.get(m)
                or r.get("group") not in ("picked", "random")
            ):
                continue
            picks, draw = days.setdefault(r["day"], ([], []))
            (picks if r["group"] == "picked" else draw).append(r)
        used = {d: pd for d, pd in sorted(days.items()) if pd[0] and pd[1]}

        def mean(rs: list[Row], m: str = m) -> float:
            return statistics.fmean(w[r[m]] * r["excess"] for r in rs)

        got: Row = {
            "picks": sum(len(p) for p, _ in used.values()),
            "draw": sum(len(d) for _, d in used.values()),
            "days": len(used),
            "no_draw": sum(1 for p, d in days.values() if p and not d),
            "no_picks": sum(1 for p, d in days.values() if d and not p),
            "picks_mean": statistics.fmean(mean(p) for p, _ in used.values()) if used else None,
            "draw_mean": statistics.fmean(mean(d) for _, d in used.values()) if used else None,
        }
        if used:
            n = len(used)
            parts = [
                (sign * w[r[m]] / (len(rs) * n), r)
                for pd in used.values()
                for sign, rs in ((1, pd[0]), (-1, pd[1]))
                for r in rs
            ]
            totals = list(day_totals(parts).values())
            se = standard_error(totals, 1, cfg.carry_days)
            luck = None if not se else t95(len(totals) - 1) * se
            gap = got["picks_mean"] - got["draw_mean"]
            got |= _verdict(gap, se, luck, _weeks(used), cfg)
        out[m] = got
    return out


def screen_version(row: Row) -> str | None:
    return (row.get("versions") or {}).get("screen")


def current_screen(records: Iterable[dict[str, Any]]) -> str | None:
    """The screen version of the newest screened day: the one the checks judge."""
    days = [(r["day"], str(r["screen"].get("version"))) for r in records if screen_groups(r)]
    return max(days)[1] if days else None


def screen_check(
    runs: Path,
    today: date,
    cfg: ScoreConfig,
    calls: Sequence[Row],
    prices: PriceFetch,
    *,
    save: bool,
    current: str | None = None,
) -> Row:
    """Both screen checks, on `current`'s days only: a new screen picks for new
    reasons, so its days are not pooled with an older one's. Check 1 needs its
    own price fetch; if that fails, check 2 (from the scored calls) still stands."""

    def mine(rows: Iterable[Row]) -> list[Row]:
        return [r for r in rows if current is None or screen_version(r) == current]

    def older(rows: Iterable[Row]) -> dict[str, int]:
        count: dict[str, int] = {}
        for r in rows:
            v = screen_version(r)
            if current is not None and v is not None and v != current:
                count[v] = count.get(v, 0) + 1
        return dict(sorted(count.items()))

    out: Row = {"version": current, "calls": on_picks(mine(calls), cfg)}
    try:
        all_done, all_opened, all_new = screen_update(runs, cfg, today, prices, save=save)
    except Exception as exc:  # the call scorecard must not go down with it
        log.warning("screen check failed (%s)", type(exc).__name__)
        error = f"check 1 could not be made ({type(exc).__name__})"
        return out | {"error": error, "earlier_calls": older(calls)}
    done, opened, new = mine(all_done), mine(all_opened), mine(all_new)
    earlier = older([*all_done, *all_opened])
    held = [r for r in new if r["status"] == "scored" and r["group"] == "held"]
    held.sort(key=lambda r: (-abs(r["excess"]), r["day"], r["ticker"]))
    return out | {
        "moves": moved(done, cfg),
        "finished": len(done),
        "open": len(opened),
        "waiting": sum(1 for r in opened if r["exit_day"] <= str(today)),
        "next_due": min(
            (r["exit_day"] for r in opened if r["exit_day"] > str(today)), default=None
        ),
        "unscored": sum(1 for r in done if r["status"] == "unscored"),
        "biggest_held": [
            {"day": r["day"], "ticker": r["ticker"], "excess": r["excess"]} for r in held[:3]
        ],
        "earlier": earlier,
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
    records = daily_records(runs)
    week_records = [
        r
        for r in records
        if date.fromisoformat(r["day"]).isocalendar()[:2] == today.isocalendar()[:2]
    ]
    # Which prompt made each scored Opus call, and which rules made each scored
    # Jev call, so a mix of versions is named rather than read as one method
    versions: dict[str, int] = {}
    rules: dict[str, int] = {}
    for r in scored:
        made = r.get("versions") or {}
        if r.get("opus"):
            v = str(made.get("call_prompt", "?"))
            versions[v] = versions.get(v, 0) + 1
        if r.get("jev"):
            v = str(made.get("rules", "?"))
            rules[v] = rules.get(v, 0) + 1
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
        "jev_rules": rules,
        "left_out": [x for r in records for x in left_out(r)],
        "screen": screen_check(
            runs, today, cfg, scored, prices, save=save, current=current_screen(records)
        ),
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


def _size(x: Any) -> str:
    return "n/a" if not isinstance(x, int | float) else f"{x * 100:.2f}%"


def _luck_lines(g: Row, first: int, name: str, more: str, less: str) -> list[str]:
    """The gap (`name` says which way round), whether it is settled and, if
    not, the range luck alone gives, as for Jev against Opus; `more` and `less`
    say what a settled gap above or below zero means."""
    head = f"Gap ({name}): {_pct(g['gap'])}."
    if g["settled"]:
        return [
            f"{head} Settled: {more if g['gap'] > 0 else less} (a gap this size is unlikely "
            "to be luck, even counting every weekly check)."
        ]
    early = f" (too early: {g['weeks']} of {first} weeks scored)" if g["weeks"] < first else ""
    if g["luck"] is None:
        return [f"{head} Not settled{early}: too few days scored to size luck yet."]
    luck = g["luck"] * 100
    out = [
        f"{head} Not settled{early}: 9 times in 10, luck alone gives a gap between "
        f"-{luck:.2f}% and +{luck:.2f}%."
    ]
    gap, bar = abs(g["gap"]), g.get("bar")
    if not bar or round(gap * 100, 2) == 0:
        return out
    if early and gap >= bar:
        out.append(
            f"The gap already clears the settle bar of {bar * 100:.2f}%; a gap can settle "
            f"only once {first} weeks are scored."
        )
    elif gap > g["luck"]:
        wait = f", and only once {first} weeks are scored" if early else ""
        out.append(
            f"The gap is outside that range, but since the card is read every week a gap "
            f"settles only at {bar * 100:.2f}%{wait}."
        )
    return out


def _days(n: int) -> str:
    return f"{n} day" if n == 1 else f"{n} days"


def screen_lines(sc: Row, first: int) -> list[str]:
    """The two screen checks in plain words; nothing before the first screened day."""
    out: list[str] = []
    if sc.get("earlier"):
        gone = ", ".join(f"{n} by {v}" for v, n in sc["earlier"].items())
        out.append(
            f"Screen checks are on {sc.get('version')} only; stock-days screened by an "
            f"earlier screen are left out ({gone})."
        )
    if sc.get("earlier_calls"):
        gone = ", ".join(f"{n} by {v}" for v, n in sc["earlier_calls"].items())
        out.append(
            f"Screen check 2 is on {sc.get('version')} only; calls on days screened by an "
            f"earlier screen are left out ({gone})."
        )
    mv = sc.get("moves") or {}
    wait = sc.get("waiting") or 0
    if sc.get("error"):
        out.append(f"Screen check 1 was not made this week: {sc['error']}.")
    elif mv.get("days"):
        out.append(
            "Screen check 1, do the stocks that pass move more: over 20 sessions, up or "
            f"down against QQQ, those that passed moved {_size(mv['passed_mean'])} on "
            f"average ({mv['passed']} stock-days), those that did not "
            f"{_size(mv['not_passed_mean'])} ({mv['not_passed']}), comparing stocks "
            f"screened the same day ({_days(mv['days'])})."
        )
        out += _luck_lines(
            mv,
            first,
            "passed less not passed",
            "stocks that pass the screen move more",
            "they move less",
        )
    elif sc.get("finished"):
        out.append(
            "Screen check 1: no finished day has priced stocks both among those that "
            "passed and those that did not yet."
        )
    elif sc.get("open") and not wait:
        due = f"; the first finishes on {sc['next_due']}" if sc.get("next_due") else ""
        out.append(
            f"Screen checks: no screened stock has finished its 20 sessions yet "
            f"({sc['open']} stock-days open{due})."
        )
    if wait:
        out.append(f"Screen check 1: {wait} stock-day(s) finished but are waiting for a price.")
    for m, name in METHODS.items():
        c = (sc.get("calls") or {}).get(m) or {}
        why = [
            f"{_days(n)} with no {side}"
            for n, side in (
                (c.get("no_draw") or 0, "random draw"),
                (c.get("no_picks") or 0, "picks"),
            )
            if n
        ]
        sat = f" {' and '.join(why)} sat out." if why else ""
        if not c.get("days"):
            if why:
                out.append(
                    f"Screen check 2 for {name}'s calls: no finished day had both picks and "
                    f"a random draw yet.{sat}"
                )
            continue
        out.append(
            f"Screen check 2, are {name}'s calls better on the screen's picks: "
            f"{_pct(c['picks_mean'])} a call on the picks ({c['picks']}), "
            f"{_pct(c['draw_mean'])} on the random draw ({c['draw']}), comparing picks "
            f"and draws from the same day ({_days(c['days'])}).{sat}"
        )
        out += _luck_lines(
            c,
            first,
            "picks less random draw",
            f"{name}'s calls do better on the picks",
            "they do worse",
        )
    if sc.get("unscored"):
        out.append(f"Screen check 1 leaves out {sc['unscored']} stock-day(s) with no price.")
    if sc.get("biggest_held"):
        big = ", ".join(
            f"{x['ticker']} {_pct(x['excess'])} (screened {x['day']})" for x in sc["biggest_held"]
        )
        out.append(f"Biggest moves against QQQ among stocks Hold by screen, newly scored: {big}.")
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
        if len(card.get("jev_rules", {})) > 1:
            mix = ", ".join(f"{v} {n}" for v, n in sorted(card["jev_rules"].items()))
            out.append(f"Jev's rules among the scored calls: {mix}.")
    groups: dict[tuple[str, str], list[str]] = {}
    for u in card["unscored"]:
        groups.setdefault((u["ticker"], u["reason"]), []).append(u["day"])
    for (ticker, reason), days in sorted(groups.items()):
        span = days[0] if len(days) == 1 else f"{min(days)} to {max(days)}"
        out.append(f"Not scored: {ticker}, {len(days)} call(s) from {span} ({reason}).")
    for x in card.get("left_out") or []:
        jev = f" Jev's {x['jev']} on {x['ticker']} that day still counts." if x.get("jev") else ""
        out.append(
            f"Left out: Opus's {x['call']} on {x['ticker']} from {x['day']}. Its prompt then "
            f"({x['prompt']}) said Short meant the stock would fall outright; since "
            f"call-v{SHORT_LAGS_QQQ_FROM}, Short means it will lag QQQ by a wide margin, "
            f"which is what the card scores.{jev}"
        )
    out += screen_lines(card.get("screen") or {}, card.get("weeks_first_verdict") or 13)
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
