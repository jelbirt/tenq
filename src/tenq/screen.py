"""The daily screen (L6): code alone picks which stocks Jev and Opus see.

A stock passes when its next earnings release, by Yahoo's earnings list
(tenq.earnings), falls in the `earnings_sessions` sessions after the screened
session (config/screen.yaml). A release on the screened day itself does not
count: it is out by the time the calls are made, and the calls' 20 sessions
start at the next open, after its jump.

At most `cap` pass; past it, the soonest release comes first, and stocks
reporting on the same day are taken in an order drawn at random, seeded by the
day. If fewer than `fill_to` pass, the rest of the slots up to `fill_to` go to
stocks drawn at random from the ones left out ("random fill"), seeded by the
day so a rerun draws the same ones. They are a fair sample of what the screen
skips: the scorecard can then check whether the models find more to act on in
screened stocks than in random ones. Every other stock is "Hold by screen". A
stock whose dates could not be read, or that has no upcoming date, is not
screened: neither passed nor held, and never drawn for the fill.

Until screen-v7 a stock passed on a new filing or a big move against QQQ.
Neither picked stocks that moved more than the ones left out over two years of
prices; an earnings date did (see config/screen.yaml). A recent Buy, Sell or
Short is not a reason to pass either.

This module asks no model anything: `python -m tenq.screen` only prints who
would pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from tenq.config import REPO_ROOT, load_watchlist
from tenq.earnings import EarningsSource, YahooEarnings, lookup
from tenq.prices import BENCHMARK, MARKET_TZ, Prices, YahooPrices, fetch
from tenq.scorecard import next_session

log = logging.getLogger(__name__)

DEFAULT_SCREEN = REPO_ROOT / "config" / "screen.yaml"
DEFAULT_UNIVERSE = REPO_ROOT / "config" / "universe.yaml"
TOP_LEVEL = {"version", "earnings_sessions", "cap", "fill_to"}
MAX_FAILED_SHARE = 0.1  # more stocks than this with no usable date: the screen is refused


@dataclass(frozen=True)
class ScreenConfig:
    version: str
    earnings_sessions: int
    cap: int
    fill_to: int
    sha256: str


def load_config(path: Path = DEFAULT_SCREEN) -> ScreenConfig:
    text = path.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or set(raw) != TOP_LEVEL:
        raise ValueError(f"{path}: expected exactly the keys {sorted(TOP_LEVEL)}")
    window, cap, fill = raw["earnings_sessions"], raw["cap"], raw["fill_to"]
    if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 20:
        raise ValueError(f"{path}: 'earnings_sessions' must be a whole number from 1 to 20")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"{path}: 'cap' must be a whole number, at least 1")
    if isinstance(fill, bool) or not isinstance(fill, int) or not 0 <= fill <= cap:
        raise ValueError(f"{path}: 'fill_to' must be a whole number from 0 to 'cap'")
    if not isinstance(raw["version"], str) or not raw["version"]:
        raise ValueError(f"{path}: 'version' must be text")
    return ScreenConfig(
        version=raw["version"],
        earnings_sessions=window,
        cap=cap,
        fill_to=fill,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def load_universe(path: Path = DEFAULT_UNIVERSE) -> tuple[str, ...]:
    """The Nasdaq-100 list plus the watchlist, watchlist first, no repeats."""
    raw = yaml.safe_load(path.read_text())
    ndx = raw.get("nasdaq_100") if isinstance(raw, dict) else None
    if not isinstance(ndx, list) or not all(isinstance(t, str) for t in ndx):
        raise ValueError(f"{path}: 'nasdaq_100' must be a list of strings")
    return tuple(dict.fromkeys([*load_watchlist().tickers, *(t.upper() for t in ndx)]))


@dataclass
class Stock:
    ticker: str
    earnings: str | None = None  # the next release after the session: "2026-11-12"
    due: bool = False  # that release falls in the window
    unchecked: list[str] = field(default_factory=list)  # why it could not be screened

    def reasons(self) -> list[str]:
        return [f"earnings due {self.earnings}"] if self.due else []


def window(day: date, sessions: int) -> tuple[date, date]:
    """The first and the last of the `sessions` sessions after `day`."""
    first = last = next_session(day)
    for _ in range(sessions - 1):
        last = next_session(last)
    return first, last


def next_release(when: Sequence[datetime], day: date) -> date | None:
    """The first release dated after the `day` session, as a New York date."""
    later = (w.astimezone(MARKET_TZ).date() for w in when)
    return min((d for d in later if d > day), default=None)


def random_fill(left: Sequence[str], slots: int, version: str, day: date) -> list[str]:
    """Up to `slots` of the left-out stocks, drawn at random but the same on a
    rerun: the draw is seeded by the screen version and the session day."""
    if slots <= 0 or not left:
        return []
    rng = random.Random(f"{version}:{day.isoformat()}")
    return rng.sample(sorted(left), min(slots, len(left)))


def screen(
    tickers: Sequence[str],
    day: date,
    cfg: ScreenConfig,
    source: EarningsSource,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Who passes for the `day` session. Never raises for one stock: one whose
    dates could not be read is noted, and it is not screened."""
    first, last = window(day, cfg.earnings_sessions)
    found, failed = lookup(tickers, source, sleep=sleep)
    stocks: list[Stock] = []
    for t in tickers:
        st = Stock(t)
        nxt = next_release(found.get(t, []), day)
        if t in failed:
            st.unchecked.append(failed[t])
        elif nxt is None:
            st.unchecked.append("no upcoming earnings date")
        else:
            st.earnings, st.due = nxt.isoformat(), nxt <= last
        stocks.append(st)
    # Past the cap: the soonest release first; a day's ties in a seeded random order.
    passed = sorted((s for s in stocks if s.due), key=lambda s: s.ticker)
    random.Random(f"{cfg.version}:{day.isoformat()}:cap").shuffle(passed)
    passed.sort(key=lambda s: str(s.earnings))
    out = [s for s in stocks if not s.due]
    left = [s.ticker for s in out if not s.unchecked]
    unscreened = [s.ticker for s in out if s.unchecked]
    filled = random_fill(left, cfg.fill_to - min(len(passed), cfg.cap), cfg.version, day)
    return {
        "kind": "screen",
        "version": cfg.version,
        "sha256": cfg.sha256,
        "day": day.isoformat(),
        "due_from": first.isoformat(),
        "due_through": last.isoformat(),
        "universe": len(tickers),
        "cap": cfg.cap,
        "chosen": [_row(s) for s in passed[: cfg.cap]],
        "over_cap": [_row(s) for s in passed[cfg.cap :]],
        "filled": filled,
        "held": [t for t in left if t not in filled],
        "unscreened": unscreened,
        "unchecked": {s.ticker: s.unchecked for s in stocks if s.unchecked},
        # every stock's next release, passed or not, so the dates can be checked later
        "next_earnings": {s.ticker: s.earnings for s in stocks if s.earnings},
        "dates_failed": len(unscreened),
    }


def refusals(result: dict[str, Any]) -> list[str]:
    """Why this screen is not to be trusted, if it is not: too many stocks
    whose earnings dates could not be read."""
    n = result["dates_failed"]
    if n > MAX_FAILED_SHARE * result["universe"]:
        return [f"{n} stocks' earnings dates could not be read"]
    return []


def _row(s: Stock) -> dict[str, Any]:
    return {**asdict(s), "reasons": s.reasons()}


def lines(result: dict[str, Any]) -> list[str]:
    chosen, over = result["chosen"], result["over_cap"]
    out = [
        f"Screen {result['version']} for the {result['day']} session "
        f"(earnings due {result['due_from']} to {result['due_through']}):",
        f"{len(chosen)} of {result['universe']} stocks pass"
        + (f"; {len(over)} more passed but are over the cap of {result['cap']}" if over else "")
        + (f"; {len(result['filled'])} more drawn at random" if result["filled"] else "")
        + f"; {len(result['held'])} are Hold by screen"
        + (
            f"; {len(result['unscreened'])} could not be screened." if result["unscreened"] else "."
        ),
        "",
        "Would go to Jev and Opus:",
    ]
    out += [f"  {i:>2}. {r['ticker']}: {'; '.join(r['reasons'])}" for i, r in enumerate(chosen, 1)]
    if over:
        out += ["", "Passed, but over the cap:"]
        out += [f"  {r['ticker']}: {'; '.join(r['reasons'])}" for r in over]
    if result["filled"]:
        out += ["", "Random fill (a check on the screen): " + ", ".join(result["filled"])]
    out += ["", "Hold by screen: " + (", ".join(result["held"]) or "none")]
    if result["unscreened"]:
        out += ["", "Could not be screened (no decision):"]
        out += [f"  {t}: {', '.join(result['unchecked'][t])}" for t in result["unscreened"]]
    return out


def universe_names(path: Path = DEFAULT_UNIVERSE) -> dict[str, str]:
    """Company names for every stock the screen can pick, the watchlist's own
    names first."""
    raw = yaml.safe_load(path.read_text())
    names = raw.get("names") if isinstance(raw, dict) else None
    out = {str(k).upper(): str(v) for k, v in (names or {}).items()}
    return out | load_watchlist().names


def _failed(reason: str) -> dict[str, Any]:
    return {"kind": "screen", "ok": False, "refused": [reason]}


def run_screen(
    now: datetime,
    *,
    prices_fetch: Callable[..., Prices] = fetch,
    earnings: EarningsSource | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """The screen for the latest session. Never raises: when it cannot be
    trusted the result says `"ok": false` and why, so the caller can fall back."""
    from tenq.daily import expected_session  # daily imports this module

    try:
        cfg, tickers = load_config(), load_universe()
        # QQQ alone: its latest bar says which session has settled
        bench = prices_fetch((), YahooPrices(), now=now).bars.get(BENCHMARK)
        if not bench:
            return _failed("no QQQ prices")
        expected = expected_session(now)
        if bench[-1].day < expected:
            return _failed(f"no QQQ bar for the {expected} session yet (latest: {bench[-1].day})")
        result = screen(tickers, bench[-1].day, cfg, earnings or YahooEarnings(), sleep=sleep)
    except Exception as e:  # any failure means fall back to the watchlist, never crash
        log.exception("the screen failed")  # the traceback stays in the log, not the email
        return _failed(f"the screen failed ({type(e).__name__})")
    refused = refusals(result)
    # Decided before anyone reads it, so a refused screen never reads as a good one.
    result["ok"] = not refused
    result["refused"] = refused
    return result


def picked(result: dict[str, Any]) -> list[str]:
    """The stocks a good screen sends to the models: those that passed, then the
    random fill."""
    return [r["ticker"] for r in result["chosen"]] + list(result["filled"])


def main(
    argv: list[str] | None = None,
    *,
    prices_fetch: Callable[..., Prices] = fetch,
    earnings: EarningsSource | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    ap = argparse.ArgumentParser(description="Show which stocks the daily screen would pass.")
    ap.add_argument("--json", type=Path, help="also write the result here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    result = run_screen(
        now or datetime.now(UTC), prices_fetch=prices_fetch, earnings=earnings, sleep=sleep
    )
    if "chosen" not in result:
        print(f"{result['refused'][0]}: nothing screened")
        return 1
    print("\n".join(lines(result)))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    if result["refused"]:
        print(f"\n{'; '.join(result['refused'])}: too many to trust this screen")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
