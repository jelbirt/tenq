"""The daily screen (L6): code alone picks which stocks Jev and Opus see.

A stock passes when either of two things holds (config/screen.yaml):
1. it filed one of `forms` after the previous session's day (8-K, 10-Q, 10-K,
   or a foreign company's 6-K, 20-F or 40-F);
2. it moved a lot against QQQ for that stock: over 1 session, or over 5, by
   at least `move_sigma` of its usual daily moves against QQQ, measured over
   the 20 sessions before the move.

At most `cap` pass; past it, new filings come first, then the bigger move.
If fewer than `fill_to` pass, the rest of the slots up to `fill_to` go to
stocks drawn at random from the ones left out ("random fill"), seeded by the
day so a rerun draws the same ones. They are a fair sample of what the screen
skips: the scorecard can then check whether the models find more to act on in
screened stocks than in random ones. Every other stock is "Hold by screen".

A recent Buy, Sell or Short is not a reason to pass: each day's call is
scored on its own, and re-asking stocks because they were asked lately would
fill the cap with the same names (see the comments in config/screen.yaml).
This module asks no model anything: `python -m tenq.screen` only prints who
would pass.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import yaml

from tenq.config import REPO_ROOT, load_watchlist
from tenq.filings import FilingMeta
from tenq.prices import (
    BENCHMARK,
    MARKET_TZ,
    Prices,
    YahooPrices,
    aligned,
    fetch,
)
from tenq.scorecard import is_session

log = logging.getLogger(__name__)

DEFAULT_SCREEN = REPO_ROOT / "config" / "screen.yaml"
DEFAULT_UNIVERSE = REPO_ROOT / "config" / "universe.yaml"
TOP_LEVEL = {"version", "forms", "routine", "move_sigma", "cap", "fill_to"}
ROUTINE_KEYS = {"sessions", "filings"}
# Newest filings looked at per stock: enough to see a day's filings and the
# routine look-back before them, even for a company that files every day.
LIST_DEPTH = 30
YARDSTICK = 20  # sessions of daily moves against QQQ that set a stock's usual move
MAX_FAILED_SHARE = 0.1  # more EDGAR lookups failing than this: the run says so and exits 1


@dataclass(frozen=True)
class ScreenConfig:
    version: str
    forms: tuple[str, ...]
    routine_sessions: int
    routine_filings: int
    move_sigma: float
    cap: int
    fill_to: int
    sha256: str


def load_config(path: Path = DEFAULT_SCREEN) -> ScreenConfig:
    text = path.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or set(raw) != TOP_LEVEL:
        raise ValueError(f"{path}: expected exactly the keys {sorted(TOP_LEVEL)}")
    forms = raw["forms"]
    if not isinstance(forms, list) or not forms or not all(isinstance(f, str) for f in forms):
        raise ValueError(f"{path}: 'forms' must be a list of form names")
    routine = raw["routine"]
    if not isinstance(routine, dict) or set(routine) != ROUTINE_KEYS:
        raise ValueError(f"{path}: 'routine' must have exactly the keys {sorted(ROUTINE_KEYS)}")
    r_sessions, r_filings = routine["sessions"], routine["filings"]
    for name, v, top in (("sessions", r_sessions, 60), ("filings", r_filings, 20)):
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= top:
            raise ValueError(f"{path}: 'routine.{name}' must be a whole number from 1 to {top}")
    sigma, cap = raw["move_sigma"], raw["cap"]
    if isinstance(sigma, bool) or not isinstance(sigma, int | float) or not 0 < sigma <= 10:
        raise ValueError(f"{path}: 'move_sigma' must be a number above 0, at most 10")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"{path}: 'cap' must be a whole number, at least 1")
    fill = raw["fill_to"]
    if isinstance(fill, bool) or not isinstance(fill, int) or not 0 <= fill <= cap:
        raise ValueError(f"{path}: 'fill_to' must be a whole number from 0 to 'cap'")
    if not isinstance(raw["version"], str) or not raw["version"]:
        raise ValueError(f"{path}: 'version' must be text")
    return ScreenConfig(
        version=raw["version"],
        forms=tuple(forms),
        routine_sessions=r_sessions,
        routine_filings=r_filings,
        move_sigma=float(sigma),
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


class Filings(Protocol):
    def list_recent(self, ticker: str, forms: tuple[str, ...], limit: int) -> list[FilingMeta]: ...


@dataclass
class Stock:
    ticker: str
    filings: list[str] = field(default_factory=list)  # "8-K filed 2026-11-06"
    routine: list[str] = field(default_factory=list)  # new filings not counted, and why
    move: float | None = None  # the bigger of the 1- and 5-session moves, in usual moves
    move_window: int | None = None
    unchecked: list[str] = field(default_factory=list)  # what could not be looked at

    def reasons(self, sigma: float) -> list[str]:
        out = list(self.filings)
        if self.move is not None and abs(self.move) >= sigma:
            way = "above" if self.move > 0 else "below"
            out.append(
                f"moved {abs(self.move):.1f} usual moves {way} QQQ over "
                f"{self.move_window} session(s)"
            )
        return out

    def passes(self, sigma: float) -> bool:
        return bool(self.reasons(sigma))

    def rank(self, sigma: float) -> tuple[bool, float]:
        """Order past the cap: a new filing first, then the bigger move."""
        big = abs(self.move) if self.move is not None and abs(self.move) >= sigma else 0.0
        return (not self.filings, -big)


def previous_session(day: date) -> date:
    d = day - timedelta(days=1)
    while not is_session(d):
        d -= timedelta(days=1)
    return d


def moves(ticker: str, prices: Prices, day: date) -> tuple[float | None, int | None, str | None]:
    """The bigger of the 1- and 5-session moves against QQQ up to the `day`
    session, in the stock's usual daily moves against QQQ, with its window; or
    a note saying why it is missing.

    The yardstick is the spread of the stock's daily return less QQQ's over the
    YARDSTICK sessions before the window tested, so the move being tested is
    not part of its own yardstick. Prices that end before `day` are not used:
    they would pass a stock today on an earlier day's move.

    A window is tested only when the stock has a bar on every QQQ session it
    and its yardstick span: a missing bar would stretch a 1-session move over
    two sessions. A window that cannot be tested is noted, so the stock counts
    as not fully checked rather than as held."""
    stock, bench = prices.bars.get(ticker), prices.bars.get(BENCHMARK)
    pairs = aligned(stock, bench) if stock and bench else []
    if not pairs or not bench:
        return None, None, "no prices"
    if pairs[-1][0].day != day:
        return None, None, f"prices end {pairs[-1][0].day.isoformat()}, not {day.isoformat()}"
    mine = [s.close for s, _ in pairs]
    theirs = [q.close for _, q in pairs]
    best: tuple[float, int] | None = None
    notes: list[str] = []
    for n in (1, 5):
        k = YARDSTICK + n + 1
        if len(pairs) < k:
            notes.append(f"too little price history for the {n}-session move")
            continue
        if [s.day for s, _ in pairs[-k:]] != [b.day for b in bench[-k:]]:
            notes.append(f"gap in prices for the {n}-session move")
            continue
        x = (mine[-1] / mine[-1 - n] - 1) - (theirs[-1] / theirs[-1 - n] - 1)
        before_s, before_q = mine[:-n][-(YARDSTICK + 1) :], theirs[:-n][-(YARDSTICK + 1) :]
        daily = [
            (b / a) - (d / c)
            for (a, b), (c, d) in zip(
                itertools.pairwise(before_s), itertools.pairwise(before_q), strict=True
            )
        ]
        usual = statistics.stdev(daily)
        if not (math.isfinite(x) and math.isfinite(usual)) or usual <= 0:
            notes.append(f"no usual move for the {n}-session move")
            continue
        z = x / (usual * math.sqrt(n))
        if best is None or abs(z) > abs(best[0]):
            best = (z, n)
    note = "; ".join(notes) or None
    if best is None:
        return None, None, note
    return best[0], best[1], note


EXHIBITS_ITEM = "9.01"  # an 8-K's exhibit list; it says nothing about the event


def kind_of(meta: FilingMeta) -> tuple[str, frozenset[str]]:
    """What a filing reports: its form, and for an 8-K the items it lists
    (2.02 earnings, 1.01 a major deal, 5.02 an executive change, and so on),
    leaving out 9.01, the exhibit list, which rides along with most of them."""
    if meta.form != "8-K":
        return meta.form, frozenset()
    items = {i.strip() for i in meta.items.split(",") if i.strip()}
    return meta.form, frozenset(items - {EXHIBITS_ITEM})


def _kind_text(kind: tuple[str, frozenset[str]]) -> str:
    form, items = kind
    if not items:
        return form
    return f"{form} item{'s' if len(items) > 1 else ''} {', '.join(sorted(items))}"


def new_filings(
    listed: Sequence[FilingMeta], cfg: ScreenConfig, since: date, look_back: date, today: date
) -> tuple[list[str], list[str]]:
    """The new filings that count as a reason to pass, and those that do not
    because they are routine for this company.

    New means filed after `since` (the previous session) up to `today`. A new
    filing is routine when the company filed the same kind of filing on at
    least `routine_filings` other days from `look_back` to `since`, the
    `routine_sessions` sessions before: a company that files it that often
    (weekly buyback notices, say) is not telling us anything new by filing it
    again. For an 8-K the kind is its items as well as its form (v4), so a
    weekly filer's purchase reports are routine but its earnings 8-K is not.
    When a run is late and several sessions' filings are new at once, they are
    not counted against each other; that only lets more through."""
    dated: list[tuple[tuple[str, frozenset[str]], date]] = []
    for m in listed:
        try:
            dated.append((kind_of(m), date.fromisoformat(m.filed)))
        except ValueError:
            continue
    counted: list[str] = []
    routine: list[str] = []
    for kind, filed in dated:
        if kind[0] not in cfg.forms or not since < filed <= today:
            continue
        before = {f for k, f in dated if k == kind and look_back <= f <= since}
        text = f"{_kind_text(kind)} filed {filed.isoformat()}"
        if len(before) >= cfg.routine_filings:
            routine.append(
                f"{text} (routine: filed on {len(before)} days in the "
                f"{cfg.routine_sessions} sessions before)"
            )
        else:
            counted.append(text)
    return counted, routine


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
    today: date,
    cfg: ScreenConfig,
    prices: Prices,
    filings: Filings | None,
) -> dict[str, Any]:
    """Who passes for the `day` session. Never raises for one stock: what
    could not be checked is noted, and that check simply does not count."""
    since = previous_session(day)
    look_back = since
    for _ in range(cfg.routine_sessions - 1):
        look_back = previous_session(look_back)
    stocks: list[Stock] = []
    price_failed = 0
    for t in tickers:
        st = Stock(t)
        st.move, st.move_window, note = moves(t, prices, day)
        if note:
            st.unchecked.append(note)
            price_failed += 1
        if filings is not None:
            try:
                listed = filings.list_recent(t, cfg.forms, LIST_DEPTH)
            except Exception as exc:
                st.unchecked.append(f"filings unavailable ({type(exc).__name__})")
                listed = []
            st.filings, st.routine = new_filings(listed, cfg, since, look_back, today)
        stocks.append(st)
    passed = sorted(
        (s for s in stocks if s.passes(cfg.move_sigma)), key=lambda s: s.rank(cfg.move_sigma)
    )
    # A stock that did not pass but could not be fully checked is not a screen
    # decision: it is neither Hold by screen nor in the random fill's pool.
    out = [s for s in stocks if not s.passes(cfg.move_sigma)]
    left = [s.ticker for s in out if not s.unchecked]
    unscreened = [s.ticker for s in out if s.unchecked]
    filled = random_fill(left, cfg.fill_to - min(len(passed), cfg.cap), cfg.version, day)
    failed = sum(any(n.startswith("filings unavailable") for n in s.unchecked) for s in stocks)
    return {
        "kind": "screen",
        "version": cfg.version,
        "sha256": cfg.sha256,
        "day": day.isoformat(),
        "since": since.isoformat(),
        "today": today.isoformat(),
        "universe": len(tickers),
        "cap": cfg.cap,
        "chosen": [_row(s, cfg) for s in passed[: cfg.cap]],
        "over_cap": [_row(s, cfg) for s in passed[cfg.cap :]],
        "filled": filled,
        "held": [t for t in left if t not in filled],
        "unscreened": unscreened,
        "unchecked": {s.ticker: s.unchecked for s in stocks if s.unchecked},
        "routine": {s.ticker: s.routine for s in stocks if s.routine},
        "filings_checked": filings is not None,
        "filings_failed": failed,
        "prices_failed": price_failed,
        "price_notes": list(prices.notes),
    }


def refusals(result: dict[str, Any]) -> list[str]:
    """Why this screen is not to be trusted, if it is not: too many stocks
    whose filings or prices could not be checked."""
    limit = MAX_FAILED_SHARE * result["universe"]
    out = []
    if result["filings_failed"] > limit:
        out.append(f"{result['filings_failed']} EDGAR lookups failed")
    if result["prices_failed"] > limit:
        out.append(f"{result['prices_failed']} stocks' prices could not be fully checked")
    return out


def _row(s: Stock, cfg: ScreenConfig) -> dict[str, Any]:
    return {**asdict(s), "reasons": s.reasons(cfg.move_sigma)}


def lines(result: dict[str, Any]) -> list[str]:
    chosen, over = result["chosen"], result["over_cap"]
    out = [
        f"Screen {result['version']} for the {result['day']} session "
        f"(new filings since {result['since']}):",
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
    if result["routine"]:
        out += ["", "New filings not counted, routine for the company:"]
        out += [f"  {t}: {'; '.join(n)}" for t, n in result["routine"].items()]
    if result["unscreened"]:
        out += ["", "Could not be screened (no decision): " + ", ".join(result["unscreened"])]
    if not result["filings_checked"]:
        out += ["", "Filings were not checked (no EDGAR lookup): new filings could add stocks."]
    if result["unchecked"]:
        out += ["", "Could not check everything for:"]
        out += [f"  {t}: {', '.join(n)}" for t, n in result["unchecked"].items()]
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
    filings: Filings | None = None,
    check_filings: bool = True,
) -> dict[str, Any]:
    """The screen for the latest session. Never raises: when it cannot be
    trusted the result says `"ok": false` and why, so the caller can fall back."""
    from tenq.daily import expected_session  # daily imports this module

    try:
        cfg, tickers = load_config(), load_universe()
        prices = prices_fetch(tickers, YahooPrices(), now=now)
        bench = prices.bars.get(BENCHMARK)
        if not bench:
            return _failed("no QQQ prices")
        expected = expected_session(now)
        if bench[-1].day < expected:
            return _failed(f"no QQQ bar for the {expected} session yet (latest: {bench[-1].day})")
        if filings is None and check_filings:
            from tenq.edgar import EdgarSource

            filings = EdgarSource()
        today = now.astimezone(MARKET_TZ).date()
        result = screen(tickers, bench[-1].day, today, cfg, prices, filings)
    except Exception as e:  # any failure means fall back to the watchlist, never crash
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
    filings: Filings | None = None,
    now: datetime | None = None,
) -> int:
    ap = argparse.ArgumentParser(description="Show which stocks the daily screen would pass.")
    ap.add_argument("--no-filings", action="store_true", help="skip the EDGAR check")
    ap.add_argument("--json", type=Path, help="also write the result here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    result = run_screen(
        now or datetime.now(UTC),
        prices_fetch=prices_fetch,
        filings=filings,
        check_filings=not args.no_filings,
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
