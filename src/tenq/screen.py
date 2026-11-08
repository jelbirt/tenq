"""The daily screen (L6): code alone picks which stocks Jev and Opus see.

A stock passes when either of two things holds (config/screen.yaml):
1. it filed an 8-K, 10-Q or 10-K after the previous session's day;
2. it moved a lot against QQQ for that stock: over 1 session, or over 5, by
   at least `move_sigma` of its own usual daily moves (20-session volatility).

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
import json
import logging
import math
import random
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
    SESSIONS_PER_YEAR,
    Prices,
    YahooPrices,
    fetch,
    price_facts,
)
from tenq.scorecard import is_session

log = logging.getLogger(__name__)

DEFAULT_SCREEN = REPO_ROOT / "config" / "screen.yaml"
DEFAULT_UNIVERSE = REPO_ROOT / "config" / "universe.yaml"
TOP_LEVEL = {"version", "forms", "move_sigma", "cap", "fill_to"}
LIST_DEPTH = 8  # newest filings looked at per stock; more than a day's worth


@dataclass(frozen=True)
class ScreenConfig:
    version: str
    forms: tuple[str, ...]
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
    def list_recent(self, ticker: str, limit: int) -> list[FilingMeta]: ...


@dataclass
class Stock:
    ticker: str
    filings: list[str] = field(default_factory=list)  # "8-K filed 2026-11-06"
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


def moves(ticker: str, prices: Prices) -> tuple[float | None, int | None, str | None]:
    """The bigger of the 1- and 5-session excess moves, in the stock's usual
    daily moves, with its window; or a note saying why it is missing."""
    facts = price_facts(ticker, prices)
    if facts is None:
        return None, None, "no prices"
    vol = facts.volatility
    if vol is None or not math.isfinite(vol) or vol <= 0:
        return None, None, "no volatility"
    daily = vol / math.sqrt(SESSIONS_PER_YEAR)
    best: tuple[float, int] | None = None
    for n in (1, 5):
        x = facts.excess.get(n)
        if x is None or not math.isfinite(x):
            continue
        z = x / (daily * math.sqrt(n))
        if best is None or abs(z) > abs(best[0]):
            best = (z, n)
    if best is None:
        return None, None, "no excess return"
    return best[0], best[1], None


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
    stocks: list[Stock] = []
    for t in tickers:
        st = Stock(t)
        st.move, st.move_window, note = moves(t, prices)
        if note:
            st.unchecked.append(note)
        if filings is not None:
            try:
                listed = filings.list_recent(t, LIST_DEPTH)
            except Exception as exc:
                st.unchecked.append(f"filings unavailable ({type(exc).__name__})")
                listed = []
            for m in listed:
                try:
                    filed = date.fromisoformat(m.filed)
                except ValueError:
                    continue
                if m.form in cfg.forms and since < filed <= today:
                    st.filings.append(f"{m.form} filed {m.filed}")
        stocks.append(st)
    passed = sorted(
        (s for s in stocks if s.passes(cfg.move_sigma)), key=lambda s: s.rank(cfg.move_sigma)
    )
    left = [s.ticker for s in stocks if not s.passes(cfg.move_sigma)]
    filled = random_fill(left, cfg.fill_to - min(len(passed), cfg.cap), cfg.version, day)
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
        "unchecked": {s.ticker: s.unchecked for s in stocks if s.unchecked},
        "filings_checked": filings is not None,
        "price_notes": list(prices.notes),
    }


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
        + f"; {len(result['held'])} are Hold by screen.",
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
    if not result["filings_checked"]:
        out += ["", "Filings were not checked (no EDGAR lookup): new filings could add stocks."]
    if result["unchecked"]:
        out += ["", "Could not check everything for:"]
        out += [f"  {t}: {', '.join(n)}" for t, n in result["unchecked"].items()]
    return out


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
    now = now or datetime.now(UTC)
    cfg, tickers = load_config(), load_universe()
    prices = prices_fetch(tickers, YahooPrices(), now=now)
    bench = prices.bars.get(BENCHMARK)
    if not bench:
        print("no QQQ prices: nothing screened")
        return 1
    if filings is None and not args.no_filings:
        from tenq.edgar import EdgarSource

        filings = EdgarSource()
    today = now.astimezone(MARKET_TZ).date()
    result = screen(tickers, bench[-1].day, today, cfg, prices, filings)
    print("\n".join(lines(result)))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
