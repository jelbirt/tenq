"""Daily closes for the watchlist and the benchmark, and the price facts computed from them.

Closes come from Yahoo through yfinance with auto_adjust on, so they are split and
dividend adjusted: a return over any window is a total return, and a stock and QQQ are
compared on the same basis. The download is adapted from ticker-sentinel's
`src/sentinel/data/prices.py` (MIT, same author); tenq has no Twelve Data key, so a
ticker Yahoo cannot serve is left out with a note instead of falling back.

Every number here is computed in code. The model only ever sees the results.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from datetime import time as clock
from typing import Any, Protocol
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

BENCHMARK = "QQQ"
# A year of sessions is enough for every window below; the scorecard asks again later.
DEFAULT_PERIOD = "1y"
WINDOWS = (1, 5, 20)
VOL_WINDOW = 20
SESSIONS_PER_YEAR = 252
ATTEMPTS = 3

# A bar dated today is the session still in progress until the close has settled
# (the same cut-off ticker-sentinel uses after a late run scored a partial day).
MARKET_TZ = ZoneInfo("America/New_York")
SESSION_SETTLED = clock(16, 30)


@dataclass(frozen=True)
class Bar:
    day: date
    close: float
    volume: float | None


@dataclass(frozen=True)
class Prices:
    bars: dict[str, tuple[Bar, ...]]
    notes: tuple[str, ...]


class PriceSource(Protocol):
    def history(self, tickers: Sequence[str], period: str) -> Any: ...


class YahooPrices:
    """One batched yfinance download; the frame is parsed by `bars_from_frame`."""

    def history(self, tickers: Sequence[str], period: str) -> Any:
        import yfinance as yf

        return yf.download(
            list(tickers), period=period, auto_adjust=True, progress=False, group_by="column"
        )


def _number(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def bars_from_frame(frame: Any, tickers: Sequence[str]) -> dict[str, tuple[Bar, ...]]:
    """Each ticker's bars, oldest first, keeping only days with a positive close."""
    if frame is None or getattr(frame, "empty", True):
        return {}
    multi = getattr(frame.columns, "nlevels", 1) > 1
    out: dict[str, tuple[Bar, ...]] = {}
    for ticker in tickers:
        if multi:
            if ("Close", ticker) not in frame.columns:
                continue
            closes, volumes = frame["Close"][ticker], frame.get(("Volume", ticker))
        else:  # one ticker comes back with flat columns
            if len(tickers) != 1 or "Close" not in frame.columns:
                continue
            closes, volumes = frame["Close"], frame.get("Volume")
        bars = []
        for stamp, raw in closes.items():
            close = _number(raw)
            if close is None or close <= 0:
                continue
            volume = _number(volumes[stamp]) if volumes is not None else None
            bars.append(Bar(day=stamp.date(), close=close, volume=volume))
        if bars:
            out[ticker] = tuple(sorted(bars, key=lambda b: b.day))
    return out


def drop_unfinished(bars: tuple[Bar, ...], now: datetime) -> tuple[Bar, ...]:
    """Leave out a bar dated today in New York until the session has settled."""
    local = now.astimezone(MARKET_TZ)
    if bars and local.time() < SESSION_SETTLED and bars[-1].day == local.date():
        return bars[:-1]
    return bars


def fetch(
    tickers: Iterable[str],
    source: PriceSource,
    *,
    period: str = DEFAULT_PERIOD,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Prices:
    """Closes for `tickers` plus the benchmark. Never raises: what is missing gets a note."""
    wanted = list(dict.fromkeys([*tickers, BENCHMARK]))
    notes: list[str] = []
    frame = None
    failure = "no attempt made"
    for attempt in range(ATTEMPTS):
        try:
            frame = source.history(wanted, period)
            if frame is not None and not getattr(frame, "empty", True):
                break
            failure = "no data returned"
        except Exception as exc:
            failure = type(exc).__name__
        log.warning("price download attempt %d failed: %s", attempt + 1, failure)
        frame = None
        if attempt + 1 < ATTEMPTS:
            sleep(2.0**attempt)
    if frame is None:
        notes.append(f"price download failed ({failure})")
    try:
        bars = bars_from_frame(frame, wanted)
    except Exception as exc:  # a reshaped frame must not stop the run
        log.warning("price frame unreadable: %s", type(exc).__name__)
        bars = {}
        notes.append("price data unreadable")
    stamp = now or datetime.now(UTC)
    bars = {t: kept for t, b in bars.items() if (kept := drop_unfinished(b, stamp))}
    missing = [t for t in wanted if t not in bars]
    if missing and not notes:
        notes.append(f"no prices for {', '.join(missing)}")
    return Prices(bars=bars, notes=tuple(notes))


def aligned(stock: Sequence[Bar], bench: Sequence[Bar]) -> list[tuple[Bar, Bar]]:
    """Pairs of bars on the days both traded, oldest first."""
    by_day = {b.day: b for b in bench}
    return [(s, by_day[s.day]) for s in stock if s.day in by_day]


def trailing_return(closes: Sequence[float], n: int) -> float | None:
    """Change over the last `n` sessions, as a fraction; None without n+1 closes."""
    if n < 1 or len(closes) <= n:
        return None
    return closes[-1] / closes[-1 - n] - 1


def volatility(closes: Sequence[float], n: int = VOL_WINDOW) -> float | None:
    """Annualised standard deviation of the last `n` daily log returns."""
    if n < 2 or len(closes) <= n:
        return None
    window = closes[-1 - n :]
    logs = [math.log(b / a) for a, b in itertools.pairwise(window)]
    mean = sum(logs) / len(logs)
    var = sum((x - mean) ** 2 for x in logs) / (len(logs) - 1)
    return math.sqrt(var) * math.sqrt(SESSIONS_PER_YEAR)


@dataclass(frozen=True)
class PriceFacts:
    """One stock's price facts on its latest session that the benchmark also traded."""

    ticker: str
    day: date
    close: float
    returns: dict[int, float | None]
    benchmark_returns: dict[int, float | None]
    excess: dict[int, float | None]
    volatility: float | None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["day"] = self.day.isoformat()
        return out


def price_facts(ticker: str, prices: Prices) -> PriceFacts | None:
    """Returns over each window against QQQ on the same days, or None without both series."""
    stock, bench = prices.bars.get(ticker), prices.bars.get(BENCHMARK)
    if not stock or not bench:
        return None
    pairs = aligned(stock, bench)
    if not pairs:
        return None
    mine = [s.close for s, _ in pairs]
    theirs = [b.close for _, b in pairs]
    returns = {n: trailing_return(mine, n) for n in WINDOWS}
    bench_returns = {n: trailing_return(theirs, n) for n in WINDOWS}
    excess: dict[int, float | None] = {}
    for n in WINDOWS:
        r, q = returns[n], bench_returns[n]
        excess[n] = None if r is None or q is None else r - q
    last = pairs[-1][0]
    return PriceFacts(
        ticker=ticker,
        day=last.day,
        close=last.close,
        returns=returns,
        benchmark_returns=bench_returns,
        excess=excess,
        volatility=volatility(mine),
    )


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def main(argv: list[str] | None = None) -> int:
    from tenq.config import load_watchlist

    parser = argparse.ArgumentParser(description="Print price facts for the watchlist")
    parser.add_argument("tickers", nargs="*", help="default: the whole watchlist")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    tickers = [t.upper() for t in args.tickers] or list(load_watchlist().tickers)
    prices = fetch(tickers, YahooPrices())
    for note in prices.notes:
        print(f"note: {note}")
    for ticker in tickers:
        facts = price_facts(ticker, prices)
        if facts is None:
            print(f"{ticker:6} no prices")
            continue
        vs = "  ".join(f"{n}d {_pct(facts.excess[n])}" for n in WINDOWS)
        vol = "n/a" if facts.volatility is None else f"{facts.volatility:.0%}"
        print(f"{ticker:6} {facts.day} {facts.close:10.2f}  vs {BENCHMARK}: {vs}  vol {vol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
