"""Earnings release dates for the daily screen (screen-v7), from Yahoo through
yfinance, the library the prices already come from. No key, no secret.

Yahoo's earnings list for a stock gives its past releases and the next one,
each with a time (16:00 New York for after the close, 07:00 before the open).
The quick calendar lookup (`Ticker.calendar`) is not used: on 2026-11-09 it
showed a release already past for 45 of the 113 stocks, until their next date
was set, while the earnings list had an upcoming date for all 113. On
2026-11-09 the list's past dates fell on EDGAR's release day (8-K item 2.02)
for 95% of releases and within a day for all but 2.

A lookup that fails, or gets no earnings table back (yfinance returns none for
a blocked or broken page), is asked once more after a pause, since Yahoo
throttles bursts; what still fails is reported, never raised, so the screen can
count it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Protocol

from tenq.prices import MARKET_TZ

RETRY_PAUSE_S = 5.0
ROWS = 12  # Yahoo's list, newest first: the next release and the past ones


class EarningsSource(Protocol):
    def dates(self, ticker: str) -> list[datetime]: ...


class YahooEarnings:
    """One stock's release times from Yahoo's earnings list."""

    def dates(self, ticker: str) -> list[datetime]:
        import yfinance as yf  # heavy import, deferred until a lookup is due

        frame = yf.Ticker(ticker).get_earnings_dates(limit=ROWS)
        if frame is None:  # yfinance logs a blocked or broken page and returns None
            raise LookupError("no earnings table")
        out: list[datetime] = []
        for ts in frame.index:
            when = ts.to_pydatetime()
            out.append(when if when.tzinfo else when.replace(tzinfo=MARKET_TZ))
        return out


def lookup(
    tickers: Sequence[str],
    source: EarningsSource,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, list[datetime]], dict[str, str]]:
    """Each stock's release times, and why the others could not be read."""
    found: dict[str, list[datetime]] = {}
    failed: dict[str, str] = {}
    for attempt in range(2):
        pending = [t for t in tickers if t not in found]
        if not pending:
            break
        if attempt:
            sleep(RETRY_PAUSE_S)
        for t in pending:
            try:
                found[t] = list(source.dates(t))
                failed.pop(t, None)
            except Exception as exc:  # one stock's failure is a note, never a crash
                failed[t] = f"earnings dates unavailable ({type(exc).__name__})"
    return found, failed
