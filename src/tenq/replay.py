"""The simple baselines replayed over past prices: how they have done, and how much they swing.

Jev and Opus cannot be replayed (the news they read is gone), but the baselines
need prices only. For every past session and every watchlist stock this makes
the baselines' calls from what was known at that close, and scores them exactly
as the scorecard scores a live call: entry at the next open, exit at the close
20 sessions later, excess over QQQ, and the same luck range from daily moves.

The baselines:
- always Hold: scores 0 by definition, the line any method must stay above;
- always Buy: the stock's excess over QQQ, every time;
- last month's winners: Buy when the stock beat QQQ over the 20 sessions
  before, Hold otherwise (the scorecard's momentum baseline).

This is background, not the bar a live method is held to: that is still the
baselines on the same days, on the scorecard, since the baselines swing a lot
from year to year. Its luck range is for years of calls; a few months of live
calls have a much wider one. And the watchlist was picked in 2026, so its
stocks are ones that did well enough to be watched: that flatters every rule
that mostly buys, always Buy most.

`uv run python -m tenq.replay` downloads the prices (no keys needed) and prints
the table; `--json PATH` also writes the numbers. Nothing in runs/ is touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from tenq.config import load_watchlist
from tenq.prices import BENCHMARK, MARKET_TZ, Bar, Prices, YahooPrices, aligned, fetch
from tenq.scorecard import (
    Made,
    ScoreConfig,
    day_totals,
    load_config,
    outcome,
    standard_error,
    t95,
)

log = logging.getLogger(__name__)

MADE_AT = time(18, 30)  # the daily job runs after the close; the next open is the entry
LOOKBACK = 20  # sessions of history behind the momentum baseline
BASELINES = {
    "always_hold": "Always Hold",
    "always_buy": "Always Buy",
    "last_months_winners": "Last month's winners",
}


def _made_at(day: date) -> datetime:
    return datetime.combine(day, MADE_AT, MARKET_TZ).astimezone(UTC)


def momentum(pairs: Sequence[tuple[Bar, Bar]], j: int) -> float | None:
    """The stock's return less QQQ's over the LOOKBACK days both traded, up to and
    including pair j: the live sheet's px excess_20d (prices.price_facts), as known
    at that close. None with fewer than LOOKBACK days behind it."""
    if j < LOOKBACK:
        return None
    (s0, q0), (s1, q1) = pairs[j - LOOKBACK], pairs[j]
    if s0.close <= 0 or q0.close <= 0:
        return None
    return (s1.close / s0.close - 1) - (q1.close / q0.close - 1)


def rows_from(prices: Prices, tickers: Sequence[str], cfg: ScoreConfig) -> list[dict[str, Any]]:
    """One scored row per stock per past session: its excess, daily moves and
    the momentum known at that close."""
    bench = prices.bars.get(BENCHMARK) or ()
    out = []
    for t in tickers:
        bars = prices.bars.get(t)
        if not bars:
            continue
        pairs = aligned(bars, bench)  # only days both traded: not listed yet, or a gap
        for j, (_, q) in enumerate(pairs):
            made = Made(
                day=q.day,
                made_at=_made_at(q.day),
                ticker=t,
                jev=None,
                jev_score=None,
                opus=None,
                opus_confidence=None,
                momentum=momentum(pairs, j),
                volatility=None,
                versions={},
            )
            got = outcome(made, prices.bars, cfg)
            if got.status != "scored" or got.excess is None:
                continue
            out.append(
                {
                    "day": q.day.isoformat(),
                    "ticker": t,
                    "excess": got.excess,
                    "moves": list(got.moves or ()),
                    "move_days": [d.isoformat() for d in got.move_days or ()],
                    "exit_day": got.exit_day.isoformat() if got.exit_day else None,
                    "momentum": made.momentum,
                }
            )
    return out


def weight(baseline: str, row: dict[str, Any]) -> float | None:
    """The baseline's call weight on a row (Buy 1, Hold 0), or None if it makes no call."""
    if baseline == "always_hold":
        return 0.0
    if baseline == "always_buy":
        return 1.0
    m = row.get("momentum")
    if not isinstance(m, int | float) or not math.isfinite(m):
        return None
    return 1.0 if m > 0 else 0.0


def summary(rows: Sequence[dict[str, Any]], baseline: str, cfg: ScoreConfig) -> dict[str, Any]:
    """Mean score per call, how often it scored above 0, and the luck range
    around that mean, measured the way the scorecard measures it."""
    pairs = [(w, r) for r in rows if (w := weight(baseline, r)) is not None]
    if not pairs:
        return {"calls": 0, "mean": None, "luck": None, "hit": None, "buys": 0}
    scores = [w * r["excess"] for w, r in pairs]
    totals = list(day_totals(pairs).values())
    se = standard_error(totals, len(pairs), cfg.carry_days)
    buys = [s for (w, _), s in zip(pairs, scores, strict=True) if w]
    return {
        "calls": len(pairs),
        "buys": len(buys),
        "mean": sum(scores) / len(scores),
        "luck": None if not se else t95(len(totals) - 1) * se,
        "hit": (sum(1 for s in buys if s > 0) / len(buys)) if buys else None,
        "first": min(r["day"] for _, r in pairs),
        "last": max(r["day"] for _, r in pairs),
    }


def replay(prices: Prices, tickers: Sequence[str], cfg: ScoreConfig) -> dict[str, Any]:
    rows = rows_from(prices, tickers, cfg)
    years = sorted({r["day"][:4] for r in rows})
    return {
        "score_version": cfg.version,
        "tickers": list(tickers),
        "stocks_scored": sorted({r["ticker"] for r in rows}),
        "rows": len(rows),
        "notes": list(prices.notes),
        "all": {b: summary(rows, b, cfg) for b in BASELINES},
        "by_year": {
            y: {b: summary([r for r in rows if r["day"][:4] == y], b, cfg) for b in BASELINES}
            for y in years
        },
    }


def _pct(x: Any) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def lines(result: dict[str, Any]) -> list[str]:
    a = result["all"]
    span = a["always_buy"]
    out = [
        f"Baselines replayed over past prices, {span.get('first')} to {span.get('last')}: "
        f"{result['rows']} stock-days, {len(result['stocks_scored'])} stocks, "
        f"scored as {result['score_version']} scores a live call.",
        "",
        f"{'':24}{'per call':>10}{'luck +/-':>10}{'Buys':>8}{'Buys beat QQQ':>15}",
    ]

    def row(label: str, s: dict[str, Any]) -> str:
        hit = "n/a" if s["hit"] is None else f"{s['hit'] * 100:.0f}%"
        luck = "n/a" if s["luck"] is None else f"{s['luck'] * 100:.2f}%"
        return f"{label:24}{_pct(s['mean']):>10}{luck:>10}{s['buys']:>8}{hit:>15}"

    for key, label in BASELINES.items():
        out.append(row(label, a[key]))
    for y, by in result["by_year"].items():
        span_y = by["always_buy"]
        out.append("")
        out.append(
            f"{y} ({span_y.get('first')} to {span_y.get('last')}, {span_y['calls']} stock-days):"
        )
        for key, label in BASELINES.items():
            if key != "always_hold":
                out.append(row(f"  {label}", by[key]))
    out += [
        "",
        "Per call is the mean score a recommendation earned (Buy counts the stock's lead over "
        "QQQ from the next open to the close 20 sessions later; Hold counts 0). Luck +/- is the "
        "90% range around it from luck alone: a mean inside +/- that of 0 could be chance.",
        "This is background, not the bar: a live method is judged against the baselines on the "
        "same days, on the scorecard. The luck range here is for years of calls; a few months "
        "of live calls have a much wider one.",
        "The watchlist was chosen in 2026, so these stocks did well enough to be watched: that "
        "flatters every rule that mostly buys, always Buy most.",
    ]
    if result["notes"]:
        out.append("Price notes: " + "; ".join(result["notes"]))
    return out


PriceSource = Callable[[list[str], str], Prices]


def _yahoo(tickers: list[str], period: str) -> Prices:
    return fetch(tickers, YahooPrices(), period=period)


def main(argv: list[str] | None = None, prices: PriceSource = _yahoo) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", type=int, choices=(1, 2, 3, 5), default=3)
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    args = ap.parse_args(argv)
    tickers = list(load_watchlist().tickers)
    got = prices(tickers, f"{args.years}y")
    if BENCHMARK not in got.bars:
        print(f"no {BENCHMARK} prices: {'; '.join(got.notes) or 'download failed'}")
        return 1
    result = replay(got, tickers, load_config())
    if not result["rows"]:
        print("no stock-day could be scored")
        return 1
    print("\n".join(lines(result)))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
