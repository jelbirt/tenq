"""How steady each method is: both calls asked several times on the same facts.

    python -m tenq.consistency                 # every watchlist stock, 3 runs each
    python -m tenq.consistency --ticker CRWD   # these stocks only
    python -m tenq.consistency --no-opus       # Jev only (pennies, no Claude usage)

Each stock's facts pack is gathered once. Then Jev (its answers, turned into a
call by the rules) and Opus's own call are each asked --repeats times on that
same pack. A method whose call changes on identical facts owes part of its
future score to luck, so this is the earliest honest read on either method,
and worth repeating after any prompt or rules change. Opus's why for Jev's
call is not asked: only the calls are compared.

The record goes to runs/consistency/<date run>-<id>.json, saved after every
stock ("complete" turns true at the end), and a summary prints: per
method, on how many stocks the call was the same every time, and how far the
score (Jev) or the confidence (Opus) moved between runs. Nothing is emailed,
and the scorecard never reads these calls.

Fails open like the daily run: if Jev is down, Claude is at its limit (or
fails twice in a row), or the run passes its 110 minute budget for Opus, that
method is skipped for the stocks left. With Opus on, a run is refused on
weekday evenings (20:00 to 02:00 UTC), when the daily run needs the same
Claude subscription, and above 60 Opus runs. Exits 1 when no call was made.
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tenq import jev, opus
from tenq.config import load_watchlist
from tenq.daily import Context, State, _write
from tenq.filings import TICKER
from tenq.pack import Pack
from tenq.rules import decide, load_rules

log = logging.getLogger(__name__)

KIND = "consistency"
CONSISTENCY_DIR = "consistency"
REPEATS = 3
MAX_REPEATS = 5
# Opus is skipped once the run passes this, so Jev still finishes every stock
# inside the workflow step's 140 minutes.
OPUS_BUDGET_S = 110 * 60
MAX_OPUS_RUNS = 60  # repeats x stocks: more cannot fit the budget
# The daily run uses the same Claude subscription from 22:30 UTC on weekdays.
# A run that could still be asking Opus then is refused, so it never uses up
# the quota the day's calls need.
DAILY_WINDOW_UTC = (20, 2)  # from 20:00 to 02:00 the next morning, Monday to Friday nights


def in_daily_window(now: datetime) -> bool:
    """Whether `now` falls in a weekday evening's daily run window (UTC)."""
    utc = now.astimezone(UTC)
    start, end = DAILY_WINDOW_UTC
    if utc.hour >= start:
        return utc.weekday() < 5  # Monday to Friday evening
    if utc.hour < end:
        return 1 <= utc.weekday() <= 5  # Tuesday to Saturday, just after midnight
    return False


def jev_runs(
    pack: Pack, ctx: Context, state: State, repeats: int, run_id: str, out: list[dict[str, Any]]
) -> None:
    """Jev's call `repeats` times on one pack, each a fresh set of answers, onto `out`."""
    for i in range(repeats):
        if state.jev_down:
            out.append({"call": None, "error": "not asked: Jev was down earlier in this run"})
            continue
        result = ctx.ask(
            pack.render(),
            ctx.questions,
            session_id=f"tenq-consistency-{run_id[:8]}-{pack.ticker}-{pack.day}-{i + 1}",
        )
        state.jev_down |= result.down
        state.jev_cost += result.cost
        call = decide(pack, result, ctx.rules, ctx.questions) if result.answers else None
        out.append(
            {
                "call": None if call is None else call.call,
                "score": None if call is None else call.score,
                "confidence": None if call is None else call.confidence,
                "error": result.error,
                "cost": result.cost,
            }
        )


def opus_runs(
    pack: Pack, ctx: Context, state: State, repeats: int, out: list[dict[str, Any]]
) -> None:
    """Opus's own call `repeats` times on one pack, onto `out`."""
    for _ in range(repeats):
        if ctx.claude is None:
            out.append({"call": None, "error": "Opus was off for this run"})
            continue
        skipped = state.opus_skipped(ctx.clock)
        if skipped:
            out.append({"call": None, "error": f"not asked: {skipped}"})
            continue
        run, own = opus.opus_call(pack, ctx.call_prompt, claude=ctx.claude)
        state.claude_ran(run)
        out.append(
            {
                "call": own.call,
                "confidence": own.confidence,
                "lean": own.lean,
                "lean_error": own.lean_error,
                "error": own.error,
                "agent": run.to_dict(),
            }
        )


def measure(
    ticker: str, company: str, ctx: Context, state: State, repeats: int, run_id: str = ""
) -> dict[str, Any]:
    """One stock's repeated calls. Never raises: a failure in code is recorded on the
    stock, after the runs it had already finished."""
    out: dict[str, Any] = {"ticker": ticker, "company": company}
    try:
        pack = ctx.gather(ticker, ctx.now)
    except Exception as exc:
        return out | {"error": f"no facts pack ({type(exc).__name__})"}
    if pack is None:
        return out | {"error": "no prices"}
    out |= {"day": pack.day.isoformat(), "jev": [], "opus": []}
    try:
        jev_runs(pack, ctx, state, repeats, run_id, out["jev"])
    except Exception as exc:
        out["jev"].append({"call": None, "error": f"failed in code ({type(exc).__name__})"})
    try:
        opus_runs(pack, ctx, state, repeats, out["opus"])
    except Exception as exc:
        out["opus"].append({"call": None, "error": f"failed in code ({type(exc).__name__})"})
    return out


def _spread(values: Sequence[float | None]) -> float | None:
    """How far a number moved across runs: highest less lowest, over runs that gave one."""
    got = [v for v in values if v is not None and math.isfinite(v)]
    return max(got) - min(got) if len(got) >= 2 else None


def _attempted(run: dict[str, Any]) -> bool:
    """The method was really asked on this run (it may still have failed)."""
    error = str(run.get("error") or "")
    return bool(run.get("call")) or not error.startswith(("not asked", "Opus was off"))


def summarize(stocks: Sequence[dict[str, Any]], method: str, measure_key: str) -> dict[str, Any]:
    """Per method: stocks with every run answered, how many kept one call, and the moves.

    A stock counts only when every run gave a call: a run that failed says
    nothing about steadiness, so it is left out rather than read as a change."""
    asked = sum(1 for stock in stocks if any(_attempted(r) for r in stock.get(method) or []))
    complete: list[dict[str, Any]] = []
    for stock in stocks:
        runs = stock.get(method) or []
        if runs and all(r.get("call") for r in runs):
            complete.append(stock)
    changed = []
    spreads: list[float] = []
    for stock in complete:
        runs = stock[method]
        calls = [r["call"] for r in runs]
        if len(set(calls)) > 1:
            changed.append({"ticker": stock["ticker"], "calls": calls})
        moved = _spread([r.get(measure_key) for r in runs])
        if moved is not None:
            spreads.append(moved)
    return {
        "asked": asked,
        "stocks": len(complete),
        "same_call": len(complete) - len(changed),
        "changed": changed,
        "measure": measure_key,
        "median_move": statistics.median(spreads) if spreads else None,
        "largest_move": max(spreads) if spreads else None,
    }


def lines(summary: dict[str, Any], repeats: int) -> list[str]:
    out = []
    for name, s in (("Jev", summary["jev"]), ("Opus", summary["opus"])):
        left = s["asked"] - s["stocks"]
        out_of = f" ({left} more left out: a run failed)" if left else ""
        if not s["stocks"]:
            out.append(f"{name}: no stock got a call on every run{out_of}")
            continue
        move = ""
        if s["median_move"] is not None:
            what = "score" if s["measure"] == "score" else "confidence"
            move = (
                f"; {what} moved by a median {s['median_move']:.2f}, "
                f"at most {s['largest_move']:.2f}"
            )
        out.append(
            f"{name}: same call on {s['same_call']} of {s['stocks']} stocks "
            f"with all {repeats} runs answered{move}{out_of}"
        )
        for c in s["changed"]:
            out.append(f"  {c['ticker']} changed: {' / '.join(c['calls'])}")
    return out


def main(argv: list[str] | None = None, *, ctx: Context | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ask both calls several times on the same facts.")
    ap.add_argument("--ticker", action="append", help="these stocks only")
    ap.add_argument(
        "--repeats", type=int, default=REPEATS, help=f"runs per method, 2 to {MAX_REPEATS}"
    )
    ap.add_argument("--no-opus", action="store_true", help="Jev only")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not 2 <= args.repeats <= MAX_REPEATS:
        print(f"--repeats must be 2 to {MAX_REPEATS}")
        return 2
    watch = load_watchlist()
    tickers = list(dict.fromkeys(t.upper() for t in args.ticker or watch.tickers))
    bad = [t for t in tickers if not TICKER.fullmatch(t) or t not in watch.tickers]
    if bad:
        print(f"not watchlist tickers: {bad}")
        return 2
    if ctx is None:
        questions = jev.load_questions()
        ctx = Context(
            questions=questions,
            rules=load_rules(questions),
            call_prompt=opus.load_call_prompt(),
            why_prompt=opus.load_why_prompt(),
            now=datetime.now(UTC),
        )
    if args.no_opus:
        ctx.claude = None
    if ctx.claude is not None:
        if args.repeats * len(tickers) > MAX_OPUS_RUNS:
            print(f"{args.repeats} runs on {len(tickers)} stocks is over {MAX_OPUS_RUNS} Opus runs")
            return 2
        if in_daily_window(ctx.now):
            print(
                "refused: the daily run uses the same Claude subscription on weekday "
                "evenings (20:00 to 02:00 UTC). Run it outside that window, or with --no-opus."
            )
            return 2
    state = State(deadline=ctx.clock() + OPUS_BUDGET_S)
    run_id = uuid.uuid4().hex
    record: dict[str, Any] = {
        "kind": KIND,
        "run_id": run_id,
        "day": None,
        "made_at": ctx.now.isoformat(),
        "started_at": time.time(),
        "ended_at": None,
        "complete": False,
        "repeats": args.repeats,
        "versions": {
            "questions": ctx.questions.version,
            "rules": ctx.rules.version,
            "rules_sha256": ctx.rules.sha256,
            "call_prompt": ctx.call_prompt.version,
        },
        "opus_on": ctx.claude is not None,
        "jev_cost": 0.0,
        "summary": None,
        "stocks": [],
    }
    path = args.runs / CONSISTENCY_DIR / f"{ctx.now.date().isoformat()}-{run_id[:8]}.json"
    stocks: list[dict[str, Any]] = record["stocks"]
    for ticker in tickers:
        stock = measure(ticker, watch.names.get(ticker, ticker), ctx, state, args.repeats, run_id)
        stocks.append(stock)
        log.info(
            "%s: Jev %s, Opus %s",
            ticker,
            [r.get("call") for r in stock.get("jev", [])],
            [r.get("call") for r in stock.get("opus", [])],
        )
        _finish(record, state, stocks)
        _write(path, record)  # after every stock: a run cut off keeps what it got
    record |= {"complete": True, "ended_at": time.time()}
    _write(path, record)
    for line in lines(record["summary"], args.repeats):
        print(line)
    print(f"Jev cost ${state.jev_cost:.4f}; record {path}")
    called = any(r.get("call") for s in stocks for m in ("jev", "opus") for r in s.get(m) or [])
    return 0 if called else 1


def _finish(record: dict[str, Any], state: State, stocks: list[dict[str, Any]]) -> None:
    days = sorted({s["day"] for s in stocks if "day" in s})
    record |= {
        "day": days[-1] if days else None,
        "jev_cost": state.jev_cost,
        "summary": {
            "jev": summarize(stocks, "jev", "score"),
            "opus": summarize(stocks, "opus", "confidence"),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
