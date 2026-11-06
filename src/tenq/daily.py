"""The daily run: both calls on every watchlist stock after the close, one
email, one record, and a trace per stock.

    python -m tenq.daily                  # every watchlist stock, the latest session
    python -m tenq.daily --no-email
    python -m tenq.daily --ticker CRWD    # a check on a few stocks (see below)

For each stock: the facts pack for its latest settled session; Jev answers
the question file and the rules turn the answers into Jev's call; Opus makes
its own call from the same pack and writes the why for Jev's call; code
checks every reason (tenq.reasons). Then one email with both calls per stock
(tenq.daily_email) and one Langfuse trace per stock, grouped by day.

The record, runs/daily/<day>.json, is written before the email goes out and
holds everything the email is rendered from, plus each stock's close and
QQQ's, which is what the scorecard needs to score the calls later. So:
- a run on a day whose record says the email went out stops at once: a
  holiday (the latest session is one already done) or a rerun sends nothing
  twice;
- a run on a day whose record says the email failed sends it again from the
  record, without asking either model again.

A run with --ticker is a check: its record is runs/daily/<day>.check-<id>.json,
it never stops or replaces the day's full run, and the scorecard ignores it.

Fails open: if Jev is down, Opus's calls still go out, marked; if Claude is
down, Jev's calls go out without the why. Exits 1 when the email fails, when
no stock got a call from either method, or when no settled session is found.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from tenq import jev, opus
from tenq.brief import Prompt
from tenq.config import load_watchlist
from tenq.daily_email import render
from tenq.deliver import send
from tenq.filings import TICKER
from tenq.jev import QuestionSet, Result
from tenq.jev_check import Tally
from tenq.llm import call_claude
from tenq.opus import Claude
from tenq.pack import Pack, gather
from tenq.prices import BENCHMARK, Prices, YahooPrices, fetch
from tenq.reasons import ReasonCheck, check_reasons
from tenq.rules import Call, Rules, decide, load_rules
from tenq.trace import send_daily_traces

log = logging.getLogger(__name__)

KIND = "daily"
DAILY_DIR = "daily"

Gather = Callable[[str, datetime], Pack | None]
Ask = Callable[..., Result]
Mailer = Callable[[Any], tuple[bool, str]]
Tracer = Callable[[dict[str, Any]], tuple[bool, str, dict[str, str]]]


@dataclass
class Context:
    """What every stock in one run shares."""

    questions: QuestionSet
    rules: Rules
    call_prompt: Prompt
    why_prompt: Prompt
    now: datetime
    gather: Gather = gather
    ask: Ask = jev.ask
    claude: Claude | None = call_claude  # None: Opus is off for this run


@dataclass
class State:
    """What one stock's failure tells the rest of the run."""

    jev_down: bool = False  # the service or the key failed: skip Jev from here on
    claude_down: bool = False  # the subscription's limit was hit: skip Opus from here on
    tally: Tally = field(default_factory=Tally)
    jev_cost: float = 0.0


class Steps:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, start: float, output: Any = None, error: str | None = None) -> None:
        self.items.append(
            {"name": name, "start": start, "end": time.time(), "output": output, "error": error}
        )


def session(now: datetime, prices: Callable[[], Prices] | None = None) -> tuple[date, float] | None:
    """The latest session QQQ has settled by `now`, and its close."""
    got = prices() if prices is not None else fetch([], YahooPrices(), now=now)
    bars = got.bars.get(BENCHMARK)
    return (bars[-1].day, bars[-1].close) if bars else None


def _checked(checked: ReasonCheck) -> dict[str, Any]:
    return {
        "kept": [{"text": r.text, "cites": list(r.cites)} for r in checked.kept],
        "dropped": [
            {"text": d.reason.text, "cites": list(d.reason.cites), "why": d.why}
            for d in checked.dropped
        ],
    }


def _call_dict(call: Call) -> dict[str, Any]:
    return call.to_dict() | {"explain": call.explain()}


def run_stock(ticker: str, company: str, ctx: Context, state: State) -> dict[str, Any]:
    """Both calls for one stock, as a record entry. Never raises."""
    steps = Steps()
    out: dict[str, Any] = {"ticker": ticker, "company": company, "steps": steps.items}
    t = time.time()
    try:
        pack = ctx.gather(ticker, ctx.now)
    except Exception as exc:  # the type only: exception text is not trusted to be clean
        steps.add("facts pack", t, None, type(exc).__name__)
        return out | {"error": f"no facts pack ({type(exc).__name__})"}
    if pack is None:
        steps.add("facts pack", t, None, "no prices")
        return out | {"error": "no prices"}
    steps.add(
        "facts pack",
        t,
        {"numbers": len(pack.numbers), "reports": len(pack.reports), "headlines": len(pack.news)},
    )
    close = pack.number(f"px:{ticker}:close")
    out |= {
        "day": pack.day.isoformat(),
        "close": None if close is None else close.value,
        "pack": pack.to_dict(),
    }

    call: Call | None = None
    if state.jev_down:
        out["jev"] = {"error": "not asked: Jev was down earlier in this run"}
    else:
        t = time.time()
        result = ctx.ask(pack.render(), ctx.questions, session_id=f"tenq-daily-{ticker}-{pack.day}")
        state.jev_down |= result.down
        state.jev_cost += result.cost
        call = decide(pack, result, ctx.rules, ctx.questions) if result.answers else None
        steps.add("jev", t, {"answers": len(result.answers)}, result.error)
        out["jev"] = result.to_dict()
    if call is not None:
        out["call"] = _call_dict(call)
        state.tally.jev.append(call)

    if ctx.claude is None:
        out["opus"] = {"error": "Opus was off for this run"}
        return out
    if state.claude_down:
        out["opus"] = {"error": "not asked: Claude's usage limit was hit earlier in this run"}
        return out
    t = time.time()
    run, own = opus.opus_call(pack, ctx.call_prompt, claude=ctx.claude)
    state.claude_down |= run.usage_limit
    steps.add("opus call", t, {"call": own.call}, own.error)
    out["opus"] = {
        "call": own.call,
        "confidence": own.confidence,
        "error": own.error,
        **_checked(check_reasons(own.reasons, pack)),
        "agent": run.to_dict(),
    }
    if own.call is not None:
        state.tally.opus.append(own.call)
    if call is not None and not state.claude_down:
        t = time.time()
        run, why = opus.explain_call(pack, call, ctx.why_prompt, claude=ctx.claude)
        state.claude_down |= run.usage_limit
        steps.add("opus why", t, {"reasons": len(why.reasons)}, why.error)
        out["why"] = {
            "error": why.error,
            **_checked(check_reasons(why.reasons, pack)),
            "agent": run.to_dict(),
        }
    elif call is not None:
        out["why"] = {"error": "not asked: Claude's usage limit was hit earlier in this run"}
    return out


def called(stock: dict[str, Any]) -> bool:
    """Either method made a call on this stock."""
    return "call" in stock or bool((stock.get("opus") or {}).get("call"))


def record_path(runs: Path, day: date, check_id: str | None = None) -> Path:
    name = day.isoformat() if check_id is None else f"{day.isoformat()}.check-{check_id}"
    return runs / DAILY_DIR / f"{name}.json"


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _emailed(record: dict[str, Any]) -> bool:
    email = record.get("email")
    return isinstance(email, dict) and email.get("sent") is True


def deliver(
    path: Path,
    record: dict[str, Any],
    *,
    email: bool,
    mailer: Mailer = send,
    tracer: Tracer = send_daily_traces,
    trace: bool = True,
) -> dict[str, Any]:
    """Email and trace a written record, writing it again after each."""
    message = render(record)
    if email:
        sent, status = mailer(message)
    else:
        sent, status = False, "email skipped: --no-email"
    record["email"] = {"sent": sent, "status": status, "subject": message.subject}
    _write(path, record)
    if trace:
        stored, trace_status, ids = tracer(record)
        record["trace"] = {"stored": stored, "status": trace_status, "ids": ids}
        _write(path, record)
    log.info("daily %s: %s, %s", record["day"], status, record["trace"]["status"])
    return record


def run_day(
    tickers: list[str],
    names: dict[str, str],
    day: date,
    bench_close: float,
    ctx: Context,
) -> dict[str, Any]:
    """Every stock's calls, as the day's record (email and trace not yet sent)."""
    run_id = str(uuid.uuid4())
    started = time.time()
    state = State()
    stocks = []
    for ticker in tickers:
        stock = run_stock(ticker, names.get(ticker, ticker), ctx, state)
        log.info(
            "%s: Jev %s, Opus %s",
            ticker,
            (stock.get("call") or {}).get("call"),
            (stock.get("opus") or {}).get("call"),
        )
        stocks.append(stock)
    return {
        "kind": KIND,
        "run_id": run_id,
        "day": day.isoformat(),
        "started_at": started,
        "ended_at": time.time(),
        "versions": {
            "questions": ctx.questions.version,
            "rules": ctx.rules.version,
            "rules_sha256": ctx.rules.sha256,
            "call_prompt": ctx.call_prompt.version,
            "why_prompt": ctx.why_prompt.version,
        },
        "benchmark": {"ticker": BENCHMARK, "day": day.isoformat(), "close": bench_close},
        "opus_on": ctx.claude is not None,
        "stocks": stocks,
        "balance": state.tally.lines(),
        "jev_cost": state.jev_cost,
        "email": {"sent": False, "status": "email pending"},
        "trace": {"stored": False, "status": "trace pending", "ids": {}},
    }


def main(
    argv: list[str] | None = None,
    *,
    ctx: Context | None = None,
    prices: Callable[[], Prices] | None = None,
    mailer: Mailer = send,
    tracer: Tracer = send_daily_traces,
) -> int:
    ap = argparse.ArgumentParser(description="Both daily calls on the watchlist, emailed.")
    ap.add_argument("--ticker", action="append", help="a check on these stocks only")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--no-opus", action="store_true", help="Jev's calls only")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

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
            rules=load_rules(questions),  # a bad file fails here, before anything is paid for
            call_prompt=opus.load_call_prompt(),
            why_prompt=opus.load_why_prompt(),
            now=datetime.now(UTC),
        )
    if args.no_opus:
        ctx.claude = None

    found = session(ctx.now, prices)
    if found is None:
        print(f"no settled {BENCHMARK} session found: nothing run")
        return 1
    day, bench_close = found
    check_id = uuid.uuid4().hex[:8] if args.ticker else None
    path = record_path(args.runs, day, check_id)
    email = not args.no_email
    if check_id is None and path.exists():
        done = _read(path)
        if done is not None and _emailed(done):
            print(f"{day}: already emailed, nothing to do")
            return 0
        if done is not None and email:
            print(f"{day}: recorded but not emailed, sending it again from the record")
            done = deliver(path, done, email=True, mailer=mailer, tracer=tracer, trace=False)
            return 0 if _emailed(done) else 1
        if done is not None:
            print(f"{day}: already recorded, nothing to do with --no-email")
            return 0

    record = run_day(tickers, watch.names, day, bench_close, ctx)
    # written before the email: once it is out, a cancelled job must still
    # leave the record that stops the next run sending it again
    _write(path, record)
    record = deliver(path, record, email=email, mailer=mailer, tracer=tracer)
    for line in record["balance"]:
        print(line)
    print(f"total Jev cost ${record['jev_cost']:.6f}")
    print(record["email"]["status"])
    print(record["trace"]["status"])
    if email and not record["email"]["sent"]:
        return 1
    return 0 if any(called(s) for s in record["stocks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
