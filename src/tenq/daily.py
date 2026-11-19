"""The daily run: three calls on each stock the screen picks after the close
(tenq.screen; the watchlist if the screen fails), one email, one record, and a
trace per stock.

    python -m tenq.daily                  # the screen's picks, the latest session
    python -m tenq.daily --ticker CRWD    # a check (see below)
    python -m tenq.daily --no-email       # a check that emails nothing

For each stock: the facts pack for its latest settled session; Jev answers
the question file and the rules turn the answers into Jev's call; Opus makes
its own call from the same pack and writes the why for Jev's call; code
checks every reason (tenq.reasons). Once Jev and Opus are done with every
stock, Kairos answers the same questions on each stock's same pack, and the
same rules turn its answers into its own call; Kairos never changes Jev's or
Opus's call, and its pass comes last so it never takes their time. Then one
email with the three calls per stock (tenq.daily_email) and one Langfuse
trace per stock, grouped by day.

The record, runs/daily/<day>.json, is written after every stock and before
the email goes out. It holds everything the email is rendered from, plus
each stock's close and QQQ's, and the time the calls were made (made_at).
Those closes are from before the calls were made, since the calls read news
up to made_at: they are kept for reference and are not the price a call is
scored from. So:
- a run on a day whose record says the email went out stops at once: a
  holiday (the latest session is one already done) or a rerun sends nothing
  twice;
- a run on a day whose record says the email failed sends it again from the
  record, without asking any model again;
- a record that is unreadable, unfinished, or whose email may have gone out
  (it was claimed for sending, or the mail server's answer was lost) is never
  run or sent again by code: a person looks first.

In the workflow the run is three steps (--stage calls, claim, send). The
claim, which marks the email as being sent, is saved to main before the send
step runs. If that save fails, nothing was sent and a rerun is safe; once it
has worked, any later failure leaves a record that asks a person. A
scheduled run that fails before its email was sent sends a short failure
notice instead (--stage notice). A run that finds no bar yet for the session
the exchange's calendar expects fails, so a late price feed is never taken
for a holiday.

A run with --ticker, --no-email or --no-opus is a check: its record is
runs/daily/<day>.check-<id>.json, it never stops or replaces the day's full
run, and the scorecard ignores it.

Fails open: if Jev is down, Opus's calls still go out, marked; if Claude is
at its limit, fails twice in a row, or the run's time budget for it runs out,
Jev's calls go out without the why. Kairos failing alone changes nothing
else: its stock shows no Kairos call and a note saying why. Once Kairos is
found down it is not asked for the rest of its pass; after its last stock,
each stock it missed while down or failing is asked once more (after a
pause), and Kairos down again ends that pass. Kairos has a time bound of its
own (KAIROS_BUDGET_S, KAIROS_LAST_S), held by the clock: an ask still going
past it is left unanswered. With no KAIROS_API_KEY it is noted as not asked.
Kairos never changes the exit code. Exits 1 when the email fails, when no
stock got a call from Jev or Opus, when the day's record needs a person, or
when no settled session is found.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import holidays

from tenq import jev, opus
from tenq.brief import Prompt
from tenq.config import load_watchlist
from tenq.daily_email import render
from tenq.deliver import Email, send
from tenq.filings import TICKER
from tenq.jev import QuestionSet, Result
from tenq.llm import AgentRun, call_claude
from tenq.model_check import Tally
from tenq.opus import Claude
from tenq.pack import Pack, gather
from tenq.prices import BENCHMARK, MARKET_TZ, SESSION_SETTLED, Prices, YahooPrices, fetch
from tenq.reasons import ReasonCheck, check_reasons
from tenq.rules import Call, Rules, decide, load_rules
from tenq.scorecard import PriceFetch, weekly, yahoo
from tenq.screen import load_universe, picked, run_screen, universe_names
from tenq.trace import send_daily_traces

log = logging.getLogger(__name__)

KIND = "daily"
DAILY_DIR = "daily"
# Opus is skipped once the run passes this. The calls step's limit is 165 min: this
# leaves room for one claude run still going (up to 5 min) and Jev on the stocks left.
OPUS_BUDGET_S = 140 * 60
# Kairos is asked after Jev and Opus are done with every stock, so it only ever takes
# time they left. Its own bound: at most KAIROS_BUDGET_S in a run, its pass, pause and
# retries together (about a minute a stock when well: 20 to 25 minutes a day), so a slow
# Kairos delays the email by that much at most; and no Kairos request starts after
# KAIROS_LAST_S from the run's start. The one still going gets jev.KAIROS_TIMEOUT_S
# (5 min) more, then is left unanswered (_within), so Kairos is done by 155 min,
# inside the calls step's 165, however slowly its server answers.
KAIROS_BUDGET_S = 45 * 60
KAIROS_LAST_S = 150 * 60
# A stock Kairos missed is asked again only this long after its last failed ask:
# its 2026-11-19 outage was 30 s.
KAIROS_PAUSE_S = 2 * 60
KAIROS_DOWN = "Kairos was down earlier in this run"
KAIROS_DOWN_AGAIN = "Kairos was down again at the end of the run"
KAIROS_OUT = "the run's time for Kairos ran out"
KAIROS_LATE = "still going when the run's time for Kairos ran out; left unanswered"
CLAUDE_FAILS = 2  # this many failed claude runs in a row: Claude is treated as down
STAGES = ("all", "calls", "claim", "send", "notice")
SHEET_NOTE = "(this record's pack, rendered; message_sha256 is of the whole message)"

Gather = Callable[[str, datetime], Pack | None]
Ask = Callable[..., Result]
Mailer = Callable[[Any], tuple[bool | None, str]]
Tracer = Callable[..., tuple[bool, str, dict[str, str]]]  # (record, skip=tickers already stored)


def ask_kairos(state: str, questions: QuestionSet, **kw: Any) -> Result:
    """Kairos through the one adapter: Jev's questions on Kairos's endpoint."""
    return jev.ask(state, questions, provider=jev.KAIROS, **kw)


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
    opus_budget_s: float = OPUS_BUDGET_S
    clock: Callable[[], float] = time.monotonic
    kairos: Ask | None = ask_kairos  # None: Kairos is off for this run
    kairos_budget_s: float = KAIROS_BUDGET_S
    kairos_last_s: float = KAIROS_LAST_S
    # an ask still going this long after Kairos's time ran out is left unanswered
    kairos_grace_s: float = jev.KAIROS_TIMEOUT_S
    sleep: Callable[[float], None] = time.sleep  # Kairos's pause before its retries


@dataclass
class State:
    """What one stock's failure tells the rest of the run."""

    deadline: float  # on ctx.clock: Opus is not asked after it
    jev_down: bool = False  # the service or the key failed: skip Jev from here on
    claude_down: str | None = None  # why Opus is skipped from here on
    claude_fails: int = 0  # failed claude runs in a row
    tally: Tally = field(default_factory=Tally)
    jev_cost: float = 0.0
    kairos_last: float = math.inf  # on ctx.clock: no Kairos request starts after it
    kairos_off: str | None = None  # why Kairos is not asked again in this run, retries too
    kairos_down: str | None = None  # why Kairos is skipped for the rest of this pass
    kairos_failed_at: float | None = None  # on ctx.clock: its last ask with no call
    kairos_spent: float = 0.0  # seconds Kairos has taken in this run
    kairos_tokens: int = 0  # Kairos's input tokens in this run (it bills on tokens)
    # each stock with a facts pack, for Kairos's pass after Jev's and Opus's
    packs: list[tuple[dict[str, Any], Pack]] = field(default_factory=list)
    # stocks Kairos missed while down or failing, for one more ask at the end
    kairos_missed: list[tuple[dict[str, Any], Pack]] = field(default_factory=list)

    def claude_ran(self, run: AgentRun) -> None:
        if run.usage_limit:
            self.claude_down = "Claude's usage limit was hit"
        elif run.error:
            self.claude_fails += 1
            if self.claude_fails >= CLAUDE_FAILS:
                self.claude_down = f"Claude failed {self.claude_fails} times in a row"
        else:
            self.claude_fails = 0

    def opus_skipped(self, clock: Callable[[], float]) -> str | None:
        if self.claude_down is None and clock() > self.deadline:
            self.claude_down = "the run's time budget for Opus ran out"
        return self.claude_down

    def kairos_left(self, ctx: Context, since: float | None = None) -> float:
        """Seconds Kairos may still take (counting an ask begun at `since`): the
        less of what is left of its budget and the time to its last start."""
        now = ctx.clock()
        spent = self.kairos_spent + (0.0 if since is None else now - since)
        return min(ctx.kairos_budget_s - spent, self.kairos_last - now)

    def kairos_skipped(self, ctx: Context) -> str | None:
        """Why Kairos is not asked now; once its time is out, for good."""
        if self.kairos_off is None and self.kairos_left(ctx) <= 0:
            self.kairos_off = KAIROS_OUT
        return self.kairos_off or self.kairos_down

    def kairos_missed_one(self, out: dict[str, Any], pack: Pack, at: float | None) -> None:
        """Keep a stock for one more ask at the end; `at` is when an ask on it
        failed (None: it was skipped while Kairos was down, never asked)."""
        self.kairos_missed.append((out, pack))
        if at is not None:
            self.kairos_failed_at = at


class Steps:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.items: list[dict[str, Any]] = [] if items is None else items

    def add(self, name: str, start: float, output: Any = None, error: str | None = None) -> None:
        self.items.append(
            {"name": name, "start": start, "end": time.time(), "output": output, "error": error}
        )


def session(now: datetime, prices: Callable[[], Prices] | None = None) -> tuple[date, float] | None:
    """The latest session QQQ has settled by `now`, and its close."""
    got = prices() if prices is not None else fetch([], YahooPrices(), now=now)
    bars = got.bars.get(BENCHMARK)
    return (bars[-1].day, bars[-1].close) if bars else None


def expected_session(now: datetime) -> date:
    """The latest NYSE session that should have settled by `now`: today from
    16:30 New York on a trading day, else the trading day before. Weekends
    and the exchange's holidays are not sessions."""
    local = now.astimezone(MARKET_TZ)
    day = local.date()
    if local.time() < SESSION_SETTLED:
        day -= timedelta(days=1)
    closed = holidays.financial_holidays("NYSE", years=range(day.year - 1, day.year + 1))
    while day.weekday() >= 5 or day in closed:
        day -= timedelta(days=1)
    return day


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


def run_stock(ticker: str, company: str, day: date, ctx: Context, state: State) -> dict[str, Any]:
    """Jev's and Opus's calls for one stock, as a record entry (Kairos's come
    later, in kairos_pass). Never raises: a failure in
    code is recorded on this stock (its type only, since exception text is not
    trusted to be clean) and the run goes on to the next."""
    steps = Steps()
    out: dict[str, Any] = {"ticker": ticker, "company": company, "steps": steps.items}
    t = time.time()
    try:
        pack = ctx.gather(ticker, ctx.now)
    except Exception as exc:
        steps.add("facts pack", t, None, type(exc).__name__)
        return out | {"error": f"no facts pack ({type(exc).__name__})"}
    if pack is None:
        steps.add("facts pack", t, None, "no prices")
        return out | {"error": "no prices"}
    try:
        _facts(out, pack, ticker, day, steps, t)
    except Exception as exc:
        return out | {"error": f"the facts pack could not be read ({type(exc).__name__})"}
    state.packs.append((out, pack))  # Kairos is asked on it after the last stock

    call: Call | None = None
    try:
        call = _jev(out, pack, ticker, ctx, state, steps)
    except Exception as exc:
        out["jev"] = (out.get("jev") or {}) | {"error": f"failed in code ({type(exc).__name__})"}

    try:
        _opus(out, pack, call, ctx, state, steps)
    except Exception as exc:
        failed = f"failed in code ({type(exc).__name__})"
        if not (out.get("opus") or {}).get("call"):
            out["opus"] = (out.get("opus") or {}) | {"call": None, "error": failed}
        if call is not None and "why" not in out:
            out["why"] = {"error": failed}
    return out


def _facts(out: dict[str, Any], pack: Pack, ticker: str, day: date, steps: Steps, t: float) -> None:
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
    if pack.day != day:
        out["note"] = f"its latest price is from {pack.day}, not {day}"


def _jev(
    out: dict[str, Any], pack: Pack, ticker: str, ctx: Context, state: State, steps: Steps
) -> Call | None:
    """Jev's call, recorded on `out` and the tally together, or not at all."""
    if state.jev_down:
        out["jev"] = {"error": "not asked: Jev was down earlier in this run"}
        return None
    t = time.time()
    result = ctx.ask(pack.render(), ctx.questions, session_id=f"tenq-daily-{ticker}-{pack.day}")
    state.jev_down |= result.down
    state.jev_cost += result.cost
    out["jev"] = result.to_dict()
    call = decide(pack, result, ctx.rules, ctx.questions) if result.answers else None
    steps.add("jev", t, {"answers": len(result.answers)}, result.error)
    if call is None:
        return None
    out["call"] = _call_dict(call)
    state.tally.jev.append(call)
    return call


def _opus(
    out: dict[str, Any], pack: Pack, call: Call | None, ctx: Context, state: State, steps: Steps
) -> None:
    if ctx.claude is None:
        out["opus"] = {"error": "Opus was off for this run"}
        return
    skipped = state.opus_skipped(ctx.clock)
    if skipped:
        out["opus"] = {"error": f"not asked: {skipped}"}
        if call is not None:
            out["why"] = {"error": f"not asked: {skipped}"}
        return
    t = time.time()
    run, own = opus.opus_call(pack, ctx.call_prompt, claude=ctx.claude)
    state.claude_ran(run)
    steps.add("opus call", t, {"call": own.call}, own.error)
    out["opus"] = {
        "call": own.call,
        "confidence": own.confidence,
        "lean": own.lean,
        "lean_error": own.lean_error,
        "error": own.error,
        **_checked(check_reasons(own.reasons, pack)),
        "agent": run.to_dict(),
    }
    if own.call is not None:
        state.tally.opus.append(own.call)
    if call is None:
        return
    skipped = state.opus_skipped(ctx.clock)
    if skipped:
        out["why"] = {"error": f"not asked: {skipped}"}
        return
    t = time.time()
    run, why = opus.explain_call(pack, call, ctx.why_prompt, claude=ctx.claude)
    state.claude_ran(run)
    steps.add("opus why", t, {"reasons": len(why.reasons)}, why.error)
    out["why"] = {
        "error": why.error,
        # what Opus was shown, without a second copy of the facts sheet: the sheet
        # is this record's pack, rendered, and the hash pins the whole message
        "message": f"{why.message.split(opus.SHEET_HEADER, 1)[0]}{opus.SHEET_HEADER} {SHEET_NOTE}",
        "message_sha256": hashlib.sha256(why.message.encode("utf-8")).hexdigest(),
        **_checked(check_reasons(why.reasons, pack)),
        "agent": run.to_dict(),
    }


def kairos_pass(
    record: dict[str, Any], ctx: Context, state: State, save: Callable[[dict[str, Any]], None]
) -> None:
    """Kairos's call on each stock with a facts pack, once Jev and Opus are
    done with every stock; then once more on each stock it missed while down
    or failing (kairos_again). Never raises. The record is saved after each."""
    for out, pack in state.packs:
        try:
            _kairos(out, pack, ctx, state)
        except Exception as exc:
            _kairos_failed(out, exc)
            state.kairos_missed_one(out, pack, ctx.clock())
        log.info("%s: Kairos %s", out["ticker"], (out.get("kairos_call") or {}).get("call"))
        record["kairos_input_tokens"] = state.kairos_tokens
        save(record)
    kairos_again(record, ctx, state, save)


def _kairos_failed(out: dict[str, Any], exc: Exception) -> None:
    out["kairos"] = (out.get("kairos") or {}) | {"error": f"failed in code ({type(exc).__name__})"}
    out.pop("kairos_call", None)


def _kairos(out: dict[str, Any], pack: Pack, ctx: Context, state: State) -> None:
    """Kairos's call on one stock in its pass. A stock it missed while down or
    failing is kept for one more ask at the end (kairos_again)."""
    skipped = state.kairos_skipped(ctx)
    if skipped:
        out["kairos"] = {"error": f"not asked: {skipped}"}
        if state.kairos_off is None:  # skipped only because Kairos is down
            state.kairos_missed_one(out, pack, None)
        return
    if _ask_kairos(out, pack, ctx, state, "kairos", KAIROS_DOWN) is None:
        state.kairos_missed_one(out, pack, ctx.clock())


def _ask_kairos(
    out: dict[str, Any], pack: Pack, ctx: Context, state: State, step: str, down: str
) -> Call | None:
    """Kairos asked Jev's questions on the same pack, and its call by the same
    rules. `down` is why the stocks after this one skip Kairos if this ask
    finds it down."""
    kairos = ctx.kairos
    assert kairos is not None  # kairos_skipped said it is on
    t, began = time.time(), ctx.clock()
    facts, sid = pack.render(), f"tenq-daily-{out['ticker']}-{pack.day}"
    try:
        result = _within(
            lambda: kairos(
                facts,
                ctx.questions,
                session_id=sid,
                time_left=lambda: state.kairos_left(ctx, began),
            ),
            max(state.kairos_left(ctx), 0.0) + ctx.kairos_grace_s,
        )
    finally:
        state.kairos_spent += ctx.clock() - began
    if result is None:
        state.kairos_off = KAIROS_OUT
        out["kairos"] = {"error": KAIROS_LATE}
        Steps(out["steps"]).add(step, t, {"answers": 0}, KAIROS_LATE)
        return None
    if result.down:
        state.kairos_down = down
    state.kairos_tokens += result.input_tokens
    out["kairos"] = result.to_dict()
    call = decide(pack, result, ctx.rules, ctx.questions) if result.answers else None
    Steps(out["steps"]).add(step, t, {"answers": len(result.answers)}, result.error)
    if call is None:
        return None
    out["kairos_call"] = _call_dict(call)
    state.tally.kairos.append(call)
    return call


def _within(ask: Callable[[], Result], seconds: float) -> Result | None:
    """ask() in a thread of its own; None once `seconds` pass with it still
    going. The socket timeout bounds each read, not a whole reply, so only
    this holds Kairos to its time. The thread left behind is a daemon: it
    starts no new round (its time_left is out) and the process does not wait
    for it. An exception in ask() is raised here."""
    done: list[Result] = []
    failed: list[Exception] = []

    def run() -> None:
        try:
            done.append(ask())
        except Exception as exc:
            failed.append(exc)

    worker = threading.Thread(target=run, name="kairos", daemon=True)
    worker.start()
    worker.join(seconds)
    if failed:
        raise failed[0]
    return done[0] if done else None


def kairos_again(
    record: dict[str, Any], ctx: Context, state: State, save: Callable[[dict[str, Any]], None]
) -> None:
    """After Kairos's last stock, ask it once more on each stock it missed
    while down or failing (never one with no facts pack: it never got that
    far), first waiting until KAIROS_PAUSE_S after its last failed ask.
    Kairos down again, or out of time, ends the pass, and each stock it did
    not reach says why. The record is saved after each ask."""
    missed, state.kairos_missed = state.kairos_missed, []
    if not missed:
        return
    state.kairos_down = None
    if state.kairos_failed_at is not None and not state.kairos_skipped(ctx):
        wait = state.kairos_failed_at + KAIROS_PAUSE_S - ctx.clock()
        if wait >= state.kairos_left(ctx):
            state.kairos_off = KAIROS_OUT
        elif wait > 0:
            began = ctx.clock()
            ctx.sleep(wait)
            state.kairos_spent += ctx.clock() - began
    for n, (out, pack) in enumerate(missed):
        skipped = state.kairos_skipped(ctx)
        if skipped:
            for left, _ in missed[n:]:
                left["kairos"]["again"] = f"not asked again: {skipped}"
            save(record)
            return
        first = out.pop("kairos")
        try:
            _ask_kairos(out, pack, ctx, state, "kairos again", KAIROS_DOWN_AGAIN)
        except Exception as exc:
            _kairos_failed(out, exc)
        out["kairos"]["first"] = first  # what Kairos's first pass recorded
        log.info("%s: Kairos again %s", out["ticker"], (out.get("kairos_call") or {}).get("call"))
        record["kairos_input_tokens"] = state.kairos_tokens
        save(record)


def called(stock: dict[str, Any]) -> bool:
    """Jev or Opus made a call on this stock. Kairos's call does not count: it never
    changes the exit code, as it never changes the other two calls."""
    return "call" in stock or bool((stock.get("opus") or {}).get("call"))


def why_no_kairos(ctx: Context) -> str | None:
    """Why Kairos is not asked at all in this run, or None when it is."""
    if ctx.kairos is None:
        return "Kairos was off for this run"
    if not os.environ.get(jev.KAIROS.key_env):
        return f"{jev.KAIROS.key_env} not set"
    return None


def record_path(runs: Path, day: date, check_id: str | None = None) -> Path:
    name = day.isoformat() if check_id is None else f"{day.isoformat()}.check-{check_id}"
    return runs / DAILY_DIR / f"{name}.json"


def _write(path: Path, record: dict[str, Any]) -> None:
    """Whole or not at all: a job stopped mid-write leaves the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    # compact: a day's record is read by code, and indenting it added about 15%
    tmp.write_text(json.dumps(record, separators=(",", ":"), default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _emailed(record: dict[str, Any]) -> bool:
    email = record.get("email")
    return isinstance(email, dict) and email.get("sent") is True


def _needs_a_person(record: dict[str, Any] | None) -> str | None:
    """Why code must not run or send this day's record again, if it must not."""
    if record is None:
        return "its record is unreadable"
    if record.get("complete") is not True:
        return "its record is unfinished (the job stopped mid-run)"
    email = record.get("email")
    if not isinstance(email, dict):
        return "its email entry is unreadable"
    sent = email.get("sent", False)
    if sent is not True and sent is not False:  # claimed, the answer lost, or a hand edit
        return f"its email may have gone out ({email.get('status')})"
    return None


def _output(**values: str) -> None:
    """Tell the workflow's later steps what to do (no-op outside Actions)."""
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.writelines(f"{k}={v}\n" for k, v in values.items())


def _claim(path: Path, record: dict[str, Any], token: str) -> None:
    """Mark the email as being sent by this job. From here a run that finds the
    record asks a person, so whatever stops the job, the email never goes twice."""
    record["email"] = {"sent": None, "status": "sending", "claim": token}
    _write(path, record)


def _mail(path: Path, record: dict[str, Any], mailer: Mailer) -> dict[str, Any]:
    message = render(record)
    sent, status = mailer(message)  # sent None: the server may have taken it
    record["email"] = {"sent": sent, "status": status, "subject": message.subject}
    _write(path, record)
    return record


def _trace(path: Path, record: dict[str, Any], tracer: Tracer) -> dict[str, Any]:
    """Trace every stock not already stored, keeping the ids of those that were."""
    done = dict((record.get("trace") or {}).get("ids") or {})
    stored, status, ids = tracer(record, skip=set(done))
    record["trace"] = {"stored": stored, "status": status, "ids": done | ids}
    _write(path, record)
    return record


def deliver(
    path: Path,
    record: dict[str, Any],
    *,
    email: bool,
    mailer: Mailer = send,
    tracer: Tracer = send_daily_traces,
) -> dict[str, Any]:
    """Email and trace a finished, written record in one go, writing it again
    after each. The workflow does the same in steps instead, saving the claim
    to main before the email goes out (see main's --stage)."""
    if email:
        _claim(path, record, "local")
        record = _mail(path, record, mailer)
    else:
        record["email"] = {"sent": False, "status": "email skipped: --no-email"}
        _write(path, record)
    if (record.get("trace") or {}).get("stored") is not True:
        record = _trace(path, record, tracer)
    log.info(
        "daily %s: %s, %s", record["day"], record["email"]["status"], record["trace"]["status"]
    )
    return record


def _summary(record: dict[str, Any]) -> None:
    for line in record["balance"]:
        print(line)
    print(f"total Jev cost ${record['jev_cost']:.6f}")
    print(f"total Kairos input tokens {record['kairos_input_tokens']}")


def _finish(record: dict[str, Any], email: bool) -> int:
    print(record["email"]["status"])
    print(record["trace"]["status"])
    if email and record["email"]["sent"] is not True:
        return 1
    return 0 if any(called(s) for s in record["stocks"]) else 1


def send_claimed(path: Path, token: str, mailer: Mailer, tracer: Tracer) -> int:
    """--stage send: email and trace a record this job claimed."""
    record = _read(path)
    email = (record or {}).get("email") or {}
    if record is None or email.get("status") != "sending" or email.get("claim") != token:
        print(f"{path.name}: not claimed by this job, nothing sent")
        return 1
    record = _mail(path, record, mailer)
    # the failure notice reads this, not the step's exit code: a sent email
    # with no calls in it still exits 1, and must not be followed by a notice
    sent = record["email"]["sent"]
    _output(emailed="yes" if sent is True else "no" if sent is False else "maybe")
    record = _trace(path, record, tracer)
    return _finish(record, email=True)


def claim(path: Path, token: str) -> int:
    """--stage claim: mark a finished, unsent record as this job's to send."""
    record = _read(path)
    why = _needs_a_person(record)
    if why is not None or record is None:
        print(f"{path.name}: {why}; not claimed")
        return 1
    if _emailed(record):
        print(f"{path.name}: already emailed; not claimed")
        return 1
    render(record)  # a record that cannot be emailed fails here, before the claim reaches main
    _claim(path, record, token)
    print(f"{path.name}: claimed for this job's email")
    return 0


def run_day(
    tickers: list[str],
    names: dict[str, str],
    day: date,
    bench_close: float,
    ctx: Context,
    *,
    check_id: str | None = None,
    screen: dict[str, Any] | None = None,
    started: float | None = None,
    save: Callable[[dict[str, Any]], None] = lambda record: None,
) -> dict[str, Any]:
    """Every stock's calls, as the day's record (email and trace not yet sent).
    `save` gets the record after every stock, unfinished until the last."""
    # The budgets run from `started` (before the screen), so screen time counts too.
    start = ctx.clock() if started is None else started
    state = State(
        deadline=start + ctx.opus_budget_s,
        kairos_last=start + ctx.kairos_last_s,
        kairos_off=why_no_kairos(ctx),
    )
    record: dict[str, Any] = {
        "kind": KIND,
        "run_id": str(uuid.uuid4()),
        "check_id": check_id,
        "day": day.isoformat(),
        # the calls read news up to this moment, which can be after `day`'s close
        "made_at": ctx.now.isoformat(),
        "started_at": time.time(),
        "ended_at": None,
        "complete": False,
        "versions": {
            "questions": ctx.questions.version,
            "rules": ctx.rules.version,
            "rules_sha256": ctx.rules.sha256,
            "call_prompt": ctx.call_prompt.version,
            "why_prompt": ctx.why_prompt.version,
            # the commit that was checked out and ran (set by the workflow from git,
            # not GITHUB_SHA, which is the commit that triggered the run), so the
            # facts sheet can be rendered again exactly
            "code": os.environ.get("TENQ_CODE_SHA", ""),
        },
        "benchmark": {"ticker": BENCHMARK, "day": day.isoformat(), "close": bench_close},
        "opus_on": ctx.claude is not None,
        # Kairos is asked (it has a key and is not off); each stock says why not
        "kairos_on": state.kairos_off is None,
        # which stocks were asked and why: the screen's result, or why it failed
        # and the watchlist ran instead; None for a check run on named stocks
        "screen": screen,
        "stocks": [],
        "balance": [],
        "jev_cost": 0.0,
        "kairos_input_tokens": 0,
        "email": {"sent": False, "status": "email pending"},
        "trace": {"stored": False, "status": "trace pending", "ids": {}},
    }
    for ticker in tickers:
        stock = run_stock(ticker, names.get(ticker, ticker), day, ctx, state)
        log.info(
            "%s: Jev %s, Opus %s",
            ticker,
            (stock.get("call") or {}).get("call"),
            (stock.get("opus") or {}).get("call"),
        )
        record["stocks"].append(stock)
        record["jev_cost"] = state.jev_cost
        save(record)
    kairos_pass(record, ctx, state, save)
    record |= {"ended_at": time.time(), "complete": True, "balance": state.tally.lines()}
    save(record)
    return record


def notice(run_url: str, mailer: Mailer = send) -> int:
    """Email that a scheduled run failed and its daily email was not sent, or
    may not have been.

    Separate from the daily email and never instead of it: the person opens
    the run and the record, then follows the README's daily section."""
    if not run_url.startswith("https://github.com/"):
        print("--run-url must be the run's GitHub address")
        return 2
    text = (
        "The daily run failed, so its daily email was not sent, or may not "
        f"have been.\n\nThe run: {run_url}\n\nWhat to do next is in the "
        "README, under The daily calls."
    )
    body = html.escape(text).replace("\n\n", "</p><p>")
    sent, status = mailer(Email("tenq daily run failed", text, f"<p>{body}</p>"))
    print(f"failure notice: {status}")
    return 0 if sent else 1


def choose(
    result: dict[str, Any], day: date, watchlist: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """The stocks to ask today: the screen's picks, or the watchlist when the
    screen failed or screened a different session. Either way the result is
    kept for the record, marked with what was used, and a failure is loud: it
    heads the email and the log."""
    if result.get("ok") and result.get("day") == day.isoformat():
        return result | {"used": "screen"}, picked(result)
    if result.get("ok"):
        result = result | {
            "ok": False,
            "refused": [f"it screened the {result.get('day')} session, not {day}"],
        }
    log.error("SCREEN FAILED (%s): running the watchlist", "; ".join(result.get("refused") or []))
    return result | {"used": "watchlist"}, list(watchlist)


def main(
    argv: list[str] | None = None,
    *,
    ctx: Context | None = None,
    prices: Callable[[], Prices] | None = None,
    screener: Callable[[datetime], dict[str, Any]] | None = None,
    score_prices: PriceFetch = yahoo,
    mailer: Mailer = send,
    tracer: Tracer = send_daily_traces,
) -> int:
    ap = argparse.ArgumentParser(
        description="The daily calls on the screen's picks (the watchlist if it fails), emailed."
    )
    ap.add_argument("--ticker", action="append", help="a check on these stocks only")
    ap.add_argument("--no-email", action="store_true", help="a check that emails nothing")
    ap.add_argument("--no-opus", action="store_true", help="a check without Opus's calls")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="all: in one go; calls, claim, send: the workflow's steps, which save the "
        "claim to main between claim and send; notice: email that a run failed",
    )
    ap.add_argument("--record", type=Path, help="claim, send: the record the calls stage wrote")
    ap.add_argument("--claim", help="claim, send: this job's token")
    ap.add_argument("--run-url", help="notice: the failed run's address")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.stage == "notice":
        if not args.run_url:
            print("--stage notice needs --run-url")
            return 2
        return notice(args.run_url, mailer)
    if args.stage in ("claim", "send"):
        if args.record is None or not args.claim:
            print(f"--stage {args.stage} needs --record and --claim")
            return 2
        if args.stage == "claim":
            return claim(args.record, args.claim)
        return send_claimed(args.record, args.claim, mailer, tracer)

    watch = load_watchlist()
    try:
        names, universe = universe_names(), set(load_universe())
    except Exception as e:  # a broken universe file: the screen fails too, so run the watchlist
        log.error("the universe could not be read (%s): watchlist only", type(e).__name__)
        names, universe = dict(watch.names), set(watch.tickers)
    tickers = list(dict.fromkeys(t.upper() for t in args.ticker or watch.tickers))
    bad = [t for t in tickers if not TICKER.fullmatch(t) or t not in universe]
    if bad:
        print(f"not on the watchlist or the Nasdaq-100: {bad}")
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
    expected = expected_session(ctx.now)
    if day < expected:
        # not a holiday: the session's bar is not posted yet. Failing here sends
        # the failure notice, where stopping quietly would skip the day unseen.
        print(
            f"no {BENCHMARK} bar for the {expected} session yet (latest: {day}); nothing run. "
            "A late price feed, or a closure the holidays package does not know yet."
        )
        return 1
    email = not args.no_email
    check = bool(args.ticker) or args.no_email or args.no_opus
    check_id = uuid.uuid4().hex[:8] if check else None
    path = record_path(args.runs, day, check_id)
    if check_id is None and path.exists():
        done = _read(path)
        if done is not None and _emailed(done):
            # a weekend or holiday: a missing bar on a trading day stopped above
            print(f"{day}: already emailed, nothing to do")
            _output(deliver="no")
            return 0
        why = _needs_a_person(done)
        if why is not None or done is None:
            print(f"{day}: {why}; nothing run or sent. See the README's daily section.")
            return 1
        print(f"{day}: recorded but not emailed, sending it again from the record")
        if args.stage == "calls":
            _output(record=str(path), deliver="yes")
            return 0
        return _finish(deliver(path, done, email=True, mailer=mailer, tracer=tracer), True)

    screened, started = None, ctx.clock()
    if not args.ticker:
        screened, tickers = choose((screener or run_screen)(ctx.now), day, list(watch.tickers))
    record = run_day(
        tickers, names, day, bench_close, ctx, check_id=check_id, screen=screened,
        started=started, save=lambda r: _write(path, r),
    )  # fmt: skip
    if check_id is None:
        # the week's last session: the scorecard rides at the top of the email
        card = weekly(args.runs, day, score_prices)
        if card is not None:
            record["scorecard"] = card
            _write(path, record)
    _summary(record)
    if args.stage == "calls" and email:
        _output(record=str(path), deliver="yes")
        return 0
    _output(deliver="no")
    return _finish(deliver(path, record, email=email, mailer=mailer, tracer=tracer), email)


if __name__ == "__main__":
    raise SystemExit(main())
