"""The model check: one live Jev call per stock, and Opus or Kairos beside it
when asked, printed: the first look at real answers and cost.

    python -m tenq.model_check CRWD DDOG
    python -m tenq.model_check --opus CRWD
    python -m tenq.model_check --kairos CRWD

With --opus it also runs task 7's two Opus calls on the same pack (Opus's own
call, and the why for Jev's call) and prints each reason tenq.reasons kept or
dropped. That needs the claude CLI and CLAUDE_CODE_OAUTH_TOKEN.

With --kairos it also asks Kairos the same questions on the same rendered pack
and prints its answers beside Jev's, with the pack's size in characters,
Kairos's tokens, the model that answered and the time taken. It first lists
the model ids Minara Cloud offers the key. That needs KAIROS_API_KEY. Kairos
gets no call and no weight here: the rules are not run on its answers. Its
reply has no cost (billing is on tokens), so none is printed. Once Kairos is
found down (a timeout, a key it refuses, a failure upstream), the rest of the
run does not ask it again.

Builds today's facts pack for each stock, asks Jev the question file over its
balanced rounds, turns the answers into a call with the rules file, and prints
the answers, the measured first-option lean, the call and what it cost. Runs
where OPENROUTER_API_KEY is set (the model-check workflow on Actions). Prints
nothing secret: the packs are public data and Jev's answers are public under
the plan's record decision. Ends with a balance line per method (mean score
before and after the reaction cut, and how many of each call), so a lean
toward one call shows the same day. Exits 1 when any stock gets no usable
call, so a failed check shows red.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tenq import jev, opus
from tenq.filings import TICKER
from tenq.jev import Answer, QuestionSet, Result
from tenq.llm import call_claude
from tenq.opus import CALLS, CallName, Claude
from tenq.pack import Pack, gather
from tenq.reasons import ReasonCheck, check_reasons
from tenq.rules import Call, Rules, decide, load_rules

MAX_TICKERS = 5  # a check, not a run: keeps a mistyped list from costing much


@dataclass
class Tally:
    """Every call one run made, for the balance lines at the end."""

    jev: list[Call] = field(default_factory=list)
    opus: list[CallName] = field(default_factory=list)
    kairos: list[Call] = field(default_factory=list)  # the daily run's; this check fills none

    def lines(self) -> list[str]:
        out: list[str] = []
        if self.jev:
            out.append(_rules_line("Jev", self.jev))
        if self.opus:
            out.append(f"balance, Opus over {len(self.opus)}: {_counts(self.opus)}")
        if self.kairos:
            out.append(_rules_line("Kairos", self.kairos))
        return out


def _rules_line(name: str, calls: list[Call]) -> str:
    """The balance of the calls the rules made from one model's answers."""
    n = len(calls)
    raw = sum(c.raw for c in calls) / n
    score = sum(c.score for c in calls) / n
    counts = _counts([c.call for c in calls])
    return (
        f"balance, {name} over {n}: mean before the reaction cut {raw:+.2f}, "
        f"mean score {score:+.2f}, {counts}"
    )


def _counts(calls: list[CallName]) -> str:
    return ", ".join(f"{name} {calls.count(name)}" for name in CALLS)


def answer_line(a: Answer) -> str:
    """One answer: its value, how sure, and the first-option lean it measured."""
    value = f"{a.value:.2f}" if isinstance(a.value, float) else str(a.value)
    bits = [f"{a.key}: {value}", f"{a.agreed}/{a.repeats} rounds agree", f"spread {a.spread:.2f}"]
    if a.margin is not None:
        bits.append(f"margin {a.margin:.2f}")
    if a.first_lift is not None:
        bits.append(f"first-option lean {a.first_lift:+.2f}")
    elif a.type != "noul":
        bits.append("lean not measured (rounds missing)")
    return "  " + ", ".join(bits)


def report(ticker: str, result: Result, call: Call | None) -> list[str]:
    lines = [
        f"== {ticker}: Jev {result.resolved_model or result.model}, questions {result.version}",
        f"  rounds answered {result.rounds} ({result.answered}), "
        f"tokens in {result.input_tokens} out {result.output_tokens}, "
        f"cost ${result.cost:.6f}, {result.duration_ms} ms",
    ]
    lines += [answer_line(a) for a in result.answers.values()]
    lines += [f"  error: {e}" for e in result.errors]
    lines += [f"  note: {n}" for n in result.notes]
    if result.error:
        lines.append(f"  no answers: {result.error}")
    if call is None:
        lines.append("  no call")
    else:
        lines += ["  " + line for line in call.explain()]
        lines += [f"  call note: {n}" for n in call.notes]
    return lines


def reason_lines(label: str, checked: ReasonCheck) -> list[str]:
    lines = [f"  {label} reason kept: {r.text} {list(r.cites)}" for r in checked.kept]
    lines += [f"  {label} reason dropped ({d.why}): {d.reason.text}" for d in checked.dropped]
    return lines


def opus_lines(pack: Pack, call: Call | None, claude: Claude) -> tuple[list[str], CallName | None]:
    """Opus's own call and the why for Jev's call, each checked, and Opus's
    call (None when it made none)."""
    _, own = opus.opus_call(pack, opus.load_call_prompt(), claude=claude)
    if own.call is None:
        lines = [f"  Opus: no call ({own.error})"]
    else:
        conf = "unreadable" if own.confidence is None else f"{own.confidence:.0%}"
        lean = f"lean {own.lean:+.2f}" if own.lean is not None else f"lean: {own.lean_error}"
        lines = [f"  Opus: {own.call} (confidence {conf}, {lean})"]
        lines += reason_lines("Opus", check_reasons(own.reasons, pack))
    if call is not None:
        _, why = opus.explain_call(pack, call, opus.load_why_prompt(), claude=claude)
        if why.error:
            lines.append(f"  why for Jev's call: none ({why.error})")
        lines += reason_lines("why", check_reasons(why.reasons, pack))
    return lines, own.call


def _value(a: Answer | None) -> str:
    if a is None:
        return "none"
    return f"{a.value:.2f}" if isinstance(a.value, float) else str(a.value)


@dataclass
class KairosRun:
    """Kairos over one check run: once found down, not asked again."""

    down: str = ""  # the error that found it down


def kairos_lines(
    state: str, questions: QuestionSet, beside: Result, session_id: str
) -> tuple[list[str], Result]:
    """Kairos asked the same questions on the same state, printed beside Jev's
    answers (`beside`), and its result."""
    result = jev.ask(state, questions, provider=jev.KAIROS, session_id=session_id)
    tokens = f"tokens in {result.input_tokens} out {result.output_tokens}"
    # Minara counts the pack once for every question asked, so its per-question
    # limit is met or not by input tokens over the questions in answered rounds
    asked = sum(len(questions.to_request(i)) for i in result.answered)
    if asked:
        tokens += f" (about {result.input_tokens // asked} a question, {asked} asked)"
    took = f"{result.duration_ms} ms"
    if result.rounds and not result.errors:  # a failed request's time is in the total
        took += f" ({result.duration_ms // result.rounds} a round)"
    lines = [
        f"  Kairos {result.resolved_model or 'did not say'} (asked for {result.model}), "
        f"pack {len(state)} characters",
        f"  Kairos rounds answered {result.rounds} ({result.answered}), {tokens}, {took}",
    ]
    lines += ["  Kairos " + answer_line(a).strip() for a in result.answers.values()]
    lines += [f"  Kairos error: {e}" for e in result.errors]
    lines += [f"  Kairos note: {n}" for n in result.notes]
    if result.error:
        lines.append(f"  Kairos no answers: {result.error}")
    lines += [
        f"  side by side, {q.key}: Jev {_value(beside.answers.get(q.key))}, "
        f"Kairos {_value(result.answers.get(q.key))}"
        for q in questions.questions
    ]
    return lines, result


def check(
    ticker: str,
    questions: QuestionSet,
    rules: Rules,
    now: datetime,
    *,
    claude: Claude | None = None,
    kairos: KairosRun | None = None,
    tally: Tally | None = None,
) -> tuple[list[str], bool, float]:
    """Lines to print, whether a call came out, and the cost. Never raises for
    one stock's data: an unknown ticker or an EDGAR outage is that stock's line."""
    try:
        pack = gather(ticker, now)
    except Exception as exc:  # the type only: exception text is not trusted to be clean
        return [f"== {ticker}: no facts pack ({type(exc).__name__}), nothing asked"], False, 0.0
    if pack is None:
        return [f"== {ticker}: no prices, nothing asked"], False, 0.0
    state, session_id = pack.render(), f"tenq-check-{ticker}-{pack.day}"
    result = jev.ask(state, questions, session_id=session_id)
    call = decide(pack, result, rules, questions) if result.answers else None
    lines = report(ticker, result, call)
    lines += [f"  pack note: {n}" for n in pack.notes]
    ok = call is not None
    if tally is not None and call is not None:
        tally.jev.append(call)
    if kairos is not None and kairos.down:
        lines.append(f"  Kairos not asked: down earlier in this run ({kairos.down})")
        ok = False
    elif kairos is not None:
        more, asked = kairos_lines(state, questions, result, session_id)
        lines, ok = lines + more, ok and bool(asked.answers)
        if asked.down:
            kairos.down = asked.errors[-1] if asked.errors else "down"
    if claude is not None:
        more, own = opus_lines(pack, call, claude)
        lines, ok = lines + more, ok and own is not None
        if tally is not None and own is not None:
            tally.opus.append(own)
    return lines, ok, result.cost


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ask Jev, and Opus or Kairos when asked, about today's facts for a few stocks."
    )
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--opus", action="store_true", help="also run the two Opus calls")
    ap.add_argument("--kairos", action="store_true", help="also ask Kairos, beside Jev")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    tickers = list(dict.fromkeys(t.upper() for t in args.tickers))
    bad = [t for t in tickers if not TICKER.fullmatch(t)]
    if bad or len(tickers) > MAX_TICKERS:
        print(f"give 1 to {MAX_TICKERS} tickers; not tickers: {bad}")
        return 2
    questions = jev.load_questions()
    rules = load_rules(questions)  # a bad file fails here, before anything is paid for
    now = datetime.now(UTC)
    ok, total, tally = True, 0.0, Tally()
    kairos = KairosRun() if args.kairos else None
    if args.kairos:
        offered = jev.models()
        listed = (", ".join(offered) or "none") if isinstance(offered, list) else offered
        print(f"Kairos models offered: {listed}")
    for ticker in tickers:
        lines, called, cost = check(
            ticker,
            questions,
            rules,
            now,
            claude=call_claude if args.opus else None,
            kairos=kairos,
            tally=tally,
        )
        print("\n".join(lines))
        ok, total = ok and called, total + cost
    for line in tally.lines():
        print(line)
    print(f"total Jev cost ${total:.6f}")  # Opus runs on the subscription
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
