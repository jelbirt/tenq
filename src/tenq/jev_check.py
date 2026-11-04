"""One live Jev call per stock, printed: the first look at real answers and cost.

    python -m tenq.jev_check CRWD DDOG

Builds today's facts pack for each stock, asks Jev the question file over its
balanced rounds, turns the answers into a call with the rules file, and prints
the answers, the measured first-option lean, the call and what it cost. Runs
where OPENROUTER_API_KEY is set (the jev-check workflow on Actions). Prints
nothing secret: the packs are public data and Jev's answers are public under
the plan's record decision. Exits 1 when any stock gets no usable call, so a
failed check shows red.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

from tenq import jev
from tenq.filings import TICKER
from tenq.jev import Answer, QuestionSet, Result
from tenq.pack import gather
from tenq.rules import Call, Rules, decide, load_rules

MAX_TICKERS = 5  # a check, not a run: keeps a mistyped list from costing much


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


def check(
    ticker: str, questions: QuestionSet, rules: Rules, now: datetime
) -> tuple[list[str], bool, float]:
    """Lines to print, whether a call came out, and the cost. Never raises for
    one stock's data: an unknown ticker or an EDGAR outage is that stock's line."""
    try:
        pack = gather(ticker, now)
    except Exception as exc:  # the type only: exception text is not trusted to be clean
        return [f"== {ticker}: no facts pack ({type(exc).__name__}), nothing asked"], False, 0.0
    if pack is None:
        return [f"== {ticker}: no prices, nothing asked"], False, 0.0
    result = jev.ask(pack.render(), questions, session_id=f"tenq-check-{ticker}-{pack.day}")
    call = decide(pack, result, rules, questions) if result.answers else None
    lines = report(ticker, result, call)
    lines += [f"  pack note: {n}" for n in pack.notes]
    return lines, call is not None, result.cost


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ask Jev about today's facts for a few stocks.")
    ap.add_argument("tickers", nargs="+")
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
    ok, total = True, 0.0
    for ticker in tickers:
        lines, called, cost = check(ticker, questions, rules, now)
        print("\n".join(lines))
        ok, total = ok and called, total + cost
    print(f"total cost ${total:.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
