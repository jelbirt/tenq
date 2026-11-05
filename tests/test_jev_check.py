from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from test_rules import _choice, _good, _noul, _pack

import tenq.jev
from tenq import jev_check
from tenq.jev import QuestionSet, Result, load_questions
from tenq.llm import call_claude
from tenq.pack import Pack
from tenq.rules import decide, load_rules


class Asked:
    def __init__(self, result: Result) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, state: str, questions: QuestionSet, **kw: Any) -> Result:
        self.calls.append((state, kw))
        return self.result


def _setup(
    monkeypatch: pytest.MonkeyPatch, result: Result, pack: Pack | None = None
) -> tuple[Asked, list[str]]:
    asked, gathered = Asked(result), []

    def gather(ticker: str, now: datetime) -> Pack | None:
        gathered.append(ticker)
        return pack

    monkeypatch.setattr(jev_check, "gather", gather)
    monkeypatch.setattr(tenq.jev, "ask", asked)
    return asked, gathered


def _measured() -> Result:
    r = _good()
    r.resolved_model, r.rounds, r.answered = "typesafe/jev-1.14", 6, [0, 1, 2, 3, 4, 5]
    r.input_tokens, r.output_tokens, r.cost, r.duration_ms = 9000, 120, 0.00038, 840
    r.answers["direction"].balanced, r.answers["direction"].first_lift = True, 0.04
    return r


def test_a_check_prints_answers_lean_call_and_cost(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = _pack()
    asked, gathered = _setup(monkeypatch, _measured(), pack)
    assert jev_check.main(["acme"]) == 0
    out = capsys.readouterr().out
    assert gathered == ["ACME"]
    state, kw = asked.calls[0]
    assert state == pack.render() and kw["session_id"] == "tenq-check-ACME-2026-11-03"
    assert "== ACME: Jev typesafe/jev-1.14" in out
    assert "tokens in 9000 out 120, cost $0.000380, 840 ms" in out
    assert "direction: good" in out and "first-option lean +0.04" in out
    assert ": Buy (confidence" in out
    assert "balance, Jev over 1: mean score +" in out
    assert "Buy 1, Hold 0, Sell 0, Short 0" in out
    assert out.rstrip().endswith("total Jev cost $0.000380")


def test_a_stock_with_no_prices_is_not_asked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked, _ = _setup(monkeypatch, _measured(), None)
    assert jev_check.main(["ACME"]) == 1
    assert asked.calls == [] and "no prices, nothing asked" in capsys.readouterr().out


def test_no_answers_is_a_failed_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failed = Result(
        load_questions().version,
        "jev-latest",
        errors=["HTTP 401"],
        down=True,
        error="no usable answer",
    )
    _setup(monkeypatch, failed, _pack())
    assert jev_check.main(["ACME"]) == 1
    out = capsys.readouterr().out
    assert "error: HTTP 401" in out and "no answers: no usable answer" in out
    assert "  no call" in out


@pytest.mark.parametrize("argv", [["AC ME"], ["../x"], ["A", "B", "C", "D", "E", "F"], ["$(id)"]])
def test_anything_but_a_few_tickers_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    asked, gathered = _setup(monkeypatch, _measured(), _pack())
    assert jev_check.main(argv) == 2
    assert gathered == [] and asked.calls == []


def test_tickers_are_uppercased_and_asked_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _, gathered = _setup(monkeypatch, _measured(), _pack())
    assert jev_check.main(["acme", "ACME", "brk.b"]) == 0
    assert gathered == ["ACME", "BRK.B"]


def test_each_answer_line_says_whether_its_lean_was_measured() -> None:
    unbalanced = _choice("direction", {"neutral": 0.1, "mixed": 0.1, "bad": 0.1, "good": 0.7})
    assert "lean not measured" in jev_check.answer_line(unbalanced)
    line = jev_check.answer_line(_noul("risk", 0.25))
    assert "risk: 0.25" in line and "lean" not in line


def test_one_stock_failing_to_gather_costs_only_its_own_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked, gathered = _setup(monkeypatch, _measured(), _pack())
    real: Callable[[str, datetime], Pack | None] = vars(jev_check)["gather"]

    def gather(ticker: str, now: datetime) -> Pack | None:
        if ticker == "CRDW":
            gathered.append(ticker)
            raise LookupError("company CRDW not found, secret-ish detail")
        return real(ticker, now)

    monkeypatch.setattr(jev_check, "gather", gather)
    assert jev_check.main(["CRDW", "ACME"]) == 1
    out = capsys.readouterr().out
    assert gathered == ["CRDW", "ACME"] and len(asked.calls) == 1
    assert "== CRDW: no facts pack (LookupError), nothing asked" in out
    assert "secret-ish" not in out
    assert out.rstrip().endswith("total Jev cost $0.000380")


def test_five_tickers_are_allowed_and_their_costs_add_up(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked, _ = _setup(monkeypatch, _measured(), _pack())
    assert jev_check.main(["A", "B", "C", "D", "E"]) == 0
    assert len(asked.calls) == 5
    assert capsys.readouterr().out.rstrip().endswith("total Jev cost $0.001900")


def test_a_broken_rules_file_fails_before_anything_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked, gathered = _setup(monkeypatch, _measured(), _pack())

    def broken(_q: QuestionSet) -> None:
        raise ValueError("rules: bad")

    monkeypatch.setattr(jev_check, "load_rules", broken)
    with pytest.raises(ValueError):
        jev_check.main(["ACME"])
    assert gathered == [] and asked.calls == []


NEWS_ID = "news:ACME:0123456789"


class Opus:
    """A fake claude: Opus's own call, then the why for Jev's call."""

    def __init__(self, *texts: str | None) -> None:
        self.texts = list(texts)
        self.systems: list[str] = []

    def __call__(self, message: str, **kw: Any) -> Any:
        from tenq.llm import INFRA, AgentRun

        self.systems.append(kw["system"])
        run = AgentRun(kw["model"], text=self.texts.pop(0))
        if run.text is None:
            run.fail(INFRA, "api error: down")
        return run


def _news_pack() -> Pack:
    from tenq.pack import Headline

    pack = _pack()
    pack.news.append(Headline(NEWS_ID, "Acme beats", "reuters.com", "2026-11-03", ""))
    return pack


def test_with_opus_both_calls_run_and_each_reason_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tenq import opus

    _setup(monkeypatch, _measured(), _news_pack())
    fake = Opus(
        '{"call": "Hold", "confidence": 0.35, "reasons": ['
        f'{{"text": "Results beat.", "cites": ["{NEWS_ID}"]}},'
        f'{{"text": "Shares rose 40%.", "cites": ["{NEWS_ID}"]}}]}}',
        f'{{"reasons": [{{"text": "The news was good.", "cites": ["{NEWS_ID}"]}},'
        f'{{"text": "Shares rose 40%.", "cites": ["{NEWS_ID}"]}}]}}',
    )
    lines, ok, _ = jev_check.check(
        "ACME",
        tenq.jev.load_questions(),
        load_rules(tenq.jev.load_questions()),
        datetime.now().astimezone(),
        claude=fake,
    )
    out = "\n".join(lines)
    assert ok
    assert fake.systems == [opus.load_call_prompt().text, opus.load_why_prompt().text]
    assert "Opus: Hold (confidence 35%)" in out
    assert f"Opus reason kept: Results beat. ['{NEWS_ID}']" in out
    assert "Opus reason dropped (a figure typed in its own words ('4'))" in out
    assert "why reason kept: The news was good." in out
    assert "why reason dropped (a figure typed in its own words ('4'))" in out


def test_with_no_jev_call_the_why_is_not_asked() -> None:
    fake = Opus('{"call": "Hold", "confidence": 0.5, "reasons": []}')
    lines, made = jev_check.opus_lines(_news_pack(), None, fake)
    assert made and len(fake.systems) == 1
    assert not any("why" in line for line in lines)


def test_with_opus_no_call_from_opus_is_a_failed_check(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(monkeypatch, _measured(), _news_pack())
    lines, ok, _ = jev_check.check(
        "ACME",
        tenq.jev.load_questions(),
        load_rules(tenq.jev.load_questions()),
        datetime.now().astimezone(),
        claude=Opus(None, None),
    )
    out = "\n".join(lines)
    assert not ok
    assert "Opus: no call (api error: down)" in out
    assert "why for Jev's call: none (api error: down)" in out


def test_opus_runs_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(monkeypatch, _measured(), _pack())
    seen: list[Any] = []
    real = jev_check.check

    def spy(*a: Any, **kw: Any) -> Any:
        seen.append(kw.get("claude"))
        return real(*a, **{**kw, "claude": None})

    monkeypatch.setattr(jev_check, "check", spy)
    jev_check.main(["ACME"])
    jev_check.main(["--opus", "ACME"])
    assert seen[0] is None and seen[1] is call_claude


def test_the_balance_lines_count_each_methods_calls() -> None:
    questions = load_questions()
    rules = load_rules(questions)
    buy = decide(_pack(), _good(), rules, questions)
    hold = decide(
        _pack(),
        _good(
            guidance=_choice("guidance", {"none": 1.0}),
            direction=_choice("direction", {"neutral": 1.0}),
        ),
        rules,
        questions,
    )
    assert buy is not None and hold is not None and (buy.call, hold.call) == ("Buy", "Hold")
    tally = jev_check.Tally(jev=[buy, hold], opus=["Hold", "Sell", "Sell"])
    mean = (buy.score + hold.score) / 2
    assert tally.lines() == [
        f"balance, Jev over 2: mean score {mean:+.2f}, Buy 1, Hold 1, Sell 0, Short 0",
        "balance, Opus over 3: Buy 0, Hold 1, Sell 2, Short 0",
    ]
    assert jev_check.Tally().lines() == []
