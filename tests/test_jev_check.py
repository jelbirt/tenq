from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from test_rules import _choice, _good, _noul, _pack

import tenq.jev
from tenq import jev_check
from tenq.jev import QuestionSet, Result
from tenq.pack import Pack


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
    assert out.rstrip().endswith("total cost $0.000380")


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
        "jev-q2", "jev-latest", errors=["HTTP 401"], down=True, error="no usable answer"
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
    assert out.rstrip().endswith("total cost $0.000380")


def test_five_tickers_are_allowed_and_their_costs_add_up(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked, _ = _setup(monkeypatch, _measured(), _pack())
    assert jev_check.main(["A", "B", "C", "D", "E"]) == 0
    assert len(asked.calls) == 5
    assert capsys.readouterr().out.rstrip().endswith("total cost $0.001900")


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
