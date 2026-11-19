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
    assert "balance, Jev over 1: mean before the reaction cut +" in out
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
    assert "Opus: Hold (confidence 35%, lean: no lean given)" in out
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
    raw, mean = (buy.raw + hold.raw) / 2, (buy.score + hold.score) / 2
    assert tally.lines() == [
        f"balance, Jev over 2: mean before the reaction cut {raw:+.2f}, "
        f"mean score {mean:+.2f}, Buy 1, Hold 1, Sell 0, Short 0",
        "balance, Opus over 3: Buy 0, Hold 1, Sell 2, Short 0",
    ]
    assert jev_check.Tally().lines() == []


class ByProvider:
    """tenq.jev.ask faked per endpoint: Jev's result, or Kairos's."""

    def __init__(self, jev: Result, kairos: Result) -> None:
        self.jev, self.kairos = jev, kairos
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, state: str, questions: QuestionSet, **kw: Any) -> Result:
        self.calls.append((state, kw))
        return self.kairos if kw.get("provider") is tenq.jev.KAIROS else self.jev


def _kairos_answered() -> Result:
    r = _good(
        material=_noul("material", 0.55),
        direction=_choice("direction", {"bad": 0.6, "good": 0.4}),
    )
    r.model, r.resolved_model = "DMIND/Kairos", "DMIND/Kairos-v4"
    r.rounds, r.answered = 6, [0, 1, 2, 3, 4, 5]
    r.input_tokens, r.output_tokens, r.duration_ms = 24000, 0, 12000
    return r


def _with_kairos(
    monkeypatch: pytest.MonkeyPatch, kairos: Result, offered: list[str] | str
) -> tuple[ByProvider, Pack, list[int]]:
    pack, listed = _pack(), []
    asked = ByProvider(_measured(), kairos)
    monkeypatch.setattr(jev_check, "gather", lambda ticker, now: pack)
    monkeypatch.setattr(tenq.jev, "ask", asked)

    def models() -> list[str] | str:
        listed.append(1)
        return offered

    monkeypatch.setattr(tenq.jev, "models", models)
    return asked, pack, listed


def test_with_kairos_it_is_asked_the_same_on_the_same_pack_and_printed_beside_jev(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked, pack, listed = _with_kairos(
        monkeypatch, _kairos_answered(), ["DMIND/Kairos", "DMIND/Kairos-v4"]
    )
    assert jev_check.main(["--kairos", "ACME"]) == 0
    out = capsys.readouterr().out
    (jev_state, jev_kw), (k_state, k_kw) = asked.calls
    assert jev_state == k_state == pack.render()
    assert "provider" not in jev_kw and k_kw["provider"] is tenq.jev.KAIROS
    assert jev_kw["session_id"] == k_kw["session_id"] == "tenq-check-ACME-2026-11-03"
    assert listed == [1]
    assert out.startswith("Kairos models offered: DMIND/Kairos, DMIND/Kairos-v4\n")
    size = len(pack.render())
    assert f"  Kairos DMIND/Kairos-v4 (asked for DMIND/Kairos), pack {size} characters" in out
    assert (
        "  Kairos rounds answered 6 ([0, 1, 2, 3, 4, 5]), tokens in 24000 out 0 "
        "(about 1142 a question, 21 asked), 12000 ms (2000 a round)"
    ) in out
    assert "  Kairos direction: bad" in out
    assert "  side by side, material: Jev 0.80, Kairos 0.55" in out
    assert "  side by side, direction: Jev good, Kairos bad" in out
    assert "  side by side, risk: Jev 0.10, Kairos 0.10" in out
    assert out.index(": Buy (confidence") < out.index("  Kairos DMIND")  # Jev's lines first
    assert out.rstrip().endswith("total Jev cost $0.000380")  # Kairos adds no cost


def test_kairos_giving_no_answer_fails_the_check_but_keeps_jevs_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    refused = Result("jev-q9", "DMIND/Kairos", duration_ms=900)
    refused.errors = ["HTTP 413 [request_too_large] state is 7012 tokens (request_id r1)"]
    refused.error = f"Kairos gave no usable answer ({refused.errors[0]})"
    _with_kairos(monkeypatch, refused, "HTTP 401")
    assert jev_check.main(["--kairos", "ACME"]) == 1
    out = capsys.readouterr().out
    assert "Kairos models offered: HTTP 401" in out
    assert ": Buy (confidence" in out  # Jev's call still printed
    assert "  Kairos did not say (asked for DMIND/Kairos)" in out
    assert "  Kairos rounds answered 0 ([]), tokens in 0 out 0, 900 ms\n" in out
    assert f"  Kairos error: {refused.errors[0]}" in out
    assert f"  Kairos no answers: {refused.error}" in out
    assert "  side by side, direction: Jev good, Kairos none" in out


def test_without_kairos_nothing_about_it_is_asked_or_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked, _, listed = _with_kairos(monkeypatch, _kairos_answered(), ["DMIND/Kairos"])
    assert jev_check.main(["ACME"]) == 0
    out = capsys.readouterr().out
    assert len(asked.calls) == 1 and "provider" not in asked.calls[0][1] and listed == []
    assert "Kairos" not in out and "side by side" not in out


def test_the_check_never_prints_the_kairos_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import dataclasses
    import json

    secret = "mk-live-CHECK-SECRET-abcdef0123456789"
    monkeypatch.setenv("KAIROS_API_KEY", secret)
    echo = json.dumps({"detail": {"error_type": secret, "message": f"key {secret} refused"}})
    seen: list[dict[str, str]] = []

    def post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        seen.append(headers)
        return 401, echo

    monkeypatch.setattr(tenq.jev, "KAIROS", dataclasses.replace(tenq.jev.KAIROS, post=post))
    monkeypatch.setattr(jev_check, "gather", lambda ticker, now: _pack())
    real = tenq.jev.ask
    monkeypatch.setattr(
        tenq.jev,
        "ask",
        lambda state, q, **kw: real(state, q, **kw) if "provider" in kw else _measured(),
    )
    monkeypatch.setattr(tenq.jev, "models", lambda: "HTTP 401 key [key] refused")
    assert jev_check.main(["--kairos", "ACME"]) == 1
    out = capsys.readouterr().out
    assert seen and seen[0]["Authorization"] == f"Bearer {secret}"  # it was sent, only sent
    assert secret not in out
    assert "  Kairos error: HTTP 401 key [key] refused" in out


def test_once_kairos_is_down_it_is_not_asked_again_in_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    down = Result("jev-q9", "DMIND/Kairos", errors=["request error TimeoutError"], down=True)
    down.error = "Kairos gave no usable answer (request error TimeoutError)"
    asked, _, _ = _with_kairos(monkeypatch, down, ["DMIND/Kairos"])
    assert jev_check.main(["--kairos", "ACME", "BETA", "GAMMA"]) == 1
    out = capsys.readouterr().out
    kairos_calls = [kw for _, kw in asked.calls if kw.get("provider") is tenq.jev.KAIROS]
    assert len(kairos_calls) == 1 and len(asked.calls) == 4  # Jev still asked for all three
    assert out.count("  Kairos error: request error TimeoutError") == 1
    note = "  Kairos not asked: down earlier in this run (request error TimeoutError)"
    assert out.count(note) == 2
    assert out.count(": Buy (confidence") == 3


def test_kairos_refusing_one_pack_is_asked_for_the_next(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    refused = Result("jev-q9", "DMIND/Kairos", errors=["HTTP 413"])
    asked, _, _ = _with_kairos(monkeypatch, refused, ["DMIND/Kairos"])
    assert jev_check.main(["--kairos", "ACME", "BETA"]) == 1
    kairos_calls = [kw for _, kw in asked.calls if kw.get("provider") is tenq.jev.KAIROS]
    assert len(kairos_calls) == 2 and "not asked" not in capsys.readouterr().out
