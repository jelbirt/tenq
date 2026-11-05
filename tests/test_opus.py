from __future__ import annotations

import json
from typing import Any

import pytest
from test_reasons import NEWS, PACK
from test_rules import QUESTIONS, RULES, _good, _pack

from tenq import opus
from tenq.llm import INFRA, AgentRun
from tenq.opus import explain_call, opus_call, parse_call, parse_why
from tenq.reasons import MAX_REASON_CHARS, Reason
from tenq.rules import decide


class FakeClaude:
    def __init__(self, text: str | None = None, error: str | None = None) -> None:
        self.text, self.error = text, error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, message: str, **kw: Any) -> AgentRun:
        self.calls.append((message, kw))
        run = AgentRun(kw["model"], text=self.text)
        if self.error:
            run.fail(INFRA, self.error)
        return run


ANSWER = json.dumps(
    {
        "call": "Hold",
        "confidence": 0.4,
        "reasons": [
            {"text": "Shares fell 21% over the past week.", "cites": ["px:ACME:ret_5d"]},
            {"text": "The outlook disappointed.", "cites": [NEWS, NEWS]},
        ],
    }
)


def test_opus_calls_from_the_pack_alone_with_no_tools() -> None:
    claude = FakeClaude("Here you go:\n```json\n" + ANSWER + "\n```")
    _, got = opus_call(PACK, opus.load_call_prompt(), claude=claude)
    message, kw = claude.calls[0]
    assert message == PACK.render()
    assert kw["mcp_servers"] == {} and kw["allowed_tools"] == []
    assert kw["model"] == "opus" and kw["timeout"] == opus.TIMEOUT_S
    assert kw["system"] == opus.load_call_prompt().text and "Jev" not in message
    assert got.call == "Hold" and got.confidence == 0.4 and got.error is None
    assert got.reasons == [
        Reason("Shares fell 21% over the past week.", ("px:ACME:ret_5d",)),
        Reason("The outlook disappointed.", (NEWS,)),  # repeated cites kept once
    ]


def test_a_failed_run_is_no_call_and_says_why() -> None:
    _, got = opus_call(PACK, opus.load_call_prompt(), claude=FakeClaude(error="api error: x"))
    assert got.call is None and got.reasons == [] and got.error == "api error: x"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (None, "no answer"),
        ("I think Buy.", "no JSON object with a call"),
        ('{"call": "Strong Buy", "reasons": []}', "not Buy, Hold, Sell or Short"),
        ('{"call": 3}', "not Buy, Hold, Sell or Short"),
    ],
)
def test_an_unusable_call_is_no_call(text: str | None, error: str) -> None:
    got = parse_call(text)
    assert got.call is None and got.error is not None and error in got.error


@pytest.mark.parametrize("confidence", [1.5, -0.1, True, "high", float("nan"), 10**400])
def test_an_unreadable_confidence_is_none_but_the_call_stands(confidence: Any) -> None:
    raw = '{"call": "sell", "confidence": %s}' % (
        json.dumps(confidence)
        if not isinstance(confidence, int) or isinstance(confidence, bool)
        else str(confidence)
    )
    got = parse_call(raw.replace("NaN", "NaN"))
    assert got.call == "Sell" and got.confidence is None


def test_reasons_are_capped_and_malformed_ones_skipped() -> None:
    many = [{"text": f"reason {i}", "cites": [NEWS]} for i in range(10)]
    junk = [
        "text",
        {"text": 5, "cites": [NEWS]},
        {"text": "no cites list", "cites": NEWS},
        {"text": "  ", "cites": [NEWS]},
        {"text": "odd cites", "cites": [5, None, " " + NEWS + " "]},
    ]
    got = parse_call(json.dumps({"call": "Buy", "reasons": junk + many}))
    assert got.reasons[0] == Reason("odd cites", (NEWS,))
    assert len(got.reasons) == opus.MAX_REASONS  # junk is skipped, not counted
    got = parse_call(json.dumps({"call": "Buy", "reasons": many}))
    assert len(got.reasons) == opus.MAX_REASONS
    cites = [f"news:ACME:{i:010d}" for i in range(20)]
    capped = parse_call(json.dumps({"call": "Buy", "reasons": [{"text": "x", "cites": cites}]}))
    assert capped.reasons[0].cites == tuple(cites[: opus.MAX_CITES])
    scan = [*(["junk"] * opus.MAX_REASONS * 4), {"text": "late", "cites": [NEWS]}]
    assert parse_call(json.dumps({"call": "Buy", "reasons": scan})).reasons == []
    edge = "x" * MAX_REASON_CHARS
    reasons = [{"text": edge, "cites": [NEWS]}, {"text": edge * 5, "cites": [NEWS]}]
    long = parse_call(json.dumps({"call": "Buy", "reasons": reasons}))
    # too long is kept just over the limit, so the reason check drops it and says why
    assert [len(r.text) for r in long.reasons] == [MAX_REASON_CHARS, MAX_REASON_CHARS + 1]


def test_reasons_without_cites_and_with_lone_surrogates_are_kept_printable() -> None:
    raw = '{"call": "Buy", "reasons": [{"text": "A \\ud800 rival cut its outlook."}]}'
    got = parse_call(raw)
    assert got.reasons == [Reason("A ? rival cut its outlook.", ())]
    got.reasons[0].text.encode("utf-8")  # printable


def test_the_why_for_jevs_call_sees_the_call_and_the_pack() -> None:
    pack = _pack()
    call = decide(pack, _good(), RULES, QUESTIONS)
    assert call is not None
    claude = FakeClaude(json.dumps({"reasons": [{"text": "The news was good.", "cites": [NEWS]}]}))
    _, why = explain_call(pack, call, opus.load_why_prompt(), claude=claude)
    message, kw = claude.calls[0]
    assert message.startswith(f"Jev's call on ACME: {call.call}.")
    assert all(line in message for line in call.explain())
    assert message.endswith(pack.render())
    assert kw["allowed_tools"] == [] and kw["system"] == opus.load_why_prompt().text
    assert why.reasons == [Reason("The news was good.", (NEWS,))] and why.error is None


@pytest.mark.parametrize(
    ("text", "error"),
    [(None, "no answer"), ('{"reasons": "none"}', "reasons list"), ("prose", "reasons list")],
)
def test_an_unusable_why_has_no_reasons(text: str | None, error: str) -> None:
    got = parse_why(text)
    assert got.reasons == [] and got.error is not None and error in got.error


def test_a_failed_why_run_says_why() -> None:
    call = decide(_pack(), _good(), RULES, QUESTIONS)
    assert call is not None
    _, got = explain_call(_pack(), call, opus.load_why_prompt(), claude=FakeClaude(error="timeout"))
    assert got.reasons == [] and got.error == "timeout"


def test_prompts_are_versioned() -> None:
    assert opus.load_call_prompt().version == "call-v5"
    assert opus.load_why_prompt().version == "why-v5"
