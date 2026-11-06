from __future__ import annotations

import json
import re
from typing import Any

import pytest
from test_reasons import NEWS, PACK
from test_rules import QUESTIONS, RULES, USUAL_5D, _good, _move, _pack

from tenq import opus
from tenq.llm import INFRA, AgentRun
from tenq.opus import explain_call, opus_call, parse_call, parse_why
from tenq.reasons import MAX_REASON_CHARS, Reason, check_reasons
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
    assert kw["log_detail"] is False
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


def test_a_reason_padded_with_spaces_is_dropped_never_cut() -> None:
    padded = "{px:ACME:ret_5d}" + " " * (MAX_REASON_CHARS * 2) + " " + "word " * 130
    got = parse_call(json.dumps({"call": "Buy", "reasons": [{"text": padded, "cites": []}]}))
    assert [len(r.text) for r in got.reasons] == [MAX_REASON_CHARS + 1]
    assert check_reasons(got.reasons, PACK).kept == []


def test_the_why_is_told_which_move_the_rules_weighed() -> None:
    moved = _pack(*_move(2.0 * USUAL_5D))
    call = decide(moved, _good(), RULES, QUESTIONS)
    assert call is not None and call.reaction_id == f"px:{moved.ticker}:excess_5d"
    message = opus.why_message(moved, call)
    id_line = f"- the market reaction was measured on {{{call.reaction_id}}}\n"
    reaction_line = next(x for x in call.explain() if x.startswith("market reaction:"))
    assert f"- {reaction_line}\n{id_line}" in message  # right under the cut it measures
    assert call.to_dict()["reaction_id"] == call.reaction_id
    still = decide(_pack(), _good(), RULES, QUESTIONS)
    assert still is not None and still.reaction_id == ""
    assert "measured on" not in opus.why_message(_pack(), still)


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
    assert opus.load_call_prompt().version == "call-v10"
    assert opus.load_why_prompt().version == "why-v10"


def test_a_short_is_judged_against_qqq_like_every_call() -> None:
    """The scorecard judges every call against QQQ, so the prompt defines a Short that way."""
    text = opus.load_call_prompt().text
    calls = {}
    for line in text.splitlines():
        for name in ("Buy", "Hold", "Sell", "Short"):
            if line.startswith(f"- {name}:"):
                calls[name] = line.lower()
    assert set(calls) == {"Buy", "Hold", "Sell", "Short"}
    for name in ("Buy", "Sell", "Short"):
        assert "qqq" in calls[name], name
    for name, line in calls.items():
        assert not re.search(r"\b(fall|falls|fell|falling|drop|decline)\b", line), name
    assert "wide margin" in calls["Short"] and "wide margin" not in calls["Sell"]


def test_both_prompts_send_claims_about_numbers_to_code() -> None:
    """Words about a number on the sheet are unchecked, so the prompts forbid them."""
    for prompt in (opus.load_call_prompt(), opus.load_why_prompt()):
        assert "never put it in your own words" in prompt.text, prompt.version
        assert "reports a later quarter than the sheet's filing" in prompt.text


def test_the_why_prompt_offers_words_code_accepts_for_counts_and_nothing() -> None:
    from tenq.reasons import figure_in

    text = " ".join(opus.load_why_prompt().text.split())
    assert '"zero", "two", "five"' in text
    assert figure_in("near zero") and figure_in("two red flags")  # what the prompt bans, code drops
    for words in ("a pair", "a few", "several", "close to neutral"):
        assert f'"{words}"' in text
    for sentence in (
        "Jev saw a pair of warnings in the latest filing.",
        "A few analysts cut their targets this week after the outlook.",
        "Several headlines pointed to slowing demand.",
        "The rules left the score close to neutral, so the call is Hold.",
    ):
        assert figure_in(sentence) is None
