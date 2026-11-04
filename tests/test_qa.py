from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import QA_DOC

from tenq import qa
from tenq.brief import load_prompt
from tenq.docstore import DocStore
from tenq.financebench import Document, Evidence, Question
from tenq.llm import INFRA, MODEL, AgentRun, ToolCall
from tenq.qa import ANSWERED, INFRA_FAILURE, MODEL_FAILURE, ask, outcome, parse_answer

QUESTION = Question(
    id="fb_0001",
    company="Acme",
    doc_name=QA_DOC,
    question_type="metrics-generated",
    question_reasoning=None,
    question="What did Acme spend on widget presses?",
    answer="$1,234",
    justification="",
    evidence=(Evidence(QA_DOC, 2, "Purchases of widget presses 1,234"),),
)


def _id(docs: DocStore, words: str) -> str:
    (p,) = [p for p in docs.passages(QA_DOC) if words in p.text]
    return p.id


def _json(answer: object, ids: object) -> str:
    return (
        "Searching first.\n```json\n" + json.dumps({"answer": answer, "passage_ids": ids}) + "\n```"
    )


def test_an_answer_resolves_its_citations_to_pages(docs: DocStore) -> None:
    presses, risks = _id(docs, "widget presses"), _id(docs, "Risk factors")
    got = parse_answer(_json("  Acme spent\n$1,234. ", [risks, presses, risks]), docs, QA_DOC)
    assert got.status == ANSWERED and got.text == "Acme spent $1,234."
    assert got.passage_ids == [risks, presses] and got.pages == [2, 3]
    assert got.dropped_ids == 0 and got.error is None


@pytest.mark.parametrize(
    "bad",
    [
        f"{QA_DOC}:9999",  # no such passage
        f"{QA_DOC}:1:2",
        "OTHER_2020_10K:1",  # another document
        "../../etc/passwd",
        f"{QA_DOC}:\uff11",  # a fullwidth digit
    ],
)
def test_citations_that_are_not_passages_of_the_document_are_dropped(
    docs: DocStore, bad: str
) -> None:
    docs.save("OTHER_2020_10K", "b" * 64, ["other company"], 0.1)
    presses = _id(docs, "widget presses")
    got = parse_answer(_json("$1,234", [bad, presses, 7, None]), docs, QA_DOC)
    assert got.status == ANSWERED and got.passage_ids == [presses] and got.pages == [2]
    assert got.dropped_ids == 1  # non-strings are not citations at all


def test_an_answer_with_no_citations_is_still_an_answer(docs: DocStore) -> None:
    got = parse_answer(_json("The document does not say.", []), docs, QA_DOC)
    assert got.status == ANSWERED and got.passage_ids == [] and got.pages == []
    assert parse_answer(_json("x", "not a list"), docs, QA_DOC).passage_ids == []


def test_citations_are_capped_and_counted_apart(
    docs: DocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = [p.id for p in docs.passages(QA_DOC)]
    assert len(ids) == 3
    monkeypatch.setattr(qa, "MAX_CITATIONS", 1)
    got = parse_answer(_json("x", [*ids, f"{QA_DOC}:9999"]), docs, QA_DOC)
    assert got.passage_ids == ids[:1] and got.over_cap == 2 and got.dropped_ids == 1


@pytest.mark.parametrize(
    "text",
    [None, "", "I could not find it.", _json("", []), _json("   ", []), _json(5, []), "{bad json"],
)
def test_an_unreadable_answer_is_a_model_failure(docs: DocStore, text: str | None) -> None:
    got = parse_answer(text, docs, QA_DOC)
    assert got.status == MODEL_FAILURE and got.error is not None and got.text is None


def test_a_failed_run_is_graded_by_its_kind(docs: DocStore) -> None:
    run = AgentRun(model="opus")
    run.fail(MODEL, "timed out after 900s")
    assert outcome(run, docs, QA_DOC).status == MODEL_FAILURE
    run.fail(INFRA, "error result: 401")
    assert outcome(run, docs, QA_DOC).status == INFRA_FAILURE
    run.usage_limit = True
    assert outcome(run, docs, QA_DOC).status == INFRA_FAILURE


def test_ask_runs_document_mode_quietly(docs: DocStore, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(prompt: str, **kw: Any) -> AgentRun:
        seen.update(kw, prompt=prompt)
        return AgentRun(model=kw["model"], text=_json("$1,234", [_id(docs, "widget presses")]))

    monkeypatch.setattr(qa, "call_claude", fake)
    doc = Document(QA_DOC, "Acme", "10k", 2020)
    run, answer = ask(QUESTION, docs, "http://x:1", load_prompt(qa.PROMPT_PATH), document=doc,
                      model="haiku")  # fmt: skip
    assert answer.status == ANSWERED and answer.pages == [2] and run.model == "haiku"
    assert seen["log_detail"] is False and seen["model"] == "haiku"
    assert seen["allowed_tools"] == ["mcp__tenq__search_filing", "mcp__tenq__read_passage"]
    env = seen["mcp_servers"]["tenq"]["env"]
    assert env == {"TENQ_DOCS": str(docs.root.resolve()), "TENQ_DOC": QA_DOC,
                   "TENQ_XERJ_URL": "http://x:1"}  # fmt: skip
    assert QUESTION.question in seen["prompt"] and "fiscal 2020" in seen["prompt"]
    assert QUESTION.answer not in seen["prompt"]  # the gold answer never reaches the model


def test_the_qa_prompt_is_versioned_and_tells_the_model_the_tools() -> None:
    prompt = load_prompt(qa.PROMPT_PATH)
    assert prompt.version == "qa-v1"
    for tool in qa.TOOLS:
        assert tool in prompt.text
    assert '"answer"' in prompt.text and '"passage_ids"' in prompt.text
    assert "\u2014" not in Path(qa.PROMPT_PATH).read_text(encoding="utf-8")


def _searched(*errors: bool) -> AgentRun:
    run = AgentRun(model="opus", text=_json("not in the document", []))
    run.tool_calls = [
        ToolCall(f"t{i}", qa.SEARCH_TOOL, {"query": "x"}, is_error=e) for i, e in enumerate(errors)
    ]
    return run


def test_an_answer_when_every_search_failed_is_infrastructure(docs: DocStore) -> None:
    assert outcome(_searched(True, True), docs, QA_DOC).status == INFRA_FAILURE
    assert outcome(_searched(True, False), docs, QA_DOC).status == ANSWERED
    assert (
        outcome(_searched(), docs, QA_DOC).status == ANSWERED
    )  # no search at all is the model's call


@pytest.mark.parametrize(
    ("text", "status", "answer"),
    [
        ('{"answer": "About  $1,234 million."}', ANSWERED, "About $1,234 million."),
        ('{"answer": "  "}', MODEL_FAILURE, None),
        ("no json here", MODEL_FAILURE, None),
    ],
)
def test_a_closed_book_answer_is_read_from_its_json(
    text: str, status: str, answer: str | None
) -> None:
    a = qa.closed_outcome(AgentRun(model="opus", text=text))
    assert (a.status, a.text) == (status, answer)
    assert a.passage_ids == [] and a.pages == []


def test_a_closed_book_failure_keeps_its_kind() -> None:
    run = AgentRun(model="opus")
    run.fail(INFRA, "api error: authentication_failed")
    assert qa.closed_outcome(run).status == INFRA_FAILURE
    run.fail(MODEL, "timeout")
    assert qa.closed_outcome(run).status == MODEL_FAILURE


def test_closed_book_has_no_tools_no_document_and_quiet_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake(prompt: str, **kw: Any) -> AgentRun:
        seen.update(kw, prompt=prompt)
        return AgentRun(model=kw["model"], text='{"answer": "$1"}')

    monkeypatch.setattr(qa, "call_claude", fake)
    _, a = qa.ask_closed(QUESTION, load_prompt(qa.CLOSED_PROMPT_PATH), model="haiku")
    assert a.status == ANSWERED
    assert seen["mcp_servers"] == {} and seen["allowed_tools"] == []
    assert seen["log_detail"] is False and seen["model"] == "haiku"
    assert QA_DOC not in seen["prompt"] and QUESTION.question in seen["prompt"]
    assert load_prompt(qa.CLOSED_PROMPT_PATH).version == "closed-v1"
