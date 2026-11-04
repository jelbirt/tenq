"""Answer one FinanceBench question from one document (slice 2's Q&A mode).

Claude gets tenq's MCP server in document mode (search_filing and
read_passage over that document alone) and the versioned prompts/qa.md, and
answers with JSON: the answer and the passage ids it rests on. A cited id
that is not a passage of the document is dropped and counted; the pages of
the rest are what the evidence-hit check compares with the gold pages.

Every question ends in one status:
- ANSWERED: an answer to grade.
- MODEL_FAILURE: the model timed out, ran out of turns or gave no readable
  answer. Graded Refusal, and counted on its own.
- INFRA_FAILURE: nothing was learned about the model (auth, CLI, usage
  limit, ...). Not finished: the question runs again on a later night.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tenq.brief import Prompt, last_json_object
from tenq.config import REPO_ROOT
from tenq.docstore import DOC_PASSAGE_ID, DocStore
from tenq.financebench import Document, Question
from tenq.llm import DEFAULT_MODEL, INFRA, AgentRun, call_claude

PROMPT_PATH = REPO_ROOT / "prompts" / "qa.md"
SERVER = "tenq"
TOOLS = ("search_filing", "read_passage")
ALLOWED_TOOLS = [f"mcp__{SERVER}__{t}" for t in TOOLS]
SEARCH_TOOL = f"mcp__{SERVER}__search_filing"
MAX_ANSWER_CHARS = 2000
MAX_CITATIONS = 10
ANSWERED = "answered"
MODEL_FAILURE = "model_failure"
INFRA_FAILURE = "infra_failure"


@dataclass
class Answer:
    status: str
    text: str | None = None
    passage_ids: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)  # zero-indexed, sorted, distinct
    dropped_ids: int = 0  # cited ids that are not passages of the document
    over_cap: int = 0  # valid citations past MAX_CITATIONS, not kept
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_answer(text: str | None, store: DocStore, doc: str) -> Answer:
    """The model's answer, with its citations resolved to pages of `doc`."""
    if not text:
        return Answer(MODEL_FAILURE, error="no answer")
    obj = last_json_object(text, "answer")
    if obj is None or not isinstance(obj.get("answer"), str) or not obj["answer"].strip():
        return Answer(MODEL_FAILURE, error="no JSON object with an answer")
    raw = obj.get("passage_ids")
    cited = [r.strip() for r in raw if isinstance(r, str)] if isinstance(raw, list) else []
    answer = Answer(ANSWERED, " ".join(obj["answer"].split())[:MAX_ANSWER_CHARS])
    pages: set[int] = set()
    passages = {p.id: p for p in store.passages(doc)}  # one read of the document
    for pid in dict.fromkeys(cited):  # distinct, in order
        m = DOC_PASSAGE_ID.match(pid)
        p = passages.get(pid) if m and m["doc"] == doc else None
        page = store.page_of(doc, p.line) if p is not None else None
        if p is None or page is None:
            answer.dropped_ids += 1
        elif len(answer.passage_ids) >= MAX_CITATIONS:
            answer.over_cap += 1
        else:
            answer.passage_ids.append(p.id)
            pages.add(page)
    answer.pages = sorted(pages)
    return answer


def mcp_servers(docs: Path, doc: str, xerj_url: str) -> dict[str, Any]:
    """The --mcp-config entry that starts tenq's server over one document."""
    return {
        SERVER: {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "tenq.mcp_server"],
            "env": {
                "TENQ_DOCS": str(docs.resolve()),
                "TENQ_DOC": doc,
                "TENQ_XERJ_URL": xerj_url,
            },
        }
    }


def user_message(q: Question, document: Document | None = None) -> str:
    about = (
        f"{document.company}, {document.doc_type} for fiscal {document.doc_period}"
        if document is not None
        else q.company
    )
    return f"Document: {q.doc_name} ({about}).\n\nQuestion: {q.question}"


def outcome(run: AgentRun, store: DocStore, doc: str) -> Answer:
    if run.error is not None:
        status = INFRA_FAILURE if run.failure == INFRA else MODEL_FAILURE
        return Answer(status, error=run.error)
    searches = [c for c in run.tool_calls if c.name == SEARCH_TOOL]
    if searches and all(c.is_error for c in searches):
        # search was down (xerj): whatever the model said, it never saw the document
        return Answer(INFRA_FAILURE, error="every search_filing call failed")
    return parse_answer(run.text, store, doc)


def ask(
    q: Question,
    store: DocStore,
    xerj_url: str,
    prompt: Prompt,
    *,
    document: Document | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[AgentRun, Answer]:
    """One question, one agent run. Logs never carry the model's text."""
    run = call_claude(
        user_message(q, document),
        system=prompt.text,
        mcp_servers=mcp_servers(store.root, q.doc_name, xerj_url),
        allowed_tools=ALLOWED_TOOLS,
        model=model,
        log_detail=False,
    )
    return run, outcome(run, store, q.doc_name)
