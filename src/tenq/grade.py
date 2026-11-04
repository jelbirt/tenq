"""The grader: one claude -p call that labels an answer against FinanceBench's gold.

It sees the question, the gold answer, the gold justification (empty for about
a third of the questions) and the answer, never which model or setup wrote
it. It has no tools. Labels are the paper's: Correct Answer, Incorrect Answer,
Refusal, so its grades and the paper's human grades compare directly.

A grade ends in the same statuses as a Q&A run: GRADED; MODEL_FAILURE (the
grader timed out or gave no readable label: no label, counted on its own);
INFRA_FAILURE (nothing was learned: the item is graded again later). An empty
answer is a Refusal without a call.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from tenq.brief import Prompt, last_json_object
from tenq.config import REPO_ROOT
from tenq.financebench import PAPER_LABELS, Question
from tenq.llm import DEFAULT_MODEL, INFRA, AgentRun, call_claude
from tenq.qa import INFRA_FAILURE, MODEL_FAILURE

PROMPT_PATH = REPO_ROOT / "prompts" / "grade.md"
CORRECT, INCORRECT, REFUSAL = PAPER_LABELS
LABELS = {"correct": CORRECT, "incorrect": INCORRECT, "refusal": REFUSAL}
GRADED = "graded"
MAX_GRADED_CHARS = 12_000  # the longest answer in the paper's files is under 11k
TAG = re.compile(
    r"<\s*/?\s*(?:question|gold_answer|gold_justification|answer_to_grade)\b[^>]*>", re.I
)
MAX_REASON_CHARS = 500
TIMEOUT_S = 300


@dataclass
class Grade:
    status: str
    label: str | None = None
    reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _inert(text: str) -> str:
    """Defuse any of the message's own tags inside a field, so an answer cannot
    close its block early and speak outside it."""
    return TAG.sub(lambda m: m.group(0).replace("<", "&lt;"), text)


def user_message(q: Question, answer: str) -> str:
    if len(answer) > MAX_GRADED_CHARS:
        answer = answer[:MAX_GRADED_CHARS] + " [answer cut here]"
    justification = q.justification.strip() or "(none given)"
    return (
        f"<question>\n{_inert(q.question)}\n</question>\n\n"
        f"<gold_answer>\n{_inert(q.answer)}\n</gold_answer>\n\n"
        f"<gold_justification>\n{_inert(justification)}\n</gold_justification>\n\n"
        f"<answer_to_grade>\n{_inert(answer)}\n</answer_to_grade>"
    )


def parse_grade(text: str | None) -> Grade:
    obj = last_json_object(text or "", "label")
    label = obj.get("label") if isinstance(obj, dict) else None
    if not isinstance(label, str) or label.strip().lower() not in LABELS:
        return Grade(MODEL_FAILURE, error="no JSON object with a known label")
    reason = obj.get("reason")
    return Grade(
        GRADED,
        LABELS[label.strip().lower()],
        " ".join(reason.split())[:MAX_REASON_CHARS] if isinstance(reason, str) else None,
    )


def outcome(run: AgentRun) -> Grade:
    if run.error is not None:
        status = INFRA_FAILURE if run.failure == INFRA else MODEL_FAILURE
        return Grade(status, error=run.error)
    return parse_grade(run.text)


def grade(
    q: Question, answer: str, prompt: Prompt, *, model: str = DEFAULT_MODEL
) -> tuple[AgentRun | None, Grade]:
    """One answer, one grader call with no tools. Logs never carry its text."""
    if not answer.strip():
        return None, Grade(GRADED, REFUSAL, "empty answer")
    run = call_claude(
        user_message(q, answer),
        system=prompt.text,
        mcp_servers={},
        allowed_tools=[],
        model=model,
        timeout=TIMEOUT_S,
        log_detail=False,
    )
    return run, outcome(run)
