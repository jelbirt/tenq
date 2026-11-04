"""One night of FinanceBench evals, resumable from the private evals repo.

Each night, until the deadline or the first infrastructure failure:

1. calibration: the grader grades the next batch of the paper's 450
   singleStore answers, until the gate (tenq.calibrate) is decided;
2. answers: Opus and Haiku in turn answer the next questions, singleStore
   (search and read on the question's own PDF) for all 150, then closed-book
   (no tools, no document) for 50; night 1 caps each model at 10, later
   nights at 30;
3. grades: once the gate has passed, the grader grades every answer not yet
   graded. It sees the question, the gold answer and the answer, never which
   model wrote it. A model failure is a Refusal, with no call.

A night after one that hit a usage limit is skipped. An infrastructure
failure (auth, CLI exit, usage limit, dead search) stops the night and is
not recorded for that item, so it runs again on a later night.

State lives in a checkout of the private evals repo (--state): JSONL records
for calibration, answers and grades, and one line per night with counts and
usage (calls, tokens, model time). The workflow commits that checkout, and
only that checkout, in one commit. Nothing here is written to the public
repo, and stdout carries counts only: a public log says nothing about how
any answer or grade came out. A crash logs its exception type and no more.

    uv run python -m tenq.evals [--state evals-state] [--hours 4.5]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tenq import calibrate
from tenq.brief import Prompt, load_prompt
from tenq.config import REPO_ROOT
from tenq.docstore import DocStore, extract_pages
from tenq.financebench import (
    DEFAULT_ROOT,
    Dataset,
    Question,
    download,
    download_results,
    load_manifest,
    pdf_path,
)
from tenq.grade import GRADED, REFUSAL
from tenq.grade import PROMPT_PATH as GRADE_PROMPT
from tenq.grade import grade as grade_one
from tenq.jsonl import append, load_lines, load_records
from tenq.llm import DEFAULT_MODEL, AgentRun
from tenq.qa import (
    ANSWERED,
    CLOSED_PROMPT_PATH,
    INFRA_FAILURE,
    MODEL_FAILURE,
    Answer,
    ask,
    ask_closed,
)
from tenq.qa import PROMPT_PATH as QA_PROMPT
from tenq.search import Xerj, xerj_node

log = logging.getLogger(__name__)

DEFAULT_STATE = REPO_ROOT / "evals-state"
MODELS = ("opus", "haiku")
SINGLE_STORE = "singlestore"
CLOSED_BOOK = "closedbook"
CLOSED_BOOK_N = 50
NIGHT_ONE_PER_MODEL = 10
PER_MODEL = 30
CALIBRATION_PER_NIGHT = 150
HOURS = 4.5


@dataclass
class Usage:
    """What a night cost: calls, tokens and model time, by phase."""

    calls: Counter[str] = field(default_factory=Counter)
    tokens: Counter[str] = field(default_factory=Counter)
    model_ms: int = 0
    usage_limit: bool = False

    def add(self, phase: str, run: AgentRun | None) -> None:
        if run is None:
            return
        self.calls[phase] += 1
        for k, v in (run.usage or {}).items():
            if isinstance(v, int) and not isinstance(v, bool):
                self.tokens[k] += v
        self.model_ms += run.duration_ms or 0
        self.usage_limit = self.usage_limit or run.usage_limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "tokens": dict(self.tokens),
            "model_ms": self.model_ms,
            "usage_limit": self.usage_limit,
        }


@dataclass
class Night:
    started_at: str
    counts: Counter[str] = field(default_factory=Counter)
    usage: Usage = field(default_factory=Usage)
    stopped: str | None = None  # "infrastructure" or "deadline"
    gate_decided: bool = False

    def record(self, seconds: float) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "ended_at": _now(),
            "seconds": round(seconds, 1),
            "counts": dict(self.counts),
            "usage": self.usage.to_dict(),
            "stopped": self.stopped,
            "gate_decided": self.gate_decided,
            "skipped": False,
        }


class State:
    """The private evals checkout: where every record lives."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calibration = root / "calibration" / "singlestore.jsonl"
        self.calibration_report = root / "calibration" / "report.json"
        self.answers = root / "answers.jsonl"
        self.grades = root / "grades.jsonl"
        self.nights = root / "nights.jsonl"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def answer_key(mode: str, model: str, qid: str) -> str:
    return f"{mode}:{model}:{qid}"


def _order(questions: list[Question], salt: str) -> list[Question]:
    return sorted(questions, key=lambda q: hashlib.sha256(f"{salt}:{q.id}".encode()).hexdigest())


def plan(questions: list[Question]) -> list[tuple[str, Question]]:
    """Every (mode, question) in run order: singleStore on all, then closed-book on 50."""
    single = [(SINGLE_STORE, q) for q in _order(questions, SINGLE_STORE)]
    closed = [(CLOSED_BOOK, q) for q in _order(questions, CLOSED_BOOK)[:CLOSED_BOOK_N]]
    return single + closed


def answered(record: dict[str, Any] | None, prompt_version: str) -> bool:
    return (
        record is not None
        and record.get("status") in (ANSWERED, MODEL_FAILURE)
        and record.get("prompt_version") == prompt_version
    )


def answer_sha(record: dict[str, Any]) -> str:
    """Ties a grade to the exact answer it graded."""
    text = f"{record.get('status')}\n{record.get('answer') or ''}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence(q: Question, pages: list[int]) -> dict[str, Any]:
    gold = sorted({e.page for e in q.evidence})
    cited = set(pages)
    return {
        "gold_pages": gold,
        "any_hit": bool(cited & set(gold)),
        "all_hit": bool(gold) and set(gold) <= cited,
    }


def answer_record(
    mode: str, model: str, q: Question, prompt: Prompt, run: AgentRun, a: Answer
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "key": answer_key(mode, model, q.id),
        "question_id": q.id,
        "mode": mode,
        "model": model,
        "resolved_model": run.resolved_model,
        "prompt_version": prompt.version,
        "status": a.status,
        "answer": a.text,
        "passage_ids": a.passage_ids,
        "pages": a.pages,
        "dropped_ids": a.dropped_ids,
        "over_cap": a.over_cap,
        "error": a.error,
        "usage": run.usage,
        "duration_ms": run.duration_ms,
        "num_turns": run.num_turns,
        "tool_calls": len(run.tool_calls),
        "at": _now(),
    }
    if mode == SINGLE_STORE:
        rec["evidence"] = evidence(q, a.pages)
    return rec


Asker = Callable[..., tuple[AgentRun, Answer]]
Grader = calibrate.Grader


class Documents:
    """Extracts and indexes a question's document on first use this night."""

    def __init__(
        self,
        ds: Dataset,
        store: DocStore,
        shas: dict[str, str],
        node: Callable[[], contextlib.AbstractContextManager[str]],
        work: Path,
    ) -> None:
        self.ds, self.store, self.shas, self.node, self.work = ds, store, shas, node, work
        self.stack = contextlib.ExitStack()
        self.url: str | None = None
        self.ready: set[str] = set()

    def __enter__(self) -> Documents:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()

    def prepare(self, doc: str) -> str:
        """The xerj url with `doc` indexed; raises if it cannot be made ready."""
        if not self.store.has(doc, self.shas[doc]):
            pages, seconds = extract_pages(self.ds.pdf(doc))
            self.store.save(doc, self.shas[doc], pages, seconds)
        if self.url is None:
            self.url = self.stack.enter_context(self.node())
        if doc not in self.ready:
            Xerj(self.url, self.store).index(doc, self.work / "xerj-state" / doc)
            self.ready.add(doc)
        return self.url


def answer_phase(
    night: Night,
    state: State,
    ds: Dataset,
    docs: Documents,
    *,
    per_model: int,
    qa_prompt: Prompt,
    closed_prompt: Prompt,
    stop: Callable[[], bool],
    asker: Asker = ask,
    closed_asker: Asker = ask_closed,
) -> bool:
    """Answer the next questions, models in turn. False if infrastructure failed."""
    records = load_records(state.answers)
    done: Counter[str] = Counter()
    for mode, q in plan(ds.questions):
        prompt = qa_prompt if mode == SINGLE_STORE else closed_prompt
        for model in MODELS:
            if done[model] >= per_model or answered(
                records.get(answer_key(mode, model, q.id)), prompt.version
            ):
                continue
            if stop():
                night.stopped = "deadline"
                return True
            if mode == SINGLE_STORE:
                try:
                    url = docs.prepare(q.doc_name)
                except Exception as e:
                    log.warning("document not ready (%s)", type(e).__name__)
                    night.counts["document_not_ready"] += 1
                    break  # the other model would hit the same document
                run, a = asker(
                    q, docs.store, url, prompt, document=ds.documents.get(q.doc_name), model=model
                )
            else:
                run, a = closed_asker(q, prompt, model=model)
            night.usage.add("answer", run)
            if a.status == INFRA_FAILURE:
                night.stopped = "infrastructure"
                return False
            append(state.answers, answer_record(mode, model, q, prompt, run, a))
            night.counts[f"answered_{mode}_{model}"] += 1
            done[model] += 1
        if all(done[m] >= per_model for m in MODELS):
            break
    return True


def grade_phase(
    night: Night,
    state: State,
    ds: Dataset,
    *,
    grade_prompt: Prompt,
    stop: Callable[[], bool],
    grader: Grader = grade_one,
) -> bool:
    """Grade every answer not yet graded. False if infrastructure failed."""
    questions = {q.id: q for q in ds.questions}
    grades = load_records(state.grades)
    for key, rec in sorted(load_records(state.answers).items()):
        sha = answer_sha(rec)
        old = grades.get(key)
        if (
            old is not None
            and old.get("answer_sha") == sha
            and calibrate.finished(old, grade_prompt.version, DEFAULT_MODEL)
        ):
            continue
        if stop():
            night.stopped = "deadline"
            return True
        q = questions[rec["question_id"]]
        if rec["status"] == MODEL_FAILURE:
            g_status, label, reason = GRADED, REFUSAL, "model failure"
        else:
            run, g = grader(q, rec.get("answer") or "", grade_prompt, model=DEFAULT_MODEL)
            g_status, label, reason = g.status, g.label or "", g.reason or ""
            night.usage.add("grade", run)
            if g_status == INFRA_FAILURE:
                night.stopped = "infrastructure"
                return False
        append(
            state.grades,
            {
                "key": key,
                "answer_sha": sha,
                "status": g_status,
                "label": label,
                "reason": reason,
                "prompt_version": grade_prompt.version,
                "model": DEFAULT_MODEL,
                "at": _now(),
            },
        )
        night.counts["graded"] += 1
    return True


def calibration_phase(
    night: Night,
    state: State,
    results: dict[str, list[Any]],
    ds: Dataset,
    *,
    grade_prompt: Prompt,
    stop: Callable[[], bool],
    grader: Grader = grade_one,
) -> tuple[bool, bool]:
    """Grade the next calibration batch. Returns (infrastructure ok, gate passed)."""
    items = calibrate.calibration_set(results)
    questions = {q.id: q for q in ds.questions}

    def gate() -> dict[str, Any]:
        rep = calibrate.report(
            items,
            load_records(state.calibration),
            prompt_version=grade_prompt.version,
            model=DEFAULT_MODEL,
        )
        state.calibration_report.parent.mkdir(parents=True, exist_ok=True)
        state.calibration_report.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        return dict(rep["gate"])

    g = gate()
    if g["decided"]:
        night.gate_decided = True
        return True, bool(g["passed"])
    count, stopped = calibrate.grade_items(
        items,
        questions,
        state.calibration,
        grade_prompt,
        model=DEFAULT_MODEL,
        limit=CALIBRATION_PER_NIGHT,
        grader=grader,
        stop=stop,
        on_run=lambda run: night.usage.add("calibration", run),
    )
    night.counts["calibrated"] += count
    g = gate()
    night.gate_decided = bool(g["decided"])
    if stopped:
        night.stopped = "infrastructure"
        return False, False
    return True, bool(g["decided"] and g["passed"])


def last_hit_a_limit(nights: list[dict[str, Any]]) -> bool:
    if not nights:
        return False
    last = nights[-1]
    return not last.get("skipped") and bool(last.get("usage", {}).get("usage_limit"))


def per_model_cap(nights: list[dict[str, Any]]) -> int:
    answered_before = any(k.startswith("answered_") for n in nights for k in n.get("counts", {}))
    return PER_MODEL if answered_before else NIGHT_ONE_PER_MODEL


def run_night(
    state: State,
    ds: Dataset,
    results: dict[str, list[Any]],
    store: DocStore,
    shas: dict[str, str],
    *,
    hours: float = HOURS,
    node: Callable[[], contextlib.AbstractContextManager[str]] = xerj_node,
    asker: Asker = ask,
    closed_asker: Asker = ask_closed,
    grader: Grader = grade_one,
    clock: Callable[[], float] = time.monotonic,
) -> Night:
    t0 = clock()
    night = Night(started_at=_now())
    nights = load_lines(state.nights)
    if last_hit_a_limit(nights):
        append(state.nights, {"started_at": night.started_at, "skipped": True})
        night.stopped = "skipped after a usage limit"
        return night

    def stop() -> bool:
        return clock() - t0 >= hours * 3600

    qa_prompt, closed_prompt = load_prompt(QA_PROMPT), load_prompt(CLOSED_PROMPT_PATH)
    grade_prompt = load_prompt(GRADE_PROMPT)
    try:
        ok, passed = calibration_phase(
            night, state, results, ds, grade_prompt=grade_prompt, stop=stop, grader=grader
        )
        if ok and not night.stopped:
            with (
                tempfile.TemporaryDirectory(prefix="tenq-evals-") as work,
                Documents(ds, store, shas, node, Path(work)) as docs,
            ):
                ok = answer_phase(
                    night,
                    state,
                    ds,
                    docs,
                    per_model=per_model_cap(nights),
                    qa_prompt=qa_prompt,
                    closed_prompt=closed_prompt,
                    stop=stop,
                    asker=asker,
                    closed_asker=closed_asker,
                )
        if ok and passed and not night.stopped:
            grade_phase(night, state, ds, grade_prompt=grade_prompt, stop=stop, grader=grader)
    finally:
        append(state.nights, night.record(clock() - t0))
    return night


def summary(night: Night) -> str:
    """Counts only: safe for a public log."""
    parts = [f"{k} {v}" for k, v in sorted(night.counts.items())] or ["nothing run"]
    calls = sum(night.usage.calls.values())
    end = f"stopped: {night.stopped}" if night.stopped else "finished"
    return f"eval night: {', '.join(parts)}; {calls} model calls; {end}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE, help="private evals checkout")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="FinanceBench download")
    ap.add_argument("--docs", type=Path, default=None, help="default: <root>/docs")
    ap.add_argument("--hours", type=float, default=HOURS, help="stop starting work after this")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        if not args.state.is_dir():
            raise FileNotFoundError("evals state checkout")
        manifest = load_manifest()
        ds = download(args.root, manifest)
        results = download_results(args.root, manifest)
        store = DocStore(args.docs or args.root / "docs")
        shas = {d: manifest.files[pdf_path(d)] for d in ds.documents}
        night = run_night(State(args.state), ds, results, store, shas, hours=args.hours)
    except Exception as e:
        # the type only: a message could quote a question, an answer or a grade
        log.error("eval night failed: %s", type(e).__name__)
        return 1
    print(summary(night))
    return 2 if night.stopped == "infrastructure" else 0


if __name__ == "__main__":
    sys.exit(main())
