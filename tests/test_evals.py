"""The nightly eval runner, on made-up questions, answers and grades: no dataset text."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from conftest import QA_DOC

from tenq import evals
from tenq.brief import Prompt
from tenq.calibrate import SINGLE_STORE
from tenq.docstore import DocStore
from tenq.evals import (
    CLOSED_BOOK,
    Night,
    State,
    answer_key,
    plan,
    run_night,
    summary,
)
from tenq.evals import (
    SINGLE_STORE as SINGLE,
)
from tenq.financebench import Dataset, Document, Evidence, GradedAnswer, Question
from tenq.grade import CORRECT, GRADED, INCORRECT, REFUSAL, Grade
from tenq.jsonl import append, load_lines, load_records
from tenq.llm import DEFAULT_MODEL, AgentRun
from tenq.qa import ANSWERED, INFRA_FAILURE, MODEL_FAILURE, Answer

SHA = "a" * 64


def _q(i: int) -> Question:
    return Question(
        f"q{i}",
        "Acme",
        QA_DOC,
        "metrics-generated",
        None,
        f"What was widget figure {i}?",
        f"${i}",
        "",
        (Evidence(QA_DOC, 2, f"widget {i}"),),
    )


QUESTIONS = [_q(i) for i in range(4)]
LABELS = [CORRECT, INCORRECT, REFUSAL, CORRECT]
RESULTS = {
    source: [
        GradedAnswer(source, f"q{i}", f"paper answer {n}-{i}", LABELS[(i + n) % 4], f"${i}")
        for i in range(4)
    ]
    for n, source in enumerate(SINGLE_STORE)
}
# the people's label for each paper answer, by its text
HUMAN = {a.model_answer: a.label for answers in RESULTS.values() for a in answers}


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return Dataset(tmp_path / "fb", QUESTIONS, {QA_DOC: Document(QA_DOC, "Acme", "10k", 2020)})


@pytest.fixture
def state(tmp_path: Path) -> State:
    (tmp_path / "evals-state").mkdir()
    return State(tmp_path / "evals-state")


@pytest.fixture(autouse=True)
def fake_index(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    indexed: list[str] = []

    class FakeXerj:
        def __init__(self, url: str, store: DocStore) -> None:
            pass

        def index(self, key: str, state_dir: Path) -> None:
            indexed.append(key)

    monkeypatch.setattr(evals, "Xerj", FakeXerj)
    return indexed


@contextlib.contextmanager
def _node() -> Iterator[str]:
    yield "http://127.0.0.1:1"


class Models:
    """Fake askers and a fake grader, recording every call."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []
        self.graded: list[str] = []
        self.answer_status = ANSWERED
        self.infra_after: int | None = None
        self.usage_limit = False
        self.grade_label: str | None = None  # None: agree with the people
        self.grade_calls: list[tuple[Question, str, dict[str, Any]]] = []
        self.grade_infra_at: int | None = None  # this grader call fails, infrastructure
        self.grade_limit_at: int | None = None  # this grader call hits a usage limit

    def _run(self, model: str) -> AgentRun:
        run = AgentRun(model=model, duration_ms=1000, usage={"input_tokens": 10})
        n = len(self.asked)
        if self.infra_after is not None and n > self.infra_after:
            run.usage_limit = self.usage_limit
        return run

    def ask(self, q: Question, store: Any, url: str, prompt: Prompt, **kw: Any) -> Any:
        self.asked.append((SINGLE, kw["model"]))
        return self._answer(kw["model"], [2])

    def ask_closed(self, q: Question, prompt: Prompt, **kw: Any) -> Any:
        self.asked.append((CLOSED_BOOK, kw["model"]))
        return self._answer(kw["model"], [])

    def _answer(self, model: str, pages: list[int]) -> tuple[AgentRun, Answer]:
        run = self._run(model)
        if self.infra_after is not None and len(self.asked) > self.infra_after:
            return run, Answer(INFRA_FAILURE, error="usage limit")
        if self.answer_status == MODEL_FAILURE:
            return run, Answer(MODEL_FAILURE, error="timeout")
        return run, Answer(ANSWERED, "the figure was $1", ["x:1"], pages)

    def grade(self, q: Question, answer: str, prompt: Prompt, **kw: Any) -> Any:
        self.graded.append(answer)
        self.grade_calls.append((q, answer, kw))
        n = len(self.graded)
        run = AgentRun(model=DEFAULT_MODEL, duration_ms=10)
        if n == self.grade_limit_at:
            run.usage_limit = True
        if n in (self.grade_infra_at, self.grade_limit_at):
            return run, Grade(INFRA_FAILURE, error="api error: rate_limit")
        label = self.grade_label or HUMAN.get(answer, CORRECT)
        return run, Grade(GRADED, label, "why")


def _night(state: State, ds: Dataset, docs: DocStore, m: Models, **kw: Any) -> Night:
    return run_night(
        state,
        ds,
        RESULTS,
        docs,
        {QA_DOC: SHA},
        node=_node,
        asker=m.ask,
        closed_asker=m.ask_closed,
        grader=m.grade,
        **kw,
    )


def test_the_plan_is_singlestore_on_all_then_closed_book_on_some() -> None:
    order = plan(QUESTIONS)
    assert [mode for mode, _ in order] == [SINGLE] * 4 + [CLOSED_BOOK] * 4
    assert {q.id for _, q in order[:4]} == {q.id for q in QUESTIONS}
    assert order == plan(list(reversed(QUESTIONS)))  # fixed order, whatever the input


def test_a_first_night_calibrates_answers_and_grades(
    state: State, ds: Dataset, docs: DocStore, fake_index: list[str]
) -> None:
    m = Models()
    night = _night(state, ds, docs, m)
    assert night.stopped is None and night.gate_decided
    assert night.counts["calibrated"] == 12
    # both models in turn, singleStore then closed-book
    assert m.asked[:2] == [(SINGLE, "opus"), (SINGLE, "haiku")]
    assert night.counts["answered_singlestore_opus"] == 4
    assert night.counts["answered_closedbook_haiku"] == 4
    assert fake_index == [QA_DOC]  # indexed once a night, however many questions
    answers = load_records(state.answers)
    assert len(answers) == 16
    rec = answers[answer_key(SINGLE, "opus", "q0")]
    assert rec["evidence"] == {"gold_pages": [2], "any_hit": True, "all_hit": True}
    assert "evidence" not in answers[answer_key(CLOSED_BOOK, "opus", "q0")]
    assert night.counts["graded"] == 16 and len(load_records(state.grades)) == 16
    (line,) = load_lines(state.nights)
    assert line["usage"]["calls"] == {"calibration": 12, "answer": 16, "grade": 16}
    assert line["usage"]["tokens"]["input_tokens"] == 160
    assert json.loads(state.calibration_report.read_text())["gate"]["passed"]


def test_the_grader_never_sees_which_model_answered(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    m = Models()
    _night(state, ds, docs, m)
    answers = load_records(state.answers)
    calls = m.grade_calls[12:]  # past calibration: tenq's own answers
    assert len(calls) == 16
    for q, answer, kw in calls:
        assert kw == {"model": DEFAULT_MODEL}  # the grader's model, nothing about the author
        assert q in QUESTIONS  # the dataset's question, untouched
        assert answer == "the figure was $1"  # byte for byte what both models said
    assert {r["answer"] for r in answers.values()} == {"the figure was $1"}


def test_a_second_night_resumes_and_redoes_nothing(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    _night(state, ds, docs, Models())
    m = Models()
    night = _night(state, ds, docs, m)
    assert m.asked == [] and m.graded == []
    assert night.gate_decided and night.stopped is None


def test_night_one_caps_each_model_lower_than_later_nights(
    state: State, ds: Dataset, docs: DocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evals, "NIGHT_ONE_PER_MODEL", 1)
    monkeypatch.setattr(evals, "PER_MODEL", 3)
    m = Models()
    _night(state, ds, docs, m)
    assert m.asked == [(SINGLE, "opus"), (SINGLE, "haiku")]
    m = Models()
    _night(state, ds, docs, m)
    assert len(m.asked) == 6


def test_an_infrastructure_failure_stops_the_night_and_records_nothing_for_it(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    m = Models()
    m.infra_after = 2
    night = _night(state, ds, docs, m)
    assert night.stopped == "infrastructure"
    assert len(load_records(state.answers)) == 2  # the failed one runs again later
    assert len(m.graded) == 12  # calibration only: no grading after the stop
    assert load_lines(state.nights)[-1]["stopped"] == "infrastructure"


def test_a_night_after_a_usage_limit_is_skipped_then_work_resumes(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    m = Models()
    m.infra_after, m.usage_limit = 0, True
    _night(state, ds, docs, m)
    assert load_lines(state.nights)[-1]["usage"]["usage_limit"]
    m = Models()
    night = _night(state, ds, docs, m)
    assert night.stopped == "skipped after a usage limit" and m.asked == []
    assert load_lines(state.nights)[-1]["skipped"]
    night = _night(state, ds, docs, Models())
    assert night.stopped is None


def test_the_deadline_stops_new_work(state: State, ds: Dataset, docs: DocStore) -> None:
    m = Models()
    night = _night(state, ds, docs, m, hours=0)
    assert night.stopped == "deadline"
    assert m.asked == [] and m.graded == []


def test_answers_are_not_graded_until_the_gate_passes(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    m = Models()
    m.grade_label = INCORRECT  # disagrees with the people: the gate fails
    night = _night(state, ds, docs, m)
    assert night.gate_decided
    assert not json.loads(state.calibration_report.read_text())["gate"]["passed"]
    assert len(load_records(state.answers)) == 16
    assert load_records(state.grades) == {}


def test_a_model_failure_is_a_refusal_with_no_grader_call(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    m = Models()
    m.answer_status = MODEL_FAILURE
    _night(state, ds, docs, m)
    assert len(m.graded) == 12  # calibration only
    grades = load_records(state.grades)
    assert len(grades) == 16 and {g["label"] for g in grades.values()} == {REFUSAL}


def test_an_answer_from_an_older_prompt_is_answered_again(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    _night(state, ds, docs, Models())
    key = answer_key(SINGLE, "opus", "q0")
    append(state.answers, {**load_records(state.answers)[key], "prompt_version": "qa-v0"})
    m = Models()
    _night(state, ds, docs, m)
    assert m.asked == [(SINGLE, "opus")]


def test_a_new_grader_version_grades_every_answer_again(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    _night(state, ds, docs, Models())
    for rec in load_records(state.grades).values():
        append(state.grades, {**rec, "prompt_version": "grade-v0"})
    m = Models()
    night = _night(state, ds, docs, m)
    assert night.counts["graded"] == 16 and len(m.graded) == 16


def test_an_answer_whose_question_left_the_dataset_is_skipped(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    _night(state, ds, docs, Models())
    rec = load_records(state.answers)[answer_key(SINGLE, "opus", "q0")]
    append(state.answers, {**rec, "key": "singlestore:opus:gone", "question_id": "gone"})
    night = _night(state, ds, docs, Models())
    assert night.counts["answer_without_question"] == 1 and night.stopped is None


@pytest.mark.parametrize("at", [3, 14])  # in calibration, then grading tenq's answers
def test_a_usage_limit_while_grading_stops_and_skips_the_next_night(
    state: State, ds: Dataset, docs: DocStore, at: int
) -> None:
    m = Models()
    m.grade_limit_at = at
    night = _night(state, ds, docs, m)
    assert night.stopped == "infrastructure"
    assert load_lines(state.nights)[-1]["usage"]["usage_limit"]
    m = Models()
    assert _night(state, ds, docs, m).stopped == "skipped after a usage limit"
    assert m.graded == [] and m.asked == []


@pytest.mark.parametrize("at", [3, 14])
def test_an_infrastructure_failure_while_grading_records_nothing_for_it(
    state: State, ds: Dataset, docs: DocStore, at: int
) -> None:
    m = Models()
    m.grade_infra_at = at
    night = _night(state, ds, docs, m)
    assert night.stopped == "infrastructure"
    recorded = len(load_records(state.calibration)) + len(load_records(state.grades))
    assert recorded == at - 1
    assert not load_lines(state.nights)[-1]["usage"]["usage_limit"]


def test_the_deadline_counts_from_the_process_start_less_a_reserve(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    m = Models()
    limit = evals.HOURS * 3600 - evals.RESERVE_S
    # inside the reserve before the hours are up: nothing new starts
    night = _night(state, ds, docs, m, clock=lambda: limit + 1, started=0.0)
    assert night.stopped == "deadline" and m.asked == [] and m.graded == []
    m = Models()
    night = _night(state, ds, docs, m, clock=lambda: limit - 1, started=0.0)
    assert night.stopped is None and m.asked


def test_a_search_node_that_will_not_start_stops_the_night(
    state: State, ds: Dataset, docs: DocStore
) -> None:
    @contextlib.contextmanager
    def dead() -> Iterator[str]:
        raise RuntimeError("xerj did not start")
        yield ""

    m = Models()
    night = run_night(
        state, ds, RESULTS, docs, {QA_DOC: SHA}, node=dead,
        asker=m.ask, closed_asker=m.ask_closed, grader=m.grade,
    )  # fmt: skip
    assert night.stopped == "infrastructure"
    assert m.asked == [] and load_records(state.answers) == {}


def test_documents_failing_to_index_in_a_row_mean_search_is_down(
    state: State, docs: DocStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [f"ACME{i}_2020_10K" for i in range(4)]
    for name in names:
        docs.save(name, SHA, ["one", "two"], 0.1)
    qs = [replace(_q(i), doc_name=name) for i, name in enumerate(names)]
    many = Dataset(tmp_path / "fb", qs, {n: Document(n, "Acme", "10k", 2020) for n in names})

    class Broken:
        def __init__(self, url: str, store: DocStore) -> None:
            pass

        def index(self, key: str, state_dir: Path) -> None:
            raise RuntimeError("xerj autoindex failed")

    monkeypatch.setattr(evals, "Xerj", Broken)
    m = Models()
    night = run_night(
        state, many, RESULTS, docs, dict.fromkeys(names, SHA), node=_node,
        asker=m.ask, closed_asker=m.ask_closed, grader=m.grade,
    )  # fmt: skip
    assert night.stopped == "infrastructure"
    assert night.counts["document_not_ready"] == evals.INDEX_FAILURES - 1
    assert m.asked == []


def test_a_changed_answer_is_graded_again(state: State, ds: Dataset, docs: DocStore) -> None:
    _night(state, ds, docs, Models())
    key = answer_key(SINGLE, "opus", "q0")
    rec = load_records(state.answers)[key]
    append(state.answers, {**rec, "answer": "a different figure"})
    m = Models()
    _night(state, ds, docs, m)
    assert m.graded == ["a different figure"]


def test_a_document_that_will_not_index_skips_its_questions(
    state: State, ds: Dataset, docs: DocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        def __init__(self, url: str, store: DocStore) -> None:
            pass

        def index(self, key: str, state_dir: Path) -> None:
            raise RuntimeError("xerj autoindex failed")

    monkeypatch.setattr(evals, "Xerj", Broken)
    m = Models()
    night = _night(state, ds, docs, m)
    assert night.counts["document_not_ready"] == 4
    assert {mode for mode, _ in m.asked} == {CLOSED_BOOK}
    assert night.stopped is None


def test_a_document_not_yet_extracted_is_extracted_first(
    state: State, ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def extract(pdf: Path) -> tuple[list[str], float]:
        calls.append(pdf)
        return ["page zero", "page one", "widget 1"], 0.1

    monkeypatch.setattr(evals, "extract_pages", extract)
    store = DocStore(tmp_path / "fresh")
    _night(state, ds, store, Models())
    assert calls == [ds.pdf(QA_DOC)]
    assert store.has(QA_DOC, SHA)


def test_the_summary_is_counts_only() -> None:
    night = Night(started_at="t")
    night.counts.update({"answered_singlestore_opus": 3, "graded": 3})
    text = summary(night)
    assert text.startswith("eval night: answered_singlestore_opus 3, graded 3")
    assert "finished" in text


def test_a_crash_logs_only_its_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = tmp_path / "not-a-state-dir-4321-widgets"
    with caplog.at_level(logging.ERROR):
        assert evals.main(["--state", str(secret)]) == 1
    assert "FileNotFoundError" in caplog.text
    assert "4321" not in caplog.text and "4321" not in capsys.readouterr().out


def test_a_crash_mid_run_logs_only_its_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "s").mkdir()

    def boom(*a: Any, **kw: Any) -> Any:
        raise ValueError("gold answer was 4,321 widgets")

    monkeypatch.setattr(evals, "load_manifest", boom)
    with caplog.at_level(logging.ERROR):
        assert evals.main(["--state", str(tmp_path / "s")]) == 1
    assert "ValueError" in caplog.text and "4,321" not in caplog.text
