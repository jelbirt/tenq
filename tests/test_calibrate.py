"""Grader calibration, on made-up answers and grades: nothing here is dataset text."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tenq import calibrate
from tenq.brief import Prompt
from tenq.calibrate import (
    FAILED,
    SINGLE_STORE,
    calibration_set,
    grade_items,
    held_out_set,
    kappa,
    kappa_interval,
    report,
    run_order,
    wilson,
)
from tenq.financebench import GradedAnswer, Question
from tenq.grade import CORRECT, GRADED, INCORRECT, REFUSAL, Grade
from tenq.jsonl import load_records
from tenq.llm import DEFAULT_MODEL, AgentRun
from tenq.qa import INFRA_FAILURE, MODEL_FAILURE

PROMPT = Prompt("grade-test", "grade it")
CURRENT = {"prompt_version": "grade-test", "model": DEFAULT_MODEL}


def _q(qid: str) -> Question:
    return Question(qid, "Acme", "ACME_2020_10K", "t", None, f"q {qid}?", "$1", "", ())


def _answers(source: str, labels: list[str]) -> list[GradedAnswer]:
    return [
        GradedAnswer(source, f"q{i}", f"answer {i}", lab, f"gold {i}")
        for i, lab in enumerate(labels)
    ]


RESULTS = {
    SINGLE_STORE[0]: _answers(SINGLE_STORE[0], [CORRECT, CORRECT, INCORRECT, REFUSAL]),
    SINGLE_STORE[1]: _answers(SINGLE_STORE[1], [CORRECT, REFUSAL, REFUSAL, INCORRECT]),
    SINGLE_STORE[2]: _answers(SINGLE_STORE[2], [INCORRECT, CORRECT, CORRECT, CORRECT]),
    "m_oracle": _answers("m_oracle", [CORRECT] * 4),
    "m_closedBook": _answers("m_closedBook", [REFUSAL] * 3 + [INCORRECT]),
    "m_sharedStore": _answers("m_sharedStore", [INCORRECT] * 2 + [CORRECT] * 2),
}
QUESTIONS = {f"q{i}": _q(f"q{i}") for i in range(4)}


def test_kappa_matches_a_worked_example() -> None:
    # po = 0.7, pe = 0.5, so kappa = 0.4
    pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
    assert kappa(pairs) == pytest.approx(0.4)
    assert kappa([("y", "y")] * 5) is None  # one class on both sides: undefined
    assert kappa([]) is None


def test_wilson_interval_matches_known_values() -> None:
    lo, hi = wilson(0, 10) or (0.0, 0.0)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson(283, 314) or (0.0, 0.0)
    assert lo == pytest.approx(0.8640, abs=1e-3) and hi == pytest.approx(0.9300, abs=1e-3)
    assert wilson(0, 0) is None


def test_the_kappa_interval_is_repeatable_and_holds_the_estimate() -> None:
    pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
    ci = kappa_interval(pairs)
    assert ci == kappa_interval(pairs)
    assert ci is not None and ci[0] < 0.4 < ci[1]


def test_the_calibration_set_is_the_three_single_store_files() -> None:
    items = calibration_set(RESULTS)
    assert len(items) == 12 and {a.source for a in items} == set(SINGLE_STORE)


def test_the_held_out_set_has_the_same_mix_from_the_other_files() -> None:
    mix = Counter(a.label for a in calibration_set(RESULTS))
    held = held_out_set(RESULTS)
    assert Counter(a.label for a in held) == mix
    assert not {a.source for a in held} & set(SINGLE_STORE)
    assert held == held_out_set(dict(reversed(list(RESULTS.items()))))  # order-free
    with pytest.raises(ValueError, match="held-out"):
        held_out_set(RESULTS, Counter({REFUSAL: 9}))


def test_the_run_order_is_fixed_whatever_order_items_come_in() -> None:
    items = calibration_set(RESULTS)
    assert run_order(items) == run_order(list(reversed(items)))
    assert run_order(items) != items


def _grader(labels: dict[str, str | None], seen: list[str]) -> Any:
    """A grader keyed on the answer text: a label, None for a model failure,
    "infra" for an infrastructure failure."""

    def grader(q: Question, answer: str, prompt: Prompt, *, model: str) -> tuple[AgentRun, Grade]:
        seen.append(answer)
        run = AgentRun(model=model, text="x")
        run.usage = {"output_tokens": 5}
        got = labels.get(answer, CORRECT)
        if got == "infra":
            return run, Grade(INFRA_FAILURE, error="api error: x")
        if got is None:
            return run, Grade(MODEL_FAILURE, error="no JSON object with a known label")
        return run, Grade(GRADED, got, "made-up reason")

    return grader


def test_items_are_graded_once_and_resumed(tmp_path: Path) -> None:
    items = calibration_set(RESULTS)
    records = tmp_path / "cal.jsonl"
    seen: list[str] = []
    assert grade_items(items, QUESTIONS, records, PROMPT, limit=5, grader=_grader({}, seen)) == (
        5,
        False,
    )
    assert grade_items(items, QUESTIONS, records, PROMPT, grader=_grader({}, seen)) == (7, False)
    assert grade_items(items, QUESTIONS, records, PROMPT, grader=_grader({}, seen)) == (0, False)
    assert len(seen) == 12
    rec = load_records(records)
    assert len(rec) == 12
    one = next(iter(rec.values()))
    assert one["prompt_version"] == "grade-test" and one["usage"] == {"output_tokens": 5}
    assert one["human"] in (CORRECT, INCORRECT, REFUSAL) and one["label"] == CORRECT


def test_an_infrastructure_failure_stops_and_is_not_recorded(tmp_path: Path) -> None:
    items = run_order(calibration_set(RESULTS))
    records = tmp_path / "cal.jsonl"
    labels: dict[str, str | None] = {items[2].model_answer: "infra", items[1].model_answer: None}
    got = grade_items(items, QUESTIONS, records, PROMPT, grader=_grader(labels, []))
    assert got == (2, True)
    rec = load_records(records)
    assert set(rec) == {items[0].key, items[1].key}
    assert rec[items[1].key]["status"] == MODEL_FAILURE  # finished: the grader's own failure
    seen: list[str] = []
    grade_items(items, QUESTIONS, records, PROMPT, grader=_grader({}, seen))
    assert seen[0] == items[2].model_answer  # the infra item runs again first


def test_a_record_cut_short_by_a_crash_is_graded_again(tmp_path: Path) -> None:
    items = calibration_set(RESULTS)
    records = tmp_path / "cal.jsonl"
    grade_items(items, QUESTIONS, records, PROMPT, limit=2, grader=_grader({}, []))
    lines = records.read_text().splitlines()
    records.write_text(lines[0] + "\n" + lines[1][:20])
    seen: list[str] = []
    assert grade_items(items, QUESTIONS, records, PROMPT, grader=_grader({}, seen))[0] == 11
    assert len(load_records(records)) == 12


def _records(items: list[GradedAnswer], label: Any = None) -> dict[str, dict[str, Any]]:
    return {
        a.key: {"key": a.key, "status": GRADED, "label": label(a) if label else a.label,
                "prompt_version": "grade-test", "model": DEFAULT_MODEL}
        for a in items
    }  # fmt: skip


def test_full_agreement_passes_the_gate() -> None:
    items = calibration_set(RESULTS)
    rep = report(items, _records(items), **CURRENT)
    assert rep["kappa"] == pytest.approx(1.0) and rep["gate"] == {
        "kappa_at_least": 0.8,
        "non_refusal_agreement_at_least": 0.9,
        "decided": True,
        "passed": True,
    }
    assert rep["non_refusal_agreement"]["n"] == 9 and rep["agreement"]["agree"] == 12


def test_a_grader_failure_is_a_disagreement_in_its_own_column() -> None:
    items = calibration_set(RESULTS)
    recs = _records(items)
    first = items[0]
    recs[first.key] = {"key": first.key, "status": MODEL_FAILURE, "label": None, **CURRENT}
    rep = report(items, recs, **CURRENT)
    assert rep["grader_failed"] == 1 and rep["matrix"][first.label][FAILED] == 1
    assert rep["agreement"]["agree"] == 11 and not rep["gate"]["passed"]


def test_refusals_graded_incorrect_fail_on_kappa_alone() -> None:
    items = calibration_set(RESULTS)
    rep = report(
        items, _records(items, lambda a: INCORRECT if a.label == REFUSAL else a.label), **CURRENT
    )
    assert rep["non_refusal_agreement"]["rate"] == 1.0
    assert rep["kappa"] < 0.8 and not rep["gate"]["passed"]


def test_the_gate_is_not_decided_until_every_item_is_finished() -> None:
    items = calibration_set(RESULTS)
    recs = _records(items)
    recs[items[3].key]["status"] = INFRA_FAILURE  # never written in practice, but not finished
    del recs[items[4].key]
    rep = report(items, recs, **CURRENT)
    assert rep["finished"] == 10 and rep["gate"]["decided"] is False
    assert rep["gate"]["passed"] is False and rep["kappa"] == pytest.approx(1.0)


def test_the_single_store_comparison_counts_correct_by_source() -> None:
    items = calibration_set(RESULTS)
    rep = report(items, _records(items, lambda a: CORRECT), **CURRENT)
    assert rep["per_source"][SINGLE_STORE[0]] == {"n": 4, "human_correct": 2, "grader_correct": 4}


def test_the_report_carries_no_answer_or_reason_text(tmp_path: Path) -> None:
    items = calibration_set(RESULTS)
    records = tmp_path / "cal.jsonl"
    grade_items(items, QUESTIONS, records, PROMPT, grader=_grader({}, []))
    text = json.dumps(report(items, load_records(records), **CURRENT))
    assert "answer 0" not in text and "made-up reason" not in text and "q q0" not in text


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        calibrate, "download", lambda root: type("D", (), {"questions": list(QUESTIONS.values())})()
    )
    monkeypatch.setattr(calibrate, "download_results", lambda root: RESULTS)
    return tmp_path


def test_the_command_prints_counts_only_and_writes_the_report(
    cli: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(calibrate, "grade", _grader({}, []))
    code = calibrate.main(
        ["--records", str(cli / "r.jsonl"), "--report", str(cli / "rep.json"), "--limit", "3"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out.strip() == "calibration singlestore: 3 graded this run, 3 of 12 finished"
    assert json.loads((cli / "rep.json").read_text())["finished"] == 3


def test_the_command_exits_2_on_an_infrastructure_failure(
    cli: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    first = run_order(calibration_set(RESULTS))[0]
    monkeypatch.setattr(calibrate, "grade", _grader({first.model_answer: "infra"}, []))
    assert calibrate.main(["--records", str(cli / "r.jsonl")]) == 2
    assert "stopped on an infrastructure failure" in capsys.readouterr().out


def test_a_crash_logs_only_its_type(
    cli: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise ValueError("quoting answer 0 and a gold answer of $1")

    monkeypatch.setattr(calibrate, "grade", boom)
    with caplog.at_level(logging.DEBUG):
        assert calibrate.main(["--records", str(cli / "r.jsonl")]) == 1
    assert "ValueError" in caplog.text and "answer 0" not in caplog.text and "$1" not in caplog.text


def test_kappa_counts_a_grader_failure_as_a_disagreement_across_three_labels() -> None:
    # 10 items; agree 7. Human margins C 4, I 3, R 3; grader C 4, I 3, R 2, failed 1.
    # pe = (4*4 + 3*3 + 3*2) / 100 = 0.31, so kappa = (0.7 - 0.31) / 0.69
    pairs = (
        [(CORRECT, CORRECT)] * 3 + [(CORRECT, FAILED)]
        + [(INCORRECT, INCORRECT)] * 2 + [(INCORRECT, CORRECT)]
        + [(REFUSAL, REFUSAL)] * 2 + [(REFUSAL, INCORRECT)]
    )  # fmt: skip
    assert kappa(pairs) == pytest.approx(0.39 / 0.69)


def test_records_from_another_prompt_version_or_model_are_stale(tmp_path: Path) -> None:
    items = calibration_set(RESULTS)
    records = tmp_path / "cal.jsonl"
    grade_items(items, QUESTIONS, records, PROMPT, grader=_grader({}, []))
    assert report(items, load_records(records), **CURRENT)["gate"]["decided"]
    v2 = Prompt("grade-v2-test", "grade it again")
    rep = report(items, load_records(records), prompt_version=v2.version, model=DEFAULT_MODEL)
    assert (rep["finished"], rep["stale"], rep["gate"]["decided"]) == (0, 12, False)
    rep = report(items, load_records(records), prompt_version="grade-test", model="haiku")
    assert rep["finished"] == 0 and not rep["gate"]["passed"]
    seen: list[str] = []
    assert grade_items(items, QUESTIONS, records, v2, grader=_grader({}, seen))[0] == 12
    rep = report(items, load_records(records), prompt_version=v2.version, model=DEFAULT_MODEL)
    assert rep["finished"] == 12 and rep["gate"]["decided"]


def test_the_grader_sees_the_gold_the_people_graded_against(tmp_path: Path) -> None:
    golds: list[str] = []

    def grader(q: Question, answer: str, prompt: Prompt, *, model: str) -> tuple[None, Grade]:
        golds.append(q.answer)
        return None, Grade(GRADED, CORRECT)

    grade_items(calibration_set(RESULTS), QUESTIONS, tmp_path / "r.jsonl", PROMPT, grader=grader)
    assert sorted(set(golds)) == ["gold 0", "gold 1", "gold 2", "gold 3"]


@pytest.mark.parametrize("cut", [20, None])  # mid-record, or after the JSON but before its newline
def test_a_line_left_open_by_a_crash_loses_no_later_record(tmp_path: Path, cut: int | None) -> None:
    items = calibration_set(RESULTS)
    records = tmp_path / "cal.jsonl"
    grade_items(items, QUESTIONS, records, PROMPT, limit=2, grader=_grader({}, []))
    first, second = records.read_text().splitlines()
    records.write_text(first + "\n" + (second[:cut] if cut else second))
    graded, _ = grade_items(items, QUESTIONS, records, PROMPT, grader=_grader({}, []))
    assert graded == (11 if cut else 10)
    assert len(load_records(records)) == 12


def test_grading_stops_when_asked_and_reports_every_call(tmp_path: Path) -> None:
    items = calibration_set(RESULTS)
    records = tmp_path / "cal.jsonl"
    runs: list[AgentRun | None] = []
    seen: list[str] = []
    count, stopped = grade_items(
        items, QUESTIONS, records, PROMPT, grader=_grader({}, seen),
        stop=lambda: len(seen) >= 2, on_run=runs.append,
    )  # fmt: skip
    assert (count, stopped) == (2, False)
    assert len(runs) == 2 and all(r is not None and r.usage for r in runs)
