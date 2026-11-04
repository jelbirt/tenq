import json
import logging
from pathlib import Path
from typing import Any

import pytest

from tenq import jev
from tenq.jev import QuestionSet, ask, average, load_questions

QUESTIONS = """\
version: test-q1
questions:
  material:
    type: noul
    instructions: Is the news material?
    criteria:
      "true": Yes.
      "false": No.
  direction:
    type: choice
    instructions: Good or bad?
    criteria:
      good: Good.
      bad: Bad.
      neutral: Neither.
  trend:
    type: score
    instructions: How is it trending?
    criteria:
      - Worse.
      - Steady.
      - Better.
"""


@pytest.fixture
def questions(tmp_path: Path) -> QuestionSet:
    path = tmp_path / "q.yaml"
    path.write_text(QUESTIONS)
    return load_questions(path)


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.delenv("JEV_MODEL", raising=False)
    return "sk-or-test-key"


def _answers(noul: float, choice: str, probs: dict[str, float], score: float) -> dict[str, Any]:
    return {
        "material": {"type": "noul", "noul": noul},
        "direction": {"type": "choice", "choice": choice, "probabilities": probs},
        "trend": {"type": "score", "score": score, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
    }


def _ok(answers: dict[str, Any], cost: float = 0.00002) -> tuple[int, str]:
    return 200, json.dumps(
        {
            "model": "typesafe/jev-1.13-20260917",
            "answers": answers,
            "usage": {"input_tokens": 500, "output_tokens": 70, "cost": cost},
        }
    )


class Server:
    def __init__(self, *replies: tuple[int, str] | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        self.calls.append((url, headers, json.loads(body)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _ask(questions: QuestionSet, server: Server, **kw: Any) -> jev.Result:
    return ask("Facts for CrowdStrike (CRWD).", questions, post=server, sleep=lambda s: None, **kw)


def test_the_committed_question_file_loads() -> None:
    q = load_questions()
    assert q.version == "jev-q1"
    assert len(q.questions) == 8
    assert {x.type for x in q.questions} == {"noul", "choice", "score"}
    assert "\u2014" not in jev.QUESTIONS_PATH.read_text(encoding="utf-8")


def test_questions_are_sent_in_the_system_one_shape(questions: QuestionSet) -> None:
    assert questions.to_request() == {
        "material": {
            "type": "noul",
            "instructions": "Is the news material?",
            "criteria": {"true": "Yes.", "false": "No."},
        },
        "direction": {
            "type": "choice",
            "instructions": "Good or bad?",
            "criteria": {"good": "Good.", "bad": "Bad.", "neutral": "Neither."},
        },
        "trend": {
            "type": "score",
            "instructions": "How is it trending?",
            "criteria": ["Worse.", "Steady.", "Better."],
        },
    }


@pytest.mark.parametrize(
    "text",
    [
        "questions: {}\n",
        "version: v\nquestions: {}\nextra: 1\n",
        "version: ''\nquestions: {a: {type: noul, instructions: x}}\n",
        "version: v\nquestions: {}\n",
        "version: v\nquestions: {Bad-Key: {type: noul, instructions: x}}\n",
        "version: v\nquestions: {a: {type: noul, instructions: ''}}\n",
        "version: v\nquestions: {a: {type: noul, instructions: x, extra: 1}}\n",
        "version: v\nquestions: {a: {type: noul, instructions: x, criteria: {yes: y}}}\n",
        "version: v\nquestions: {a: {type: choice, instructions: x, criteria: {only: o}}}\n",
        "version: v\nquestions: {a: {type: choice, instructions: x, criteria: {Up: u, dn: d}}}\n",
        "version: v\nquestions: {a: {type: score, instructions: x, criteria: [one]}}\n",
        "version: v\nquestions: {a: {type: score, instructions: x, criteria: {a: b, c: d}}}\n",
        "version: v\nquestions: {a: {type: box, instructions: x}}\n",
    ],
)
def test_a_malformed_question_file_is_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_questions(path)


def test_no_key_is_no_call(questions: QuestionSet, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    server = Server()
    result = _ask(questions, server)
    assert server.calls == []
    assert result.answers == {}
    assert result.error == "Jev skipped: OPENROUTER_API_KEY not set"


def test_each_question_is_asked_three_times_and_averaged(questions: QuestionSet) -> None:
    server = Server(
        _ok(_answers(0.9, "good", {"good": 0.8, "bad": 0.1, "neutral": 0.1}, 2.0)),
        _ok(_answers(0.6, "good", {"good": 0.6, "bad": 0.3, "neutral": 0.1}, 1.5)),
        _ok(_answers(0.6, "bad", {"good": 0.4, "bad": 0.5, "neutral": 0.1}, 1.0)),
    )
    result = _ask(questions, server, session_id="run-1")
    assert len(server.calls) == 3 and result.repeats == 3 and result.errors == []
    url, headers, body = server.calls[0]
    assert url == "https://openrouter.ai/api/v1/systemone"
    assert headers["Authorization"] == "Bearer sk-or-test-key"
    assert body == {
        "model": "jev-latest",
        "state": "Facts for CrowdStrike (CRWD).",
        "questions": questions.to_request(),
        "session_id": "run-1",
    }
    material = result.answers["material"]
    assert material.value == pytest.approx(0.7) and material.spread == pytest.approx(0.3)
    direction = result.answers["direction"]
    assert direction.value == "good"
    assert direction.probabilities == pytest.approx({"good": 0.6, "bad": 0.3, "neutral": 0.1})
    assert direction.spread == pytest.approx(0.4)  # good's probability: 0.8 down to 0.4
    trend = result.answers["trend"]
    assert trend.value == pytest.approx(1.5) and trend.spread == pytest.approx(1.0)
    assert trend.probabilities == pytest.approx({"0": 0.1, "1": 0.2, "2": 0.7})
    assert result.resolved_model == "typesafe/jev-1.13-20260917"
    assert (result.input_tokens, result.output_tokens) == (1500, 210)
    assert result.cost == pytest.approx(0.00006)
    assert result.error is None


def test_the_model_is_a_setting(questions: QuestionSet, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_MODEL", "typesafe/jev-1.13")
    server = Server(_ok(_answers(0.5, "good", {}, 1.0)))
    result = _ask(questions, server, repeats=1)
    assert server.calls[0][2]["model"] == "typesafe/jev-1.13" == result.model


def test_a_rate_limit_is_retried_once(questions: QuestionSet) -> None:
    slept: list[float] = []
    server = Server((429, "{}"), _ok(_answers(0.5, "good", {}, 1.0)))
    result = ask("s", questions, repeats=1, post=server, sleep=slept.append)
    assert len(server.calls) == 2 and slept == [2.0]
    assert result.repeats == 1 and result.errors == []


def test_a_failed_repeat_is_left_out_of_the_average(questions: QuestionSet) -> None:
    server = Server(
        _ok(_answers(0.9, "good", {}, 2.0)),
        (503, "busy"),
        (503, "still busy"),
        TimeoutError("slow"),
        TimeoutError("slow"),
    )
    result = _ask(questions, server)
    assert result.repeats == 1
    assert result.errors == ["HTTP 503", "request error TimeoutError"]
    assert result.answers["material"].value == 0.9 and result.answers["material"].spread == 0
    assert result.error is None


@pytest.mark.parametrize("status", [401, 402, 403])
def test_a_bad_key_or_no_credit_stops_at_once(questions: QuestionSet, status: int) -> None:
    server = Server((status, '{"error": {"message": "no"}}'))
    result = _ask(questions, server)
    assert len(server.calls) == 1
    assert result.answers == {} and result.error == f"Jev gave no usable answer (HTTP {status})"


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((200, "not json"), "unreadable response"),
        ((200, "[1, 2]"), "response without answers"),
        ((200, '{"answers": []}'), "response without answers"),
        ((200, '{"answers": {}}'), None),
    ],
)
def test_an_unreadable_response_is_an_error(
    questions: QuestionSet, reply: tuple[int, str], error: str | None
) -> None:
    result = _ask(questions, Server(reply), repeats=1)
    assert result.answers == {}
    assert result.error == "Jev gave no usable answer" + (f" ({error})" if error else "")


def test_an_answer_out_of_range_or_of_the_wrong_kind_is_dropped(questions: QuestionSet) -> None:
    bad = {
        "material": {"noul": 1.5},
        "direction": {"choice": "sideways", "probabilities": {"sideways": 1.0}},
        "trend": {"score": 7},
    }
    worse = {"material": {"noul": True}, "direction": "good", "trend": {"score": float("nan")}}
    good = _answers(0.2, "bad", {"bad": 0.7, "good": 0.2, "neutral": 0.1, "other": 5}, 0.0)
    out = average(questions, [bad, worse, good])
    assert {k: a.repeats for k, a in out.items()} == {"material": 1, "direction": 1, "trend": 1}
    assert out["direction"].probabilities == {"good": 0.2, "bad": 0.7, "neutral": 0.1}


def test_a_question_no_repeat_answered_is_left_out(questions: QuestionSet) -> None:
    out = average(questions, [{"material": {"noul": 0.4}}, {}])
    assert set(out) == {"material"}
    assert out["material"].repeats == 1


def test_a_choice_without_probabilities_counts_its_pick(questions: QuestionSet) -> None:
    out = average(
        questions,
        [
            {"direction": {"choice": "bad"}},
            {"direction": {"choice": "bad"}},
            {"direction": {"choice": "good"}},
        ],
    )
    d = out["direction"]
    assert d.value == "bad"
    assert d.probabilities == pytest.approx({"good": 1 / 3, "bad": 2 / 3, "neutral": 0.0})
    assert d.spread == 1.0


def test_a_tied_choice_goes_to_the_earlier_label(questions: QuestionSet) -> None:
    out = average(questions, [{"direction": {"choice": "bad"}}, {"direction": {"choice": "good"}}])
    assert out["direction"].value == "good"


def test_a_long_state_is_cut_and_noted(questions: QuestionSet) -> None:
    server = Server(_ok(_answers(0.5, "good", {}, 1.0)))
    result = ask("x" * (jev.MAX_STATE_CHARS + 10), questions, repeats=1, post=server)
    assert len(server.calls[0][2]["state"]) == jev.MAX_STATE_CHARS
    assert result.errors == [f"state cut to {jev.MAX_STATE_CHARS} characters"]


def test_logs_and_the_record_never_carry_the_key_state_or_reply(
    questions: QuestionSet, key: str, caplog: pytest.LogCaptureFixture
) -> None:
    secret_state = "CONFIDENTIAL-STATE-7731"
    secret_reply = "CONFIDENTIAL-REPLY-7731"
    server = Server(
        (400, secret_reply),
        (200, secret_reply),
        TimeoutError(secret_reply),
        TimeoutError(secret_reply),
    )
    with caplog.at_level(logging.DEBUG):
        result = ask(secret_state, questions, post=server, sleep=lambda s: None)
    text = caplog.text + json.dumps(result.to_dict())
    for secret in (key, secret_state, secret_reply):
        assert secret not in text
    assert result.errors == ["HTTP 400", "unreadable response", "request error TimeoutError"]


def test_the_result_is_saved_as_plain_json(questions: QuestionSet) -> None:
    server = Server(_ok(_answers(0.9, "good", {"good": 1.0}, 2.0)))
    saved = json.loads(json.dumps(_ask(questions, server, repeats=1).to_dict()))
    assert saved["version"] == "test-q1"
    assert saved["answers"]["direction"]["value"] == "good"
    assert saved["answers"]["material"]["probabilities"] == {}
