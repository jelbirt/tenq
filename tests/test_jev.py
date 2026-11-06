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


def _answers(
    noul: float,
    choice: str,
    probs: dict[str, float],
    score: float,
    levels: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One reply. On odd repeats the score levels were sent reversed, so a
    reply means level 0 as the file's last level: pass those as Jev sees them."""
    return {
        "material": {"type": "noul", "noul": noul},
        "direction": {"type": "choice", "choice": choice, "probabilities": probs},
        "trend": {
            "type": "score",
            "score": score,
            "probabilities": levels or {"0": 0.1, "1": 0.2, "2": 0.7},
        },
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
    assert q.version == "jev-q8"
    assert len(q.questions) == 5
    assert {x.type for x in q.questions} == {"noul", "choice"}
    assert "\u2014" not in jev.QUESTIONS_PATH.read_text(encoding="utf-8")


def test_jev_reads_text_only_and_judges_surprise_not_cheer() -> None:
    """jev-q8: no question asks Jev to compare figures; routine news is neutral both ways."""
    by_key = {x.key: x for x in load_questions().questions}
    assert "fundamentals" not in by_key  # the filing trend is read in code (rules.trend)
    for q in by_key.values():
        assert "points" not in q.instructions and "%" not in q.instructions, q.key
    direction = by_key["direction"].instructions
    assert "than investors" in direction and "count as neutral" in direction
    assert "does not tie to a shortfall or a change in plans" in direction
    # a headline that gives only figures ("EPS $0.93 vs $0.81 est.") is not Jev's to compare
    guidance = by_key["guidance"].instructions
    for text, fallback in ((direction, "counts as neutral"), (guidance, "counts as issued")):
        assert "Never compare figures yourself" in text and fallback in text
    labels = {q.key: q for q in load_questions().questions}["guidance"].criteria
    assert "figures only" in labels["issued"] and "The text says" in labels["mixed"]


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


def test_each_question_is_asked_in_balanced_rounds_and_averaged(questions: QuestionSet) -> None:
    server = Server(
        _ok(_answers(0.9, "good", {"good": 0.8, "bad": 0.1, "neutral": 0.1}, 2.0)),
        # repeat 2 saw the levels reversed: its 0.5 is the file's 1.5
        _ok(
            _answers(
                0.6,
                "good",
                {"good": 0.6, "bad": 0.3, "neutral": 0.1},
                0.5,
                {"0": 0.7, "1": 0.2, "2": 0.1},
            )
        ),
        _ok(_answers(0.6, "bad", {"good": 0.4, "bad": 0.5, "neutral": 0.1}, 1.0)),
        # round 4 asks only the score (reversed); its other answers are ignored
        _ok(_answers(0.0, "bad", {"bad": 1.0}, 0.5, {"0": 0.7, "1": 0.2, "2": 0.1})),
    )
    result = _ask(questions, server, session_id="run-1")
    assert len(server.calls) == 4 and result.rounds == 4 and result.errors == []
    url, headers, body = server.calls[0]
    assert url == "https://openrouter.ai/api/v1/systemone"
    assert headers["Authorization"] == "Bearer sk-or-test-key"
    assert body == {
        "model": "jev-latest",
        "state": "Facts for CrowdStrike (CRWD).",
        "questions": questions.to_request(0),
        "session_id": "run-1",
    }
    sent = [c[2]["questions"] for c in server.calls]
    assert [list(q["direction"]["criteria"]) for q in sent[:3]] == [
        ["good", "bad", "neutral"],
        ["bad", "neutral", "good"],
        ["neutral", "good", "bad"],
    ]
    assert "direction" not in sent[3] and "material" not in sent[3]
    assert [q["trend"]["criteria"][0] for q in sent] == ["Worse.", "Better.", "Worse.", "Better."]
    material = result.answers["material"]
    assert material.value == pytest.approx(0.7) and material.spread == pytest.approx(0.3)
    direction = result.answers["direction"]
    assert direction.value == "good"
    assert direction.probabilities == pytest.approx({"good": 0.6, "bad": 0.3, "neutral": 0.1})
    assert direction.spread == pytest.approx(0.4)  # good's probability: 0.8 down to 0.4
    assert direction.margin == pytest.approx(0.3)  # good 0.6 against bad 0.3
    assert direction.agreed == 2  # the third repeat picked bad
    assert material.margin is None and material.agreed == 3
    assert material.balanced and material.first_lift is None  # true/false has no order
    # first shown: good 0.8, bad 0.3, neutral 0.1; a fair share is 1/3
    assert direction.balanced and direction.first_lift == pytest.approx(0.4 - 1 / 3)
    trend = result.answers["trend"]
    assert trend.value == pytest.approx(1.5) and trend.spread == pytest.approx(1.0)
    assert trend.probabilities == pytest.approx({"0": 0.1, "1": 0.2, "2": 0.7})
    assert trend.agreed == 3  # levels 2, 1.5, 1, 1.5 round half up to 2, 2, 1, 2; the mean to 2
    assert trend.balanced and trend.first_lift == pytest.approx(0.0)  # 1.5 either way
    assert result.resolved_model == "typesafe/jev-1.13-20260917"
    assert (result.input_tokens, result.output_tokens) == (2000, 280)
    assert result.cost == pytest.approx(0.00008)
    assert result.error is None


def test_the_model_is_a_setting(questions: QuestionSet, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_MODEL", "typesafe/jev-1.13")
    server = Server(_ok(_answers(0.5, "good", {}, 1.0)))
    result = _ask(questions, server, rounds=1)
    assert server.calls[0][2]["model"] == "typesafe/jev-1.13" == result.model


def test_a_rate_limit_is_retried_once(questions: QuestionSet) -> None:
    slept: list[float] = []
    server = Server((429, "{}"), _ok(_answers(0.5, "good", {}, 1.0)))
    result = ask("s", questions, rounds=1, post=server, sleep=slept.append)
    assert len(server.calls) == 2 and slept == [2.0]
    assert result.rounds == 1 and result.errors == []


def test_a_failed_repeat_is_left_out_of_the_average(questions: QuestionSet) -> None:
    server = Server(
        _ok(_answers(0.9, "good", {}, 2.0)),
        (429, "slow down"),
        (429, "slow down"),
        _ok(_answers(0.5, "good", {}, 2.0)),
        _ok(_answers(0.5, "good", {}, 0.0)),
    )
    result = _ask(questions, server)
    assert len(server.calls) == 5 and result.rounds == 3
    assert result.answered == [0, 2, 3]  # the failed round keeps its place
    assert result.errors == ["HTTP 429"]
    assert result.answers["material"].value == pytest.approx(0.7)
    assert result.answers["material"].spread == pytest.approx(0.4)
    direction = result.answers["direction"]
    assert not direction.balanced and direction.first_lift is None  # "bad" never led
    assert result.error is None


@pytest.mark.parametrize(
    ("first", "error"),
    [((503, "down"), "HTTP 503"), (TimeoutError("slow"), "request error TimeoutError")],
)
def test_a_service_still_down_after_its_retry_ends_the_repeats(
    questions: QuestionSet, first: tuple[int, str] | Exception, error: str
) -> None:
    slept: list[float] = []
    server = Server(first, first)
    result = ask("s", questions, post=server, sleep=slept.append)
    assert len(server.calls) == 2 and slept == [2.0]  # one retry, no sleep after the last
    assert result.errors == [error]
    assert result.error == f"Jev gave no usable answer ({error})"
    assert result.down  # the caller skips Jev for the rest of the run


@pytest.mark.parametrize("status", [400, 404, 413])
def test_a_request_the_service_refuses_is_not_repeated(questions: QuestionSet, status: int) -> None:
    server = Server((status, "{}"))
    result = _ask(questions, server)
    assert len(server.calls) == 1 and result.errors == [f"HTTP {status}"]
    assert not result.down  # this request was bad; the next stock's may not be


@pytest.mark.parametrize("status", [401, 402, 403])
def test_a_bad_key_or_no_credit_stops_at_once(questions: QuestionSet, status: int) -> None:
    server = Server((status, '{"error": {"message": "no"}}'))
    result = _ask(questions, server)
    assert len(server.calls) == 1
    assert result.answers == {} and result.error == f"Jev gave no usable answer (HTTP {status})"
    assert result.down


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
    result = _ask(questions, Server(reply), rounds=1)
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
    # firsts were good, bad, neutral: one of three picks was the first-listed label
    assert d.balanced and d.first_lift == pytest.approx(1 / 3 - 1 / 3)


def test_a_tied_choice_goes_to_the_earlier_label(questions: QuestionSet) -> None:
    out = average(questions, [{"direction": {"choice": "bad"}}, {"direction": {"choice": "good"}}])
    assert out["direction"].value == "good"


def test_a_long_state_is_cut_and_noted(questions: QuestionSet) -> None:
    server = Server(_ok(_answers(0.5, "good", {}, 1.0)))
    result = ask("x" * (jev.MAX_STATE_CHARS + 10), questions, rounds=1, post=server)
    assert len(server.calls[0][2]["state"]) == jev.MAX_STATE_CHARS
    assert result.notes == [f"state cut to {jev.MAX_STATE_CHARS} characters"]
    assert result.errors == []


def test_logs_and_the_record_never_carry_the_key_state_or_reply(
    questions: QuestionSet, key: str, caplog: pytest.LogCaptureFixture
) -> None:
    secret_state = "CONFIDENTIAL-STATE-7731"
    secret_reply = "CONFIDENTIAL-REPLY-7731"
    replies = Server(
        (200, secret_reply), (429, secret_reply), (429, secret_reply), (400, secret_reply)
    )
    hung = Server(TimeoutError(secret_reply), TimeoutError(secret_reply))
    renamed = Server((200, json.dumps({"model": secret_reply + " x" * 20, "answers": {}})))
    with caplog.at_level(logging.DEBUG):
        results = [
            ask(secret_state, questions, post=server, sleep=lambda s: None)
            for server in (replies, hung, renamed)
        ]
    text = caplog.text + json.dumps([r.to_dict() for r in results])
    for secret in (key, secret_state, secret_reply):
        assert secret not in text
    assert results[0].errors == ["unreadable response", "HTTP 429", "HTTP 400"]
    assert results[1].errors == ["request error TimeoutError"]
    assert results[2].resolved_model == ""


HUGE = "1" + "0" * 400


@pytest.mark.parametrize(
    "body",
    [
        '{"answers": {"material": {"noul": ' + HUGE + "}}}",
        '{"answers": {"trend": {"score": ' + HUGE + "}}}",
        '{"answers": {"direction": {"choice": "good", "probabilities": {"good": ' + HUGE + "}}}}",
        '{"answers": {"material": {"noul": 0.5}}, "usage": {"input_tokens": '
        + HUGE
        + ', "cost": '
        + HUGE
        + "}}",
        "[" * 200_000,
        '{"answers": ' + "[" * 200_000 + "1" + "]" * 200_000 + "}",
        '{"answers": {"material": {"noul": 0.5}}, "model": ["x"]}',
        '{"answers": {"trend": {"score": 1, "probabilities": {"' + "9" * 5000 + '": 0.1}}}}',
    ],
)
def test_a_hostile_reply_is_never_a_crash(questions: QuestionSet, body: str) -> None:
    result = _ask(questions, Server(*[(200, body)] * questions.rounds()))
    assert isinstance(result, jev.Result)
    assert result.input_tokens == 0 and result.cost == 0.0
    assert result.resolved_model == ""


def test_an_unforeseen_failure_in_averaging_is_an_error(
    questions: QuestionSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any) -> None:
        raise KeyError("x")

    monkeypatch.setattr(jev, "average", boom)
    result = _ask(questions, Server(_ok(_answers(0.5, "good", {}, 1.0))), rounds=1)
    assert result.answers == {}
    assert result.errors == ["unreadable response (KeyError)"]
    assert result.error == "Jev gave no usable answer (unreadable response (KeyError))"


@pytest.mark.parametrize(
    ("level", "kept"), [(0, True), (2, True), (2.5, False), (3, False), (-1, False)]
)
def test_a_score_outside_the_levels_is_dropped(
    questions: QuestionSet, level: float, kept: bool
) -> None:
    out = average(questions, [{"trend": {"score": level}}])
    assert ("trend" in out) is kept


def test_the_result_is_saved_as_plain_json(questions: QuestionSet) -> None:
    server = Server(_ok(_answers(0.9, "good", {"good": 1.0}, 2.0)))
    saved = json.loads(json.dumps(_ask(questions, server, rounds=1).to_dict()))
    assert saved["version"] == "test-q1"
    assert saved["answers"]["direction"]["value"] == "good"
    assert saved["answers"]["material"]["probabilities"] == {}


def test_a_reversed_score_reply_is_mapped_back(questions: QuestionSet) -> None:
    (trend,) = [q for q in questions.questions if q.key == "trend"]
    raw = {
        "score": 0.0,
        "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1, "9": 1.0, "x": 1.0, "01": 1.0},
    }
    assert trend.unflip(raw, 1) == {"score": 2.0, "probabilities": {"2": 0.6, "1": 0.3, "0": 0.1}}
    assert trend.unflip(raw, 2) is raw  # even repeats were sent in file order
    assert trend.unflip({"score": "high"}, 1) == {"score": "high"}  # left for _one to drop


def test_varied_score_replies_average_in_file_order(questions: QuestionSet) -> None:
    server = Server(
        _ok(_answers(0.5, "good", {"good": 1.0}, 2.0, {"0": 0.0, "1": 0.2, "2": 0.8})),
        _ok(_answers(0.5, "good", {"good": 1.0}, 0.0, {"0": 0.9, "1": 0.1, "2": 0.0})),
        _ok(_answers(0.5, "good", {"good": 1.0}, 2.0, {"0": 0.0, "1": 0.0, "2": 1.0})),
        _ok(_answers(0.5, "good", {"good": 1.0}, 0.0, {"0": 1.0, "1": 0.0, "2": 0.0})),
    )
    result = _ask(questions, server)
    assert result.errors == [] and len(server.calls) == 4
    trend = result.answers["trend"]
    assert trend.value == pytest.approx(2.0) and trend.spread == 0.0 and trend.agreed == 4
    assert trend.probabilities == pytest.approx({"0": 0.0, "1": 0.075, "2": 0.925})
    assert trend.balanced and trend.first_lift == pytest.approx(0.0)


def test_a_redirect_is_never_followed_with_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import email.message
    import io
    import urllib.error
    import urllib.request

    seen: list[str] = []

    class Opener:
        def open(self, req: Any, timeout: float) -> Any:
            seen.append(req.full_url)
            hdrs = email.message.Message()
            raise urllib.error.HTTPError(req.full_url, 307, "moved", hdrs, io.BytesIO(b"{}"))

    monkeypatch.setattr(jev, "_OPENER", Opener())
    assert jev._post(jev.URL, {"Authorization": "Bearer k"}, b"{}") == (307, "{}")
    assert seen == [jev.URL]
    monkeypatch.undo()
    redirects = [
        h
        for h in vars(jev._OPENER)["handlers"]
        if isinstance(h, urllib.request.HTTPRedirectHandler)
    ]
    assert [type(h) for h in redirects] == [jev._NoRedirect]


def test_every_option_leads_exactly_once_in_the_committed_file() -> None:
    q = load_questions()
    assert q.rounds() == 6  # guidance's six labels set the request count
    for question in q.questions:
        sent = [q.to_request(r).get(question.key) for r in range(q.rounds())]
        asked = [s for s in sent if s is not None]
        assert len(asked) == question.rounds()
        if question.type == "choice":
            firsts = [next(iter(s["criteria"])) for s in asked]
            assert sorted(firsts) == sorted(question.labels())
            for position in range(len(question.labels())):
                at = [list(s["criteria"])[position] for s in asked]
                assert sorted(at) == sorted(question.labels())
        if question.type == "score":
            heads = [s["criteria"][0] for s in asked]
            assert heads.count(question.criteria[0]) == heads.count(question.criteria[-1])


def test_a_lean_toward_the_first_option_is_measured(questions: QuestionSet) -> None:
    q = load_questions()
    (direction,) = [x for x in q.questions if x.key == "direction"]
    labels = direction.labels()
    # a model that gives whichever label is shown first 0.55 and the rest 0.15
    replies = [
        {
            "direction": {
                "choice": direction.first(r),
                "probabilities": {
                    lab: 0.55 if lab == direction.first(r) else 0.15 for lab in labels
                },
            }
        }
        for r in range(direction.rounds())
    ]
    out = average(q, replies)
    answer = out["direction"]
    assert answer.balanced and answer.first_lift == pytest.approx(0.55 - 0.25)
    assert answer.probabilities == pytest.approx({lab: 0.25 for lab in labels})  # cancelled
    # a score (none is committed now; the test set has one) that always lands
    # on the first-listed level: 0 in file order, top reversed
    (trend,) = [x for x in questions.questions if x.key == "trend"]
    top = len(trend.criteria) - 1
    raw = [{"trend": {"score": 0.0}} for _ in range(trend.rounds())]
    flipped = [questions.unflip(r, i) for i, r in enumerate(raw)]
    lean = average(questions, flipped)["trend"]
    assert lean.balanced and lean.first_lift == pytest.approx(top / 2)
    assert lean.value == pytest.approx(top / 2)  # the middle, as a fair answer would be


def test_a_choice_too_large_to_balance_is_refused(tmp_path: Path) -> None:
    labels = ", ".join(f"l{i}: x" for i in range(jev.MAX_LABELS + 1))
    path = tmp_path / "q.yaml"
    path.write_text(
        f"version: v\nquestions: {{a: {{type: choice, instructions: x, criteria: {{{labels}}}}}}}\n"
    )
    with pytest.raises(ValueError, match="more than"):
        load_questions(path)


def test_a_zero_round_cap_asks_nothing(questions: QuestionSet) -> None:
    server = Server()
    result = _ask(questions, server, rounds=0)
    assert server.calls == [] and result.answers == {}
