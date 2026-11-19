"""Kairos through the System One adapter, on canned replies only.

Every reply here is made up in the test; nothing reaches Minara Cloud. The
opener itself refuses, so a test that would touch the network fails instead.
"""

from __future__ import annotations

import email.message
import io
import json
import logging
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from test_jev import QUESTIONS, _answers

from tenq import jev
from tenq.jev import KAIROS, QuestionSet, ask, load_questions, short_error

SECRET = "mk-live-KAIROS-SECRET-0123456789abcdef"


class NoNetwork(BaseException):
    """Not an Exception, so the adapter's catch-all cannot swallow it."""


class Refuse:
    def open(self, *a: Any, **kw: Any) -> Any:
        raise NoNetwork("tests must not reach Minara Cloud")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jev, "_OPENER", Refuse())
    monkeypatch.setenv("KAIROS_API_KEY", SECRET)
    monkeypatch.delenv("KAIROS_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.delenv("JEV_MODEL", raising=False)


Reply = tuple[int, str] | tuple[int, str, str] | Exception


class Server:
    """Canned replies in order; a third item is the Retry-After header."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes
    ) -> tuple[int, str] | tuple[int, str, str]:
        self.calls.append((url, headers, json.loads(body)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def questions(tmp_path: Path) -> QuestionSet:
    path = tmp_path / "q.yaml"
    path.write_text(QUESTIONS)
    return load_questions(path)


def _ok(answers: dict[str, Any]) -> tuple[int, str]:
    """A good Kairos reply: tokens in and (as usual) none out, and no cost."""
    return 200, json.dumps(
        {
            "model": "DMIND/Kairos-v4",
            "answers": answers,
            "usage": {"input_tokens": 600, "output_tokens": 0},
        }
    )


def _kairos(questions: QuestionSet, server: Server, **kw: Any) -> jev.Result:
    slept: list[float] = kw.pop("slept", [])
    return ask(
        "Facts for Acme (ACME).",
        questions,
        provider=KAIROS,
        post=server,
        sleep=slept.append,
        **kw,
    )


def test_a_good_reply_is_averaged_like_jevs(questions: QuestionSet) -> None:
    replies = [_ok(_answers(0.8, "good", {"good": 0.7, "bad": 0.2, "neutral": 0.1}, 2.0))] * 4
    result = _kairos(questions, Server(*replies), session_id="tenq-check-ACME-2026-11-19")
    assert result.errors == [] and result.error is None
    assert result.model == "DMIND/Kairos" and result.resolved_model == "DMIND/Kairos-v4"
    assert result.rounds == 4 and result.answered == [0, 1, 2, 3]
    assert result.answers["material"].value == pytest.approx(0.8)
    assert result.answers["direction"].value == "good"
    assert result.input_tokens == 2400 and result.output_tokens == 0
    assert result.cost == 0.0  # billed on tokens: no cost is made up


def test_kairos_is_sent_what_jev_is_sent(questions: QuestionSet) -> None:
    reply = _ok(_answers(0.5, "good", {}, 1.0))
    kairos, jev_server = Server(*[reply] * 4), Server(*[reply] * 4)
    _kairos(questions, kairos, session_id="tenq-check-ACME-2026-11-19")
    ask(
        "Facts for Acme (ACME).",
        questions,
        post=jev_server,
        session_id="tenq-check-ACME-2026-11-19",
    )
    assert [c[0] for c in kairos.calls] == [jev.KAIROS_URL] * 4
    assert [c[0] for c in jev_server.calls] == [jev.URL] * 4
    for (_, kh, kb), (_, jh, jb) in zip(kairos.calls, jev_server.calls, strict=True):
        assert kh == {"Authorization": f"Bearer {SECRET}", "Content-Type": "application/json"}
        assert jh["Authorization"] == "Bearer sk-or-test-key"
        assert kb["model"] == "DMIND/Kairos" and jb["model"] == "jev-latest"
        assert {k: v for k, v in kb.items() if k != "model"} == {
            k: v for k, v in jb.items() if k != "model"
        }
    assert kairos.calls[0][2]["session_id"] == "tenq-check-ACME-2026-11-19"


def test_the_kairos_model_is_a_setting(
    questions: QuestionSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROS_MODEL", "DMIND/Kairos-27b-v4")
    server = Server(_ok(_answers(0.5, "good", {}, 1.0)))
    result = _kairos(questions, server, rounds=1)
    assert server.calls[0][2]["model"] == "DMIND/Kairos-27b-v4" == result.model


def test_a_missing_key_skips_kairos_with_a_note(
    questions: QuestionSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAIROS_API_KEY")
    server = Server()
    result = _kairos(questions, server)
    assert server.calls == [] and result.answers == {}
    assert result.error == "Kairos skipped: KAIROS_API_KEY not set"


RELAY = {
    "detail": {
        "error_type": "request_too_large",
        "message": "state is 7012 tokens; the limit is 4096",
        "code": "context_length",
        "request_id": "req_relay_1",
        "upstream_request_id": "up_9",
    }
}
EARLIER = {"error": {"message": "Invalid channel for this key"}}
UPSTREAM = {"error": {"code": "cluster_busy", "message": "Try again soon", "request_id": "r-77"}}
VALIDATION = {
    "detail": [
        {
            "type": "extra_forbidden",
            "loc": ["body", "session_id"],
            "msg": "Extra inputs are not permitted",
            "input": "tenq-check-ACME-2026-11-19",
        }
    ]
}


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (
            413,
            RELAY,
            "HTTP 413 [request_too_large context_length] state is 7012 tokens; the limit is "
            "4096 (request_id req_relay_1, upstream_request_id up_9)",
        ),
        (403, EARLIER, "HTTP 403 Invalid channel for this key"),
        (503, UPSTREAM, "HTTP 503 [cluster_busy] Try again soon (request_id r-77)"),
        (
            422,
            VALIDATION,
            "HTTP 422 [extra_forbidden] body.session_id: Extra inputs are not permitted",
        ),
    ],
)
def test_each_error_shape_keeps_its_type_message_and_request_ids(
    questions: QuestionSet, status: int, body: dict[str, Any], error: str
) -> None:
    server = Server(*[(status, json.dumps(body))] * 2)
    result = _kairos(questions, server)
    assert result.errors == [error]
    assert result.answers == {}  # no default answer stands in
    assert result.error == f"Kairos gave no usable answer ({error})"


@pytest.mark.parametrize("status", [413, 422])
def test_a_request_kairos_refuses_ends_the_rounds_without_retry(
    questions: QuestionSet, status: int
) -> None:
    slept: list[float] = []
    server = Server((status, json.dumps(RELAY)))
    result = _kairos(questions, server, slept=slept)
    assert len(server.calls) == 1 and slept == []
    assert len(result.errors) == 1 and result.errors[0].startswith(f"HTTP {status} [")
    assert not result.down  # a too-large pack says nothing about the next stock's


def test_a_rate_limit_is_waited_out_at_the_delay_it_names(questions: QuestionSet) -> None:
    slept: list[float] = []
    server = Server(
        (429, json.dumps(UPSTREAM), "7"),  # the Retry-After header
        (429, json.dumps({"error": {"message": "slow down", "retry_after": 3}})),
        (429, "{}", "900"),  # too long: capped
        _ok(_answers(0.5, "good", {}, 1.0)),
    )
    result = _kairos(questions, server, rounds=1, slept=slept)
    assert slept == [7.0, 3.0, jev.KAIROS_MAX_WAIT_S]
    assert result.rounds == 1 and result.errors == [] and result.error is None


def test_a_rate_limit_naming_no_delay_backs_off_a_bounded_number_of_times(
    questions: QuestionSet,
) -> None:
    slept: list[float] = []
    server = Server(*[(429, "{}", "Wed, 21 Oct 2026 07:28:00 GMT")] * 5)
    result = _kairos(questions, server, rounds=1, slept=slept)
    assert slept == [5.0, 10.0, 20.0]  # a date is not read; the backoff doubles
    assert len(server.calls) == 1 + jev.KAIROS_RATE_RETRIES
    assert result.errors == ["HTTP 429"] and result.answers == {}


def test_one_rate_limit_then_success_costs_one_wait(questions: QuestionSet) -> None:
    slept: list[float] = []
    server = Server((429, "{}"), *[_ok(_answers(0.5, "good", {}, 1.0))] * 4)
    result = _kairos(questions, server, slept=slept)
    assert slept == [5.0] and result.rounds == 4 and result.errors == []


@pytest.mark.parametrize(
    ("first", "retried"),
    [
        ((502, "bad gateway"), True),
        (ConnectionResetError("reset"), True),
        (TimeoutError("slow"), False),  # already waited the whole timeout
    ],
)
def test_an_upstream_failure_is_tried_once_more_but_a_timeout_is_not(
    questions: QuestionSet, first: tuple[int, str] | Exception, retried: bool
) -> None:
    slept: list[float] = []
    server = Server(first, first)
    result = _kairos(questions, server, slept=slept)
    assert len(server.calls) == (2 if retried else 1)
    assert slept == ([2.0] if retried else [])
    assert result.down and result.answers == {}


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((200, "not json"), "unreadable response"),
        ((200, "[1, 2]"), "response without answers"),
        ((200, '{"answers": "yes"}'), "response without answers"),
        ((200, json.dumps(UPSTREAM)), "response without answers [cluster_busy] Try again soon"),
        ((200, "[" * 200_000), "unreadable response"),
    ],
)
def test_a_malformed_reply_is_a_recorded_error(
    questions: QuestionSet, reply: tuple[int, str], error: str
) -> None:
    result = _kairos(questions, Server(reply), rounds=1)
    assert result.answers == {}
    assert result.errors[0].startswith(error)
    assert result.error is not None and result.error.startswith("Kairos gave no usable answer")


def test_a_reply_with_unusable_answers_is_no_answer(questions: QuestionSet) -> None:
    bad = {"material": {"noul": 1.5}, "direction": {"choice": "sideways"}, "trend": "high"}
    result = _kairos(questions, Server(_ok(bad)), rounds=1)
    assert result.answers == {} and result.error == "Kairos gave no usable answer"


def test_the_key_never_reaches_the_errors_notes_logs_or_record(
    questions: QuestionSet, caplog: pytest.LogCaptureFixture
) -> None:
    echoes = [
        {"detail": {"error_type": SECRET, "message": f"bad key {SECRET}", "request_id": SECRET}},
        {"error": {"message": f"Authorization: Bearer {SECRET} refused"}},
        {"error": {"code": SECRET, "message": f"key {SECRET[:30]}... is not valid"}},
        {"detail": [{"type": "x", "loc": ["header", SECRET], "msg": SECRET, "input": SECRET}]},
    ]
    with caplog.at_level(logging.DEBUG):
        results = [
            _kairos(questions, Server(*[(status, json.dumps(body))] * 2))
            for status in (401, 422)
            for body in echoes
        ]
        results.append(_kairos(questions, Server(TimeoutError(SECRET))))
        listed = jev.models(post=lambda url, headers: (403, json.dumps(echoes[0])))
    text = caplog.text + json.dumps([r.to_dict() for r in results]) + str(listed)
    assert SECRET not in text and SECRET[:30] not in text
    assert all(r.errors and r.error for r in results)
    assert results[0].errors == ["HTTP 401 bad key [key]"]
    assert results[1].errors == ["HTTP 401 Authorization: [redacted] refused"]


def test_an_error_message_is_cleaned_and_cut(questions: QuestionSet) -> None:
    long = {"error": {"message": "line one\nline\ttwo\x00 " + "word " * 100}}
    (error,) = _kairos(questions, Server((400, json.dumps(long)))).errors
    assert error.startswith("HTTP 400 line one line two word")
    assert len(error) <= len("HTTP 400 ") + jev.MAX_MESSAGE
    assert "\n" not in error and "\x00" not in error


def test_kairos_posts_through_the_no_redirect_opener_with_its_own_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, float]] = []

    class Opener:
        def open(self, req: Any, timeout: float) -> Any:
            seen.append((req.full_url, timeout))
            hdrs = email.message.Message()
            hdrs["Retry-After"] = "12"
            raise urllib.error.HTTPError(req.full_url, 429, "busy", hdrs, io.BytesIO(b"{}"))

    monkeypatch.setattr(jev, "_OPENER", Opener())
    assert KAIROS.post(jev.KAIROS_URL, {"Authorization": "Bearer k"}, b"{}") == (429, "{}", "12")
    assert seen == [(jev.KAIROS_URL, jev.KAIROS_TIMEOUT_S)]


def test_jev_still_gets_its_session_id_its_timeout_and_its_bare_errors(
    questions: QuestionSet,
) -> None:
    server = Server((413, json.dumps(RELAY)))
    result = ask("s", questions, post=server, session_id="tenq-daily-ACME-2026-11-19")
    assert server.calls[0][2]["session_id"] == "tenq-daily-ACME-2026-11-19"
    assert result.errors == ["HTTP 413"]  # Jev's record keeps only the status
    assert jev.JEV.post is jev._post and jev.TIMEOUT_S == 30


@pytest.mark.parametrize(
    ("reply", "listed"),
    [
        ((200, '{"data": [{"id": "DMIND/Kairos"}, {"id": "kairos-27b-v4"}, {"id": 7}]}'), None),
        ((200, '["DMIND/Kairos", "bad id with spaces"]'), ["DMIND/Kairos"]),
        ((200, "{}"), "unreadable model list"),
        ((401, json.dumps(EARLIER)), "HTTP 401 Invalid channel for this key"),
    ],
)
def test_the_model_list_is_read_or_a_short_error(
    reply: tuple[int, str], listed: list[str] | str | None
) -> None:
    got = jev.models(post=lambda url, headers: reply)
    assert got == (listed if listed is not None else ["DMIND/Kairos", "kairos-27b-v4"])


def test_the_model_list_needs_the_key_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, headers: dict[str, str]) -> tuple[int, str]:
        assert url == jev.KAIROS_MODELS_URL and headers == {"Authorization": f"Bearer {SECRET}"}
        raise TimeoutError(SECRET)

    assert jev.models(post=boom) == "request error TimeoutError"
    monkeypatch.delenv("KAIROS_API_KEY")
    assert jev.models(post=boom) == "KAIROS_API_KEY not set"


def test_a_rate_limit_that_outlasts_its_waits_ends_the_stocks_rounds(
    questions: QuestionSet,
) -> None:
    slept: list[float] = []
    server = Server(*[(429, "{}", "60")] * 30)
    result = _kairos(questions, server, slept=slept)
    assert len(server.calls) == 1 + jev.KAIROS_RATE_RETRIES  # one round's worth, not six
    assert slept == [60.0] * jev.KAIROS_RATE_RETRIES
    assert result.errors == ["HTTP 429"] and not result.down  # the next stock may get through


def test_a_rate_limit_delay_at_the_top_of_the_reply_is_honored(questions: QuestionSet) -> None:
    slept: list[float] = []
    server = Server(
        (429, json.dumps({"detail": "rate limited", "retry_after": 30})),
        _ok(_answers(0.5, "good", {}, 1.0)),
    )
    _kairos(questions, server, rounds=1, slept=slept)
    assert slept == [30.0]


def test_an_upstream_failure_after_a_rate_limit_is_not_tried_again(
    questions: QuestionSet,
) -> None:
    slept: list[float] = []
    server = Server((429, "{}"), (503, "{}"))
    result = _kairos(questions, server, slept=slept)
    assert slept == [5.0] and len(server.calls) == 2
    assert result.down and result.errors == ["HTTP 503"]


def test_a_connect_timeout_is_a_timeout_and_not_tried_again(
    questions: QuestionSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[float] = []

    class Stuck:
        def open(self, req: Any, timeout: float) -> Any:
            opened.append(timeout)
            raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(jev, "_OPENER", Stuck())
    slept: list[float] = []
    result = ask("s", questions, provider=KAIROS, sleep=slept.append)
    assert opened == [jev.KAIROS_TIMEOUT_S] and slept == []
    assert result.errors == ["request error TimeoutError"] and result.down


def test_a_model_name_holding_the_key_is_not_kept(questions: QuestionSet) -> None:
    status, text = _ok(_answers(0.5, "good", {}, 1.0))
    reply = json.loads(text)
    reply["model"] = f"x/{SECRET}"
    result = _kairos(questions, Server((status, json.dumps(reply))), rounds=1)
    assert result.resolved_model == "" and result.answers
    listed = jev.models(post=lambda url, headers: (200, json.dumps([SECRET, "DMIND/Kairos"])))
    assert listed == ["DMIND/Kairos"]


def test_no_round_starts_once_the_callers_time_is_used_up(questions: QuestionSet) -> None:
    server = Server()
    result = _kairos(questions, server, time_left=lambda: 0.0)
    assert server.calls == [] and result.answers == {} and not result.down
    assert result.error == "Kairos gave no usable answer (out of time after 0 of 4 rounds)"
    assert result.notes == ["out of time after 0 of 4 rounds"]


def test_rounds_answered_before_the_time_ran_out_are_kept(questions: QuestionSet) -> None:
    left = iter([10.0, 10.0, 0.0])
    server = Server(*[_ok(_answers(0.8, "good", {"good": 1.0}, 2.0))] * 2)
    result = _kairos(questions, server, time_left=lambda: next(left))
    assert len(server.calls) == 2 and result.answered == [0, 1]
    assert result.answers["material"].value == pytest.approx(0.8) and result.error is None
    assert result.notes == ["out of time after 2 of 4 rounds"]


def test_a_retry_wait_longer_than_the_time_left_is_not_waited(questions: QuestionSet) -> None:
    slept: list[float] = []
    server = Server((429, "{}", "7"))
    result = _kairos(questions, server, slept=slept, time_left=lambda: 5.0)
    assert slept == [] and len(server.calls) == 1
    assert result.errors == ["HTTP 429"] and result.answers == {}
    assert "out of time after 0 of 4 rounds" in result.notes


BUSY = json.dumps(
    {
        "error": {
            "type": "channel_unavailable",
            "code": "evil.example",
            "message": 'Service busy, see https://bit.ly/x9 or <a href="https://evil.example/p">status</a>',
            "request_id": "req-123",
        }
    }
)


def test_the_short_error_keeps_none_of_the_replys_own_words(questions: QuestionSet) -> None:
    result = _kairos(questions, Server((503, BUSY), (503, BUSY)), slept=[])
    assert result.down and result.error is not None
    assert "https://bit.ly/x9" in result.error and "req-123" in result.error  # kept in full
    assert (
        short_error(result.error) == "Kairos gave no usable answer (HTTP 503 [channel_unavailable])"
    )


@pytest.mark.parametrize(
    ("error", "short"),
    [
        ("HTTP 503", "HTTP 503"),
        ("Kairos gave no usable answer (HTTP 429)", "Kairos gave no usable answer (HTTP 429)"),
        (
            "Kairos gave no usable answer (HTTP 500 Busy [x] see http://a.b (request_id r))",
            "Kairos gave no usable answer (HTTP 500)",
        ),
        (
            "Kairos gave no usable answer (response without answers [bad_shape] see a.b)",
            "Kairos gave no usable answer (response without answers [bad_shape])",
        ),
        (
            "Kairos gave no usable answer (HTTP 400 [a.b c:d e/f] x)",
            "Kairos gave no usable answer (HTTP 400)",
        ),
        (
            "Kairos gave no usable answer (request error TimeoutError)",
            "Kairos gave no usable answer (request error TimeoutError)",
        ),
        (
            "not asked: Kairos was down earlier in this run",
            "not asked: Kairos was down earlier in this run",
        ),
    ],
)
def test_the_short_error_of_each_shape(error: str, short: str) -> None:
    assert short_error(error) == short
