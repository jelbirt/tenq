from __future__ import annotations

import logging
import socket
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tenq.trace import QUIET_LOGGERS, connect, main, wait_found


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def test_wait_found_looks_until_the_deadline() -> None:
    clock = Clock()
    calls: list[float] = []

    def never(trace_id: str) -> bool:
        calls.append(clock.now)
        return False

    assert wait_found(never, "t", wait=20, poll=5, sleep=clock.sleep, clock=clock) == (False, "")
    assert calls == [0, 5, 10, 15, 20]


def test_wait_found_stops_at_the_first_hit_and_keeps_the_last_error() -> None:
    clock = Clock()
    answers: list[object] = [TimeoutError(), False, True]

    def flaky(trace_id: str) -> bool:
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return bool(a)

    assert wait_found(flaky, "t", wait=60, poll=5, sleep=clock.sleep, clock=clock) == (True, "")
    assert clock.now == 10

    def down(trace_id: str) -> bool:
        raise TimeoutError

    assert wait_found(down, "t", wait=5, poll=5, sleep=clock.sleep, clock=clock) == (
        False,
        "TimeoutError",
    )


def test_check_without_langfuse_env_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(k, raising=False)
    assert main(["--check"]) == 1
    assert "trace skipped" in capsys.readouterr().out


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*a: Any, **k: Any) -> None:
        raise OSError("tests never touch the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)


def _key() -> str:
    return f"pk-{uuid.uuid4()}"  # the SDK keeps one client per key


def test_the_exporters_failure_lines_never_reach_the_log(
    offline: None, caplog: pytest.LogCaptureFixture
) -> None:
    connect(_key(), "sk", "https://secret-host.example", exporter=InMemorySpanExporter())
    for name in QUIET_LOGGERS:
        logging.getLogger(f"{name}.sub").warning("error at secret-host.example")
    assert "secret-host" not in caplog.text


def test_a_lookup_is_one_bounded_request(offline: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    pages: list[int] = []

    def endless(self: Langfuse, **kwargs: Any) -> Iterator[Any]:
        calls.append(kwargs)
        while True:  # empty pages that each point to another, as a cold scan can
            pages.append(1)
            yield SimpleNamespace(data=[], meta=SimpleNamespace(cursor="more"))

    monkeypatch.setattr(Langfuse, "iter_observation_pages", endless)
    tracing = connect(_key(), "sk", "https://lf.example", exporter=InMemorySpanExporter())
    assert tracing.found("abc") is False
    assert len(pages) == 1
    assert calls[0]["trace_id"] == "abc"
    assert calls[0]["request_options"] == {"timeout_in_seconds": 15}
    assert calls[0]["from_start_time"] > datetime.now(UTC) - timedelta(days=2)


def test_one_public_key_is_one_connection(offline: None) -> None:
    key = _key()
    first = connect(key, "sk", "https://lf.example", exporter=InMemorySpanExporter())
    assert connect(key, "sk", "https://lf.example") is first
    with pytest.raises(ValueError, match="another host"):
        connect(key, "sk", "https://other.example")
