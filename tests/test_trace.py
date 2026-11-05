from __future__ import annotations

import pytest

from tenq.trace import main, wait_found


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
