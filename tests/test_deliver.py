from __future__ import annotations

from typing import Any, ClassVar

import pytest

from tenq.deliver import Email, send

EMAIL = Email("tenq: CRWD", "text", "<p>html</p>")


class FakeSMTP:
    sent: ClassVar[list[Any]] = []
    fail: ClassVar[Exception | None] = None

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.addr = (host, port)

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def starttls(self) -> None:
        pass

    def login(self, user: str, password: str) -> None:
        if FakeSMTP.fail:
            raise FakeSMTP.fail

    def send_message(self, msg: Any) -> None:
        FakeSMTP.sent.append((self.addr, msg))


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch) -> type[FakeSMTP]:
    FakeSMTP.sent, FakeSMTP.fail = [], None
    monkeypatch.setattr("tenq.deliver.smtplib.SMTP", FakeSMTP)
    monkeypatch.setattr("tenq.deliver.smtplib.SMTP_SSL", FakeSMTP)
    for k, v in {
        "SMTP_USER": "me@example.com",
        "SMTP_PASS": "hunter2",
        "RECIPIENT_EMAILS": "a@example.com, b@example.com",
    }.items():
        monkeypatch.setenv(k, v)
    for k in ("SMTP_HOST", "SMTP_PORT"):
        monkeypatch.delenv(k, raising=False)
    return FakeSMTP


def test_sends_to_every_recipient_and_reports_only_a_count(smtp: type[FakeSMTP]) -> None:
    ok, status = send(EMAIL)
    assert ok and status == "email sent to 2 recipient(s)"
    ((addr, msg),) = smtp.sent
    assert addr == ("smtp.gmail.com", 587)
    assert msg["To"] == "a@example.com, b@example.com" and msg["Subject"] == "tenq: CRWD"


@pytest.mark.parametrize("unset", ["SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAILS"])
def test_missing_settings_skip_without_sending(
    smtp: type[FakeSMTP], monkeypatch: pytest.MonkeyPatch, unset: str
) -> None:
    monkeypatch.setenv(unset, "")  # an unset Actions secret arrives as an empty string
    ok, status = send(EMAIL)
    assert not ok and status.startswith("email skipped") and smtp.sent == []


def test_a_failure_reports_no_credentials(smtp: type[FakeSMTP]) -> None:
    smtp.fail = OSError("535 auth failed for me@example.com hunter2")
    ok, status = send(EMAIL)
    assert (ok, status) == (False, "email delivery failed: OSError")


def test_a_bad_port_setting_fails_open(
    smtp: type[FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMTP_PORT", "five-eight-seven")
    assert send(EMAIL) == (False, "email delivery failed: ValueError")
