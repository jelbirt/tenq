from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_rules import QUESTIONS, RULES, _good

from tenq import daily, jev, opus
from tenq.config import load_watchlist
from tenq.daily import Context, main, record_path
from tenq.daily_email import DISCLAIMER, agreement, render, sources, subject_for
from tenq.deliver import Email
from tenq.jev import Result
from tenq.llm import INFRA, AgentRun
from tenq.pack import Headline, Number, Pack
from tenq.prices import BENCHMARK, Bar, Prices
from tenq.trace import Tracing, emit_daily, send_daily_traces

DAY = date(2026, 11, 4)
NOW = datetime(2026, 11, 4, 22, 30, tzinfo=UTC)
WATCH = load_watchlist()


def _news(ticker: str) -> str:
    return f"news:{ticker}:0123456789"


def _pack(ticker: str, now: datetime) -> Pack:
    return Pack(
        ticker,
        WATCH.names.get(ticker, ticker),
        DAY,
        now,
        numbers=[
            Number(f"px:{ticker}:close", f"{ticker} close", 241.0, "usd"),
            Number(f"fin:{ticker}:revenue:growth_change", "revenue growth less", 0.03, "points"),
        ],
        news=[
            Headline(
                _news(ticker),
                "<b>Outlook</b> raised",
                "reuters.com",
                "2026-11-04",
                "",
                link="https://www.reuters.com/a?b=1&c=2",
            )
        ],
    )


def _prices() -> Prices:
    return Prices({BENCHMARK: (Bar(DAY, 600.5, None),)}, ())


OWN = json.dumps(
    {
        "call": "Hold",
        "confidence": 0.5,
        "lean": 0.1,
        "reasons": [
            {"text": "The outlook was raised <script>.", "cites": ["NEWS"]},
            {"text": "Shares jumped 12% this week.", "cites": ["NEWS"]},  # typed: dropped
        ],
    }
)
WHY = json.dumps({"reasons": [{"text": "The rules read the news as good.", "cites": ["NEWS"]}]})


class Claude:
    """Opus's own call and its why, for whichever stock the message is about."""

    def __init__(self, limit_after: int | None = None) -> None:
        self.calls: list[str] = []
        self.limit_after = limit_after

    def __call__(self, message: str, **kw: Any) -> AgentRun:
        self.calls.append(message)
        ticker = next(t for t in WATCH.tickers if _news(t) in message)
        why = "How the rules reached it" in message
        text = (WHY if why else OWN).replace("NEWS", _news(ticker))
        run = AgentRun(kw["model"], text=text)
        if self.limit_after is not None and len(self.calls) > self.limit_after:
            run = AgentRun(kw["model"])
            run.fail(INFRA, "usage limit")
            run.usage_limit = True
        return run


class Jev:
    def __init__(self, down_after: int | None = None) -> None:
        self.asked: list[str] = []
        self.down_after = down_after

    def __call__(self, state: str, questions: Any, **kw: Any) -> Result:
        self.asked.append(kw["session_id"])
        if self.down_after is not None and len(self.asked) > self.down_after:
            return Result(QUESTIONS.version, "jev-latest", down=True, error="HTTP 503")
        good = _good()
        good.cost = 0.001
        return good


ALWAYS = set(range(1, 100))  # Kairos down at every ask


class Kairos:
    """Kairos's answers, the same as Jev's fake gives, with tokens; down for the
    asks numbered in `down` (1-based, counting every ask in the run), and up
    but with no usable answer for those in `empty`."""

    def __init__(
        self,
        down: set[int] | None = None,
        clock: Clock | None = None,
        takes: float = 30.0,
        empty: set[int] | None = None,
    ) -> None:
        self.asked: list[str] = []
        self.seen: list[tuple[str, Any]] = []  # the facts and questions it was sent
        self.left: list[float] = []  # the time left it was told at each ask
        self.down = down or set()
        self.empty = empty or set()
        self.clock = clock
        self.takes = takes

    def __call__(self, state: str, questions: Any, **kw: Any) -> Result:
        self.asked.append(kw["session_id"])
        self.seen.append((state, questions))
        self.left.append(kw["time_left"]())
        if self.clock is not None:
            self.clock.now += self.takes
        if len(self.asked) in self.down:
            return Result(QUESTIONS.version, "DMIND/Kairos", down=True, error="HTTP 503")
        if len(self.asked) in self.empty:
            error = "Kairos gave no usable answer (HTTP 429)"
            return Result(QUESTIONS.version, "DMIND/Kairos", error=error)
        good = _good()
        good.model, good.resolved_model = "DMIND/Kairos", "DMIND/Kairos-0.9"
        good.input_tokens, good.output_tokens = 1000, 50
        return good


class Clock:
    """ctx.clock and ctx.sleep together: a sleep moves the clock."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Outbox:
    def __init__(self, ok: bool = True, seen: Path | None = None) -> None:
        self.sent: list[Email] = []
        self.ok = ok
        self.seen = seen
        self.on_disk: list[bool] = []

    def __call__(self, email: Email) -> tuple[bool, str]:
        if self.seen is not None:
            self.on_disk.append(self.seen.exists())
        self.sent.append(email)
        return (True, "sent to 1 recipient") if self.ok else (False, "email failed: SMTPError")


def _no_trace(record: dict[str, Any], **kw: Any) -> tuple[bool, str, dict[str, str]]:
    return False, "trace skipped: test", {}


@pytest.fixture(autouse=True)
def kairos_key(no_kairos_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The daily run asks Kairos only with a key: a test that wants none deletes it."""
    monkeypatch.setenv("KAIROS_API_KEY", "test-kairos-key")


def _ctx(
    jev: Jev | None = None, claude: Claude | None = None, kairos: Kairos | None = None
) -> Context:
    clock = kairos.clock if kairos is not None and kairos.clock is not None else Clock()
    return Context(
        questions=QUESTIONS,
        rules=RULES,
        call_prompt=opus.load_call_prompt(),
        why_prompt=opus.load_why_prompt(),
        now=NOW,
        gather=_pack,
        ask=jev or Jev(),
        claude=claude if claude is not None else Claude(),
        kairos=kairos if kairos is not None else Kairos(),
        clock=clock,
        sleep=clock.sleep,
    )


def _run(tmp_path: Path, *argv: str, **kw: Any) -> int:
    kw.setdefault("ctx", _ctx())
    kw.setdefault("mailer", Outbox())
    kw.setdefault("tracer", _no_trace)
    return main(["--runs", str(tmp_path), *argv], prices=_prices, **kw)


def _record(tmp_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(record_path(tmp_path, DAY).read_text())
    return out


def test_a_day_makes_both_calls_on_every_stock_and_checks_every_reason(tmp_path: Path) -> None:
    outbox = Outbox(seen=record_path(tmp_path, DAY))
    assert _run(tmp_path, mailer=outbox) == 0
    rec = _record(tmp_path)
    assert [s["ticker"] for s in rec["stocks"]] == list(WATCH.tickers)
    assert rec["day"] == "2026-11-04" and rec["benchmark"]["close"] == 600.5
    crwd = rec["stocks"][0]
    assert crwd["close"] == 241.0 and crwd["day"] == "2026-11-04"
    assert crwd["call"]["call"] == "Buy" and crwd["call"]["explain"]
    assert crwd["opus"]["call"] == "Hold" and crwd["opus"]["confidence"] == 0.5
    assert [r["text"] for r in crwd["opus"]["kept"]] == ["The outlook was raised <script>."]
    assert "figure" in crwd["opus"]["dropped"][0]["why"]
    assert crwd["why"]["kept"][0]["text"] == "The rules read the news as good."
    assert crwd["pack"]["ticker"] == "CRWD"  # the facts behind the calls, for later
    assert rec["versions"]["rules"] == RULES.version
    assert rec["jev_cost"] == pytest.approx(0.001 * len(WATCH.tickers))
    assert rec["email"]["sent"] is True and len(outbox.sent) == 1
    assert outbox.on_disk == [True]  # written before the email went out


def test_a_second_run_on_an_emailed_day_does_nothing(tmp_path: Path) -> None:
    assert _run(tmp_path) == 0
    jev, claude, outbox = Jev(), Claude(), Outbox()
    assert _run(tmp_path, ctx=_ctx(jev, claude), mailer=outbox) == 0
    assert jev.asked == [] and claude.calls == [] and outbox.sent == []


def test_a_failed_email_is_resent_from_the_record_without_asking_again(tmp_path: Path) -> None:
    assert _run(tmp_path, mailer=Outbox(ok=False)) == 1
    assert _record(tmp_path)["email"]["sent"] is False
    jev, claude, outbox = Jev(), Claude(), Outbox()
    assert _run(tmp_path, ctx=_ctx(jev, claude), mailer=outbox) == 0
    assert jev.asked == [] and claude.calls == []
    assert len(outbox.sent) == 1 and _record(tmp_path)["email"]["sent"] is True


def test_a_check_run_never_stands_in_for_the_day(tmp_path: Path) -> None:
    assert _run(tmp_path, "--ticker", "crwd") == 0
    checks = list((tmp_path / "daily").glob("2026-11-04.check-*.json"))
    assert len(checks) == 1 and not record_path(tmp_path, DAY).exists()
    assert [s["ticker"] for s in json.loads(checks[0].read_text())["stocks"]] == ["CRWD"]
    assert _run(tmp_path) == 0 and record_path(tmp_path, DAY).exists()


def test_jev_down_still_sends_opus_and_stops_asking_jev(tmp_path: Path) -> None:
    jev = Jev(down_after=1)
    outbox = Outbox()
    assert _run(tmp_path, ctx=_ctx(jev), mailer=outbox) == 0
    rec = _record(tmp_path)
    assert len(jev.asked) == 2  # the first call, then the one that found it down
    assert "call" in rec["stocks"][0] and "call" not in rec["stocks"][1]
    assert "not asked" in rec["stocks"][2]["jev"]["error"]
    assert all(s["opus"]["call"] == "Hold" for s in rec["stocks"])
    assert "Jev made no call" in outbox.sent[0].text


def test_claude_at_its_limit_still_sends_jev_without_the_why(tmp_path: Path) -> None:
    claude = Claude(limit_after=2)  # CRWD's call and why, then the limit
    outbox = Outbox()
    assert _run(tmp_path, ctx=_ctx(claude=claude), mailer=outbox) == 0
    rec = _record(tmp_path)
    assert len(claude.calls) == 3
    assert rec["stocks"][0]["why"]["kept"]
    assert rec["stocks"][1]["opus"]["call"] is None
    assert "usage limit" in rec["stocks"][2]["opus"]["error"]
    assert all(s["call"]["call"] == "Buy" for s in rec["stocks"])
    assert "Opus made no call" in outbox.sent[0].text


def _checks(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in (tmp_path / "daily").glob("*.check-*.json")]


def test_no_opus_runs_jev_alone_as_a_check(tmp_path: Path) -> None:
    claude = Claude()
    assert _run(tmp_path, "--no-opus", ctx=_ctx(claude=claude)) == 0
    [check] = _checks(tmp_path)
    assert claude.calls == [] and check["opus_on"] is False
    assert not record_path(tmp_path, DAY).exists()  # the day's own run is still to come


def test_a_stock_with_no_facts_is_noted_and_the_rest_go_on(tmp_path: Path) -> None:
    ctx = _ctx()

    def gather(ticker: str, now: datetime) -> Pack | None:
        if ticker == "DDOG":
            raise RuntimeError("secret-bearing message")
        return None if ticker == "SNOW" else _pack(ticker, now)

    ctx.gather = gather
    assert _run(tmp_path, ctx=ctx) == 0
    by = {s["ticker"]: s for s in _record(tmp_path)["stocks"]}
    assert by["DDOG"]["error"] == "no facts pack (RuntimeError)"
    assert "secret" not in json.dumps(by["DDOG"])
    assert by["SNOW"]["error"] == "no prices" and "call" in by["CRWD"]


def test_exit_codes(tmp_path: Path) -> None:
    assert _run(tmp_path, "--ticker", "ZZZZ") == 2
    assert _run(tmp_path, mailer=Outbox(ok=False)) == 1
    assert main(["--runs", str(tmp_path)], ctx=_ctx(), prices=lambda: Prices({}, ())) == 1


def test_no_call_at_all_fails_the_run(tmp_path: Path) -> None:
    ctx = _ctx(Jev(down_after=0), kairos=Kairos(down=ALWAYS))
    ctx.claude = None
    assert _run(tmp_path, ctx=ctx) == 1


def test_no_email_is_a_check_that_sends_nothing_and_never_takes_the_day(tmp_path: Path) -> None:
    outbox = Outbox()
    assert _run(tmp_path, "--no-email", mailer=outbox) == 0
    [check] = _checks(tmp_path)
    assert outbox.sent == [] and check["email"]["status"].startswith("email skipped")
    assert check["check_id"] and not record_path(tmp_path, DAY).exists()
    jev = Jev()
    assert _run(tmp_path, ctx=_ctx(jev), mailer=outbox) == 0
    assert len(jev.asked) == len(WATCH.tickers) and len(outbox.sent) == 1


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("{", "unreadable"),
        (json.dumps({"day": "2026-11-04", "complete": False, "stocks": []}), "unfinished"),
        (
            json.dumps(
                {
                    "day": "2026-11-04",
                    "complete": True,
                    "email": {"sent": None, "status": "sending"},
                }
            ),
            "may have gone out",
        ),
    ],
)
def test_a_record_that_needs_a_person_is_never_run_or_sent_again(
    tmp_path: Path, text: str, said: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = record_path(tmp_path, DAY)
    path.parent.mkdir(parents=True)
    path.write_text(text)
    jev, claude, outbox = Jev(), Claude(), Outbox()
    assert _run(tmp_path, ctx=_ctx(jev, claude), mailer=outbox) == 1
    assert jev.asked == [] and claude.calls == [] and outbox.sent == []
    assert said in capsys.readouterr().out and path.read_text() == text


def test_the_record_is_saved_after_every_stock_and_left_whole(tmp_path: Path) -> None:
    seen: list[int] = []
    ctx = _ctx()
    inner = ctx.gather

    def gather(ticker: str, now: datetime) -> Pack | None:
        path = record_path(tmp_path, DAY)
        if path.exists():
            rec = json.loads(path.read_text())  # always whole JSON
            assert rec["complete"] is False
            seen.append(len(rec["stocks"]))
        return inner(ticker, now)

    ctx.gather = gather
    assert _run(tmp_path, ctx=ctx) == 0
    assert seen == list(range(1, len(WATCH.tickers)))
    assert _record(tmp_path)["complete"] is True
    assert not list((tmp_path / "daily").glob(".*.tmp"))


def test_a_resend_also_sends_a_trace_that_never_went_out(tmp_path: Path) -> None:
    assert _run(tmp_path, mailer=Outbox(ok=False)) == 1
    traced: list[str] = []

    def tracer(record: dict[str, Any], **kw: Any) -> tuple[bool, str, dict[str, str]]:
        traced.append(record["day"])
        return True, "traces stored: 20 of 20", {}

    assert _run(tmp_path, tracer=tracer) == 0
    assert traced == ["2026-11-04"]
    assert _record(tmp_path)["trace"]["stored"] is True
    assert _run(tmp_path, tracer=tracer) == 0 and traced == ["2026-11-04"]


def _stage(tmp_path: Path, stage: str, token: str, **kw: Any) -> int:
    path = str(record_path(tmp_path, DAY))
    return _run(tmp_path, "--stage", stage, "--record", path, "--claim", token, **kw)


def _outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def test_the_workflow_makes_the_calls_then_claims_then_sends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    outbox = Outbox()
    assert _run(tmp_path, "--stage", "calls", mailer=outbox) == 0
    assert _outputs(out) == {"record": str(record_path(tmp_path, DAY)), "deliver": "yes"}
    assert outbox.sent == [] and _record(tmp_path)["email"]["status"] == "email pending"

    assert _stage(tmp_path, "claim", "job-1") == 0
    assert _record(tmp_path)["email"] == {"sent": None, "status": "sending", "claim": "job-1"}
    assert _stage(tmp_path, "claim", "job-2") == 1  # a second job cannot take it
    assert _stage(tmp_path, "send", "job-2", mailer=outbox) == 1  # nor send it
    assert outbox.sent == []

    assert _stage(tmp_path, "send", "job-1", mailer=outbox) == 0
    assert len(outbox.sent) == 1 and _record(tmp_path)["email"]["sent"] is True
    assert _outputs(out)["emailed"] == "yes"
    out.unlink()
    assert _run(tmp_path, "--stage", "calls", mailer=outbox) == 0
    assert _outputs(out) == {"deliver": "no"} and len(outbox.sent) == 1


def test_a_claimed_record_is_never_run_or_sent_again(tmp_path: Path) -> None:
    assert _run(tmp_path, "--stage", "calls") == 0
    assert _stage(tmp_path, "claim", "job-1") == 0  # then the job stopped before sending
    jev, claude, outbox = Jev(), Claude(), Outbox()
    assert _run(tmp_path, ctx=_ctx(jev, claude), mailer=outbox) == 1
    assert jev.asked == [] and claude.calls == [] and outbox.sent == []


def test_the_stages_that_send_need_a_record_and_a_claim(tmp_path: Path) -> None:
    assert _run(tmp_path, "--stage", "send") == 2
    assert _run(tmp_path, "--stage", "claim", "--record", "x.json") == 2


def test_an_email_that_may_have_gone_out_is_never_sent_again(tmp_path: Path) -> None:
    def lost(email: Email) -> tuple[bool | None, str]:
        return None, "email may have gone out: TimeoutError while sending"

    assert _run(tmp_path, mailer=lost) == 1
    assert _record(tmp_path)["email"]["sent"] is None
    outbox = Outbox()
    assert _run(tmp_path, mailer=outbox) == 1 and outbox.sent == []


def test_a_failed_send_is_claimed_again_and_resent_through_the_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert _run(tmp_path, "--stage", "calls") == 0
    assert _stage(tmp_path, "claim", "job-1") == 0
    assert _stage(tmp_path, "send", "job-1", mailer=Outbox(ok=False)) == 1
    assert _record(tmp_path)["email"]["sent"] is False
    out.unlink()
    jev, outbox = Jev(), Outbox()
    assert _run(tmp_path, "--stage", "calls", ctx=_ctx(jev)) == 0  # the rerun asks no one
    assert jev.asked == [] and _outputs(out)["deliver"] == "yes"
    assert _stage(tmp_path, "claim", "job-2") == 0
    assert _stage(tmp_path, "send", "job-2", mailer=outbox) == 0
    assert len(outbox.sent) == 1 and _record(tmp_path)["email"]["sent"] is True


def test_the_claim_refuses_an_emailed_day_and_a_lost_send_needs_a_person(tmp_path: Path) -> None:
    def lost(email: Email) -> tuple[bool | None, str]:
        return None, "email may have gone out: TimeoutError while sending"

    assert _run(tmp_path, "--stage", "calls") == 0
    assert _stage(tmp_path, "claim", "job-1") == 0
    assert _stage(tmp_path, "send", "job-1", mailer=lost) == 1
    assert _record(tmp_path)["email"]["sent"] is None
    assert _stage(tmp_path, "claim", "job-2") == 1

    other = tmp_path / "other"
    assert _run(other) == 0  # emailed in one go
    assert _stage(other, "claim", "job-3") == 1


@pytest.mark.parametrize("sent", ["true", 1, "yes"])
def test_a_hand_edit_that_is_not_true_or_false_needs_a_person(tmp_path: Path, sent: Any) -> None:
    assert _run(tmp_path, mailer=Outbox(ok=False)) == 1
    path = record_path(tmp_path, DAY)
    rec = json.loads(path.read_text())
    rec["email"]["sent"] = sent
    path.write_text(json.dumps(rec))
    outbox = Outbox()
    assert _run(tmp_path, mailer=outbox) == 1 and outbox.sent == []


def test_a_record_that_cannot_be_emailed_is_never_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(tmp_path, "--stage", "calls") == 0

    def broken(record: dict[str, Any]) -> Email:
        raise KeyError("stocks")

    monkeypatch.setattr(daily, "render", broken)
    with pytest.raises(KeyError):
        _stage(tmp_path, "claim", "job-1")
    assert _record(tmp_path)["email"]["status"] == "email pending"


def test_a_code_failure_reading_the_facts_is_noted_on_that_stock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = daily._facts

    def facts(out: dict[str, Any], pack: Pack, ticker: str, *a: Any) -> None:
        if ticker == "DDOG":
            raise ValueError("secret-bearing message")
        real(out, pack, ticker, *a)

    monkeypatch.setattr(daily, "_facts", facts)
    assert _run(tmp_path) == 0
    by = {s["ticker"]: s for s in _record(tmp_path)["stocks"]}
    assert by["DDOG"]["error"] == "the facts pack could not be read (ValueError)"
    assert "call" not in by["DDOG"] and "secret" not in json.dumps(by)
    assert by["CRWD"]["call"]["call"] == "Buy"


class Raising(Claude):
    """A code failure while Opus is asked about one stock."""

    def __call__(self, message: str, **kw: Any) -> AgentRun:
        if _news("SNOW") in message:
            raise RuntimeError("secret-bearing message")
        return super().__call__(message, **kw)


def test_a_code_failure_in_one_stock_is_noted_and_the_rest_go_on(tmp_path: Path) -> None:
    jev = Jev()

    def ask(state: str, questions: Any, **kw: Any) -> Result:
        if "DDOG" in kw["session_id"]:
            raise RuntimeError("secret-bearing message")
        return jev(state, questions, **kw)

    ctx = _ctx(claude=Raising())
    ctx.ask = ask
    assert _run(tmp_path, ctx=ctx) == 0
    by = {s["ticker"]: s for s in _record(tmp_path)["stocks"]}
    assert by["DDOG"]["jev"]["error"] == "failed in code (RuntimeError)"
    assert by["DDOG"]["opus"]["call"] == "Hold"  # Opus's own call still made
    assert "failed in code" in by["SNOW"]["opus"]["error"]
    assert by["SNOW"]["call"]["call"] == "Buy"  # Jev's call still made
    assert "secret" not in json.dumps(by)
    assert by["CRWD"]["opus"]["call"] == "Hold" and by["CRWD"]["why"]["kept"]


def test_a_resend_traces_only_the_stocks_not_yet_stored(tmp_path: Path) -> None:
    def half(record: dict[str, Any], **kw: Any) -> tuple[bool, str, dict[str, str]]:
        return False, "traces sent, 1 of 20 found", {"CRWD": "a" * 32}

    assert _run(tmp_path, mailer=Outbox(ok=False), tracer=half) == 1
    skipped: list[set[str]] = []

    def rest(record: dict[str, Any], **kw: Any) -> tuple[bool, str, dict[str, str]]:
        skipped.append(set(kw["skip"]))
        return True, "traces stored: 19 of 19", {"DDOG": "b" * 32}

    assert _run(tmp_path, tracer=rest) == 0
    assert skipped == [{"CRWD"}]
    assert _record(tmp_path)["trace"]["ids"] == {"CRWD": "a" * 32, "DDOG": "b" * 32}


def test_daily_traces_skip_the_stocks_already_stored(tmp_path: Path) -> None:
    rec = _day_record(tmp_path)
    every = {s["ticker"] for s in rec["stocks"]}
    assert send_daily_traces(rec, skip=every) == (True, "traces already stored", {})


class Failing(Claude):
    """Every claude run from the given one on fails without hitting the limit."""

    def __init__(self, fail_from: int) -> None:
        super().__init__()
        self.fail_from = fail_from

    def __call__(self, message: str, **kw: Any) -> AgentRun:
        run = super().__call__(message, **kw)
        if len(self.calls) >= self.fail_from:
            run = AgentRun(kw["model"])
            run.fail(INFRA, "timed out after 300s")
        return run


def test_two_failed_claude_runs_in_a_row_stop_asking_claude(tmp_path: Path) -> None:
    claude = Failing(fail_from=3)  # CRWD's call and why work, DDOG's two fail
    assert _run(tmp_path, ctx=_ctx(claude=claude)) == 0
    rec = _record(tmp_path)
    assert len(claude.calls) == 4
    assert "failed 2 times in a row" in rec["stocks"][2]["opus"]["error"]
    assert "failed 2 times in a row" in rec["stocks"][2]["why"]["error"]
    assert all(s["call"]["call"] == "Buy" for s in rec["stocks"])


def test_one_failed_claude_run_is_not_a_breaker(tmp_path: Path) -> None:
    class Once(Claude):
        def __call__(self, message: str, **kw: Any) -> AgentRun:
            run = super().__call__(message, **kw)
            if len(self.calls) == 1:
                run = AgentRun(kw["model"])
                run.fail(INFRA, "exit 1")
            return run

    claude = Once()
    assert _run(tmp_path, ctx=_ctx(claude=claude)) == 0
    assert len(claude.calls) == 2 * len(WATCH.tickers)


def test_opus_stops_when_the_runs_time_budget_is_spent(tmp_path: Path) -> None:
    ticks = iter(range(10_000))
    ctx = _ctx()
    ctx.opus_budget_s = 2.5  # one tick per clock read: CRWD's two checks, then over
    ctx.clock = lambda: float(next(ticks))
    assert _run(tmp_path, ctx=ctx) == 0
    rec = _record(tmp_path)
    assert rec["stocks"][0]["opus"]["call"] == "Hold" and rec["stocks"][0]["why"]["kept"]
    assert "time budget" in rec["stocks"][1]["opus"]["error"]
    assert "time budget" in rec["stocks"][1]["why"]["error"]


def test_the_why_message_is_recorded_and_a_lagging_price_is_noted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx()
    inner = ctx.gather

    def gather(ticker: str, now: datetime) -> Pack | None:
        pack = inner(ticker, now)
        return replace(pack, day=date(2026, 11, 3)) if pack and ticker == "DDOG" else pack

    ctx.gather = gather
    outbox = Outbox()
    monkeypatch.setenv("GITHUB_SHA", "the-commit-that-triggered-it")
    monkeypatch.setenv("TENQ_CODE_SHA", "abc123")
    assert _run(tmp_path, ctx=ctx, mailer=outbox) == 0
    rec = _record(tmp_path)
    assert "How the rules reached it" in rec["stocks"][0]["why"]["message"]
    assert rec["made_at"] == NOW.isoformat()
    # the facts sheet is not stored twice; the hash pins the message Opus was shown
    assert isinstance(ctx.claude, Claude)
    shown = next(m for m in ctx.claude.calls if "How the rules reached it" in m)
    ticker = next(t for t in WATCH.tickers if _news(t) in shown)
    why = next(s for s in rec["stocks"] if s["ticker"] == ticker)["why"]
    head, sheet = shown.split(opus.SHEET_HEADER, 1)
    assert why["message"].startswith(head + opus.SHEET_HEADER)
    assert sheet.strip() not in why["message"] and len(why["message"]) < len(head) + 200
    assert why["message_sha256"] == hashlib.sha256(shown.encode("utf-8")).hexdigest()
    assert "\n " not in record_path(tmp_path, DAY).read_text()  # written compact
    assert rec["versions"]["code"] == "abc123"
    # nothing is lost: the record's pack renders the message again, hash and all
    stock = next(s for s in rec["stocks"] if s["ticker"] == ticker)
    again = opus.rebuild_why_message(why["message"], stock["pack"])
    assert again == shown
    assert hashlib.sha256(again.encode("utf-8")).hexdigest() == why["message_sha256"]
    assert rec["stocks"][1]["note"] == "its latest price is from 2026-11-03, not 2026-11-04"
    assert "Note: its latest price is from 2026-11-03" in outbox.sent[0].text


# the email


def _day_record(tmp_path: Path) -> dict[str, Any]:
    assert _run(tmp_path) == 0
    return _record(tmp_path)


def test_a_check_says_so_in_its_subject_and_its_traces(tmp_path: Path) -> None:
    outbox = Outbox()
    assert _run(tmp_path, "--ticker", "CRWD", mailer=outbox) == 0
    assert outbox.sent[0].subject.startswith("tenq daily check, session 2026-11-04")
    [check] = _checks(tmp_path)
    tracing, exporter = _memory()
    emit_daily(check["stocks"][0], check, tracing.tracer)
    root = next(s for s in exporter.get_finished_spans() if s.name == "daily CRWD")
    attrs = root.attributes or {}
    assert attrs["session.id"] == f"daily-2026-11-04-check-{check['check_id']}"
    assert "check" in list(attrs["langfuse.trace.tags"])  # type: ignore[arg-type]


def test_the_email_shows_all_three_calls_and_escapes_untrusted_text(tmp_path: Path) -> None:
    email = render(_day_record(tmp_path))
    assert email.subject == "tenq daily, session 2026-11-04: all three agree on 0 of 20"
    # Jev: 0.559 of the 0.70 its news and guidance can reach (rules-v9); Kairos
    # gave the same answers, so the same rules make the same call
    assert "Stock | Jev | Opus | Kairos | agreement" in email.text
    assert "CRWD | Buy 80% | Hold 50% | Buy 80% | Opus differs" in email.text
    assert "  Kairos: Buy, confidence 80%\n    " in email.text
    assert "<b>Kairos: Buy</b>, confidence 80%" in email.html
    assert "The outlook was raised <script>." in email.text
    assert "<script>" not in email.html and "&lt;script&gt;" in email.html
    assert "&lt;b&gt;Outlook&lt;/b&gt;" in email.html
    assert "href='https://www.reuters.com/a?b=1&amp;c=2'" in email.html
    assert "Shares jumped" not in email.text  # a dropped reason is never shown
    assert "1 reason(s) failed the checks" in email.text
    assert "Not investment advice" in email.text and "Not investment advice" in email.html
    assert "Opus: Hold, confidence 50%, lean +0.10" in email.text
    assert "confidence 50%, lean +0.10" in email.html


@pytest.mark.parametrize(
    ("opus", "shown"),
    [
        ({"lean": -0.4}, ", lean -0.40"),
        ({"lean": None, "lean_error": "no lean given"}, ", lean not kept (no lean given)"),
        ({}, ""),  # a record made before Opus gave a lean
    ],
)
def test_the_email_shows_opus_lean_or_why_it_was_not_kept(opus: dict[str, Any], shown: str) -> None:
    from tenq.daily_email import _lean

    assert _lean(opus) == shown


def test_only_http_links_are_shown() -> None:
    pack = {
        "news": [
            {"id": "n1", "title": "a", "site": "", "link": "javascript:alert(1)"},
            {"id": "n2", "title": "b", "site": "x.com", "link": "https://x.com/b"},
        ],
        "reports": [
            {"filed": "2026-10-01", "url": "https://www.sec.gov/r", "excerpts": [{"id": "e1"}]}
        ],
        "filing": {"form": "10-Q", "period": "2026-07-31", "url": "ftp://sec.gov/f"},
        "numbers": [{"id": "fin:A:revenue:yoy_pct"}, {"id": "px:A:close"}],
    }
    got = sources(pack)
    assert got["n1"] == ("a", None) and got["n2"] == ("b (x.com)", "https://x.com/b")
    assert got["e1"] == ("8-K filed 2026-10-01", "https://www.sec.gov/r")
    assert got["fin:A:revenue:yoy_pct"] == ("the 10-Q for the quarter ended 2026-07-31", None)
    assert "px:A:close" not in got


def test_a_6k_excerpt_is_labelled_as_a_6k() -> None:
    """L11: the record keeps each report's form, so the email names a 6-K as one;
    a record from before L11 has no form and held only 8-Ks."""
    report = {"filed": "2026-10-01", "url": "https://www.sec.gov/r", "excerpts": [{"id": "e1"}]}
    six_k = sources({"reports": [report | {"form": "6-K"}]})
    assert six_k["e1"] == ("6-K filed 2026-10-01", "https://www.sec.gov/r")
    assert sources({"reports": [report]})["e1"][0] == "8-K filed 2026-10-01"


# the traces


def _memory() -> tuple[Tracing, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Tracing(provider.get_tracer("tenq"), lambda: None, lambda trace_id: True), exporter


def test_each_stock_is_its_own_trace_grouped_by_day(tmp_path: Path) -> None:
    rec = _day_record(tmp_path)
    tracing, exporter = _memory()
    ids = {s["ticker"]: emit_daily(s, rec, tracing.tracer) for s in rec["stocks"][:2]}
    spans = exporter.get_finished_spans()
    assert len(set(ids.values())) == 2
    crwd = [s for s in spans if format(s.context.trace_id, "032x") == ids["CRWD"]]
    names = {s.name: s for s in crwd}
    assert set(names) == {"daily CRWD", "facts pack", "jev", "opus call", "opus why", "kairos"}
    jev = names["jev"].attributes or {}
    assert jev["langfuse.observation.type"] == "generation"
    assert json.loads(str(jev["langfuse.observation.cost_details"])) == {"total": 0.001}
    assert names["opus call"].attributes["langfuse.observation.type"] == "generation"  # type: ignore[index]
    root = names["daily CRWD"].attributes or {}
    assert root["session.id"] == "daily-2026-11-04"
    assert root["langfuse.trace.name"] == "daily"


def test_daily_traces_are_skipped_without_env_and_never_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _day_record(tmp_path)
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    assert send_daily_traces(rec) == (False, "trace skipped: Langfuse env not set", {})
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://lf.example/")

    def boom(*a: Any) -> Tracing:
        raise RuntimeError("https://lf.example/ down")

    stored, status, ids = send_daily_traces(rec, connect=boom)
    assert (stored, status, ids) == (False, "trace failed: RuntimeError", {})


def test_daily_traces_report_any_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _day_record(tmp_path)
    rec["stocks"] = rec["stocks"][:2]
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://lf.example/")
    tracing, _ = _memory()
    first: list[str] = []

    def found(trace_id: str) -> bool:
        first.append(trace_id)
        return len(first) == 1

    clock = iter(range(0, 10_000, 5))
    stored, status, ids = send_daily_traces(
        rec,
        connect=lambda *a: Tracing(tracing.tracer, lambda: None, found),
        wait=20,
        sleep=lambda s: None,
        clock=lambda: float(next(clock)),
    )
    assert stored is False and set(ids) == {"CRWD"}
    assert status.startswith("traces sent, 1 of 2 found") and status.endswith("missing DDOG")


def test_the_daily_module_reads_no_secret_into_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("SMTP_PASS", "OPENROUTER_API_KEY", "LANGFUSE_SECRET_KEY", "EDGAR_IDENTITY"):
        monkeypatch.setenv(key, f"sentinel-{key}")
    _day_record(tmp_path)
    assert "sentinel-" not in record_path(tmp_path, DAY).read_text()
    assert daily.KIND == "daily"


RUN_URL = "https://github.com/jelbirt/tenq/actions/runs/1"


def test_the_failure_notice_links_the_run_and_touches_no_record(tmp_path: Path) -> None:
    outbox = Outbox()
    assert (
        main(["--stage", "notice", "--run-url", RUN_URL, "--runs", str(tmp_path)], mailer=outbox)
        == 0
    )
    (email,) = outbox.sent
    assert email.subject == "tenq daily run failed"
    assert RUN_URL in email.text and RUN_URL in email.html
    assert "may not have been" in email.text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        (["--stage", "notice"], 2),
        (["--stage", "notice", "--run-url", "https://evil.example/x"], 2),
    ],
)
def test_the_failure_notice_needs_the_runs_address(argv: list[str], code: int) -> None:
    outbox = Outbox()
    assert main(argv, mailer=outbox) == code
    assert outbox.sent == []


def test_a_failure_notice_that_cannot_be_sent_fails() -> None:
    assert main(["--stage", "notice", "--run-url", RUN_URL], mailer=Outbox(ok=False)) == 1


def test_the_workflow_runs_on_weekdays_and_notices_only_a_failed_scheduled_run() -> None:
    import yaml

    flow = yaml.safe_load(Path(".github/workflows/daily.yml").read_text())
    on = flow[True]  # YAML 1.1 reads the key "on" as true
    assert on["schedule"] == [{"cron": "30 22 * * 1-5"}]
    assert "workflow_dispatch" in on
    steps = {s.get("name"): s for s in flow["jobs"]["daily"]["steps"]}
    assert steps["send the email and the traces"]["id"] == "send"
    notice = steps["failure notice"]
    assert notice["if"] == (
        "${{ (failure() || cancelled()) && github.event_name == 'schedule' && "
        "steps.send.outputs.emailed != 'yes' }}"
    )
    assert "--stage notice" in notice["run"]
    assert set(notice["env"]) == {"RUN_URL", "SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAILS"}
    # the record's code version is the commit checked out, read from git right
    # after the checkout, before any step can change it
    runs = [s.get("run", "") for s in flow["jobs"]["daily"]["steps"]]
    pin = 'echo "TENQ_CODE_SHA=$(git rev-parse HEAD)" >> "$GITHUB_ENV"'
    checkout = flow["jobs"]["daily"]["steps"][0]
    assert checkout["uses"].startswith("actions/checkout") and checkout["with"]["ref"] == "main"
    assert runs.index(pin) == 1


@pytest.mark.parametrize(("ok", "emailed"), [(True, "yes"), (False, "no")])
def test_the_send_step_says_whether_the_email_went_even_when_it_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ok: bool, emailed: str
) -> None:
    # every stock failed to get a call: the email still goes, the step exits 1,
    # and only its "emailed" output keeps the failure notice from following it
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    nobody = _ctx(Jev(down_after=0), Claude(limit_after=0), Kairos(down=ALWAYS))
    assert _run(tmp_path, "--stage", "calls", ctx=nobody) == 0
    assert _stage(tmp_path, "claim", "job-1") == 0
    out.unlink()
    assert _stage(tmp_path, "send", "job-1", mailer=Outbox(ok=ok)) == 1
    assert _outputs(out) == {"emailed": emailed}


def test_a_lost_mail_server_answer_says_maybe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert _run(tmp_path, "--stage", "calls") == 0
    assert _stage(tmp_path, "claim", "job-1") == 0
    out.unlink()
    lost = lambda email: (None, "email may have gone out: TimeoutError while sending")  # noqa: E731
    assert _stage(tmp_path, "send", "job-1", mailer=lost) == 1
    assert _outputs(out) == {"emailed": "maybe"}


def _at(y: int, m: int, d: int, hour: int, minute: int = 0) -> datetime:
    return datetime(y, m, d, hour, minute, tzinfo=UTC)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (_at(2026, 11, 4, 22, 30), date(2026, 11, 4)),  # Wednesday, after the settle
        (_at(2026, 11, 4, 21, 0), date(2026, 11, 3)),  # 16:00 New York: not settled yet
        (_at(2026, 11, 4, 21, 30), date(2026, 11, 4)),  # 16:30 New York: settled
        (_at(2026, 11, 7, 22, 30), date(2026, 11, 6)),  # Saturday
        (_at(2026, 11, 9, 14, 0), date(2026, 11, 6)),  # Monday morning
        (_at(2026, 11, 26, 22, 30), date(2026, 11, 25)),  # Thanksgiving
        (_at(2026, 11, 27, 22, 30), date(2026, 11, 27)),  # the early close after it is a session
        (_at(2026, 1, 19, 22, 30), date(2026, 1, 16)),  # Martin Luther King Jr. Day
        (_at(2026, 1, 2, 14, 0), date(2025, 12, 31)),  # across New Year's Day and the year
        (_at(2026, 7, 6, 20, 15), date(2026, 7, 2)),  # 16:15 summer time; July 3 observed
        (_at(2026, 7, 6, 20, 30), date(2026, 7, 6)),  # 16:30 summer time: settled
        (_at(2026, 11, 5, 5, 30), date(2026, 11, 4)),  # a late run past New York midnight
        (_at(2026, 4, 3, 22, 30), date(2026, 4, 2)),  # Good Friday: closed, not a federal holiday
        (_at(2028, 1, 3, 14, 0), date(2027, 12, 31)),  # New Year's on a Saturday: Friday is open
    ],
)
def test_the_expected_session_follows_the_exchange_calendar(now: datetime, expected: date) -> None:
    assert daily.expected_session(now) == expected


def test_a_trading_day_with_no_bar_yet_fails_and_runs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jev, claude, outbox = Jev(), Claude(), Outbox()
    stale = lambda: Prices({BENCHMARK: (Bar(date(2026, 11, 3), 598.0, None),)}, ())  # noqa: E731
    argv = ["--runs", str(tmp_path)]
    assert main(argv, ctx=_ctx(jev, claude), prices=stale, mailer=outbox, tracer=_no_trace) == 1
    assert (
        "no QQQ bar for the 2026-11-04 session yet (latest: 2026-11-03)" in capsys.readouterr().out
    )
    assert jev.asked == [] and claude.calls == [] and outbox.sent == []
    assert list(tmp_path.iterdir()) == []


def test_a_holiday_finds_the_last_session_emailed_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eve = date(2026, 11, 25)
    bars = lambda: Prices({BENCHMARK: (Bar(eve, 610.0, None),)}, ())  # noqa: E731
    outbox = Outbox()
    argv = ["--runs", str(tmp_path)]
    day_before = replace(_ctx(), now=_at(2026, 11, 25, 22, 30))
    assert main(argv, ctx=day_before, prices=bars, mailer=outbox, tracer=_no_trace) == 0
    assert len(outbox.sent) == 1

    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    jev = Jev()
    thanksgiving = replace(_ctx(jev), now=_at(2026, 11, 26, 22, 30))
    assert main(argv, ctx=thanksgiving, prices=bars, mailer=outbox, tracer=_no_trace) == 0
    assert _outputs(out) == {"deliver": "no"} and len(outbox.sent) == 1 and jev.asked == []


def test_an_unemailed_day_is_not_resent_while_a_newer_session_is_missing(tmp_path: Path) -> None:
    # the missing-session check comes before the resend: else a late bar would
    # resend the old day, exit 0 and skip the new day unseen
    old = date(2026, 11, 3)
    bars = lambda: Prices({BENCHMARK: (Bar(old, 598.0, None),)}, ())  # noqa: E731
    argv = ["--runs", str(tmp_path)]
    that_evening = replace(_ctx(), now=_at(2026, 11, 3, 22, 30))
    assert main(argv, ctx=that_evening, prices=bars, mailer=Outbox(ok=False), tracer=_no_trace) == 1
    path = record_path(tmp_path, old)
    before = path.read_text()
    assert json.loads(before)["email"]["sent"] is False

    outbox = Outbox()
    assert main(argv, ctx=_ctx(), prices=bars, mailer=outbox, tracer=_no_trace) == 1
    assert outbox.sent == [] and path.read_text() == before


def test_a_check_run_on_a_late_feed_fails_too(tmp_path: Path) -> None:
    stale = lambda: Prices({BENCHMARK: (Bar(date(2026, 11, 3), 598.0, None),)}, ())  # noqa: E731
    argv = ["--runs", str(tmp_path), "--ticker", "CRWD"]
    assert main(argv, ctx=_ctx(), prices=stale, mailer=Outbox(), tracer=_no_trace) == 1
    assert list(tmp_path.iterdir()) == []


def test_a_pack_comes_back_from_its_dict() -> None:
    pack = _pack("CRWD", NOW)
    assert pack is not None
    again = Pack.from_dict(json.loads(json.dumps(pack.to_dict())))
    assert again == pack and again.render() == pack.render()


def test_the_why_trace_shows_the_whole_message_opus_was_sent(tmp_path: Path) -> None:
    rec = _day_record(tmp_path)
    stock = next(s for s in rec["stocks"] if (s.get("why") or {}).get("message"))
    tracing, exporter = _memory()
    emit_daily(stock, rec, tracing.tracer)
    span = next(s for s in exporter.get_finished_spans() if s.name == "opus why")
    shown = str((span.attributes or {})["langfuse.observation.input"])
    assert shown.startswith(stock["why"]["message"].split(opus.SHEET_HEADER, 1)[0])
    assert hashlib.sha256(shown.encode("utf-8")).hexdigest() == stock["why"]["message_sha256"]
    # a pack the code cannot read still traces, with the stored head
    broken = {**stock, "pack": {}}
    emit_daily(broken, rec, tracing.tracer)
    spans = [s for s in exporter.get_finished_spans() if s.name == "opus why"]
    assert (
        str((spans[-1].attributes or {})["langfuse.observation.input"]) == (stock["why"]["message"])
    )


def _screen(day: date, chosen: list[str], filled: list[str] = (), **extra: Any) -> Any:  # type: ignore[assignment]
    def screener(now: datetime) -> dict[str, Any]:
        return {
            "kind": "screen",
            "version": "screen-v6",
            "day": day.isoformat(),
            "universe": 113,
            "chosen": [{"ticker": t, "reasons": [f"{t} moved"]} for t in chosen],
            "over_cap": [],
            "filled": list(filled),
            "held": ["AAPL", "PEP"],
            "unscreened": [],
            "ok": True,
            "refused": [],
        } | extra

    return screener


def test_the_email_counts_stocks_held_because_they_just_reported(tmp_path: Path) -> None:
    outbox = Outbox()
    reported = {"PEP": "2026-11-06", "MSFT": "2026-11-05"}  # MSFT drawn, so not held
    screener = _screen(DAY, ["TEAM"], ["MSFT"], just_reported=reported)
    assert _run(tmp_path, screener=screener, mailer=outbox) == 0
    [mail] = outbox.sent
    assert "; 2 Hold by screen (1 just reported, no next date yet)." in mail.text


def test_a_good_screen_decides_which_stocks_the_models_judge(tmp_path: Path) -> None:
    outbox = Outbox()
    screener = _screen(DAY, ["TEAM", "AAPL"], ["MSFT"])
    assert _run(tmp_path, screener=screener, mailer=outbox) == 0
    record = _record(tmp_path)
    assert [s["ticker"] for s in record["stocks"]] == ["TEAM", "AAPL", "MSFT"]
    assert record["screen"]["used"] == "screen"
    assert record["screen"]["held"] == ["AAPL", "PEP"]
    [mail] = outbox.sent
    assert not mail.subject.startswith("SCREEN FAILED")
    assert (
        "Screen screen-v6: 2 of 113 stocks passed, 1 more drawn at random as a check;" in mail.text
    )
    assert "  TEAM: TEAM moved" in mail.text
    assert "Random fill: MSFT" in mail.text
    assert "SCREEN FAILED" not in mail.text and "#b00" not in mail.html
    assert "just reported" not in mail.text


@pytest.mark.parametrize(
    ("result", "why"),
    [
        ({"kind": "screen", "ok": False, "refused": ["no QQQ prices"]}, "no QQQ prices"),
        (
            {"kind": "screen", "ok": False, "refused": ["the screen failed (OSError)"]},
            "the screen failed (OSError)",
        ),
        (
            _screen(date(2026, 11, 3), ["TEAM"])(NOW),
            "it screened the 2026-11-03 session, not 2026-11-04",
        ),
    ],
)
def test_a_failed_screen_runs_the_watchlist_and_says_so_first(
    tmp_path: Path, result: dict[str, Any], why: str, caplog: pytest.LogCaptureFixture
) -> None:
    outbox = Outbox()
    assert _run(tmp_path, screener=lambda now: result, mailer=outbox) == 0
    record = _record(tmp_path)
    assert [s["ticker"] for s in record["stocks"]] == list(WATCH.tickers)
    assert record["screen"]["used"] == "watchlist"
    assert record["screen"]["ok"] is False
    [mail] = outbox.sent
    assert mail.subject.startswith("SCREEN FAILED + tenq daily, session 2026-11-04")
    assert mail.text.startswith(f"SCREEN FAILED: {why}.\nThe 20 watchlist stocks were asked")
    assert mail.html.index("#b00") < mail.html.index("tenq daily calls")
    assert "SCREEN FAILED" in caplog.text


def test_a_check_on_named_stocks_skips_the_screen_and_takes_any_universe_stock(
    tmp_path: Path,
) -> None:
    def never(now: datetime) -> dict[str, Any]:
        raise AssertionError("a check on named stocks must not screen")

    outbox = Outbox()
    assert _run(tmp_path, "--ticker", "AAPL", screener=never, mailer=outbox) == 0
    [check] = _checks(tmp_path)
    assert [s["ticker"] for s in check["stocks"]] == ["AAPL"]
    assert check["screen"] is None
    assert "Screen" not in outbox.sent[0].text
    assert _run(tmp_path, "--ticker", "ZZZZ", screener=never) == 2


def test_a_check_without_named_stocks_still_uses_the_screen(tmp_path: Path) -> None:
    assert _run(tmp_path, "--no-email", screener=_screen(DAY, ["TEAM", "AAPL"])) == 0
    [check] = _checks(tmp_path)
    assert check["screen"]["used"] == "screen"
    assert [s["ticker"] for s in check["stocks"]] == ["TEAM", "AAPL"]


def test_a_screened_stock_off_the_watchlist_shows_its_company(tmp_path: Path) -> None:
    outbox = Outbox()
    assert _run(tmp_path, screener=_screen(DAY, ["AAPL"]), mailer=outbox) == 0
    assert _record(tmp_path)["stocks"][0]["company"] == "Apple"
    assert "AAPL (Apple)" in outbox.sent[0].text


def _spy_on_gather(monkeypatch: pytest.MonkeyPatch, foreign: bool = False) -> dict[str, Any]:
    """Stub everything gather fetches; return what it passed on."""
    from tenq import pack as pack_module
    from tenq.fetch import Reports
    from tenq.news import News

    seen: dict[str, Any] = {}

    class Source:
        pass

    def collect(names: dict[str, str | None], since: datetime) -> News:
        seen["news"] = names
        return News({}, ())

    def build(ticker: str, company: str, **kw: Any) -> str:
        seen["pack"] = company
        seen["foreign"] = kw["foreign"]
        return "pack"

    monkeypatch.setattr("tenq.edgar.EdgarSource", Source)
    monkeypatch.setattr("tenq.fetch.sync_reports", lambda *a, **k: Reports([], foreign))
    monkeypatch.setattr("tenq.current.sync_current", lambda *a, **k: type("S", (), {"notes": ()})())
    monkeypatch.setattr("tenq.prices.fetch", lambda *a, **k: Prices({}, ()))
    monkeypatch.setattr("tenq.news.collect", collect)
    monkeypatch.setattr(pack_module, "build", build)
    return seen


def test_the_facts_pack_and_news_get_the_company_of_any_screened_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tenq import pack as pack_module

    seen = _spy_on_gather(monkeypatch)
    assert pack_module.gather("AAPL", NOW) == "pack"  # type: ignore[comparison-overlap]
    assert seen == {
        "news": {"AAPL": "Apple"},
        "pack": "Apple",
        "foreign": False,
    }


def test_the_sheet_hears_when_edgar_lists_a_foreign_companys_yearly_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tenq import pack as pack_module

    seen = _spy_on_gather(monkeypatch, foreign=True)
    pack_module.gather("PDD", NOW)
    assert seen["foreign"] is True


def test_a_broken_universe_file_still_lets_the_watchlist_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tenq import pack as pack_module

    def broken() -> dict[str, str]:
        raise ValueError("bad universe")

    seen = _spy_on_gather(monkeypatch)
    monkeypatch.setattr("tenq.screen.universe_names", broken)
    assert pack_module.gather("CRWD", NOW) == "pack"  # type: ignore[comparison-overlap]
    assert seen["pack"] == WATCH.names["CRWD"]


def test_the_screens_time_counts_against_the_opus_budget(tmp_path: Path) -> None:
    ticks = iter(range(10_000))
    ctx = _ctx()
    ctx.opus_budget_s = 2.5
    ctx.clock = lambda: float(next(ticks))

    def slow(now: datetime) -> Any:
        for _ in range(3):  # the screen uses up the whole budget
            next(ticks)
        return _screen(DAY, list(WATCH.tickers))(now)

    assert _run(tmp_path, ctx=ctx, screener=slow) == 0
    assert "time budget" in _record(tmp_path)["stocks"][0]["opus"]["error"]


# Kairos, the third method: asked after Jev and Opus are done with every stock


FOUR = WATCH.tickers[:4]
FIXTURE = Path(__file__).parent / "fixtures" / "daily-two-methods"


def _sid(stock: dict[str, Any]) -> str:
    return f"tenq-daily-{stock['ticker']}-2026-11-04"


def _four(kairos: Kairos, ctx: Context | None = None) -> tuple[dict[str, Any], list[Any]]:
    """Four stocks through run_day; the record and a copy of it at each save."""
    saved: list[Any] = []
    ctx = ctx or _ctx(kairos=kairos)
    rec = daily.run_day(
        list(FOUR), WATCH.names, DAY, 500.0, ctx, save=lambda r: saved.append(json.dumps(r))
    )
    return rec, [json.loads(r) for r in saved]


def test_kairos_makes_a_third_call_on_every_stock_from_jevs_facts_and_questions(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    jev_fake, kairos = Jev(), Kairos()
    sent: list[tuple[str, Any]] = []

    def ask(state: str, questions: Any, **kw: Any) -> Result:
        order.append("jev")
        sent.append((state, questions))
        return jev_fake(state, questions, **kw)

    ctx = _ctx(kairos=kairos)
    ctx.ask = ask

    def ask_kairos(state: str, questions: Any, **kw: Any) -> Result:
        order.append("kairos")
        return kairos(state, questions, **kw)

    ctx.kairos = ask_kairos
    outbox = Outbox()
    assert _run(tmp_path, ctx=ctx, mailer=outbox) == 0
    rec = _record(tmp_path)
    stocks = rec["stocks"]
    assert rec["kairos_on"] is True and len(stocks) == 20
    assert kairos.seen == sent and kairos.seen[0][1] is QUESTIONS  # the same pack, the same file
    assert order == ["jev"] * 20 + ["kairos"] * 20  # Kairos only once Jev and Opus are done
    assert kairos.asked == [_sid(s) for s in stocks]
    for stock in stocks:
        # the same answers through the same rules: the same call as Jev's
        assert stock["kairos_call"]["call"] == stock["call"]["call"]
        assert stock["kairos_call"]["explain"] == stock["call"]["explain"]
        assert stock["kairos"]["model"] == "DMIND/Kairos"
        assert stock["kairos"]["resolved_model"] == "DMIND/Kairos-0.9"
        assert "first" not in stock["kairos"]
        assert [s["name"] for s in stock["steps"]][-1] == "kairos"
    assert rec["kairos_input_tokens"] == 20 * 1000
    assert any(line.startswith("balance, Kairos over 20: ") for line in rec["balance"])
    assert outbox.sent[0].subject == "tenq daily, session 2026-11-04: all three agree on 0 of 20"


def test_kairos_down_skips_its_pass_then_asks_each_missed_stock_once_more() -> None:
    clock = Clock()
    kairos = Kairos(down={2}, clock=clock)
    rec, saved = _four(kairos)
    a, b, c, d = rec["stocks"]
    # down at the 2nd stock (30 s in): the 3rd and 4th are skipped; after
    # the pause, the 2nd to the 4th are each asked once more
    assert kairos.asked == [_sid(a), _sid(b), _sid(b), _sid(c), _sid(d)]
    assert clock.slept == [daily.KAIROS_PAUSE_S]
    assert all("kairos_call" in s for s in rec["stocks"])
    assert "first" not in a["kairos"]
    assert b["kairos"]["first"]["down"] is True and b["kairos"]["first"]["error"] == "HTTP 503"
    for stock in (c, d):
        assert stock["kairos"]["first"] == {"error": f"not asked: {daily.KAIROS_DOWN}"}
    assert [s["name"] for s in b["steps"]][-2:] == ["kairos", "kairos again"]
    assert [s["name"] for s in c["steps"]][-1] == "kairos again"
    assert rec["kairos_input_tokens"] == 4 * 1000
    # saved after each stock, each Kairos ask, and each one more ask
    made = [sum("kairos_call" in s for s in r["stocks"]) for r in saved]
    assert made == [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 4]
    text = render(rec).text
    assert text.count("Kairos's call came when it was asked again at the end of the run") == 3
    assert "asked again at the end of the run (HTTP 503)." in text


def test_kairos_down_again_ends_the_pass_of_one_more_asks(tmp_path: Path) -> None:
    kairos = Kairos(down={2, 3}, clock=Clock())
    rec, _ = _four(kairos)
    a, b, c, d = rec["stocks"]
    assert kairos.asked == [_sid(a), _sid(b), _sid(b)]
    assert "kairos_call" in a and not any("kairos_call" in s for s in (b, c, d))
    assert b["kairos"]["error"] == "HTTP 503" and b["kairos"]["first"]["error"] == "HTTP 503"
    for stock in (c, d):
        assert stock["kairos"] == {
            "error": f"not asked: {daily.KAIROS_DOWN}",
            "again": f"not asked again: {daily.KAIROS_DOWN_AGAIN}",
        }
    text = render(rec).text
    assert "Kairos made no call (asked again at the end of the run: HTTP 503)." in text
    assert (
        f"Kairos made no call (not asked: {daily.KAIROS_DOWN}; "
        f"not asked again: {daily.KAIROS_DOWN_AGAIN})."
    ) in text


@pytest.mark.parametrize(
    ("budget", "last", "told"),
    [
        (70.0, daily.KAIROS_LAST_S, [70.0, 40.0, 10.0]),  # its budget in the run
        (daily.KAIROS_BUDGET_S, 50.0, [50.0, 20.0]),  # its last start, from the run's start
    ],
)
def test_kairos_stops_at_its_time_bound_and_is_not_asked_again(
    budget: float, last: float, told: list[float]
) -> None:
    kairos = Kairos(clock=Clock())  # each ask takes 30 s
    ctx = replace(_ctx(kairos=kairos), kairos_budget_s=budget, kairos_last_s=last)
    rec, _ = _four(kairos, ctx)
    assert kairos.left == told  # each ask is told the time it has, and stops by it
    for stock in rec["stocks"][len(told) :]:
        assert stock["kairos"] == {"error": "not asked: the run's time for Kairos ran out"}
    assert all("call" in s and s["opus"]["call"] for s in rec["stocks"])


def test_a_pause_longer_than_kairos_time_left_asks_nothing_more() -> None:
    clock = Clock()
    kairos = Kairos(down={1}, clock=clock)
    ctx = replace(_ctx(kairos=kairos), kairos_budget_s=100.0)
    rec, _ = _four(kairos, ctx)
    assert len(kairos.asked) == 1 and clock.slept == []
    for stock in rec["stocks"]:
        assert stock["kairos"]["again"] == "not asked again: the run's time for Kairos ran out"


def test_the_pause_counts_from_kairos_last_failed_ask_down_or_not() -> None:
    clock = Clock()
    kairos = Kairos(empty={2}, clock=clock)  # no usable answer at stock 2, but not down
    rec, _ = _four(kairos)
    a, b, c, d = rec["stocks"]
    # stock 2's ask ended at 60 s, the pass at 120 s: the pause ends at 180 s
    assert kairos.asked == [_sid(a), _sid(b), _sid(c), _sid(d), _sid(b)]
    assert clock.slept == [60.0]
    assert b["kairos"]["first"]["error"] == "Kairos gave no usable answer (HTTP 429)"
    assert all("kairos_call" in s for s in rec["stocks"])


def test_kairos_time_running_out_ends_its_pass_of_one_more_asks() -> None:
    kairos = Kairos(down={1}, clock=Clock())
    # 30 s for the first ask, 120 s of pause, then 40 s: one more ask fits, and one starts
    ctx = replace(_ctx(kairos=kairos), kairos_budget_s=190.0)
    rec, _ = _four(kairos, ctx)
    a, b, c, d = rec["stocks"]
    assert kairos.asked == [_sid(a), _sid(a), _sid(b)] and kairos.left == [190.0, 40.0, 10.0]
    assert "kairos_call" in a and "kairos_call" in b
    for stock in (c, d):
        assert stock["kairos"]["again"] == f"not asked again: {daily.KAIROS_OUT}"


class Stuck:
    """A Kairos ask that never ends on its own (a server trickling its reply)."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.asked = 0

    def __call__(self, *a: Any, **kw: Any) -> Result:
        self.asked += 1
        self.release.wait(10)
        return _good()


def test_a_kairos_ask_past_its_time_is_left_unanswered() -> None:
    stuck = Stuck()
    ctx = replace(_ctx(), kairos=stuck, kairos_budget_s=1.0, kairos_grace_s=0.05)
    try:
        rec, _ = _four(Kairos(), ctx)
    finally:
        stuck.release.set()
    a, *rest = rec["stocks"]
    assert stuck.asked == 1
    assert a["kairos"] == {
        "error": daily.KAIROS_LATE,
        "again": f"not asked again: {daily.KAIROS_OUT}",
    }
    assert [s["error"] for s in a["steps"] if s["name"] == "kairos"] == [daily.KAIROS_LATE]
    for stock in rest:
        assert stock["kairos"] == {"error": f"not asked: {daily.KAIROS_OUT}"}
    assert rec["complete"] is True
    assert all("call" in s and s["opus"]["call"] for s in rec["stocks"])


def test_kairos_time_bound_ends_inside_the_calls_steps_limit() -> None:
    import yaml

    flow = yaml.safe_load(Path(".github/workflows/daily.yml").read_text())
    [calls] = [s for s in flow["jobs"]["daily"]["steps"] if s.get("id") == "calls"]
    # the last Kairos request starts by KAIROS_LAST_S and is left unanswered after its grace
    assert daily.Context.kairos_grace_s == jev.KAIROS_TIMEOUT_S
    assert calls["timeout-minutes"] * 60 > daily.KAIROS_LAST_S + jev.KAIROS_TIMEOUT_S
    assert daily.KAIROS_LAST_S > daily.OPUS_BUDGET_S  # Opus's time is never Kairos's
    assert calls["env"]["KAIROS_API_KEY"] == "${{ secrets.KAIROS_API_KEY }}"
    others = [s for s in flow["jobs"]["daily"]["steps"] if s is not calls]
    assert not any("KAIROS_API_KEY" in (s.get("env") or {}) for s in others)


class Broken:
    def __call__(self, *a: Any, **kw: Any) -> Result:
        raise RuntimeError("Kairos adapter bug")


@pytest.mark.parametrize("kairos", [lambda: Kairos(down=ALWAYS), Broken])
def test_a_kairos_failure_changes_no_other_call_and_not_the_exit_code(
    tmp_path: Path, kairos: Any
) -> None:
    assert _run(tmp_path / "well", ctx=_ctx(kairos=Kairos())) == 0
    well = json.loads(record_path(tmp_path / "well", DAY).read_text())
    ctx = _ctx()
    ctx.kairos = kairos()
    outbox = Outbox()
    assert _run(tmp_path / "ill", ctx=ctx, mailer=outbox) == 0
    ill = json.loads(record_path(tmp_path / "ill", DAY).read_text())
    for good, bad in zip(well["stocks"], ill["stocks"], strict=True):
        assert bad["call"] == good["call"] and "kairos_call" not in bad
        for key in ("call", "confidence", "lean", "kept", "dropped"):
            assert bad["opus"][key] == good["opus"][key]
        assert bad["why"]["kept"] == good["why"]["kept"]
    if isinstance(ctx.kairos, Broken):  # a failure in code is asked once more too
        failed = {"error": "failed in code (RuntimeError)"}
        assert ill["stocks"][0]["kairos"] == failed | {"first": failed}
    assert "| no call |" in outbox.sent[0].text


def test_a_kairos_call_alone_still_fails_the_run(tmp_path: Path) -> None:
    # Kairos never changes the exit code: with no call from Jev or Opus on any
    # stock the run exits 1, as on main, though Kairos made every call
    ctx = _ctx(Jev(down_after=0), kairos=Kairos())
    ctx.claude = None
    assert _run(tmp_path, ctx=ctx) == 1
    stocks = _record(tmp_path)["stocks"]
    assert all("kairos_call" in s for s in stocks)
    assert not any(daily.called(s) for s in stocks)


def test_without_a_key_kairos_is_noted_as_not_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAIROS_API_KEY")
    kairos, outbox = Kairos(), Outbox()
    assert _run(tmp_path, ctx=_ctx(kairos=kairos), mailer=outbox) == 0
    rec = _record(tmp_path)
    assert kairos.asked == [] and rec["kairos_on"] is False and rec["kairos_input_tokens"] == 0
    for stock in rec["stocks"]:
        assert stock["kairos"] == {"error": "not asked: KAIROS_API_KEY not set"}
    assert "Kairos made no call (not asked: KAIROS_API_KEY not set)." in outbox.sent[0].text
    # no Kairos call today: the subject keeps the two-call wording
    assert (
        outbox.sent[0].subject == "tenq daily, session 2026-11-04: the two calls agree on 0 of 20"
    )
    # the real adapter, keyless, is Kairos's and asks nothing
    result = daily.ask_kairos("Facts.", QUESTIONS)
    assert result.error == "Kairos skipped: KAIROS_API_KEY not set" and not result.down


def _calls(jev: str | None, opus: str | None, kairos: str | None) -> dict[str, Any]:
    stock: dict[str, Any] = {"ticker": "ACME"}
    if jev:
        stock["call"] = {"call": jev}
    if opus:
        stock["opus"] = {"call": opus}
    if kairos:
        stock["kairos_call"] = {"call": kairos}
    return stock


@pytest.mark.parametrize(
    ("jev_call", "opus_call", "kairos_call", "shown"),
    [
        ("Buy", "Buy", "Buy", "all agree"),
        ("Buy", "Buy", "Sell", "Kairos differs"),
        ("Hold", "Short", "Hold", "Opus differs"),
        ("Sell", "Buy", "Buy", "Jev differs"),
        ("Buy", "Hold", "Sell", "all differ"),
        ("Buy", "Buy", None, "agree"),
        (None, "Hold", "Hold", "agree"),
        ("Buy", None, "Sell", "differ"),
        ("Buy", None, None, ""),
        (None, None, "Sell", ""),
        (None, None, None, ""),
    ],
)
def test_the_agreement_column(
    jev_call: str | None, opus_call: str | None, kairos_call: str | None, shown: str
) -> None:
    assert agreement(_calls(jev_call, opus_call, kairos_call)) == shown


def test_the_subject_counts_only_stocks_with_all_three_calls() -> None:
    stocks = [
        _calls("Buy", "Buy", "Buy"),
        _calls("Buy", "Hold", "Buy"),
        _calls("Sell", "Sell", "Sell"),
        _calls("Hold", "Hold", None),  # two calls agreeing: not counted
    ]
    record = {"day": "2026-11-04", "kairos_on": True, "stocks": stocks}
    assert subject_for(record) == "tenq daily, session 2026-11-04: all three agree on 2 of 3"


def test_with_no_kairos_call_the_subject_counts_the_two_calls() -> None:
    stocks = [
        _calls("Buy", "Buy", None),
        _calls("Buy", "Hold", None),
        _calls("Sell", None, None),  # one call: not counted
    ]
    record = {"day": "2026-11-04", "kairos_on": True, "stocks": stocks}
    assert subject_for(record) == "tenq daily, session 2026-11-04: the two calls agree on 1 of 2"
    # one Kairos call anywhere turns the day to three-call counting
    stocks.append(_calls("Hold", "Hold", "Hold"))
    assert subject_for(record) == "tenq daily, session 2026-11-04: all three agree on 1 of 1"


def test_kairos_down_all_day_keeps_the_two_call_subject(tmp_path: Path) -> None:
    outbox = Outbox()
    ctx = _ctx(kairos=Kairos(down=ALWAYS))
    assert _run(tmp_path, ctx=ctx, mailer=outbox) == 0
    rec = _record(tmp_path)
    assert rec["kairos_on"] is True and not any("kairos_call" in s for s in rec["stocks"])
    email = outbox.sent[0]
    assert email.subject == "tenq daily, session 2026-11-04: the two calls agree on 0 of 20"
    # the table and the notes keep their three-method layout
    assert "Stock | Jev | Opus | Kairos | agreement" in email.text
    assert "Kairos made no call (" in email.text


def test_the_disclaimer_names_the_three_methods() -> None:
    assert "Three methods" in DISCLAIMER and "Kairos" in DISCLAIMER and "Opus" in DISCLAIMER


def test_a_record_from_before_kairos_renders_exactly_as_it_did() -> None:
    record = json.loads((FIXTURE / "record.json").read_text())
    assert not any("kairos" in key for key in record)
    email = render(record)
    assert email.subject == (FIXTURE / "subject.txt").read_text()
    assert email.text == (FIXTURE / "text.txt").read_text()
    assert email.html == (FIXTURE / "html.txt").read_text()


def test_a_record_from_before_kairos_is_resent_exactly_as_it_was(tmp_path: Path) -> None:
    record = json.loads((FIXTURE / "record.json").read_text())
    path = record_path(tmp_path, DAY)
    daily._write(path, record | {"email": {"sent": False, "status": "email failed"}})
    assert daily.claim(path, "job-1") == 0
    outbox = Outbox()
    assert daily.send_claimed(path, "job-1", outbox, _no_trace) == 0
    [email] = outbox.sent
    assert email.subject == (FIXTURE / "subject.txt").read_text()
    assert email.text == (FIXTURE / "text.txt").read_text()
    assert email.html == (FIXTURE / "html.txt").read_text()


def test_each_kairos_ask_is_its_own_generation_in_the_stocks_trace() -> None:
    rec, _ = _four(Kairos(down={2}, clock=Clock()))
    tracing, exporter = _memory()
    emit_daily(rec["stocks"][1], rec, tracing.tracer)
    names = {s.name: s.attributes or {} for s in exporter.get_finished_spans()}
    first, again = names["kairos"], names["kairos again"]
    assert first["langfuse.observation.type"] == again["langfuse.observation.type"] == "generation"
    assert first["langfuse.observation.model.name"] == "DMIND/Kairos"  # down: no model resolved
    assert json.loads(str(first["langfuse.observation.output"]))["call"] is None
    assert again["langfuse.observation.model.name"] == "DMIND/Kairos-0.9"
    assert json.loads(str(again["langfuse.observation.usage_details"])) == {
        "input": 1000,
        "output": 50,
    }
    assert json.loads(str(again["langfuse.observation.output"]))["call"]["call"] == "Buy"
    root = names[f"daily {rec['stocks'][1]['ticker']}"]
    assert json.loads(str(root["langfuse.observation.output"]))["kairos"] == "Buy"


def test_the_words_of_a_kairos_reply_never_reach_the_email() -> None:
    from test_kairos import BUSY

    def kairos(state: str, questions: Any, **kw: Any) -> Result:
        def post(url: str, headers: dict[str, str], data: bytes) -> tuple[int, str]:
            return 503, BUSY

        return jev.ask(state, questions, provider=jev.KAIROS, post=post, sleep=lambda s: None, **kw)

    rec, _ = _four(Kairos(), replace(_ctx(), kairos=kairos))
    first = rec["stocks"][0]["kairos"]
    assert "https://bit.ly/x9" in first["error"]  # the record keeps it whole
    email = render(rec)
    for part in (email.subject, email.text, email.html):
        for word in (
            "bit.ly", "evil.example", "Service busy", "req-123", "&lt;a ", '<a href="',
            "channel_unavailable",
        ):  # fmt: skip
            assert word not in part
    short = "HTTP 503"
    assert (
        f"Kairos made no call (asked again at the end of the run: Kairos gave no usable answer ({short}))."
        in email.text
    )
    # only the first stock was asked; the rest were skipped while Kairos was down
    assert email.text.count(short) == email.html.count(short) == 1


def test_a_kairos_call_asked_again_shows_its_first_error_short() -> None:
    from tenq.daily_email import _kairos_notes

    error = "Kairos gave no usable answer (HTTP 503 [busy] see https://bit.ly/x9)"
    stock = {"kairos_call": {"call": "Buy"}, "kairos": {"first": {"error": error}}}
    assert _kairos_notes(stock) == [
        "Kairos's call came when it was asked again at the end of the run "
        "(Kairos gave no usable answer (HTTP 503))."
    ]
