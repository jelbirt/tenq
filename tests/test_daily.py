from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_rules import QUESTIONS, RULES, _good

from tenq import daily, opus
from tenq.config import load_watchlist
from tenq.daily import Context, main, record_path
from tenq.daily_email import render, sources
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


def _no_trace(record: dict[str, Any]) -> tuple[bool, str, dict[str, str]]:
    return False, "trace skipped: test", {}


def _ctx(jev: Jev | None = None, claude: Claude | None = None) -> Context:
    return Context(
        questions=QUESTIONS,
        rules=RULES,
        call_prompt=opus.load_call_prompt(),
        why_prompt=opus.load_why_prompt(),
        now=NOW,
        gather=_pack,
        ask=jev or Jev(),
        claude=claude if claude is not None else Claude(),
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


def test_no_opus_runs_jev_alone(tmp_path: Path) -> None:
    claude = Claude()
    assert _run(tmp_path, "--no-opus", ctx=_ctx(claude=claude)) == 0
    assert claude.calls == [] and _record(tmp_path)["opus_on"] is False


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
    ctx = _ctx(Jev(down_after=0))
    ctx.claude = None
    assert _run(tmp_path, ctx=ctx) == 1


def test_no_email_records_but_sends_nothing(tmp_path: Path) -> None:
    outbox = Outbox()
    assert _run(tmp_path, "--no-email", mailer=outbox) == 0
    assert outbox.sent == [] and _record(tmp_path)["email"]["status"].startswith("email skipped")


# the email


def _day_record(tmp_path: Path) -> dict[str, Any]:
    assert _run(tmp_path, "--no-email") == 0
    return _record(tmp_path)


def test_the_email_shows_both_calls_and_escapes_untrusted_text(tmp_path: Path) -> None:
    email = render(_day_record(tmp_path))
    assert email.subject == "tenq daily, session 2026-11-04: the two calls agree on 0 of 20"
    assert "CRWD | Buy 83% | Hold 50% | differ" in email.text
    assert "The outlook was raised <script>." in email.text
    assert "<script>" not in email.html and "&lt;script&gt;" in email.html
    assert "&lt;b&gt;Outlook&lt;/b&gt;" in email.html
    assert "href='https://www.reuters.com/a?b=1&amp;c=2'" in email.html
    assert "Shares jumped" not in email.text  # a dropped reason is never shown
    assert "1 reason(s) failed the checks" in email.text
    assert "Not investment advice" in email.text and "Not investment advice" in email.html


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
    assert set(names) == {"daily CRWD", "facts pack", "jev", "opus call", "opus why"}
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
    assert stored is False and set(ids) == {"CRWD", "DDOG"}
    assert status.startswith("traces sent, 1 of 2 found") and status.endswith("missing DDOG")


def test_the_daily_module_reads_no_secret_into_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("SMTP_PASS", "OPENROUTER_API_KEY", "LANGFUSE_SECRET_KEY", "EDGAR_IDENTITY"):
        monkeypatch.setenv(key, f"sentinel-{key}")
    _day_record(tmp_path)
    assert "sentinel-" not in record_path(tmp_path, DAY).read_text()
    assert daily.KIND == "daily"
