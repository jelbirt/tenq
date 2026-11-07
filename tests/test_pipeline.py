from __future__ import annotations

import html
import json
import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tenq.brief import Claim, Draft, Prompt
from tenq.checker import Checked, CheckResult, Dropped
from tenq.deliver import DISCLAIMER, Email, render
from tenq.filings import Fact, FilingMeta, FilingStore, Passage
from tenq.llm import AgentRun, ToolCall
from tenq.metrics import metrics_for
from tenq.run import brief_filing, briefed, email_state, record_path, run_ticker
from tenq.trace import Tracing, connect, emit, send_trace

LATEST = "CRWD-0001535527-26-000031"
GOOD = Claim("{fin:CRWD:revenue:yoy_pct}, as customers adopted more modules.", f"{LATEST}:23")
GOOD_SENT = "CRWD's revenue for the quarter ended 2026-07-31 rose 25.83% from a year earlier, as customers adopted more modules."
BAD = Claim("Revenue grew 40%.", f"{LATEST}:23")
PROMPT = Prompt("brief-test", "rules")


class StoreSource:
    """A FilingSource that lists what the fixture store already holds."""

    def __init__(self, store: FilingStore) -> None:
        self.store = store

    def list_filings(self, ticker: str, limit: int) -> list[FilingMeta]:
        return self.store.filings(ticker)[:limit]

    def fetch(self, meta: FilingMeta) -> tuple[str, list[Fact]]:
        raise AssertionError("everything is already saved")

    def accepted_at(self, meta: FilingMeta) -> str:
        return "2026-11-03T16:05:00-05:00"


def _agent(error: str | None = None) -> AgentRun:
    run = AgentRun(
        model="opus", text=None if error else "{}", error=error, started_at=1.0, ended_at=2.0
    )
    run.tool_calls = [ToolCall("t1", "mcp__tenq__get_metrics", {}, "{...}")]
    run.usage = {"input_tokens": 100, "output_tokens": 50}
    return run


def _briefer(claims: list[Claim], error: str | None = None) -> Any:
    def go(
        root: Path, meta: FilingMeta, url: str, prompt: Prompt, model: str
    ) -> tuple[AgentRun, Draft]:
        return _agent(error), Draft(claims=[] if error else claims, error=error)

    return go


class Outbox:
    def __init__(self, ok: bool = True) -> None:
        self.sent: list[Email] = []
        self.ok = ok

    def __call__(self, email: Email) -> tuple[bool, str]:
        self.sent.append(email)
        return self.ok, "sent" if self.ok else "smtp down"


def _no_trace(record: dict[str, Any]) -> tuple[bool, str]:
    json.dumps(record, default=str)  # the record must serialise
    return False, "trace skipped"


def test_a_checked_brief_is_emailed_and_recorded(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_SHA", "the-commit-that-triggered-it")
    monkeypatch.setenv("TENQ_CODE_SHA", "abc123")
    meta = store.filings("CRWD")[0]
    outbox = Outbox()
    rec = brief_filing(
        meta, store, tmp_path / "runs", "http://x", prompt=PROMPT,
        briefer=_briefer([GOOD, BAD]), sender=outbox, tracer=_no_trace,
    )  # fmt: skip
    assert rec["mode"] == "brief"
    (kept,) = rec["check"]["kept"]
    assert kept["claim"] == GOOD_SENT and kept["draft"] == GOOD.claim
    assert kept["numbers"] == ("fin:CRWD:revenue:yoy_pct",)
    assert "a figure typed" in rec["check"]["dropped"][0]["reason"]
    (email,) = outbox.sent
    assert GOOD_SENT in email.text and "40%" not in email.text and "{" not in email.text
    assert "1 claim(s) failed the checks" in email.text
    assert DISCLAIMER in email.text and html.escape(DISCLAIMER) in email.html
    assert email.subject.startswith("tenq: CRWD 10-Q") and "+25.8% YoY" in email.subject
    saved = json.loads(record_path(tmp_path / "runs", meta).read_text())
    assert saved["run_id"] == rec["run_id"] and saved["prompt_version"] == "brief-test"
    assert saved["code"] == "abc123"  # the commit checked out, not the trigger's
    assert saved["agent"]["usage"]["output_tokens"] == 50


@pytest.mark.parametrize("url", [None, "http://x"])
def test_no_search_or_a_failed_claude_still_emails_numbers(
    store: FilingStore, tmp_path: Path, url: str | None
) -> None:
    meta = store.filings("CRWD")[0]
    outbox = Outbox()
    rec = brief_filing(
        meta, store, tmp_path, url, prompt=PROMPT,
        briefer=_briefer([], error="timed out after 900s"), sender=outbox, tracer=_no_trace,
    )  # fmt: skip
    assert rec["mode"] == "numbers-only"
    (email,) = outbox.sent
    assert "numbers only" in email.text and "Revenue" in email.text
    assert "$1.47B" in email.text


def test_model_and_filing_text_are_escaped_in_html(
    store: FilingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = store.filings("CRWD")[0]
    table = metrics_for(store, "CRWD")
    evil = Checked("<script>x</script> grew", f"{LATEST}:23", (), "<script>x</script> grew")
    real = store.passage(evil.passage_id)
    assert real is not None
    planted = Passage(real.id, real.doc, real.line, "<img src=x onerror=alert(1)> revenue")
    monkeypatch.setattr(store, "passage", lambda pid: planted)
    email = render(meta, table, CheckResult(kept=[evil]), store)
    assert "<script>" not in email.html and "&lt;script&gt;" in email.html
    assert "<img" not in email.html and "&lt;img" in email.html


def _memory_tracing(found: Any = lambda trace_id: True) -> tuple[Tracing, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Tracing(provider.get_tracer("tenq"), lambda: None, found), exporter


def _record(store: FilingStore, tmp_path: Path) -> dict[str, Any]:
    meta = store.filings("CRWD")[0]
    return brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT,
        briefer=_briefer([GOOD]), sender=Outbox(), tracer=_no_trace,
    )  # fmt: skip


def _set_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://lf.example/")


def test_trace_has_trace_steps_generation_and_tools(store: FilingStore, tmp_path: Path) -> None:
    rec = _record(store, tmp_path)
    tracing, exporter = _memory_tracing()
    trace_id = emit(rec, tracing.tracer)
    spans = {s.name: s for s in exporter.get_finished_spans()}

    root, gen, tool = spans["brief"], spans["claude -p"], spans["tool mcp__tenq__get_metrics"]
    assert root.parent is None and len(trace_id) == 32
    assert {format(s.context.trace_id, "032x") for s in spans.values()} == {trace_id}
    assert gen.parent is not None and gen.parent.span_id == root.context.span_id
    assert tool.parent is not None and tool.parent.span_id == gen.context.span_id
    g = dict(gen.attributes or {})
    assert g["langfuse.observation.type"] == "generation"
    assert g["langfuse.observation.model.name"] == "opus"
    assert json.loads(str(g["langfuse.observation.usage_details"]))["output"] == 50
    assert g["langfuse.observation.metadata.prompt_version"] == "brief-test"
    assert g["langfuse.trace.name"] == "brief"  # trace attributes on every span
    assert g["session.id"] == "CRWD"
    assert g["langfuse.trace.metadata.run_id"] == rec["run_id"]  # the dashed form, searchable
    assert dict(tool.attributes or {})["langfuse.observation.type"] == "tool"
    # each span keeps the time it really took, not the time it was sent
    assert gen.start_time == 1_000_000_000 and gen.end_time == 2_000_000_000
    assert root.start_time == int(rec["started_at"] * 1e9)
    assert all((s.end_time or 0) >= (s.start_time or 0) for s in spans.values())


def test_failed_steps_are_marked_as_errors(store: FilingStore, tmp_path: Path) -> None:
    meta = store.filings("CRWD")[0]
    rec = brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT,
        briefer=_briefer([], error="timed out after 900s"), sender=Outbox(), tracer=_no_trace,
    )  # fmt: skip
    tracing, exporter = _memory_tracing()
    emit(rec, tracing.tracer)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    for name in ("claude", "claude -p"):
        attrs = dict(spans[name].attributes or {})
        assert attrs["langfuse.observation.level"] == "ERROR"
        assert attrs["langfuse.observation.status_message"] == "timed out after 900s"


def test_the_langfuse_sdk_sends_tenq_spans(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the real SDK client, with its upload swapped for memory and sockets blocked:
    # proves the SDK exports tenq's spans (its default filter would drop them)
    def no_network(*a: Any, **k: Any) -> None:
        raise OSError("tests never touch the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    rec = _record(store, tmp_path)
    exporter = InMemorySpanExporter()
    tracing = connect(f"pk-{uuid.uuid4()}", "sk", "https://lf.example", exporter=exporter)
    trace_id = emit(rec, tracing.tracer)
    tracing.flush()
    spans = exporter.get_finished_spans()
    assert {s.name for s in spans} >= {"brief", "claude -p", "check", "email"}
    assert {format(s.context.trace_id, "032x") for s in spans} == {trace_id}
    assert all(s.instrumentation_scope and s.instrumentation_scope.name == "tenq" for s in spans)


def test_a_trace_counts_as_stored_only_once_langfuse_has_it(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _record(store, tmp_path)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    assert send_trace(rec) == (False, "trace skipped: Langfuse env not set")
    _set_langfuse_env(monkeypatch)
    hosts: list[str] = []
    lookups: list[str] = []

    def found_second_time(trace_id: str) -> bool:
        lookups.append(trace_id)
        return len(lookups) >= 2

    tracing, exporter = _memory_tracing(found_second_time)

    def connected(public: str, secret: str, host: str) -> Tracing:
        hosts.append(host)
        return tracing

    ok, status = send_trace(rec, connect=connected, sleep=lambda s: None)
    trace_id = format(exporter.get_finished_spans()[0].context.trace_id, "032x")
    assert (ok, status) == (True, f"trace stored: {trace_id}")
    assert lookups == [trace_id, trace_id] and hosts == ["https://lf.example"]
    assert "lf.example" not in status and "sk" not in status

    never, _ = _memory_tracing(lambda trace_id: False)
    ok, status = send_trace(rec, connect=lambda *a: never, wait=0, sleep=lambda s: None)
    assert not ok and status.startswith("trace sent, not found after 0s: ")


def test_send_trace_fails_open(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _record(store, tmp_path)
    _set_langfuse_env(monkeypatch)

    def down(public: str, secret: str, host: str) -> Tracing:
        raise OSError("down")

    assert send_trace(rec, connect=down) == (False, "trace failed: OSError")
    tracing, _ = _memory_tracing()
    assert send_trace({"run_id": "r"}, connect=lambda *a: tracing) == (
        False,
        "trace failed: KeyError",
    )

    def broken(trace_id: str) -> bool:
        raise ConnectionError("lookup down")

    dark, _ = _memory_tracing(broken)
    ok, status = send_trace(rec, connect=lambda *a: dark, wait=0, sleep=lambda s: None)
    assert not ok and "last lookup failed: ConnectionError" in status


SECRETS = {
    "SMTP_USER": "smtp-user-sentinel@example.com",
    "SMTP_PASS": "smtp-pass-sentinel",
    "RECIPIENT_EMAILS": "reader-sentinel@example.com",
    "LANGFUSE_PUBLIC_KEY": "pk-sentinel",
    "LANGFUSE_SECRET_KEY": "sk-sentinel",
    "LANGFUSE_HOST": "https://langfuse-sentinel.example",
    "EDGAR_IDENTITY": "edgar-sentinel@example.com",
}


def test_no_secret_reaches_the_public_record_or_the_trace(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    tracing, exporter = _memory_tracing()

    meta = store.filings("CRWD")[0]
    rec = brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT, briefer=_briefer([GOOD, BAD]),
        sender=Outbox(), tracer=lambda r: send_trace(r, connect=lambda *a: tracing),
    )  # fmt: skip
    assert rec["trace"]["sent"] is True and rec["trace"]["status"].startswith("trace stored: ")
    written = record_path(tmp_path, meta).read_text()
    sent = json.dumps([dict(s.attributes or {}) for s in exporter.get_finished_spans()])
    for value in SECRETS.values():
        assert value not in written and value not in sent
        # the host part of an address too: no domain of a secret leaks
        assert value.split("@")[-1].split("//")[-1] not in written


def test_a_record_is_on_disk_before_the_trace_is_sent(store: FilingStore, tmp_path: Path) -> None:
    meta = store.filings("CRWD")[0]
    seen: list[dict[str, Any]] = []

    def tracer(record: dict[str, Any]) -> tuple[bool, str]:
        seen.append(json.loads(record_path(tmp_path, meta).read_text()))
        return True, "trace stored: abc"

    brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT,
        briefer=_briefer([GOOD]), sender=Outbox(), tracer=tracer,
    )  # fmt: skip
    assert seen[0]["email"]["sent"] is True and seen[0]["trace"]["status"] == "trace pending"
    assert json.loads(record_path(tmp_path, meta).read_text())["trace"]["sent"] is True


@contextmanager
def _fake_node() -> Iterator[str]:
    yield "http://127.0.0.1:1"


@contextmanager
def _dead_node() -> Iterator[str]:
    raise RuntimeError("xerj not found on PATH")
    yield ""  # pragma: no cover


def test_run_briefs_only_unrecorded_filings_once(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tenq.run.Xerj.index", lambda self, ticker, state: None)
    runs = tmp_path / "runs"
    outbox = Outbox()
    kw: dict[str, Any] = dict(
        prompt=PROMPT, briefer=_briefer([GOOD]), sender=outbox, tracer=_no_trace
    )
    recs = run_ticker("CRWD", StoreSource(store), store, runs, node=_fake_node, **kw)
    assert [r["filing"]["accession"] for r in recs] == ["0001535527-26-000031"]
    assert run_ticker("CRWD", StoreSource(store), store, runs, node=_fake_node, **kw) == []
    recs = run_ticker("CRWD", StoreSource(store), store, runs, backfill=3, node=_fake_node, **kw)
    assert [r["filing"]["period"] for r in recs] == ["2026-01-31", "2026-04-30"]  # oldest first
    assert len(outbox.sent) == 3
    assert [s["name"] for s in recs[0]["steps"]][:2] == ["fetch", "index"]


def test_run_without_xerj_goes_numbers_only(store: FilingStore, tmp_path: Path) -> None:
    outbox = Outbox()
    (rec,) = run_ticker(
        "CRWD", StoreSource(store), store, tmp_path, node=_dead_node,
        prompt=PROMPT, briefer=_briefer([GOOD]), sender=outbox, tracer=_no_trace,
    )  # fmt: skip
    assert rec["mode"] == "numbers-only" and rec["agent"] is None
    index = next(s for s in rec["steps"] if s["name"] == "index")
    assert "xerj not found" in index["error"]


@pytest.mark.parametrize("email_on, ok", [(True, False), (False, True)])
def test_a_filing_whose_email_did_not_go_out_is_tried_again(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, email_on: bool, ok: bool
) -> None:
    monkeypatch.setattr("tenq.run.Xerj.index", lambda self, ticker, state: None)
    kw: dict[str, Any] = dict(prompt=PROMPT, briefer=_briefer([GOOD]), tracer=_no_trace)
    (first,) = run_ticker(
        "CRWD", StoreSource(store), store, tmp_path, node=_fake_node,
        sender=Outbox(ok=ok), email=email_on, **kw,
    )  # fmt: skip
    assert first["email"]["sent"] is False  # SMTP down, or a run with email off
    outbox = Outbox()
    (again,) = run_ticker("CRWD", StoreSource(store), store, tmp_path, node=_fake_node,
                          sender=outbox, **kw)  # fmt: skip
    assert again["filing"] == first["filing"] and len(outbox.sent) == 1
    assert briefed(record_path(tmp_path, store.filings("CRWD")[0]))


@pytest.mark.parametrize("email_on, ok", [(False, True), (True, False)])
def test_a_rerun_that_sent_nothing_keeps_the_sent_record(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, email_on: bool, ok: bool
) -> None:
    monkeypatch.setattr("tenq.run.Xerj.index", lambda self, ticker, state: None)
    kw: dict[str, Any] = dict(prompt=PROMPT, briefer=_briefer([GOOD]), tracer=_no_trace)
    (first,) = run_ticker("CRWD", StoreSource(store), store, tmp_path, node=_fake_node,
                          sender=Outbox(), **kw)  # fmt: skip
    meta = store.filings("CRWD")[0]
    (rerun,) = run_ticker(
        "CRWD", StoreSource(store), store, tmp_path, node=_fake_node, accession=meta.accession,
        sender=Outbox(ok=ok), email=email_on, **kw,
    )  # fmt: skip
    assert rerun["email"]["sent"] is False
    saved = json.loads(record_path(tmp_path, meta).read_text())
    assert saved["run_id"] == first["run_id"] and briefed(record_path(tmp_path, meta))
    side = tmp_path / "CRWD" / f"{meta.accession}.rerun-{rerun['run_id']}.json"
    assert json.loads(side.read_text())["run_id"] == rerun["run_id"]
    outbox = Outbox()
    assert run_ticker("CRWD", StoreSource(store), store, tmp_path, node=_fake_node,
                      sender=outbox, **kw) == [] and outbox.sent == []  # fmt: skip


def test_an_unreadable_record_does_not_count_as_briefed(tmp_path: Path) -> None:
    (tmp_path / "r.json").write_text("{not json")
    (tmp_path / "s.json").write_text('{"email": "sent"}')
    assert not briefed(tmp_path / "r.json") and not briefed(tmp_path / "s.json")
    assert not briefed(tmp_path / "missing.json")


def test_a_named_accession_must_be_saved(store: FilingStore, tmp_path: Path) -> None:
    with pytest.raises(LookupError):
        run_ticker("CRWD", StoreSource(store), store, tmp_path, accession="0000000000-00-000000",
                   node=_fake_node, prompt=PROMPT)  # fmt: skip


def test_dropped_reasons_are_kept_in_order() -> None:
    r = CheckResult(dropped=[Dropped(BAD, "a"), Dropped(GOOD, "b")])
    assert [d.reason for d in r.dropped] == ["a", "b"]


def _brief(store: FilingStore, runs: Path, **kw: Any) -> dict[str, Any]:
    meta = store.filings("CRWD")[0]
    kw.setdefault("sender", Outbox())
    kw.setdefault("briefer", _briefer([GOOD]))
    kw.setdefault("tracer", _no_trace)
    return brief_filing(meta, store, runs, "http://x", prompt=PROMPT, **kw)


def test_the_record_reaches_main_before_the_email_goes_out(
    store: FilingStore, tmp_path: Path
) -> None:
    seen: list[str] = []
    outbox = Outbox()

    def claimer(path: Path) -> bool:
        record = json.loads(path.read_text())
        seen.append(record["email"]["status"])
        assert not outbox.sent  # nothing sent before the claim is saved
        return True

    rec = _brief(store, tmp_path / "runs", sender=outbox, claimer=claimer)
    assert seen == ["sending"] and len(outbox.sent) == 1 and rec["email"]["sent"] is True


def test_a_claim_that_could_not_be_saved_sends_nothing(store: FilingStore, tmp_path: Path) -> None:
    outbox = Outbox()
    rec = _brief(store, tmp_path / "runs", sender=outbox, claimer=lambda p: False)
    assert not outbox.sent and rec["email"]["sent"] is False
    assert "could not be saved to main first" in rec["email"]["status"]
    # a plain failure: the next run tries again
    assert email_state(record_path(tmp_path / "runs", store.filings("CRWD")[0])) == "retry"


def test_email_off_claims_nothing(store: FilingStore, tmp_path: Path) -> None:
    def claimer(path: Path) -> bool:
        raise AssertionError("no claim when no email goes out")

    rec = _brief(store, tmp_path / "runs", email=False, claimer=claimer)
    assert rec["email"] == {**rec["email"], "sent": False, "status": "email off for this run"}


@pytest.mark.parametrize(
    ("content", "state"),
    [
        (None, "none"),
        ('{"email": {"sent": true}}', "sent"),
        ('{"email": {"sent": false}}', "retry"),
        ('{"email": {"sent": null, "status": "sending"}}', "person"),
        ('{"email": {"sent": null, "status": "email may have gone out"}}', "person"),
        ('{"email": {"sent": "yes"}}', "person"),
        ('{"email": {"se', "person"),  # cut off mid-write
    ],
)
def test_email_state(tmp_path: Path, content: str | None, state: str) -> None:
    path = tmp_path / "r.json"
    if content is not None:
        path.write_text(content)
    assert email_state(path) == state


def test_a_record_that_may_have_gone_out_is_never_resent(
    store: FilingStore, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    meta = store.filings("CRWD")[0]
    path = record_path(runs, meta)
    path.parent.mkdir(parents=True)
    path.write_text('{"email": {"sent": null, "status": "email may have gone out"}}')
    outbox = Outbox()
    held: list[str] = []
    got = run_ticker(
        "CRWD", StoreSource(store), store, runs, node=_fake_node, held=held,
        prompt=PROMPT, briefer=_briefer([GOOD]), sender=outbox, tracer=_no_trace,
        claimer=lambda p: True,
    )  # fmt: skip
    assert got == [] and not outbox.sent and held == [f"CRWD {meta.accession}"]
    assert "may have gone out" in path.read_text()  # left for a person


def test_a_maybe_sent_answer_is_recorded_as_needing_a_person(
    store: FilingStore, tmp_path: Path
) -> None:
    rec = _brief(
        store, tmp_path / "runs", claimer=lambda p: True,
        sender=lambda m: (None, "email may have gone out: TimeoutError while sending"),
    )  # fmt: skip
    assert rec["email"]["sent"] is None
    assert email_state(record_path(tmp_path / "runs", store.filings("CRWD")[0])) == "person"


def test_records_are_written_whole(tmp_path: Path) -> None:
    from tenq.run import _write

    path = tmp_path / "a" / "r.json"
    _write(path, {"email": {"sent": True}})
    assert json.loads(path.read_text()) == {"email": {"sent": True}}
    assert [p.name for p in path.parent.iterdir()] == ["r.json"]  # no temp file left


def test_a_rerun_of_a_held_filing_never_replaces_its_record(
    store: FilingStore, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    path = record_path(runs, store.filings("CRWD")[0])
    path.parent.mkdir(parents=True)
    held = '{"email": {"sent": null, "status": "email may have gone out"}}'
    path.write_text(held)
    rec = _brief(store, runs, claimer=lambda p: True)
    assert path.read_text() == held  # left for a person
    (rerun,) = path.parent.glob("*.rerun-*.json")
    assert json.loads(rerun.read_text())["run_id"] == rec["run_id"]


def test_a_held_filing_turns_the_job_red(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tenq.run

    def held_one(ticker: str, *a: object, held: list[str], **kw: object) -> list[object]:
        held.append(f"{ticker} 0001")
        return []

    monkeypatch.setattr(tenq.run, "run_ticker", held_one)
    monkeypatch.setattr("tenq.edgar.EdgarSource", lambda: None)
    args = ["--ticker", "CRWD", "--root", str(tmp_path), "--runs", str(tmp_path / "runs")]
    assert tenq.run.main(args) == 1
    monkeypatch.setattr(tenq.run, "run_ticker", lambda *a, **kw: [])
    assert tenq.run.main(args) == 0
