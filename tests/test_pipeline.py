from __future__ import annotations

import html
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tenq.brief import Claim, Draft, NumberRef, Prompt
from tenq.checker import CheckResult, Dropped
from tenq.deliver import DISCLAIMER, Email, render
from tenq.filings import Fact, FilingMeta, FilingStore, Passage
from tenq.llm import AgentRun, ToolCall
from tenq.metrics import metrics_for
from tenq.run import brief_filing, briefed, record_path, run_ticker
from tenq.trace import build_payload, send_trace

LATEST = "CRWD-0001535527-26-000031"
GOOD = Claim(
    "Revenue grew 26% from a year earlier.",
    f"{LATEST}:23",
    (NumberRef("revenue", "yoy_pct", 0.25830401932671315),),
)
BAD = Claim("Revenue grew 40%.", f"{LATEST}:23", (NumberRef("revenue", "yoy_pct", 0.4),))
PROMPT = Prompt("brief-test", "rules")


class StoreSource:
    """A FilingSource that lists what the fixture store already holds."""

    def __init__(self, store: FilingStore) -> None:
        self.store = store

    def list_filings(self, ticker: str, limit: int) -> list[FilingMeta]:
        return self.store.filings(ticker)[:limit]

    def fetch(self, meta: FilingMeta) -> tuple[str, list[Fact]]:
        raise AssertionError("everything is already saved")


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


def test_a_checked_brief_is_emailed_and_recorded(store: FilingStore, tmp_path: Path) -> None:
    meta = store.filings("CRWD")[0]
    outbox = Outbox()
    rec = brief_filing(
        meta, store, tmp_path / "runs", "http://x", prompt=PROMPT,
        briefer=_briefer([GOOD, BAD]), sender=outbox, tracer=_no_trace,
    )  # fmt: skip
    assert rec["mode"] == "brief"
    assert [c["claim"] for c in rec["check"]["kept"]] == [GOOD.claim]
    assert "claim says" in rec["check"]["dropped"][0]["reason"]
    (email,) = outbox.sent
    assert GOOD.claim in email.text and "40%" not in email.text
    assert "1 claim(s) failed the checks" in email.text
    assert DISCLAIMER in email.text and html.escape(DISCLAIMER) in email.html
    assert email.subject.startswith("tenq: CRWD 10-Q") and "+25.8% YoY" in email.subject
    saved = json.loads(record_path(tmp_path / "runs", meta).read_text())
    assert saved["run_id"] == rec["run_id"] and saved["prompt_version"] == "brief-test"
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
    evil = Claim("<script>x</script> grew", f"{LATEST}:23")
    real = store.passage(evil.passage_id)
    assert real is not None
    planted = Passage(real.id, real.doc, real.line, "<img src=x onerror=alert(1)> revenue")
    monkeypatch.setattr(store, "passage", lambda pid: planted)
    email = render(meta, table, CheckResult(kept=[evil]), store)
    assert "<script>" not in email.html and "&lt;script&gt;" in email.html
    assert "<img" not in email.html and "&lt;img" in email.html


def test_trace_batch_has_trace_steps_generation_and_tools(
    store: FilingStore, tmp_path: Path
) -> None:
    meta = store.filings("CRWD")[0]
    rec = brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT,
        briefer=_briefer([GOOD]), sender=Outbox(), tracer=_no_trace,
    )  # fmt: skip
    spans = build_payload(rec)["resourceSpans"][0]["scopeSpans"][0]["spans"]

    def attrs(span: dict[str, Any]) -> dict[str, Any]:
        return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}

    root = spans[0]
    assert root["name"] == "brief" and "parentSpanId" not in root
    assert {s["traceId"] for s in spans} == {root["traceId"]} and len(root["traceId"]) == 32
    gen = next(s for s in spans if s["name"] == "claude -p")
    g = attrs(gen)
    assert g["langfuse.observation.type"] == "generation"
    assert json.loads(g["langfuse.observation.usage_details"])["output"] == 50
    assert g["langfuse.observation.metadata.prompt_version"] == "brief-test"
    assert g["langfuse.trace.name"] == "brief"  # trace attributes on every span
    assert g["langfuse.trace.metadata.run_id"] == rec["run_id"]  # the dashed form, searchable
    tool = next(s for s in spans if s["name"] == "tool mcp__tenq__get_metrics")
    assert tool["parentSpanId"] == gen["spanId"] and gen["parentSpanId"] == root["spanId"]
    assert attrs(tool)["langfuse.observation.type"] == "tool"
    assert all(int(s["endTimeUnixNano"]) >= int(s["startTimeUnixNano"]) for s in spans)


def _boom(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    raise OSError("down")


def test_send_trace_fails_open(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = store.filings("CRWD")[0]
    rec = brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT,
        briefer=_briefer([GOOD]), sender=Outbox(), tracer=_no_trace,
    )  # fmt: skip
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    assert send_trace(rec)[0] is False
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://lf.example")
    assert send_trace(rec, post=_boom) == (False, "trace failed: OSError")
    assert send_trace({"run_id": "r"}, post=_boom) == (False, "trace failed: ValueError")


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
    posted: list[bytes] = []

    def post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        posted.append(body)
        return 200, "{}"

    meta = store.filings("CRWD")[0]
    rec = brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT, briefer=_briefer([GOOD, BAD]),
        sender=Outbox(), tracer=lambda r: send_trace(r, post=post),
    )  # fmt: skip
    assert rec["trace"] == {"sent": True, "status": "trace sent"}
    written = record_path(tmp_path, meta).read_text()
    for value in SECRETS.values():
        assert value not in written
        assert value.encode() not in posted[0]
        # the host part of an address too: no domain of a secret leaks
        assert value.split("@")[-1].split("//")[-1] not in written


def test_a_record_is_on_disk_before_the_trace_is_sent(store: FilingStore, tmp_path: Path) -> None:
    meta = store.filings("CRWD")[0]
    seen: list[dict[str, Any]] = []

    def tracer(record: dict[str, Any]) -> tuple[bool, str]:
        seen.append(json.loads(record_path(tmp_path, meta).read_text()))
        return True, "trace sent"

    brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT,
        briefer=_briefer([GOOD]), sender=Outbox(), tracer=tracer,
    )  # fmt: skip
    assert seen[0]["email"]["sent"] is True and seen[0]["trace"]["status"] == "trace pending"
    assert json.loads(record_path(tmp_path, meta).read_text())["trace"]["sent"] is True


def test_send_trace_posts_and_reads_partial_errors(
    store: FilingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = store.filings("CRWD")[0]
    rec = brief_filing(
        meta, store, tmp_path, "http://x", prompt=PROMPT,
        briefer=_briefer([GOOD]), sender=Outbox(), tracer=_no_trace,
    )  # fmt: skip
    for k, v in {
        "LANGFUSE_PUBLIC_KEY": "pk",
        "LANGFUSE_SECRET_KEY": "sk",
        "LANGFUSE_HOST": "https://lf.example/",
    }.items():
        monkeypatch.setenv(k, v)
    seen: dict[str, Any] = {}

    def post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        seen.update(url=url, headers=headers, payload=json.loads(body))
        return 200, json.dumps({"partialSuccess": {"rejectedSpans": 2, "errorMessage": "bad"}})

    ok, status = send_trace(rec, post=post)
    assert seen["url"] == "https://lf.example/api/public/otel/v1/traces"
    assert seen["headers"]["Authorization"].startswith("Basic ") and "sk" not in status
    assert seen["headers"]["x-langfuse-ingestion-version"] == "4"
    assert "resourceSpans" in seen["payload"]
    assert not ok and "2 span(s) rejected" in status

    def ok_post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        return 200, "{}"

    assert send_trace(rec, post=ok_post)[0] is True

    def http_error(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        return 401, "unauthorized"

    assert send_trace(rec, post=http_error)[0] is False


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
