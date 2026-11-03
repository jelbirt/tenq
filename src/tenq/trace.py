"""Send one run's trace to Langfuse over OpenTelemetry (OTLP/HTTP, JSON).

One trace per briefed filing: a root span for the run, a span per pipeline
step, a generation for the Claude run (model, tokens, time, prompt version)
and a tool span per call under it. The run record
(runs/<TICKER>/<accession>.json) is the permanent copy; Langfuse is the
browsable one. Sending never raises: a tracing outage must not stop an email.

Langfuse's older /api/public/ingestion batch API is shut down on Langfuse
Cloud from 2026-11-16; OTLP is its replacement.

Env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

MAX_FIELD_CHARS = 20_000
OTLP_PATH = "/api/public/otel/v1/traces"
OTEL_ERROR = 2  # OTLP status code
USAGE_KEYS = {  # claude's usage keys to Langfuse's usage_details keys
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "cache_creation_input_tokens": "cache_creation_input_tokens",
}

Post = Callable[[str, dict[str, str], bytes], tuple[int, str]]


def _nanos(t: float) -> str:
    return str(int(t * 1_000_000_000))


def _json(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + " [cut]"


def _value(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, list):
        return {"arrayValue": {"values": [_value(x) for x in v]}}
    return {"stringValue": v if isinstance(v, str) else _json(v)}


def _attrs(d: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _value(v)} for k, v in d.items() if v is not None]


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


def build_spans(record: dict[str, Any]) -> list[dict[str, Any]]:
    """OTLP spans for one run record (see tenq.run for its shape)."""
    trace_id = uuid.UUID(record["run_id"]).hex
    filing = record["filing"]
    # trace-level attributes go on every span, as Langfuse asks
    common = {
        "langfuse.trace.name": "brief",
        "langfuse.session.id": filing["ticker"],
        "langfuse.trace.tags": [filing["ticker"], filing["form"], record["mode"]],
        "langfuse.version": record["prompt_version"],
        "langfuse.trace.metadata.run_id": record["run_id"],
        "langfuse.trace.metadata.accession": filing["accession"],
        "langfuse.trace.metadata.period": filing["period"],
    }

    def span(
        name: str,
        start: float,
        end: float,
        parent: str | None,
        attrs: dict[str, Any],
        error: str | None = None,
        span_id: str | None = None,
    ) -> dict[str, Any]:
        level = {"langfuse.observation.level": "ERROR"} if error else {}
        out: dict[str, Any] = {
            "traceId": trace_id,
            "spanId": span_id or _span_id(),
            "name": name,
            "kind": 1,
            "startTimeUnixNano": _nanos(start),
            "endTimeUnixNano": _nanos(max(end, start)),
            "attributes": _attrs({**common, **attrs, **level}),
            "status": {"code": OTEL_ERROR, "message": error} if error else {},
        }
        if parent:
            out["parentSpanId"] = parent
        return out

    root = _span_id()
    spans = [
        span(
            "brief",
            record["started_at"],
            record["ended_at"],
            None,
            {
                "langfuse.observation.type": "span",
                "langfuse.observation.input": _json(
                    {k: filing[k] for k in ("ticker", "form", "accession", "period")}
                ),
                "langfuse.observation.output": _json(
                    {
                        "mode": record["mode"],
                        "claims_kept": len(record["check"]["kept"]),
                        "claims_dropped": len(record["check"]["dropped"]),
                        "email": record["email"]["status"],
                    }
                ),
                "langfuse.observation.metadata.model": record["model"],
            },
            span_id=root,
        )
    ]
    for step in record["steps"]:
        spans.append(
            span(
                step["name"],
                step["start"],
                step["end"],
                root,
                {
                    "langfuse.observation.type": "span",
                    "langfuse.observation.output": _json(step.get("output")),
                },
                step.get("error"),
            )
        )
    agent = record.get("agent")
    if agent:
        gen = _span_id()
        usage = agent.get("usage") or {}
        spans.append(
            span(
                "claude -p",
                agent["started_at"],
                agent["ended_at"],
                root,
                {
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.model.name": agent.get("resolved_model")
                    or agent["model"],
                    "langfuse.observation.input": _json(record["prompt_input"]),
                    "langfuse.observation.output": _json(agent.get("text") or ""),
                    "langfuse.observation.usage_details": json.dumps(
                        {v: int(usage.get(k) or 0) for k, v in USAGE_KEYS.items()}
                    ),
                    "langfuse.observation.metadata.prompt_version": record["prompt_version"],
                    "langfuse.observation.metadata.num_turns": agent.get("num_turns"),
                    "langfuse.observation.metadata.api_duration_ms": agent.get("api_duration_ms"),
                    "langfuse.observation.metadata.tools_offered": agent.get("tools_offered"),
                },
                agent.get("error"),
                span_id=gen,
            )
        )
        for call in agent.get("tool_calls") or []:
            t = call.get("at") or agent["started_at"]
            spans.append(
                span(
                    f"tool {call['name']}",
                    t,
                    call.get("done_at") or t,
                    gen,
                    {
                        "langfuse.observation.type": "tool",
                        "langfuse.observation.input": _json(call.get("input")),
                        "langfuse.observation.output": _json(call.get("output")),
                    },
                    "tool error" if call.get("is_error") else None,
                )
            )
    return spans


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs({"service.name": "tenq"})},
                "scopeSpans": [{"scope": {"name": "tenq"}, "spans": build_spans(record)}],
            }
        ]
    }


def _post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.status), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # a 4xx body says what was wrong: keep it
        return exc.code, exc.read().decode("utf-8", "replace")


def send_trace(record: dict[str, Any], post: Post = _post) -> tuple[bool, str]:
    """Returns (sent, status); never raises."""
    public = os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
    secret = os.environ.get("LANGFUSE_SECRET_KEY") or ""
    host = (os.environ.get("LANGFUSE_HOST") or "").rstrip("/")
    if not (public and secret and host):
        return False, "trace skipped: Langfuse env not set"
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json",
        "x-langfuse-ingestion-version": "4",  # else Langfuse may hold the spans ~10 minutes
    }
    try:
        body = json.dumps(build_payload(record), default=str).encode()
        status, text = post(f"{host}{OTLP_PATH}", headers, body)
    except Exception as exc:
        log.warning("trace failed: %s", type(exc).__name__)
        return False, f"trace failed: {type(exc).__name__}"
    rejected = 0
    with contextlib.suppress(ValueError, AttributeError, TypeError):
        rejected = int((json.loads(text).get("partialSuccess") or {}).get("rejectedSpans") or 0)
    if status >= 300 or rejected:
        log.warning("trace rejected: HTTP %s, %s span(s): %s", status, rejected, text[:1000])
        return False, f"trace rejected: HTTP {status}, {rejected} span(s) rejected"
    # the host stays out of the status: the record is committed to a public repo
    log.info("trace sent: %s", trace_url(host, record["run_id"]))
    return True, "trace sent"


def trace_url(host: str, run_id: str) -> str:
    return f"{host}/trace/{uuid.UUID(run_id).hex}"
