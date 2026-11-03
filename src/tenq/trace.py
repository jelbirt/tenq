"""Send one run's trace to Langfuse through its public ingestion API.

One trace per briefed filing: a span per pipeline step, a generation for the
Claude run (model, tokens, time, prompt version) and a span per tool call
under it. The run record (runs/<TICKER>/<accession>.json) is the permanent
copy; Langfuse is the browsable one. Sending never raises: a tracing outage
must not stop an email.

Env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

MAX_FIELD_CHARS = 20_000

Post = Callable[[str, dict[str, str], bytes], tuple[int, str]]


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def _clip(value: Any) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return value if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + " [cut]"


def _event(kind: str, body: dict[str, Any], t: float) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "type": kind, "timestamp": _iso(t), "body": body}


def build_batch(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Ingestion events for one run record (see tenq.run for its shape)."""
    trace_id = record["run_id"]
    filing = record["filing"]
    started = record["started_at"]
    events = [
        _event(
            "trace-create",
            {
                "id": trace_id,
                "name": "brief",
                "timestamp": _iso(started),
                "input": {k: filing[k] for k in ("ticker", "form", "accession", "period")},
                "output": {
                    "mode": record["mode"],
                    "claims_kept": len(record["check"]["kept"]),
                    "claims_dropped": len(record["check"]["dropped"]),
                    "email": record["email"]["status"],
                },
                "metadata": {"prompt_version": record["prompt_version"], "model": record["model"]},
                "tags": [filing["ticker"], filing["form"], record["mode"]],
                "version": record["prompt_version"],
            },
            started,
        )
    ]
    for step in record["steps"]:
        events.append(
            _event(
                "span-create",
                {
                    "id": str(uuid.uuid4()),
                    "traceId": trace_id,
                    "name": step["name"],
                    "startTime": _iso(step["start"]),
                    "endTime": _iso(step["end"]),
                    "output": _clip(step.get("output")),
                    "level": "ERROR" if step.get("error") else "DEFAULT",
                    "statusMessage": step.get("error"),
                },
                step["start"],
            )
        )
    agent = record.get("agent")
    if agent:
        gen_id = str(uuid.uuid4())
        usage = agent.get("usage") or {}
        events.append(
            _event(
                "generation-create",
                {
                    "id": gen_id,
                    "traceId": trace_id,
                    "name": "claude -p",
                    "startTime": _iso(agent["started_at"]),
                    "endTime": _iso(agent["ended_at"]),
                    "model": agent.get("resolved_model") or agent["model"],
                    "input": record["prompt_input"],
                    "output": _clip(agent.get("text")),
                    "usageDetails": {
                        k: int(usage.get(k) or 0)
                        for k in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_input_tokens",
                            "cache_creation_input_tokens",
                        )
                    },
                    "metadata": {
                        "prompt_version": record["prompt_version"],
                        "num_turns": agent.get("num_turns"),
                        "api_duration_ms": agent.get("api_duration_ms"),
                        "tools_offered": agent.get("tools_offered"),
                    },
                    "level": "ERROR" if agent.get("error") else "DEFAULT",
                    "statusMessage": agent.get("error"),
                },
                agent["started_at"],
            )
        )
        for call in agent.get("tool_calls") or []:
            t = call.get("at") or agent["started_at"]
            events.append(
                _event(
                    "span-create",
                    {
                        "id": str(uuid.uuid4()),
                        "traceId": trace_id,
                        "parentObservationId": gen_id,
                        "name": f"tool {call['name']}",
                        "startTime": _iso(t),
                        "endTime": _iso(call.get("done_at") or t),
                        "input": call.get("input"),
                        "output": _clip(call.get("output")),
                        "level": "ERROR" if call.get("is_error") else "DEFAULT",
                    },
                    t,
                )
            )
    return events


def _post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.status), resp.read().decode("utf-8", "replace")


def send_trace(record: dict[str, Any], post: Post = _post) -> tuple[bool, str]:
    """Returns (sent, status); never raises."""
    public = os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
    secret = os.environ.get("LANGFUSE_SECRET_KEY") or ""
    host = (os.environ.get("LANGFUSE_HOST") or "").rstrip("/")
    if not (public and secret and host):
        return False, "trace skipped: Langfuse env not set"
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
    try:
        body = json.dumps({"batch": build_batch(record)}, default=str).encode()
        status, text = post(f"{host}/api/public/ingestion", headers, body)
    except Exception as exc:
        log.warning("trace failed: %s", type(exc).__name__)
        return False, f"trace failed: {type(exc).__name__}"
    errors: list[Any] = []
    with contextlib.suppress(ValueError, AttributeError):
        errors = json.loads(text).get("errors") or []
    if status >= 300 or errors:
        log.warning("trace rejected: HTTP %s, %s error(s): %s", status, len(errors), text[:300])
        return False, f"trace rejected: HTTP {status}, {len(errors)} error(s)"
    # the host stays out of the status: the record is committed to a public repo
    log.info("trace sent: %s", trace_url(host, record["run_id"]))
    return True, "trace sent"


def trace_url(host: str, trace_id: str) -> str:
    return f"{host}/trace/{trace_id}"
