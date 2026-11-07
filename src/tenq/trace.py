"""Send one run's trace to Langfuse, then confirm that Langfuse stored it.

One trace per briefed filing: a root span for the run, a span per pipeline
step, a generation for the Claude run (model, tokens, time, prompt version)
and a tool span per call under it. The run record
(runs/<TICKER>/<accession>.json) is the permanent copy; Langfuse is the
browsable one.

The trace is built after the run, from its record, so every span keeps the
time it really took. The spans are made with the OpenTelemetry API on a
tracer provider of tenq's own; the Langfuse SDK is attached to that provider
and does all the sending (wire format, auth, endpoint, retries). An upload
being accepted is not proof the trace was stored, so after sending, the trace
is looked up by id through Langfuse's API, and the record says the trace is
in Langfuse only once that lookup finds it.

Sending never raises: a tracing outage must not stop an email.

Env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.

    python -m tenq.trace --check    # send one small test trace and look it up
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from langfuse import Langfuse, LangfuseOtelSpanAttributes
from opentelemetry import trace as otel
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span, Status, StatusCode, Tracer

log = logging.getLogger(__name__)

A = LangfuseOtelSpanAttributes
MAX_FIELD_CHARS = 20_000
SCOPE = "tenq"
FIND_WAIT_S = 120.0  # how long to keep looking for a sent trace
FIND_POLL_S = 5.0
LOOKUP_TIMEOUT_S = 15  # one lookup request
LOOKUP_WINDOW = timedelta(days=1)  # how far back a lookup searches (a run is shorter)
USAGE_KEYS = {  # claude's usage keys to Langfuse's usage_details keys
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "cache_creation_input_tokens": "cache_creation_input_tokens",
}


@dataclass(frozen=True)
class Tracing:
    """Where spans go: a tracer to make them, a flush to send them, and a
    lookup that says whether Langfuse has stored a trace id."""

    tracer: Tracer
    flush: Callable[[], None]
    found: Callable[[str], bool]


Connect = Callable[[str, str, str], Tracing]


def _ns(t: float) -> int:
    return int(t * 1_000_000_000)


def _json(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + " [cut]"


def _attrs(d: dict[str, Any]) -> dict[str, Any]:
    """OpenTelemetry attribute values: scalars and string lists as they are,
    anything else as JSON text; None is left out."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        scalar = isinstance(v, (bool, int, float, str))
        if scalar or (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            out[k] = v
        else:
            out[k] = _json(v)
    return out


def trace_id_hex(span: Span) -> str:
    return format(span.get_span_context().trace_id, "032x")


def emit(record: dict[str, Any], tracer: Tracer) -> str:
    """Make one run record's spans on `tracer` (see tenq.run for the record's
    shape) and return the trace id as 32 hex characters."""
    filing = record["filing"]
    # trace-level attributes go on every span, as Langfuse asks
    common = {
        A.TRACE_NAME: "brief",
        A.TRACE_SESSION_ID: filing["ticker"],
        A.TRACE_TAGS: [filing["ticker"], filing["form"], record["mode"]],
        A.VERSION: record["prompt_version"],
        f"{A.TRACE_METADATA}.run_id": record["run_id"],
        f"{A.TRACE_METADATA}.accession": filing["accession"],
        f"{A.TRACE_METADATA}.period": filing["period"],
    }

    def start(
        name: str, at: float, parent: Context, attrs: dict[str, Any], error: str | None = None
    ) -> Span:
        level = {A.OBSERVATION_LEVEL: "ERROR", A.OBSERVATION_STATUS_MESSAGE: error} if error else {}
        span = tracer.start_span(
            name,
            context=parent,
            start_time=_ns(at),
            attributes=_attrs({**common, **attrs, **level}),
        )
        if error:
            span.set_status(Status(StatusCode.ERROR, error))
        return span

    def end(span: Span, at: float, start_at: float) -> None:
        span.end(end_time=_ns(max(at, start_at)))

    root = start(
        "brief",
        record["started_at"],
        Context(),  # an empty context: always a new trace
        {
            A.OBSERVATION_TYPE: "span",
            A.OBSERVATION_INPUT: _json(
                {k: filing[k] for k in ("ticker", "form", "accession", "period")}
            ),
            A.OBSERVATION_OUTPUT: _json(
                {
                    "mode": record["mode"],
                    "claims_kept": len(record["check"]["kept"]),
                    "claims_dropped": len(record["check"]["dropped"]),
                    "email": record["email"]["status"],
                }
            ),
            f"{A.OBSERVATION_METADATA}.model": record["model"],
        },
    )
    under_root = otel.set_span_in_context(root)
    for step in record["steps"]:
        span = start(
            step["name"],
            step["start"],
            under_root,
            {A.OBSERVATION_TYPE: "span", A.OBSERVATION_OUTPUT: _json(step.get("output"))},
            step.get("error"),
        )
        end(span, step["end"], step["start"])
    agent = record.get("agent")
    if agent:
        usage = agent.get("usage") or {}
        gen = start(
            "claude -p",
            agent["started_at"],
            under_root,
            {
                A.OBSERVATION_TYPE: "generation",
                A.OBSERVATION_MODEL: agent.get("resolved_model") or agent["model"],
                A.OBSERVATION_INPUT: _json(record["prompt_input"]),
                A.OBSERVATION_OUTPUT: _json(agent.get("text") or ""),
                A.OBSERVATION_USAGE_DETAILS: json.dumps(
                    {v: int(usage.get(k) or 0) for k, v in USAGE_KEYS.items()}
                ),
                f"{A.OBSERVATION_METADATA}.prompt_version": record["prompt_version"],
                f"{A.OBSERVATION_METADATA}.num_turns": agent.get("num_turns"),
                f"{A.OBSERVATION_METADATA}.api_duration_ms": agent.get("api_duration_ms"),
                f"{A.OBSERVATION_METADATA}.tools_offered": agent.get("tools_offered"),
            },
            agent.get("error"),
        )
        under_gen = otel.set_span_in_context(gen)
        for call in agent.get("tool_calls") or []:
            t = call.get("at") or agent["started_at"]
            tool = start(
                f"tool {call['name']}",
                t,
                under_gen,
                {
                    A.OBSERVATION_TYPE: "tool",
                    A.OBSERVATION_INPUT: _json(call.get("input")),
                    A.OBSERVATION_OUTPUT: _json(call.get("output")),
                },
                "tool error" if call.get("is_error") else None,
            )
            end(tool, call.get("done_at") or t, t)
        end(gen, agent["ended_at"], agent["started_at"])
    end(root, record["ended_at"], record["started_at"])
    return trace_id_hex(root)


def _export_all(span: ReadableSpan) -> bool:
    # the provider is tenq's own, so every span on it is one of tenq's
    return True


_connected: dict[str, tuple[str, Tracing]] = {}

# The OTLP exporter logs a failed upload's error text, which names the host,
# and the host is a secret while Actions logs are public. tenq logs its own
# host-free line for every failure, so the libraries' lines are silenced.
QUIET_LOGGERS = ("opentelemetry.exporter.otlp", "urllib3", "httpx", "httpcore")


def _quiet() -> None:
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def connect(
    public: str, secret: str, host: str, *, exporter: SpanExporter | None = None
) -> Tracing:
    """The Langfuse SDK on a tracer provider of tenq's own. One per public
    key: the SDK keeps one client per public key, and a second provider for
    the same key would get no exporter, so its spans would go nowhere."""
    _quiet()
    if public in _connected:
        was, tracing = _connected[public]
        if was != host:
            raise ValueError("this public key is already connected to another host")
        return tracing
    provider = TracerProvider(resource=Resource.create({"service.name": SCOPE}))
    client = Langfuse(
        public_key=public,
        secret_key=secret,
        base_url=host,
        tracer_provider=provider,
        should_export_span=_export_all,
        span_exporter=exporter,
    )

    def found(trace_id: str) -> bool:
        # one request per lookup: only the first page is read, over a recent
        # window, with a short timeout, so one lookup cannot outlast the wait
        pages = client.iter_observation_pages(
            trace_id=trace_id,
            limit=1,
            max_items=1,
            from_start_time=datetime.now(UTC) - LOOKUP_WINDOW,
            request_options={"timeout_in_seconds": LOOKUP_TIMEOUT_S},
        )
        page = next(iter(pages), None)
        return bool(page and page.data)

    tracing = Tracing(tracer=provider.get_tracer(SCOPE), flush=client.flush, found=found)
    _connected[public] = (host, tracing)
    return tracing


def wait_found(
    found: Callable[[str], bool],
    trace_id: str,
    *,
    wait: float = FIND_WAIT_S,
    poll: float = FIND_POLL_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[bool, str]:
    """Look the trace up until Langfuse has it or `wait` seconds pass.
    Returns (found, the last lookup error's type or "")."""
    deadline = clock() + wait
    error = ""
    while True:
        try:
            if found(trace_id):
                return True, ""
            error = ""
        except Exception as exc:  # a failed lookup is retried like a miss
            error = type(exc).__name__
        if clock() >= deadline:
            return False, error
        sleep(poll)


def _env() -> tuple[str, str, str] | None:
    public = os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
    secret = os.environ.get("LANGFUSE_SECRET_KEY") or ""
    host = (os.environ.get("LANGFUSE_HOST") or "").rstrip("/")
    return (public, secret, host) if public and secret and host else None


def _send(
    make: Callable[[Tracer], str], connect: Connect, wait: float, sleep: Callable[[float], None]
) -> tuple[bool, str]:
    env = _env()
    if env is None:
        return False, "trace skipped: Langfuse env not set"
    try:
        tracing = connect(*env)
        trace_id = make(tracing.tracer)
        tracing.flush()
    except Exception as exc:
        log.warning("trace failed: %s", type(exc).__name__)
        return False, f"trace failed: {type(exc).__name__}"
    ok, error = wait_found(tracing.found, trace_id, wait=wait, sleep=sleep)
    # the host stays out of the status and the log: the record is committed
    # to a public repo, and the host is a secret
    if ok:
        log.info("trace stored in Langfuse: %s", trace_id)
        return True, f"trace stored: {trace_id}"
    why = f", last lookup failed: {error}" if error else ""
    log.warning("trace %s not found in Langfuse after %.0fs%s", trace_id, wait, why)
    return False, f"trace sent, not found after {wait:.0f}s{why}: {trace_id}"


def send_trace(
    record: dict[str, Any],
    *,
    connect: Connect = connect,
    wait: float = FIND_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, str]:
    """Send one run's trace and confirm it is stored. Returns (stored,
    status); never raises."""
    return _send(lambda tracer: emit(record, tracer), connect, wait, sleep)


def _agent_attrs(agent: dict[str, Any], prompt_version: str) -> dict[str, Any]:
    usage = agent.get("usage") or {}
    return {
        A.OBSERVATION_TYPE: "generation",
        A.OBSERVATION_MODEL: agent.get("resolved_model") or agent.get("model") or "",
        A.OBSERVATION_OUTPUT: _json(agent.get("text") or ""),
        A.OBSERVATION_USAGE_DETAILS: json.dumps(
            {v: int(usage.get(k) or 0) for k, v in USAGE_KEYS.items()}
        ),
        f"{A.OBSERVATION_METADATA}.prompt_version": prompt_version,
        f"{A.OBSERVATION_METADATA}.api_duration_ms": agent.get("api_duration_ms"),
    }


def emit_daily(stock: dict[str, Any], record: dict[str, Any], tracer: Tracer) -> str:
    """One stock's daily trace on `tracer` (see tenq.daily for the record's
    shape): a root span, a span per step, and a generation for Jev's answers
    and for each Opus run. Returns the trace id as 32 hex characters."""
    versions = record["versions"]
    ticker = stock["ticker"]
    jev_call = (stock.get("call") or {}).get("call")
    opus_call = (stock.get("opus") or {}).get("call")
    common = {
        A.TRACE_NAME: "daily",
        A.TRACE_SESSION_ID: f"daily-{record['day']}"
        + (f"-check-{record['check_id']}" if record.get("check_id") else ""),
        A.TRACE_TAGS: [ticker, "daily", *(["check"] if record.get("check_id") else [])],
        A.VERSION: versions["rules"],
        f"{A.TRACE_METADATA}.run_id": record["run_id"],
        f"{A.TRACE_METADATA}.day": record["day"],
    }

    def start(name: str, at: float, parent: Context, attrs: dict[str, Any], error: Any) -> Span:
        level = (
            {A.OBSERVATION_LEVEL: "ERROR", A.OBSERVATION_STATUS_MESSAGE: str(error)}
            if error
            else {}
        )
        span = tracer.start_span(
            name,
            context=parent,
            start_time=_ns(at),
            attributes=_attrs({**common, **attrs, **level}),
        )
        if error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
        return span

    steps = stock.get("steps") or []
    begin = steps[0]["start"] if steps else record["started_at"]
    finish = steps[-1]["end"] if steps else begin
    root = start(
        f"daily {ticker}",
        begin,
        Context(),
        {
            A.OBSERVATION_TYPE: "span",
            A.OBSERVATION_INPUT: _json(stock.get("pack") or {"ticker": ticker}),
            A.OBSERVATION_OUTPUT: _json(
                {
                    "jev": jev_call,
                    "opus": opus_call,
                    "why_kept": len((stock.get("why") or {}).get("kept") or []),
                    "opus_kept": len((stock.get("opus") or {}).get("kept") or []),
                }
            ),
        },
        stock.get("error"),
    )
    under = otel.set_span_in_context(root)
    agents = {
        "opus call": ((stock.get("opus") or {}).get("agent"), versions["call_prompt"]),
        "opus why": ((stock.get("why") or {}).get("agent"), versions["why_prompt"]),
    }
    for step in steps:
        name = step["name"]
        attrs: dict[str, Any] = {
            A.OBSERVATION_TYPE: "span",
            A.OBSERVATION_OUTPUT: _json(step.get("output")),
        }
        if name == "jev":
            jev = stock.get("jev") or {}
            attrs = {
                A.OBSERVATION_TYPE: "generation",
                A.OBSERVATION_MODEL: jev.get("resolved_model") or jev.get("model") or "",
                A.OBSERVATION_OUTPUT: _json(
                    {"answers": jev.get("answers"), "call": stock.get("call")}
                ),
                A.OBSERVATION_USAGE_DETAILS: json.dumps(
                    {
                        "input": int(jev.get("input_tokens") or 0),
                        "output": int(jev.get("output_tokens") or 0),
                    }
                ),
                A.OBSERVATION_COST_DETAILS: json.dumps({"total": float(jev.get("cost") or 0.0)}),
                f"{A.OBSERVATION_METADATA}.questions": versions["questions"],
                f"{A.OBSERVATION_METADATA}.rounds": jev.get("rounds"),
            }
        elif name in agents and isinstance(agents[name][0], dict):
            agent, version = agents[name]
            attrs = _agent_attrs(dict(agent or {}), version)
            if name == "opus why" and (stock.get("why") or {}).get("message"):
                attrs[A.OBSERVATION_INPUT] = _json(_why_input(stock))
        span = start(name, step["start"], under, attrs, step.get("error"))
        end_at = step["end"]
        span.end(end_time=_ns(max(end_at, step["start"])))
    root.end(end_time=_ns(max(finish, begin)))
    return trace_id_hex(root)


def _why_input(stock: dict[str, Any]) -> str:
    """The why message Opus was sent: the record keeps its head, the trace shows it whole."""
    stored = str(stock["why"]["message"])
    try:
        from tenq.opus import rebuild_why_message

        return rebuild_why_message(stored, stock["pack"])
    except Exception:  # a trace never fails the run; the stored head is still worth showing
        return stored


def send_daily_traces(
    record: dict[str, Any],
    *,
    skip: Collection[str] = (),
    connect: Connect = connect,
    wait: float = FIND_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[bool, str, dict[str, str]]:
    """One trace per stock not in `skip` (already stored), sent together, then
    each looked up within one shared wait. Returns (all stored, status, trace
    id per ticker found stored); never raises."""
    stocks = [s for s in record["stocks"] if s["ticker"] not in skip]
    if not stocks:
        return True, "traces already stored", {}
    env = _env()
    if env is None:
        return False, "trace skipped: Langfuse env not set", {}
    try:
        tracing = connect(*env)
        ids = {s["ticker"]: emit_daily(s, record, tracing.tracer) for s in stocks}
        tracing.flush()
    except Exception as exc:
        log.warning("daily traces failed: %s", type(exc).__name__)
        return False, f"trace failed: {type(exc).__name__}", {}
    deadline = clock() + wait
    missing = []
    for ticker, trace_id in ids.items():
        ok, _ = wait_found(
            tracing.found, trace_id, wait=max(0.0, deadline - clock()), sleep=sleep, clock=clock
        )
        if not ok:
            missing.append(ticker)
    stored = len(ids) - len(missing)
    if not missing:
        log.info("daily traces stored in Langfuse: %d", stored)
        return True, f"traces stored: {stored} of {len(ids)}", ids
    log.warning("daily traces not found after %.0fs: %s", wait, ", ".join(missing))
    found = {t: i for t, i in ids.items() if t not in missing}
    return (
        False,
        f"traces sent, {stored} of {len(ids)} found after {wait:.0f}s; "
        f"missing {', '.join(missing)}",
        found,
    )


def check_span(tracer: Tracer) -> str:
    """One small trace that only says tracing works."""
    now = time.time()
    span = tracer.start_span(
        "trace check",
        context=Context(),
        start_time=_ns(now),
        attributes={
            A.TRACE_NAME: "trace check",
            A.TRACE_TAGS: ["check"],
            A.OBSERVATION_TYPE: "span",
            A.OBSERVATION_OUTPUT: "tracing works",
        },
    )
    span.end(end_time=_ns(now))
    return trace_id_hex(span)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="send one test trace and look it up")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.check:
        ap.error("nothing to do: pass --check")
    ok, status = _send(check_span, connect, FIND_WAIT_S, time.sleep)
    print(status)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
