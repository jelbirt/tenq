"""The one place tenq calls a model.

Version 0 runs the unmodified Claude Code CLI (`claude -p`) on the owner's
claude.ai subscription (CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`).
Swapping in an API key later changes this module only. Call pattern adapted
from ticker-sentinel's news/llm.py (MIT, same author).

Guardrails:
- built-in tools emptied (`--tools ""`) and ambient MCP config ignored
  (`--strict-mcp-config`); the only server is tenq's own read-only one, and
  only its three tools are allowed. The init event's tool list is checked: a
  run that was offered anything else is discarded.
- no settings, hooks or project files: settings sources are empty and the
  CLI runs in an empty temp directory.
- a subprocess timeout; every failure (CLI missing, auth, timeout, error
  result, no text) comes back as an AgentRun with `error` set and `text`
  None, so the caller FAILS OPEN to a numbers-only email.

Each failure also says its kind. A model failure (MODEL: timed out after the
model started, empty answer, out of turns) is the model's own result; an
infrastructure failure (INFRA: CLI missing or exiting, auth, API errors, a
hang before the model started, a missing init event, an unreadable stream,
tools outside the allowlist, allowed tools never offered because tenq's
server did not start) says nothing about the model.
`usage_limit` marks an infrastructure failure from the plan's usage limit.
Evals grade the first kind as a refusal and stop on the second.

The stream-json events are kept in summary form (tool calls, usage, timing)
for the trace and the run record.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tenq.env import PRIVATE_ENV, scrubbed

log = logging.getLogger(__name__)

DEFAULT_MODEL = "opus"
TIMEOUT_S = 900
MAX_OUTPUT_TOKENS = "16000"
TOOL_OUTPUT_CHARS = 2000  # per tool result kept in the run record
MODEL = "model"
INFRA = "infra"
MODEL_RESULT_SUBTYPES = {"error_max_turns"}  # the model ran out of turns
USAGE_LIMIT = re.compile(r"usage limit|hit your limit|limit reached|rate[ _]?limit", re.I)


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    output: str = ""
    is_error: bool = False
    at: float | None = None  # when the call reached the stream
    done_at: float | None = None  # when its result did


@dataclass
class AgentRun:
    model: str
    text: str | None = None
    error: str | None = None
    failure: str | None = None  # MODEL or INFRA, set with error
    usage_limit: bool = False
    tools_offered: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    num_turns: int | None = None
    duration_ms: int | None = None
    api_duration_ms: int | None = None
    resolved_model: str | None = None
    started_at: float = 0.0
    ended_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fail(self, kind: str, error: str) -> None:
        self.error, self.failure, self.text = error, kind, None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    """The content blocks of a message event, skipping any odd shapes."""
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def parse_stream(stdout: str, run: AgentRun, times: Sequence[float] | None = None) -> None:
    """Fill `run` from claude's stream-json output (one JSON event per line).

    `times`, when given, holds the arrival time of each line; tool calls take
    their start and end times from it.
    """
    chunks: list[str] = []
    calls: dict[str, ToolCall] = {}
    result: dict[str, Any] | None = None
    saw_init = False
    limited = False
    api_error: str | None = None  # on the last assistant message
    for i, line in enumerate(stdout.splitlines()):
        at = times[i] if times is not None and i < len(times) else None
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            tools = event.get("tools")
            run.tools_offered = [str(t) for t in tools] if isinstance(tools, list) else []
            run.resolved_model = event.get("model")
            saw_init = isinstance(tools, list)
        elif kind == "assistant":
            # the CLI marks an API failure on the message: "rate_limit",
            # "authentication_failed", ...
            api_error = str(event.get("error") or "") or None
            limited = limited or "rate_limit" in (api_error or "")
            # the text of every assistant message, not just the last: a long
            # answer split by an output-token continuation is otherwise cut
            for b in _blocks(event):
                if b.get("type") == "text" and b.get("text"):
                    chunks.append(b["text"])
                elif b.get("type") == "tool_use":
                    call = ToolCall(
                        str(b.get("id")), str(b.get("name")), b.get("input") or {}, at=at
                    )
                    calls[call.id] = call
                    run.tool_calls.append(call)
        elif kind == "user":
            for b in _blocks(event):
                if b.get("type") == "tool_result":
                    done = calls.get(str(b.get("tool_use_id")))
                    if done is not None:
                        done.output = _content_text(b.get("content"))[:TOOL_OUTPUT_CHARS]
                        done.is_error = bool(b.get("is_error"))
                        done.done_at = at
        elif kind == "result":
            result = event
    run.usage_limit = limited
    if result is None:
        run.fail(INFRA, "no result event")
        return
    if not saw_init:  # without it the tool allowlist cannot be verified
        run.fail(INFRA, "no init event listing the tools offered")
        return
    run.usage = result.get("usage") or {}
    run.num_turns = result.get("num_turns")
    run.duration_ms = result.get("duration_ms")
    run.api_duration_ms = result.get("duration_api_ms")
    if result.get("is_error") or str(result.get("subtype", "success")) != "success":
        said = str(result.get("result"))[:200]
        run.usage_limit = limited or bool(USAGE_LIMIT.search(said))
        own = result.get("subtype") in MODEL_RESULT_SUBTYPES and not run.usage_limit
        run.fail(MODEL if own else INFRA, f"error result: {said}")
        return
    if api_error:  # an API failure the result did not flag: its text is the CLI's, not the model's
        run.fail(INFRA, f"api error: {api_error}")
        return
    text = "".join(chunks).strip() or str(result.get("result") or "").strip()
    if not text:
        run.fail(MODEL, "empty response")
        return
    run.text = text


class StreamTimeout(Exception):
    """The process was killed at the timeout; carries what it had streamed."""

    def __init__(self, stdout: str, times: list[float]) -> None:
        super().__init__("timed out")
        self.stdout, self.times = stdout, times


def model_started(stdout: str) -> bool:
    """Whether a stream got as far as the model: an init event, then an
    assistant message. A hang before that is the CLI's or the API's."""
    init = False
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            init = True
        elif event.get("type") == "assistant" and init and not event.get("error"):
            return True
    return False


def _run_streaming(
    cmd: list[str], prompt: str, env: dict[str, str], cwd: str, timeout: int
) -> tuple[str, list[float], str, int]:
    """Run `cmd`, timestamping each stdout line as it arrives.

    Returns (stdout, arrival time per line, stderr, exit code); raises
    StreamTimeout, with the lines read so far, after killing the process.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
    )
    lines: list[str] = []
    times: list[float] = []
    err: list[str] = []

    def read_out() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            times.append(time.time())
            lines.append(line)

    def read_err() -> None:
        assert proc.stderr is not None
        err.append(proc.stderr.read())

    readers = [threading.Thread(target=f, daemon=True) for f in (read_out, read_err)]
    for r in readers:
        r.start()
    try:
        assert proc.stdin is not None
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except BrokenPipeError:  # it exited before reading; its exit code says why
            pass
        code = proc.wait(timeout=timeout)
    except BaseException as exc:
        proc.kill()
        proc.wait()
        for r in readers:
            r.join(timeout=5)
        if isinstance(exc, subprocess.TimeoutExpired):
            raise StreamTimeout("".join(lines), list(times)) from exc
        raise
    for r in readers:
        r.join(timeout=5)
    return "".join(lines), times, "".join(err), code


def call_claude(
    prompt: str,
    *,
    system: str,
    mcp_servers: dict[str, Any],
    allowed_tools: list[str],
    model: str = DEFAULT_MODEL,
    timeout: int = TIMEOUT_S,
    log_detail: bool = True,
) -> AgentRun:
    """One headless agent run. Never raises: failures set `error` and `failure`.

    With `log_detail` off, warnings name the failure but carry none of the
    CLI's text, which can quote the model (an eval's answers stay private).
    """
    run = AgentRun(model=model, started_at=time.time())
    with tempfile.TemporaryDirectory(prefix="tenq-claude-") as cwd:
        config = Path(cwd) / "mcp.json"
        config.write_text(json.dumps({"mcpServers": mcp_servers}))
        cmd = [
            "claude", "-p", "--model", model,
            "--output-format", "stream-json", "--verbose",  # stream-json needs verbose
            "--tools", "",
            "--strict-mcp-config", "--mcp-config", str(config),
            "--allowedTools", ",".join(allowed_tools),
            "--setting-sources", "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--system-prompt", system,
        ]  # fmt: skip
        # the run's other secrets stay out of the model's process and its MCP server
        env = scrubbed(PRIVATE_ENV)
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = MAX_OUTPUT_TOKENS
        try:
            stdout, times, stderr, code = _run_streaming(cmd, prompt, env, cwd, timeout)
        except FileNotFoundError:
            run.fail(INFRA, "claude CLI not installed")
        except StreamTimeout as t:
            with contextlib.suppress(Exception):  # the partial stream is for the record only
                parse_stream(t.stdout, run, t.times)
            # a model that took too long is the model's failure; a CLI or API
            # that never got the model going is not
            run.fail(MODEL if model_started(t.stdout) else INFRA, f"timed out after {timeout}s")
        except Exception as exc:  # never let the model path break a run
            run.fail(INFRA, f"{type(exc).__name__}: {exc}")
        else:
            try:
                parse_stream(stdout, run, times)
            except Exception as exc:  # an unexpected stream shape must not break a run
                run.fail(INFRA, f"unreadable stream: {type(exc).__name__}")
            if code != 0 and run.error is None:
                run.fail(INFRA, f"exit {code}")
            if code != 0 and USAGE_LIMIT.search(stderr):
                run.usage_limit = True
            if run.error and code != 0:
                # the CLI prints auth/config errors on stdout as well as stderr
                if log_detail:
                    log.warning("claude exit %s: stderr=%r", code, stderr[-300:])
                else:
                    log.warning("claude exit %s", code)
    run.ended_at = time.time()
    used = {c.name for c in run.tool_calls}
    extra = sorted((set(run.tools_offered) | used) - set(allowed_tools))
    if extra:
        run.fail(INFRA, f"offered tools outside the allowlist: {extra}")
    missing = sorted(set(allowed_tools) - set(run.tools_offered))
    if missing and run.error is None:  # tenq's server did not start: the model had no tools
        run.fail(INFRA, f"allowed tools not offered: {missing}")
    if run.error:
        shown = run.error if log_detail else run.error.split(":")[0]
        log.warning("claude run failed (%s): %s", run.failure, shown)
    return run
