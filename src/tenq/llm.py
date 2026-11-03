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

The stream-json events are kept in summary form (tool calls, usage, timing)
for the trace and the run record.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODEL = "opus"
TIMEOUT_S = 900
MAX_OUTPUT_TOKENS = "16000"
# the run's other secrets stay out of the model's process and its MCP server
PRIVATE_ENV = ("SMTP_", "LANGFUSE_", "RECIPIENT_EMAILS", "EDGAR_IDENTITY")
TOOL_OUTPUT_CHARS = 2000  # per tool result kept in the run record


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    output: str = ""
    is_error: bool = False


@dataclass
class AgentRun:
    model: str
    text: str | None = None
    error: str | None = None
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


def parse_stream(stdout: str, run: AgentRun) -> None:
    """Fill `run` from claude's stream-json output (one JSON event per line)."""
    chunks: list[str] = []
    calls: dict[str, ToolCall] = {}
    result: dict[str, Any] | None = None
    saw_init = False
    for line in stdout.splitlines():
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
            # the text of every assistant message, not just the last: a long
            # answer split by an output-token continuation is otherwise cut
            for b in _blocks(event):
                if b.get("type") == "text" and b.get("text"):
                    chunks.append(b["text"])
                elif b.get("type") == "tool_use":
                    call = ToolCall(str(b.get("id")), str(b.get("name")), b.get("input") or {})
                    calls[call.id] = call
                    run.tool_calls.append(call)
        elif kind == "user":
            for b in _blocks(event):
                if b.get("type") == "tool_result":
                    done = calls.get(str(b.get("tool_use_id")))
                    if done is not None:
                        done.output = _content_text(b.get("content"))[:TOOL_OUTPUT_CHARS]
                        done.is_error = bool(b.get("is_error"))
        elif kind == "result":
            result = event
    if result is None:
        run.error = "no result event"
        return
    if not saw_init:  # without it the tool allowlist cannot be verified
        run.error = "no init event listing the tools offered"
        return
    run.usage = result.get("usage") or {}
    run.num_turns = result.get("num_turns")
    run.duration_ms = result.get("duration_ms")
    run.api_duration_ms = result.get("duration_api_ms")
    if result.get("is_error"):
        run.error = f"error result: {str(result.get('result'))[:200]}"
        return
    text = "".join(chunks).strip() or str(result.get("result") or "").strip()
    if not text:
        run.error = "empty response"
        return
    run.text = text


def call_claude(
    prompt: str,
    *,
    system: str,
    mcp_servers: dict[str, Any],
    allowed_tools: list[str],
    model: str = DEFAULT_MODEL,
    timeout: int = TIMEOUT_S,
) -> AgentRun:
    """One headless agent run. Never raises: failures set `error`."""
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
        env = {k: v for k, v in os.environ.items() if not k.startswith(PRIVATE_ENV)}
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = MAX_OUTPUT_TOKENS
        try:
            done = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cwd,
            )
        except FileNotFoundError:
            run.error = "claude CLI not installed"
        except subprocess.TimeoutExpired:
            run.error = f"timed out after {timeout}s"
        except Exception as exc:  # never let the model path break a run
            run.error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                parse_stream(done.stdout, run)
            except Exception as exc:  # an unexpected stream shape must not break a run
                run.error = f"unreadable stream: {type(exc).__name__}"
            if done.returncode != 0 and run.error is None:
                run.error = f"exit {done.returncode}"
            if run.error and done.returncode != 0:
                # the CLI prints auth/config errors on stdout as well as stderr
                log.warning(
                    "claude exit %s: stderr=%r", done.returncode, (done.stderr or "")[-300:]
                )
    run.ended_at = time.time()
    used = {c.name for c in run.tool_calls}
    extra = sorted((set(run.tools_offered) | used) - set(allowed_tools))
    if extra:
        run.error, run.text = f"offered tools outside the allowlist: {extra}", None
    if run.error:
        log.warning("claude run failed: %s", run.error)
    return run
