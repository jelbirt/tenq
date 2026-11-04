from __future__ import annotations

import json
import logging
import os
import stat
import sys
from pathlib import Path

import pytest

from tenq.llm import INFRA, MODEL, AgentRun, call_claude, parse_stream

ALLOWED = ["mcp__tenq__get_metrics", "mcp__tenq__search_filing", "mcp__tenq__read_passage"]


def _events(
    tools: list[str] = ALLOWED, *, error: bool = False, text: str = '{"claims": []}'
) -> str:
    # the event shapes claude -p --output-format stream-json emits (seen in a live probe)
    events = [
        {"type": "system", "subtype": "init", "tools": tools, "model": "claude-opus-x"},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "t1", "name": "mcp__tenq__get_metrics", "input": {}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "{}"}]},
        ]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {"type": "result", "subtype": "success", "is_error": error, "num_turns": 2,
         "duration_ms": 1200, "duration_api_ms": 900, "result": "boom" if error else text,
         "usage": {"input_tokens": 10, "output_tokens": 20}},
    ]  # fmt: skip
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_parse_stream_keeps_text_tool_calls_and_usage() -> None:
    run = AgentRun(model="opus")
    parse_stream("not json\n" + _events(), run)
    assert run.error is None
    assert run.text == 'Looking.{"claims": []}'
    assert run.tools_offered == ALLOWED and run.resolved_model == "claude-opus-x"
    (call,) = run.tool_calls
    assert (call.name, call.output, call.is_error) == ("mcp__tenq__get_metrics", "{}", False)
    assert run.usage["output_tokens"] == 20 and run.num_turns == 2 and run.duration_ms == 1200


def test_parse_stream_error_result_and_missing_result() -> None:
    run = AgentRun(model="opus")
    parse_stream(_events(error=True), run)
    assert run.text is None and run.error is not None and "boom" in run.error
    run = AgentRun(model="opus")
    parse_stream(_events().rsplit("\n", 2)[0], run)
    assert run.error == "no result event"


def test_a_stream_without_init_cannot_prove_its_tools() -> None:
    run = AgentRun(model="opus")
    parse_stream(_events().split("\n", 1)[1], run)
    assert run.text is None and run.error is not None and "init" in run.error


@pytest.mark.parametrize(
    "line",
    [
        {"type": "assistant", "message": {"content": ["hi"]}},
        {"type": "assistant", "message": "x"},
        {"type": "user", "message": {"content": 5}},
        {"type": "system", "subtype": "init", "tools": 5},
    ],
)
def test_odd_event_shapes_do_not_raise(line: dict[str, object]) -> None:
    run = AgentRun(model="opus")
    parse_stream(json.dumps(line) + "\n" + _events(), run)
    assert run.tool_calls  # the rest of the stream is still read


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `claude` on PATH that records its argv and stdin, then replays a stream."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "claude"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "out = os.environ['FAKE_OUT']\n"
        "json.dump({'argv': sys.argv[1:], 'stdin': sys.stdin.read(), 'cwd': os.getcwd(),"
        " 'env': sorted(os.environ)},"
        " open(out + '.call', 'w'))\n"
        "time.sleep(float(os.environ.get('FAKE_SLEEP', '0')))\n"
        "sys.stdout.write(open(out).read())\n"
        "sys.exit(int(os.environ.get('FAKE_EXIT', '0')))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    out = tmp_path / "stream.jsonl"
    monkeypatch.setenv("FAKE_OUT", str(out))
    return out


def _call(**kw: object) -> AgentRun:
    return call_claude(
        "brief CRWD",
        system="rules",
        mcp_servers={"tenq": {"type": "stdio", "command": "x"}},
        allowed_tools=ALLOWED,
        **kw,  # type: ignore[arg-type]
    )


def test_call_runs_locked_down_and_parses(fake_claude: Path) -> None:
    fake_claude.write_text(_events())
    run = _call(model="opus")
    assert run.error is None and run.text is not None
    call = json.loads(Path(f"{fake_claude}.call").read_text())
    argv = call["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--allowedTools") + 1] == ",".join(ALLOWED)
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--system-prompt") + 1] == "rules"
    assert call["stdin"] == "brief CRWD"
    assert Path(call["cwd"]).name.startswith("tenq-claude-")  # an empty temp dir
    assert run.ended_at >= run.started_at > 0


def test_run_secrets_never_reach_the_model_process(
    fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in ("SMTP_PASS", "LANGFUSE_SECRET_KEY", "RECIPIENT_EMAILS", "EDGAR_IDENTITY"):
        monkeypatch.setenv(k, "secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "token")
    fake_claude.write_text(_events())
    _call()
    env = json.loads(Path(f"{fake_claude}.call").read_text())["env"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" in env
    assert not [k for k in env if k.startswith(("SMTP_", "LANGFUSE_", "RECIPIENT", "EDGAR"))]


def test_a_run_offered_other_tools_is_discarded(fake_claude: Path) -> None:
    fake_claude.write_text(_events([*ALLOWED, "Bash"]))
    run = _call()
    assert run.text is None and run.error is not None and "Bash" in run.error


def test_a_run_that_called_another_tool_is_discarded(fake_claude: Path) -> None:
    fake_claude.write_text(_events().replace('"name": "mcp__tenq__get_metrics"', '"name": "Bash"'))
    run = _call()
    assert run.text is None and run.error is not None and "Bash" in run.error


def test_failures_fail_open(fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_claude.write_text("Invalid API key\n")
    monkeypatch.setenv("FAKE_EXIT", "1")
    run = _call()
    assert run.text is None and run.error == "no result event"

    monkeypatch.setenv("FAKE_EXIT", "0")
    monkeypatch.setenv("FAKE_SLEEP", "5")
    run = _call(timeout=1)
    assert run.text is None and run.error == "timed out after 1s"

    monkeypatch.setenv("PATH", "/nonexistent")
    run = _call()
    assert run.text is None and run.error == "claude CLI not installed"


def test_tool_calls_take_their_times_from_the_stream() -> None:
    run = AgentRun(model="opus")
    lines = _events().splitlines()
    parse_stream("\n".join(lines), run, [10.0 + i for i in range(len(lines))])
    (call,) = run.tool_calls
    assert (call.at, call.done_at) == (11.0, 12.0)


def test_streamed_lines_are_timestamped_as_they_arrive(fake_claude: Path) -> None:
    fake_claude.write_text(_events())
    run = _call()
    (call,) = run.tool_calls
    assert call.at is not None and call.done_at is not None
    assert run.started_at <= call.at <= call.done_at <= run.ended_at


def test_xerj_sees_none_of_the_run_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    from tenq.env import MODEL_ENV, PRIVATE_ENV, scrubbed

    for k in ("SMTP_PASS", "LANGFUSE_SECRET_KEY", "RECIPIENT_EMAILS", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.setenv(k, "x")
    env = scrubbed(PRIVATE_ENV + MODEL_ENV)
    assert "PATH" in env
    assert not {"SMTP_PASS", "LANGFUSE_SECRET_KEY", "RECIPIENT_EMAILS"} & env.keys()
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def _result(**over: object) -> str:
    """The stream with its result event replaced by `over` applied to it."""
    *head, last = _events().strip().split("\n")
    return "\n".join([*head, json.dumps({**json.loads(last), **over})]) + "\n"


@pytest.mark.parametrize(
    ("stream", "kind", "limited"),
    [
        (_result(is_error=True, result="Failed to authenticate. API Error: 401"), INFRA, False),
        (_result(is_error=True, result="You've hit your limit · resets 5pm (UTC)"), INFRA, True),
        (_result(is_error=True, result="Claude AI usage limit reached|1762000000"), INFRA, True),
        (_result(is_error=True, subtype="error_max_turns", result="max turns"), MODEL, False),
        (_result(subtype="error_during_execution", result="x"), INFRA, False),
        (_result(result="", is_error=False).replace('"text": "{\\"claims\\": []}"', '"text": ""')
         .replace('"text": "Looking."', '"text": ""'), MODEL, False),
        (_events().split("\n", 1)[1], INFRA, False),  # no init event
        (_events().rsplit("\n", 2)[0], INFRA, False),  # no result event
    ],
)  # fmt: skip
def test_each_failure_says_its_kind(stream: str, kind: str, limited: bool) -> None:
    run = AgentRun(model="opus")
    parse_stream(stream, run)
    assert run.text is None and run.error is not None
    assert (run.failure, run.usage_limit) == (kind, limited)


def test_a_rate_limited_message_marks_the_usage_limit() -> None:
    stream = _result(is_error=True, result="API Error").replace(
        '{"type": "assistant", "message": {"content": [{"type": "text"',
        '{"type": "assistant", "error": "rate_limit", "message": {"content": [{"type": "text"',
    )
    run = AgentRun(model="opus")
    parse_stream(stream, run)
    assert (run.failure, run.usage_limit) == (INFRA, True)


def test_a_maxed_out_model_with_a_usage_limit_is_still_infrastructure() -> None:
    run = AgentRun(model="opus")
    parse_stream(_result(is_error=True, subtype="error_max_turns", result="usage limit"), run)
    assert (run.failure, run.usage_limit) == (INFRA, True)


def test_a_good_run_has_no_failure(fake_claude: Path) -> None:
    fake_claude.write_text(_events())
    run = _call()
    assert run.error is None and run.failure is None and not run.usage_limit


def test_call_failures_say_their_kind(fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_claude.write_text("not a stream\n")
    monkeypatch.setenv("FAKE_EXIT", "1")
    assert _call().failure == INFRA
    fake_claude.write_text(_events())
    assert _call().failure == INFRA  # a clean stream but a non-zero exit
    monkeypatch.setenv("FAKE_EXIT", "0")
    monkeypatch.setenv("FAKE_SLEEP", "5")
    assert _call(timeout=1).failure == MODEL
    monkeypatch.setenv("PATH", "/nonexistent")
    assert _call().failure == INFRA
    monkeypatch.setenv("PATH", f"{fake_claude.parent / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLEEP", "0")
    fake_claude.write_text(_events([*ALLOWED, "Bash"]))
    assert _call().failure == INFRA


def test_quiet_logs_carry_no_model_text(
    fake_claude: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "the answer is 1,577 million"
    fake_claude.write_text(_result(is_error=True, result=secret))
    monkeypatch.setenv("FAKE_EXIT", "1")
    with caplog.at_level(logging.DEBUG, logger="tenq.llm"):
        run = _call(log_detail=False)
    assert run.error is not None and secret in run.error  # kept for the private record
    assert caplog.records and secret not in caplog.text and "1,577" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="tenq.llm"):
        _call()
    assert secret in caplog.text  # briefs keep the detail; their records are public anyway
