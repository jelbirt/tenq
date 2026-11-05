"""Opus's side of the daily call: its own call from the facts pack, and the "why".

Two separate claude -p runs per stock, neither with any tools (the facts pack
is the whole input):

1. opus_call: Opus reads only the pack and makes its own Buy/Hold/Sell/Short
   call with a confidence and cited reasons. It never sees Jev's answers, so
   the two calls stay independent and the scorecard can compare them.
2. explain_call: given the pack and how the rules reached Jev's call, Opus
   writes the cited reasons for that call, since Jev itself returns no text.

Both answers are only parsed here. Whether a reason may be sent is the job
of tenq.reasons, which also writes each figure into it from the pack. Both
fail open: a failed run gives an error and no reasons, and the daily email
goes out without them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from tenq.brief import Prompt, last_json_object, load_prompt
from tenq.config import REPO_ROOT
from tenq.llm import DEFAULT_MODEL, AgentRun, call_claude
from tenq.pack import Pack
from tenq.reasons import MAX_REASON_CHARS, Reason
from tenq.rules import Call

CALL_PROMPT = REPO_ROOT / "prompts" / "call.md"
WHY_PROMPT = REPO_ROOT / "prompts" / "why.md"
CallName = Literal["Buy", "Hold", "Sell", "Short"]
CALLS: tuple[CallName, ...] = ("Buy", "Hold", "Sell", "Short")
MAX_REASONS = 4
MAX_CITES = 6
TIMEOUT_S = 300  # no tools: one read and one answer

Claude = Callable[..., AgentRun]


@dataclass
class OpusCall:
    call: CallName | None = None
    confidence: float | None = None
    reasons: list[Reason] = field(default_factory=list)
    error: str | None = None  # why there is no call


@dataclass
class Why:
    reasons: list[Reason] = field(default_factory=list)
    error: str | None = None  # why there are no reasons


def load_call_prompt() -> Prompt:
    return load_prompt(CALL_PROMPT)


def load_why_prompt() -> Prompt:
    return load_prompt(WHY_PROMPT)


def _printable(text: str) -> str:
    """A lone surrogate from a JSON escape parses but cannot be printed or sent."""
    return text.encode("utf-8", "replace").decode("utf-8")


def parse_reasons(raw: Any) -> list[Reason]:
    """Well-formed reasons only, capped; whether one may be sent is tenq.reasons' job."""
    if not isinstance(raw, list):
        return []
    out: list[Reason] = []
    for r in raw[: MAX_REASONS * 4]:
        if len(out) == MAX_REASONS:
            break
        if not isinstance(r, dict):
            continue
        text, cites = r.get("text"), r.get("cites", [])
        if not isinstance(text, str) or not isinstance(cites, list):
            continue
        ids = tuple(
            dict.fromkeys(_printable(c).strip() for c in cites[:MAX_CITES] if isinstance(c, str))
        )
        # spaces collapsed first, then kept just over the limit, so an over-long
        # reason reaches tenq.reasons and is dropped with a why, never cut short
        text = " ".join(_printable(text).split())[: MAX_REASON_CHARS + 1]
        if text:
            out.append(Reason(text, ids))
    return out


def _call_name(raw: Any) -> CallName | None:
    if not isinstance(raw, str):
        return None
    return next((c for c in CALLS if c.lower() == raw.strip().lower()), None)


def _confidence(raw: Any) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    try:
        v = float(raw)
    except OverflowError:
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def parse_call(text: str | None) -> OpusCall:
    if not text:
        return OpusCall(error="no answer")
    obj = last_json_object(text, "call")
    if obj is None:
        return OpusCall(error="no JSON object with a call")
    name = _call_name(obj.get("call"))
    if name is None:
        return OpusCall(error="the call is not Buy, Hold, Sell or Short")
    return OpusCall(name, _confidence(obj.get("confidence")), parse_reasons(obj.get("reasons")))


def parse_why(text: str | None) -> Why:
    if not text:
        return Why(error="no answer")
    obj = last_json_object(text, "reasons")
    if obj is None or not isinstance(obj.get("reasons"), list):
        return Why(error="no JSON object with a reasons list")
    return Why(parse_reasons(obj["reasons"]))


def call_message(pack: Pack) -> str:
    return pack.render()


def why_message(pack: Pack, call: Call) -> str:
    lines = call.explain()
    if call.reaction_id:
        # the sheet has moves over several windows; name the one the rules used
        at = next((i for i, x in enumerate(lines) if x.startswith("market reaction:")), -1)
        lines.insert(at + 1, f"the market reaction was measured on {{{call.reaction_id}}}")
    how = "\n".join(f"- {line}" for line in lines)
    return (
        f"Jev's call on {pack.ticker}: {call.call}.\n"
        f"How the rules reached it:\n{how}\n\n"
        f"The facts sheet it was made from:\n\n{pack.render()}"
    )


def _run(claude: Claude, message: str, prompt: Prompt, model: str) -> AgentRun:
    return claude(
        message,
        system=prompt.text,
        mcp_servers={},
        allowed_tools=[],
        model=model,
        timeout=TIMEOUT_S,
        log_detail=False,
    )


def opus_call(
    pack: Pack, prompt: Prompt, *, model: str = DEFAULT_MODEL, claude: Claude = call_claude
) -> tuple[AgentRun, OpusCall]:
    """Opus's own call from the pack alone. Never raises."""
    run = _run(claude, call_message(pack), prompt, model)
    return run, parse_call(run.text) if run.error is None else OpusCall(error=run.error)


def explain_call(
    pack: Pack,
    call: Call,
    prompt: Prompt,
    *,
    model: str = DEFAULT_MODEL,
    claude: Claude = call_claude,
) -> tuple[AgentRun, Why]:
    """Cited reasons for Jev's call. Never raises."""
    run = _run(claude, why_message(pack, call), prompt, model)
    return run, parse_why(run.text) if run.error is None else Why(error=run.error)
