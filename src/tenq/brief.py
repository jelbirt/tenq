"""Ask Claude for a brief on one filing and parse its answer.

The prompt is a versioned file (prompts/brief.md); its version goes into the
trace and the run record. The model's answer is untrusted until the checker
(tenq.checker) has passed each claim and code has written its numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tenq.config import REPO_ROOT
from tenq.filings import FilingMeta
from tenq.llm import DEFAULT_MODEL, AgentRun, call_claude

PROMPT_PATH = REPO_ROOT / "prompts" / "brief.md"
TOOLS = ("get_metrics", "search_filing", "read_passage")
SERVER = "tenq"
ALLOWED_TOOLS = [f"mcp__{SERVER}__{t}" for t in TOOLS]
MAX_CLAIMS = 8
VERSION_LINE = re.compile(r"<!--\s*prompt-version:\s*([\w.\-]+)\s*-->")


@dataclass(frozen=True)
class Prompt:
    version: str
    text: str
    sha256: str  # of the file's bytes as read, so a test can pin the text to its version


def load_prompt(path: Path = PROMPT_PATH) -> Prompt:
    data = path.read_bytes()
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    m = VERSION_LINE.search(text)
    if m is None:
        raise ValueError(f"{path} has no prompt-version line")
    body = VERSION_LINE.sub("", text, count=1).strip()
    return Prompt(m.group(1), body, hashlib.sha256(data).hexdigest())


@dataclass(frozen=True)
class Claim:
    """As the model wrote it: text with {fin:...} placeholders, and one passage."""

    claim: str
    passage_id: str


@dataclass
class Draft:
    claims: list[Claim] = field(default_factory=list)
    error: str | None = None  # why the answer could not be used at all


def last_json_object(text: str, key: str) -> Any:
    """The last JSON object in `text` that has `key`, allowing a ```json fence."""
    dec = json.JSONDecoder()
    found: Any = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict) and key in obj:
            found = obj
    return found


def parse_answer(text: str | None) -> Draft:
    """Claims from the model's answer. Malformed entries are skipped here;
    whether a well-formed claim is true is the checker's job."""
    if not text:
        return Draft(error="no answer")
    obj = last_json_object(text, "claims")
    if obj is None or not isinstance(obj.get("claims"), list):
        return Draft(error="no JSON object with a claims list")
    draft = Draft()
    for raw in obj["claims"][:MAX_CLAIMS]:
        if not isinstance(raw, dict):
            continue
        claim, pid = raw.get("claim"), raw.get("passage_id")
        if not isinstance(claim, str) or not isinstance(pid, str):
            continue
        # never cut here: a claim too long is the checker's to drop whole
        draft.claims.append(Claim(" ".join(claim.split()), pid.strip()))
    return draft


def mcp_servers(root: Path, meta: FilingMeta, xerj_url: str) -> dict[str, Any]:
    """The --mcp-config entry that starts tenq's server for this one filing."""
    return {
        SERVER: {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "tenq.mcp_server"],
            "env": {
                "TENQ_ROOT": str(root.resolve()),
                "TENQ_TICKER": meta.ticker,
                "TENQ_ACCESSION": meta.accession,
                "TENQ_XERJ_URL": xerj_url,
            },
        }
    }


def user_message(meta: FilingMeta) -> str:
    return (
        f"Write the brief for {meta.company} ({meta.ticker}), {meta.form} for the "
        f"period ended {meta.period}, accession {meta.accession}."
    )


def write_brief(
    root: Path,
    meta: FilingMeta,
    xerj_url: str,
    prompt: Prompt,
    model: str = DEFAULT_MODEL,
) -> tuple[AgentRun, Draft]:
    run = call_claude(
        user_message(meta),
        system=prompt.text,
        mcp_servers=mcp_servers(root, meta, xerj_url),
        allowed_tools=ALLOWED_TOOLS,
        model=model,
    )
    draft = parse_answer(run.text) if run.error is None else Draft(error=run.error)
    return run, draft
