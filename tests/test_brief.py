from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from tenq.brief import (
    ALLOWED_TOOLS,
    PROMPT_PATH,
    Claim,
    load_prompt,
    mcp_servers,
    parse_answer,
    user_message,
)
from tenq.filings import FilingStore

# Every brief prompt version since pinning began, by the sha256 of its file, so
# the prompt edited without a version bump fails here. Versions before this
# predate the pin.
KNOWN_BRIEF_PROMPTS = {
    "brief-v4": "dc6c6a27d1d434350345d6dde046f0a6679ccc9028349a08809ba3945253150c",
}


def test_the_shipped_prompt_has_a_version_and_names_every_tool() -> None:
    p = load_prompt()
    assert p.version == "brief-v4"
    assert p == load_prompt(PROMPT_PATH)  # the file hashed is the file the code reads
    assert p.version in KNOWN_BRIEF_PROMPTS, "new prompt version: add it and its hash above"
    got = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    assert got == KNOWN_BRIEF_PROMPTS[p.version], "brief.md changed: bump its version"
    assert "prompt-version" not in p.text
    for tool in ("get_metrics", "search_filing", "read_passage"):
        assert tool in p.text
    assert "\u2014" not in p.text


def test_a_prompt_without_a_version_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "p.md"
    f.write_text("no version here")
    with pytest.raises(ValueError):
        load_prompt(f)


def test_allowed_tools_are_tenqs_three() -> None:
    assert ALLOWED_TOOLS == [
        "mcp__tenq__get_metrics",
        "mcp__tenq__search_filing",
        "mcp__tenq__read_passage",
    ]


GOOD = {
    "claims": [
        {
            "claim": "{fin:CRWD:revenue:yoy_pct},  driven\nby new customers.",
            "passage_id": "CRWD-0001535527-26-000031:23 ",
        }
    ]
}


def test_parse_answer_reads_fenced_or_prefixed_json() -> None:
    for text in (
        json.dumps(GOOD),
        "Here is the brief:\n```json\n" + json.dumps(GOOD, indent=2) + "\n```",
    ):
        draft = parse_answer(text)
        assert draft.error is None
        assert draft.claims == [
            Claim(
                "{fin:CRWD:revenue:yoy_pct}, driven by new customers.",
                "CRWD-0001535527-26-000031:23",
            )
        ]


def test_parse_answer_skips_malformed_entries() -> None:
    answer = {
        "claims": [
            "just a string",
            {"claim": 5, "passage_id": "x"},
            {"claim": "ok", "passage_id": None},
            {"claim": "ok", "passage_id": "p:1", "numbers": "ignored"},
        ]
    }
    assert parse_answer(json.dumps(answer)).claims == [Claim("ok", "p:1")]


def test_parse_answer_never_cuts_a_long_claim() -> None:
    long = "word " * 400
    (claim,) = parse_answer(json.dumps({"claims": [{"claim": long, "passage_id": "p:1"}]})).claims
    assert claim.claim == long.strip()  # the checker drops it whole


def test_the_prompt_asks_for_ids_and_no_typed_figures() -> None:
    text = load_prompt().text
    assert "{fin:" in text and "code_writes" in text
    assert '"numbers"' not in text


def test_deeply_nested_json_does_not_raise() -> None:
    text = '{"claims": [{"claim": "' + "[" * 10_000 + '"}]}' + "{" + "[" * 10_000
    assert isinstance(parse_answer(text).claims, list)


@pytest.mark.parametrize("text", [None, "", "no json", '{"summary": "x"}', '{"claims": 3}'])
def test_unusable_answers_are_errors(text: str | None) -> None:
    draft = parse_answer(text)
    assert draft.error is not None and draft.claims == []


def test_mcp_config_scopes_the_server_to_one_filing(store: FilingStore) -> None:
    meta = store.filings("CRWD")[1]
    cfg = mcp_servers(store.root, meta, "http://127.0.0.1:1234")["tenq"]
    assert cfg["command"] == sys.executable
    assert cfg["env"]["TENQ_ACCESSION"] == meta.accession
    assert cfg["env"]["TENQ_TICKER"] == "CRWD"
    assert Path(cfg["env"]["TENQ_ROOT"]).is_absolute()
    assert meta.accession in user_message(meta)
