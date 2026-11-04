from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tenq.brief import (
    ALLOWED_TOOLS,
    Claim,
    NumberRef,
    load_prompt,
    mcp_servers,
    parse_answer,
    user_message,
)
from tenq.filings import FilingStore


def test_the_shipped_prompt_has_a_version_and_names_every_tool() -> None:
    p = load_prompt()
    assert p.version == "brief-v2"
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
            "claim": "Revenue  grew\n26%.",
            "passage_id": "CRWD-0001535527-26-000031:23 ",
            "numbers": [{"metric": "revenue", "field": "yoy_pct", "value": 0.2583}],
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
                "Revenue grew 26%.",
                "CRWD-0001535527-26-000031:23",
                (NumberRef("revenue", "yoy_pct", 0.2583),),
            )
        ]


def test_parse_answer_skips_malformed_entries() -> None:
    answer = {
        "claims": [
            "just a string",
            {"claim": 5, "passage_id": "x"},
            {"claim": "ok", "passage_id": "p:1", "numbers": [
                {"metric": "revenue", "field": "current", "value": "1.4B"},
                {"metric": "revenue", "field": "current", "value": True},
                {"metric": "revenue", "field": "current", "value": 1},
            ]},
        ]
    }  # fmt: skip
    (claim,) = parse_answer(json.dumps(answer)).claims
    assert claim.numbers == (NumberRef("revenue", "current", 1.0),)


def test_parse_answer_drops_non_finite_and_huge_values() -> None:
    nums = '[{"metric": "revenue", "field": "current", "value": %s}]'
    for value in ("NaN", "Infinity", "-Infinity", "1" + "0" * 400):
        text = '{"claims": [{"claim": "ok", "passage_id": "p:1", "numbers": %s}]}' % (nums % value)
        (claim,) = parse_answer(text).claims
        assert claim.numbers == ()


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
