from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

import anyio
from mcp.types import CallToolResult

from tenq.brief import load_prompt, user_message
from tenq.filings import FilingMeta, FilingStore
from tenq.mcp_server import build_server
from tenq.search import SearchHit

LATEST = "CRWD-0001535527-26-000031"

# Code tells the brief model more than brief.md: the server's instructions, each
# tool's name, title, description and arguments, and the user turn that orders
# the brief. Their sha256 is pinned to brief.md's version, so changing any of
# them needs a new brief version. Descriptions are hashed as inspect.cleandoc
# leaves them, which is how Python 3.13 and later store docstrings, so the hash
# is the same on any Python; only an indent-only edit goes unseen.
KNOWN_BRIEF_CODE_TEXT = {
    "brief-v4": "5ec6627abd3f9e24d1155f2dc0383e0aa75d1cb83328f9eae3bd73ece521cb49",
}


class FakeSearch:
    def __init__(self, store: FilingStore) -> None:
        self.store = store
        self.calls: list[tuple[str, str, int]] = []

    def search(self, ticker: str, query: str, k: int = 5) -> list[SearchHit]:
        self.calls.append((ticker, query, k))
        p = self.store.passages(self.store.filings(ticker)[0])[1]
        return [SearchHit(p.id, 3.14159, p.text)]


def _call(store: FilingStore, name: str, args: dict[str, Any], accession: str | None = None) -> Any:
    server = build_server(store, "CRWD", FakeSearch(store), accession)

    async def go() -> Any:
        result = await server.call_tool(name, args)
        assert isinstance(result, CallToolResult) and not result.is_error, result
        return result

    result = anyio.run(go)
    if result.structured_content is not None:
        sc = result.structured_content
        return sc.get("result", sc)
    return json.loads(result.content[0].text)


def test_exactly_three_read_only_tools(store: FilingStore) -> None:
    server = build_server(store, "CRWD", FakeSearch(store))
    tools = anyio.run(server.list_tools)
    assert sorted(t.name for t in tools) == ["get_metrics", "read_passage", "search_filing"]
    for t in tools:
        assert t.annotations is not None and t.annotations.read_only_hint


def test_get_metrics_returns_the_newest_filing_table(store: FilingStore) -> None:
    table = _call(store, "get_metrics", {})
    assert table["accession"] == "0001535527-26-000031"
    (revenue,) = [n for n in table["numbers"] if n["id"] == "fin:CRWD:revenue:current"]
    assert revenue["code_writes"].startswith("CRWD's revenue was $")
    assert "value" not in revenue  # the model gets ids and code's words, no figure to copy


def test_search_returns_citable_ids_and_clamps_k(store: FilingStore) -> None:
    fake = FakeSearch(store)
    server = build_server(store, "CRWD", fake)
    anyio.run(server.call_tool, "search_filing", {"query": "revenue", "k": 500})
    assert fake.calls == [("CRWD", "revenue", 20)]
    (hit,) = _call(store, "search_filing", {"query": "revenue"})
    assert hit["passage_id"].startswith(f"{LATEST}:")
    assert hit["score"] == 3.142
    assert store.passage(hit["passage_id"]) is not None


def test_read_passage_found_and_not_found(store: FilingStore) -> None:
    got = _call(store, "read_passage", {"passage_id": f"{LATEST}:1"})
    assert got["found"] and got["text"].startswith("ITEM 2. MANAGEMENT")
    assert _call(store, "read_passage", {"passage_id": f"{LATEST}:2"})["found"] is False
    assert _call(store, "read_passage", {"passage_id": "../../etc/passwd"})["found"] is False


def test_read_passage_is_limited_to_the_run_ticker(store: FilingStore, tmp_path: Any) -> None:
    other = store.root / "ZZZ" / "text"
    other.mkdir(parents=True)
    (other / "ZZZ-0000000001-26-000001.txt").write_text("other company\n")
    assert store.passage("ZZZ-0000000001-26-000001:1") is not None
    got = _call(store, "read_passage", {"passage_id": "ZZZ-0000000001-26-000001:1"})
    assert got["found"] is False


OLDER = "0001535527-26-000025"


def test_a_backfilled_run_cannot_see_later_filings(store: FilingStore) -> None:
    table = _call(store, "get_metrics", {}, accession=OLDER)
    assert table["accession"] == OLDER
    newest = _call(store, "read_passage", {"passage_id": f"{LATEST}:1"}, accession=OLDER)
    assert newest["found"] is False
    assert _call(store, "search_filing", {"query": "revenue"}, accession=OLDER) == []


def test_unknown_accession_is_refused(store: FilingStore) -> None:
    import pytest

    with pytest.raises(LookupError):
        build_server(store, "CRWD", FakeSearch(store), "0000000000-00-000000")


def test_the_real_entrypoint_serves_over_stdio(store: FilingStore) -> None:
    import os
    import sys

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {
        **os.environ,
        "TENQ_ROOT": str(store.root),
        "TENQ_TICKER": "crwd",
        "TENQ_XERJ_URL": "http://127.0.0.1:9",  # unused by the two tools called here
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "tenq.mcp_server"], env=env)

    async def go() -> tuple[set[str], Any]:
        async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            got = await session.call_tool("read_passage", {"passage_id": f"{LATEST}:1"})
            return tools, got

    tools, got = anyio.run(go)
    assert tools == {"get_metrics", "search_filing", "read_passage"}
    assert isinstance(got, CallToolResult) and not got.is_error


def test_what_code_tells_the_brief_model_cannot_change_without_a_new_version(
    store: FilingStore,
) -> None:
    server = build_server(store, "CRWD", FakeSearch(store))
    tools = sorted(anyio.run(server.list_tools), key=lambda t: t.name)
    meta = FilingMeta(
        "CRWD",
        1535527,
        "CrowdStrike Holdings, Inc.",
        "10-Q",
        "0001535527-26-000031",
        "2026-09-03",
        "2026-07-31",
        "https://www.sec.gov/",
    )
    told = {
        "instructions": server.instructions,
        "tools": [
            [t.name, t.title, inspect.cleandoc(t.description or ""), t.input_schema] for t in tools
        ],
        "user": user_message(meta),
    }
    got = hashlib.sha256(json.dumps(told, sort_keys=True).encode()).hexdigest()
    version = load_prompt().version
    assert version in KNOWN_BRIEF_CODE_TEXT, "new brief version: add its hash above"
    assert got == KNOWN_BRIEF_CODE_TEXT[version], "what code tells the brief changed: bump brief.md"
