from __future__ import annotations

import json
from typing import Any

import anyio
from conftest import QA_DOC
from mcp.types import CallToolResult

from tenq.docstore import DocStore
from tenq.filings import FilingStore
from tenq.mcp_server import build_doc_server, build_server
from tenq.search import SearchHit

LATEST = "CRWD-0001535527-26-000031"


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
    assert table["metrics"]["revenue"]["current"] > 0


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


class FakeDocSearch:
    """Returns one hit from the document and one from another, best first."""

    def __init__(self, docs: DocStore) -> None:
        self.docs = docs
        self.calls: list[tuple[str, str, int]] = []

    def search(self, key: str, query: str, k: int = 5) -> list[SearchHit]:
        self.calls.append((key, query, k))
        (p,) = [p for p in self.docs.passages(QA_DOC) if "widget presses" in p.text]
        return [SearchHit("OTHER_2020_10K:3", 9.0, "elsewhere"), SearchHit(p.id, 2.0, p.text)]


def _doc_call(docs: DocStore, name: str, args: dict[str, Any]) -> Any:
    server = build_doc_server(docs, QA_DOC, FakeDocSearch(docs))

    async def go() -> Any:
        result = await server.call_tool(name, args)
        assert isinstance(result, CallToolResult) and not result.is_error, result
        return result

    result = anyio.run(go)
    if result.structured_content is not None:
        sc = result.structured_content
        return sc.get("result", sc)
    return json.loads(result.content[0].text)


def test_document_mode_has_only_search_and_read(docs: DocStore) -> None:
    server = build_doc_server(docs, QA_DOC, FakeDocSearch(docs))
    tools = anyio.run(server.list_tools)
    assert sorted(t.name for t in tools) == ["read_passage", "search_filing"]
    for t in tools:
        assert t.annotations is not None and t.annotations.read_only_hint


def test_document_search_keeps_this_document_and_gives_pages(docs: DocStore) -> None:
    fake = FakeDocSearch(docs)
    server = build_doc_server(docs, QA_DOC, fake)
    anyio.run(server.call_tool, "search_filing", {"query": "presses", "k": 99})
    assert fake.calls == [(QA_DOC, "presses", 20)]
    (hit,) = _doc_call(docs, "search_filing", {"query": "presses"})
    assert hit["passage_id"].startswith(f"{QA_DOC}:") and hit["page"] == 2


def test_document_read_gives_the_page_and_stays_in_the_document(docs: DocStore) -> None:
    (p,) = [p for p in docs.passages(QA_DOC) if "Risk factors" in p.text]
    got = _doc_call(docs, "read_passage", {"passage_id": p.id})
    assert got["found"] and got["page"] == 3 and got["text"].startswith("Risk factors")
    docs.save("OTHER_2020_10K", "b" * 64, ["other company"], 0.1)
    (q,) = docs.passages("OTHER_2020_10K")
    assert _doc_call(docs, "read_passage", {"passage_id": q.id})["found"] is False
    assert _doc_call(docs, "read_passage", {"passage_id": "../x:1"})["found"] is False


def test_an_unsaved_document_is_refused(docs: DocStore) -> None:
    import pytest

    with pytest.raises(LookupError):
        build_doc_server(docs, "NOPE_2020_10K", FakeDocSearch(docs))


def test_the_real_entrypoint_serves_a_document_over_stdio(docs: DocStore) -> None:
    import os
    import sys

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    (p,) = [p for p in docs.passages(QA_DOC) if "Risk factors" in p.text]
    env = {
        **os.environ,
        "TENQ_DOCS": str(docs.root),
        "TENQ_DOC": QA_DOC,
        "TENQ_XERJ_URL": "http://127.0.0.1:9",  # unused by the tool called here
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "tenq.mcp_server"], env=env)

    async def go() -> tuple[set[str], Any]:
        async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            got = await session.call_tool("read_passage", {"passage_id": p.id})
            return tools, got

    tools, got = anyio.run(go)
    assert tools == {"search_filing", "read_passage"}
    assert isinstance(got, CallToolResult) and not got.is_error
