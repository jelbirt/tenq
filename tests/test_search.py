from __future__ import annotations

import shutil
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

import tenq.search as search
from tenq.filings import FilingStore, index_prefix
from tenq.search import Xerj, hits_from_response, text_field, xerj_node

LATEST = "CRWD-0001535527-26-000031"


def _hit(path: str, line: int, score: float, body: str = "...") -> dict[str, object]:
    # the fields xerj autoindex writes, as seen in a live response
    return {
        "_index": "tenq-crwd-t-docs",
        "_score": score,
        "_source": {"ax_path": path, "start_line": str(line), "body": body, "section": "3"},
    }


def test_hits_map_to_tenq_passages_best_first_once_each(store: FilingStore) -> None:
    passages = store.passages(store.filings("CRWD")[0])
    p1, p2 = passages[1], passages[3]
    resp = {
        "hits": {
            "hits": [
                _hit(f"{LATEST}.txt", p2.line, 9.0),
                _hit(f"{LATEST}.txt", p2.line, 8.0),  # second chunk of the same passage
                _hit(f"{LATEST}.txt", p1.line, 7.0),
            ]
        }
    }
    hits = hits_from_response(resp, store, "revenue", k=5)
    assert [h.passage_id for h in hits] == [p2.id, p1.id]
    assert hits[0].score == 9.0
    assert hits[0].text == p2.text


def test_a_chunk_spanning_passages_maps_to_the_one_matching_the_query(
    store: FilingStore,
) -> None:
    passages = store.passages(store.filings("CRWD")[0])
    heading, para = passages[0], passages[1]  # "ITEM 2. MANAGEMENT..." then its first paragraph
    span = "\n" * (para.line - heading.line + 1)  # chunk runs from the heading into the paragraph
    resp = {"hits": {"hits": [_hit(f"{LATEST}.txt", heading.line, 5.0, body=span)]}}
    (hit,) = hits_from_response(resp, store, "revised previously issued statements", k=5)
    assert hit.passage_id == para.id
    (hit,) = hits_from_response(resp, store, "management discussion", k=5)
    assert hit.passage_id == heading.id


def test_hits_drop_what_does_not_resolve(store: FilingStore) -> None:
    resp = {
        "hits": {
            "hits": [
                _hit("CRWD-0001535527-99-999999.txt", 1, 9.0),  # no such filing
                _hit(f"{LATEST}.txt", 10**6, 8.0),  # past the end
                {"_score": 7.0, "_source": {"body": "no path"}},
            ]
        }
    }
    assert hits_from_response(resp, store, "revenue", k=5) == []
    assert hits_from_response({}, store, "revenue", k=5) == []


def test_hits_stop_at_k(store: FilingStore) -> None:
    passages = store.passages(store.filings("CRWD")[0])
    resp = {"hits": {"hits": [_hit(f"{LATEST}.txt", p.line, 1.0) for p in passages]}}
    assert len(hits_from_response(resp, store, "revenue", k=2)) == 2


def test_a_repeat_chunk_does_not_fall_back_to_a_neighbour(store: FilingStore) -> None:
    passages = store.passages(store.filings("CRWD")[0])
    heading, para = passages[0], passages[1]
    span = "\n" * (para.line - heading.line + 1)
    resp = {
        "hits": {
            "hits": [
                _hit(f"{LATEST}.txt", para.line, 9.0),
                # covers the heading too, but its best passage was already returned
                _hit(f"{LATEST}.txt", heading.line, 5.0, body=span),
            ]
        }
    }
    hits = hits_from_response(resp, store, "revised previously issued statements", k=5)
    assert [h.passage_id for h in hits] == [para.id]


def test_a_text_schema_chunk_spans_to_its_end_line(store: FilingStore) -> None:
    # for some corpora xerj names the field `text` and records `end_line` (an int)
    passages = store.passages(store.filings("CRWD")[0])
    heading, para = passages[0], passages[1]
    hit = _hit(f"{LATEST}.txt", heading.line, 5.0)
    src = hit["_source"]
    assert isinstance(src, dict)
    del src["body"]
    src.update(text="ITEM 2.", start_line=heading.line, end_line=para.line)
    (got,) = hits_from_response({"hits": {"hits": [hit]}}, store, "revised previously issued", k=5)
    assert got.passage_id == para.id


@pytest.mark.parametrize(
    ("properties", "field"),
    [({"body": {}, "start_line": {}}, "body"), ({"text": {}, "impl": {}}, "text")],
)
def test_the_text_field_is_read_from_the_mapping(properties: dict[str, object], field: str) -> None:
    mapping = {"tenq-crwd-t-docs": {"mappings": {"properties": properties}}}
    assert text_field(mapping) == field


def test_a_key_with_no_index_has_no_hits(
    store: FilingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(url: str, body: object = None, timeout: float = 10) -> object:
        raise urllib.error.HTTPError(url, 404, "no such index", Message(), None)

    monkeypatch.setattr(search, "_request", missing)
    x = Xerj("http://127.0.0.1:1", store)
    assert x.search("CRWD", "revenue") == []
    assert x._fields == {}  # nothing cached: a later index is still found


def test_a_mapping_without_a_text_field_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="no text field"):
        text_field({"i": {"mappings": {"properties": {"start_line": {}}}}})


def test_hits_from_another_ticker_are_dropped(store: FilingStore) -> None:
    p = store.passages(store.filings("CRWD")[0])[1]
    resp = {"hits": {"hits": [_hit(f"{LATEST}.txt", p.line, 1.0)]}}
    assert hits_from_response(resp, store, "revenue", k=5, key="CRW") == []
    assert len(hits_from_response(resp, store, "revenue", k=5, key="CRWD")) == 1


def test_index_prefix_is_xerj_shaped_and_never_a_sibling() -> None:
    assert index_prefix("CRWD") == "tenq-crwd-t"
    assert index_prefix("BRK.B") == "tenq-brk-b-t"
    assert not index_prefix("BRK.B").startswith(index_prefix("BRK") + "-")
    for bad in ("crwd", "A,B", "A*", "../X", ""):
        with pytest.raises(ValueError):
            index_prefix(bad)


def test_node_refuses_a_port_already_in_use() -> None:
    import socket

    from tenq import search

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        if shutil.which("xerj") is None:
            pytest.skip("xerj not installed")
        with pytest.raises(RuntimeError, match="in use"), search.xerj_node(port):
            pass


@pytest.mark.skipif(shutil.which("xerj") is None, reason="xerj not installed")
def test_xerj_indexes_and_finds_a_passage(store: FilingStore, tmp_path: Path) -> None:
    """A real xerj node on loopback: index the fixture text, search it."""
    with xerj_node() as url:
        xerj = Xerj(url, store)
        xerj.index("CRWD", tmp_path / "xerj-state")
        hits = xerj.search("CRWD", "revised previously issued financial statements", k=3)
    assert hits
    assert hits[0].passage_id.startswith(f"{LATEST}:")
    assert store.passage(hits[0].passage_id) is not None
    assert "revised" in hits[0].text
