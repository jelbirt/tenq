from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tenq.filings import FilingStore
from tenq.search import Xerj, hits_from_response, index_prefix, xerj_node

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


def test_hits_from_another_ticker_are_dropped(store: FilingStore) -> None:
    p = store.passages(store.filings("CRWD")[0])[1]
    resp = {"hits": {"hits": [_hit(f"{LATEST}.txt", p.line, 1.0)]}}
    assert hits_from_response(resp, store, "revenue", k=5, ticker="CRW") == []
    assert len(hits_from_response(resp, store, "revenue", k=5, ticker="CRWD")) == 1


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
