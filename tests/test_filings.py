from __future__ import annotations

from pathlib import Path

from tenq.filings import (
    MAX_PASSAGE_LINES,
    Fact,
    FilingMeta,
    FilingStore,
    normalise_text,
    split_passages,
)

LATEST = "CRWD-0001535527-26-000031"


def _meta(accession: str = "0000000001-26-000001", period: str = "2026-07-31") -> FilingMeta:
    return FilingMeta("TEST", 1, "Test Co", "10-Q", accession, "2026-08-27", period, "https://x")


def test_normalise_is_deterministic_and_collapses_blank_runs() -> None:
    raw = "Title  \r\n\r\n\r\n\nBody line\t\n\n"
    assert normalise_text(raw) == "Title\n\nBody line\n"
    assert normalise_text(normalise_text(raw)) == normalise_text(raw)


def test_passage_ids_are_file_and_first_line() -> None:
    text = "alpha\nbeta\n\ngamma\n"
    ps = split_passages("DOC-0000000001-26-000001", text)
    assert [(p.id, p.text) for p in ps] == [
        ("DOC-0000000001-26-000001:1", "alpha\nbeta"),
        ("DOC-0000000001-26-000001:4", "gamma"),
    ]


def test_long_blocks_are_capped() -> None:
    text = "\n".join(f"row {i}" for i in range(MAX_PASSAGE_LINES + 5))
    ps = split_passages("D-0000000001-26-000001", text)
    assert [p.line for p in ps] == [1, MAX_PASSAGE_LINES + 1]


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    meta = _meta()
    fact = Fact("us-gaap:GrossProfit", 5.0, "usd", "2026-05-01", "2026-07-31", None, -3)
    assert not store.has("TEST", meta.accession)
    store.save(meta, "Heading\n\nSome text.\n", [fact])
    assert store.has("TEST", meta.accession)
    assert store.filings("TEST") == [meta]
    assert store.facts(meta) == [fact]
    assert store.passage(f"{meta.doc}:3") is not None


def test_fixture_store_lists_newest_first(store: FilingStore) -> None:
    metas = store.filings("CRWD")
    assert len(metas) == 6
    assert metas[0].accession == "0001535527-26-000031"
    assert [m.period for m in metas] == sorted((m.period for m in metas), reverse=True)


def test_passage_lookup_on_real_text(store: FilingStore) -> None:
    p = store.passage(f"{LATEST}:1")
    assert p is not None
    assert p.text.startswith("ITEM 2. MANAGEMENT")


def test_passage_lookup_rejects_bad_ids(store: FilingStore) -> None:
    assert store.passage(f"{LATEST}:2") is None  # a blank line, not a passage start
    assert store.passage(f"{LATEST}:999999") is None
    assert store.passage("CRWD-0001535527-99-999999:1") is None  # no such filing
    assert store.passage("../../etc/passwd:1") is None
    assert store.passage("not an id") is None


def test_passage_lookup_stays_inside_the_store(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "text"
    outside.mkdir()
    (outside / "..-0000000000-00-000000.txt").write_text("secret\n")
    store = FilingStore(root)
    assert store.passage("..-0000000000-00-000000:1") is None
    assert store.passage_at("..-0000000000-00-000000", 1) is None
    assert (
        store.passage(
            "CRWD-\u0661\u0661\u0661\u0661\u0661\u0661\u0661\u0661\u0661\u0661-26-000010:1"
        )
        is None
    )


def test_passage_at_maps_any_line_to_its_passage(store: FilingStore) -> None:
    first = store.passages(store.filings("CRWD")[0])[3]
    inside = store.passage_at(LATEST, first.line)
    assert inside == first
    assert store.passage_at(LATEST, 10**6) is None
