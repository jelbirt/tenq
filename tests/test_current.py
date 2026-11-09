from datetime import date
from pathlib import Path

import pytest

from tenq import current
from tenq.current import describe, sync_current
from tenq.filings import FilingMeta, FilingStore


def _meta(accession: str, filed: str, items: str = "2.02,9.01") -> FilingMeta:
    return FilingMeta(
        ticker="CRWD",
        cik=1535527,
        company="CrowdStrike Holdings, Inc.",
        form="8-K",
        accession=accession,
        filed=filed,
        period=filed,
        url="https://www.sec.gov/x",
        items=items,
    )


NEW = _meta("0001535527-26-000040", "2026-11-03")
MID = _meta("0001535527-26-000039", "2026-11-01", "5.02")
OLD = _meta("0001535527-26-000030", "2026-09-01")


class Source:
    def __init__(self, listed: list[FilingMeta], broken: set[str] | None = None) -> None:
        self.listed = listed
        self.broken = broken or set()
        self.fetched: list[str] = []

    def list_current(self, ticker: str, limit: int) -> list[FilingMeta]:
        assert ticker == "CRWD"
        return self.listed[:limit]

    def fetch_current(self, meta: FilingMeta) -> str:
        self.fetched.append(meta.accession)
        if meta.accession in self.broken:
            raise TimeoutError("slow")
        return f"FORM 8-K\n\nItem 2.02 Results\n\nEXHIBIT 99.1\n\nRevenue grew ({meta.accession})\n"

    def accepted_at(self, meta: FilingMeta) -> str:
        return "2026-11-03T16:05:00-05:00"


def test_8ks_filed_since_the_date_are_saved_with_passage_ids(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    source = Source([NEW, MID, OLD])
    got = sync_current(source, store, "CRWD", date(2026, 11, 1))
    assert [m.accession for m in got.reports] == [NEW.accession, MID.accession]
    assert got.new == got.reports and got.notes == []
    assert source.fetched == [NEW.accession, MID.accession]  # OLD never fetched
    saved = store.filings("CRWD")
    assert {m.items for m in saved} == {"2.02,9.01", "5.02"}
    assert {m.accepted for m in saved} == {"2026-11-03T16:05:00-05:00"}
    assert store.facts(saved[0]) == []  # 8-Ks carry no checked numbers
    exhibit = [p for p in store.passages(NEW) if "Revenue grew" in p.text]
    assert exhibit and exhibit[0].id == f"CRWD-{NEW.accession}:7"


def test_a_saved_8k_is_reported_again_but_not_refetched(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    sync_current(Source([NEW]), store, "CRWD", date(2026, 11, 1))
    again = Source([NEW])
    got = sync_current(again, store, "CRWD", date(2026, 11, 1))
    assert again.fetched == [] and got.new == []
    assert got.reports == store.filings("CRWD")  # as saved, with its acceptance time


def test_a_failed_8k_is_a_note_and_the_others_still_save(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    got = sync_current(Source([NEW, MID], {NEW.accession}), store, "CRWD", date(2026, 11, 1))
    assert [m.accession for m in got.reports] == [MID.accession]
    assert got.notes == [f"8-K {NEW.accession} unavailable for CRWD"]
    assert not store.has("CRWD", NEW.accession)  # next run retries it


def test_a_failed_listing_is_a_note(tmp_path: Path) -> None:
    class Down(Source):
        def list_current(self, ticker: str, limit: int) -> list[FilingMeta]:
            raise ConnectionError("edgar down")

    got = sync_current(Down([]), FilingStore(tmp_path), "CRWD", date(2026, 11, 1))
    assert got.reports == [] and got.notes == ["current reports (8-K, 6-K) unavailable for CRWD"]


def test_items_get_plain_names() -> None:
    assert describe("2.02,9.01") == [
        "Item 2.02 results of operations",
        "Item 9.01 financial statements and exhibits",
    ]
    assert describe(" 1.05 , 6.10") == [
        "Item 1.05 material cybersecurity incident",
        "Item 6.10 other item",
    ]
    assert describe("") == []


def test_the_cli_lists_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import tenq.edgar

    monkeypatch.setattr(tenq.edgar, "EdgarSource", lambda: Source([NEW]))
    assert current.main(["crwd", "--since", "2026-11-01", "--root", str(tmp_path)]) == 0
    assert f"2026-11-03 {NEW.accession} Item 2.02 results of operations" in capsys.readouterr().out


def test_an_8k_without_a_filing_date_is_a_note(tmp_path: Path) -> None:
    undated = _meta("0001535527-26-000041", "")
    got = sync_current(Source([undated, NEW]), FilingStore(tmp_path), "CRWD", date(2026, 11, 1))
    assert [m.accession for m in got.reports] == [NEW.accession]
    assert got.notes == [f"8-K {undated.accession} has no filing date"]


def test_a_full_listing_since_the_date_says_older_ones_were_not_checked(tmp_path: Path) -> None:
    got = sync_current(Source([NEW, MID]), FilingStore(tmp_path), "CRWD", date(2026, 11, 1), 2)
    assert got.notes == [
        "more than 2 current reports since 2026-11-01 for CRWD; older ones not checked"
    ]
    short = sync_current(Source([NEW, OLD]), FilingStore(tmp_path), "CRWD", date(2026, 11, 1), 2)
    assert short.notes == []  # the listing reached back past the date


def test_a_failed_save_is_a_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FilingStore(tmp_path)

    def full(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", full)
    got = sync_current(Source([NEW]), store, "CRWD", date(2026, 11, 1))
    assert got.reports == [] and got.notes == [f"8-K {NEW.accession} unavailable for CRWD"]


def test_8k_indexes_are_tagged_apart_from_10q_ones(tmp_path: Path) -> None:
    assert FilingStore(tmp_path).index_prefix("CRWD") == "tenq-crwd-t"
    assert FilingStore(tmp_path, tag=current.CURRENT_TAG).index_prefix("CRWD") == "tenq-crwd-c"
