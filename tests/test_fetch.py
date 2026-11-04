from __future__ import annotations

import math
from pathlib import Path

from tenq.edgar import facts_from_frame
from tenq.fetch import sync
from tenq.filings import Fact, FilingMeta, FilingStore


class FakeSource:
    def __init__(self, metas: list[FilingMeta]) -> None:
        self.metas = metas
        self.fetched: list[str] = []
        self.broken: set[str] = set()

    def list_filings(self, ticker: str, limit: int) -> list[FilingMeta]:
        return self.metas[:limit]

    def fetch(self, meta: FilingMeta) -> tuple[str, list[Fact]]:
        self.fetched.append(meta.accession)
        if meta.accession in self.broken:
            raise RuntimeError("EDGAR hiccup")
        return f"Text of {meta.accession}\n", []

    def accepted_at(self, meta: FilingMeta) -> str:
        return "2026-11-03T16:05:00-05:00"


def _meta(n: int) -> FilingMeta:
    return FilingMeta(
        "TEST", 1, "Test Co", "10-Q", f"0000000001-26-00000{n}", "2026-01-01",
        f"2026-0{n}-30", "https://x",
    )  # fmt: skip


def test_sync_saves_new_filings_and_skips_saved_ones(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    source = FakeSource([_meta(3), _meta(2), _meta(1)])
    assert [m.accession for m in sync(source, store, "TEST", depth=2)] == [
        "0000000001-26-000003",
        "0000000001-26-000002",
    ]
    source.metas.insert(0, _meta(4))
    new = sync(source, store, "TEST", depth=2)
    assert [m.accession for m in new] == ["0000000001-26-000004"]
    assert source.fetched[-1] == "0000000001-26-000004"
    assert len(source.fetched) == 3


def test_sync_skips_a_failed_filing_and_retries_it_next_run(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    source = FakeSource([_meta(2), _meta(1)])
    source.broken.add("0000000001-26-000002")
    assert [m.accession for m in sync(source, store, "TEST", depth=2)] == ["0000000001-26-000001"]
    source.broken.clear()
    assert [m.accession for m in sync(source, store, "TEST", depth=2)] == ["0000000001-26-000002"]


def test_sync_saves_each_filings_acceptance_time(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    sync(FakeSource([_meta(1)]), store, "TEST", depth=1)
    assert [m.accepted for m in store.filings("TEST")] == ["2026-11-03T16:05:00-05:00"]


def test_an_unknown_acceptance_time_still_saves_the_filing(tmp_path: Path) -> None:
    class NoStamp(FakeSource):
        def accepted_at(self, meta: FilingMeta) -> str:
            raise TimeoutError("slow header")

    store = FilingStore(tmp_path)
    assert [m.accession for m in sync(NoStamp([_meta(1)]), store, "TEST", depth=1)] == [
        "0000000001-26-000001"
    ]
    assert [m.accepted for m in store.filings("TEST")] == [""]


def test_facts_from_frame_keeps_the_most_precise_duplicate() -> None:
    base = {
        "concept": "us-gaap:Revenues",
        "unit_ref": "USD",
        "is_dimensioned": False,
        "period_start": "2025-05-01",
        "period_end": "2025-07-31",
        "period_instant": None,
    }
    rows = [
        {**base, "numeric_value": 1.0e9, "decimals": "-8"},  # rounded, MD&A
        {**base, "numeric_value": 1009.8e6, "decimals": "-5"},  # statement
        {**base, "numeric_value": 1.01e9, "decimals": "-7"},
    ]
    (fact,) = facts_from_frame(iter(rows))
    assert fact.value == 1009.8e6
    assert fact.decimals == -5


def test_facts_from_frame_keeps_undimensioned_numbers_once() -> None:
    base = {
        "concept": "us-gaap:GrossProfit",
        "numeric_value": 1096895000.0,
        "unit_ref": "usd",
        "period_start": "2026-05-01",
        "period_end": "2026-07-31",
        "period_instant": math.nan,
        "decimals": "-3",
        "is_dimensioned": False,
    }
    rows = [
        base,
        dict(base),  # the same fact on a second statement
        {**base, "is_dimensioned": True, "numeric_value": 1.0},
        {**base, "concept": "us-gaap:SomeTextBlock", "numeric_value": math.nan},
        {**base, "concept": "dei:EntityCommonStockSharesOutstanding"},
    ]
    facts = facts_from_frame(iter(rows))
    assert facts == [
        Fact("us-gaap:GrossProfit", 1096895000.0, "usd", "2026-05-01", "2026-07-31", None, -3)
    ]
