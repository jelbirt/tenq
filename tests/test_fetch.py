from __future__ import annotations

import math
from pathlib import Path

from tenq.edgar import facts_from_frame
from tenq.fetch import sync, sync_foreign, sync_reports
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


REV = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"


def _arm(acc: str, form: str, period: str, filed: str) -> FilingMeta:
    return FilingMeta("ARM", 2, "Arm Holdings plc", form, acc, filed, period, "https://x")


def _rev(start: str, end: str, currency: str = "USD", concept: str = REV) -> Fact:
    return Fact(concept, 1.0, "USD", start, end, None, -6, currency)


# ARM's pattern, newest first: results 6-Ks for its first three quarters, the
# 20-F for the year (its fourth-quarter results 6-K is not tagged, so EDGAR's
# listing never offers it).
Q2 = _arm("q2", "6-K", "2026-09-30", "2026-11-04")
Q1 = _arm("q1", "6-K", "2026-06-30", "2026-07-29")
FY = _arm("fy", "20-F", "2026-03-31", "2026-05-26")
Q3 = _arm("q3", "6-K", "2025-12-31", "2026-02-04")
Q2_25 = _arm("q2-25", "6-K", "2025-09-30", "2025-11-05")
FACTS = {
    "q2": [_rev("2026-04-01", "2026-09-30"), _rev("2026-07-01", "2026-09-30")],
    "q1": [_rev("2026-04-01", "2026-06-30")],
    "fy": [_rev("2025-04-01", "2026-03-31")],
    "q3": [_rev("2025-04-01", "2025-12-31"), _rev("2025-10-01", "2025-12-31")],
    "q2-25": [_rev("2025-07-01", "2025-09-30")],
}


class Foreign(FakeSource):
    def __init__(self, metas: list[FilingMeta], facts: dict[str, list[Fact]]) -> None:
        super().__init__(metas)
        self.facts = facts
        self.listed: list[int] = []

    def list_filings(self, ticker: str, limit: int) -> list[FilingMeta]:
        return [m for m in self.metas if m.form in ("10-Q", "10-K")][:limit]

    def list_foreign(self, ticker: str, limit: int) -> list[FilingMeta]:
        self.listed.append(limit)
        return [m for m in self.metas if m.form not in ("10-Q", "10-K")][:limit]

    def fetch(self, meta: FilingMeta) -> tuple[str, list[Fact]]:
        self.fetched.append(meta.accession)
        if meta.accession in self.broken:
            raise RuntimeError("EDGAR hiccup")
        return f"Text of {meta.accession}\n", self.facts.get(meta.accession, [])


def _saved(store: FilingStore) -> list[tuple[str, str]]:
    return [(m.accession, m.form) for m in store.filings("ARM")]


def test_tagged_quarterly_6ks_and_the_20f_are_saved_like_10qs_and_a_10k(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    source = Foreign([Q2, Q1, FY, Q3, Q2_25], FACTS)
    new = sync_foreign(source, store, "ARM", depth=6)
    assert [m.accession for m in new] == ["q2", "q1", "fy", "q3", "q2-25"]
    assert {a for a, _ in _saved(store)} == {"q2", "q1", "fy", "q3", "q2-25"}
    assert [m.accepted for m in new] == ["2026-11-03T16:05:00-05:00"] * 5
    assert store.facts(new[0]) == FACTS["q2"]


def test_only_the_newest_reports_up_to_the_depth_are_saved(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    older = _arm("fy-25", "20-F", "2025-03-31", "2025-05-28")  # past the window
    facts = {**FACTS, "fy-25": [_rev("2024-04-01", "2025-03-31")]}
    source = Foreign([Q2, Q1, FY, Q3, Q2_25, older], facts)
    assert [m.accession for m in sync_foreign(source, store, "ARM", depth=3)] == ["q2", "q1", "fy"]
    assert not {"q3", "q2-25", "fy-25"} & set(source.fetched)


def test_a_6k_without_a_dollar_quarter_is_not_a_quarterly_report(tmp_path: Path) -> None:
    """A cover page, a half-year (MNDY), euros, or names the sheet does not read
    (IFRS, TRI): none of them counts, and the window skips past it."""
    half = _arm("half", "6-K", "2026-06-30", "2026-08-10")
    euro = _arm("euro", "6-K", "2026-06-30", "2026-08-11")
    ifrs = _arm("ifrs", "6-K", "2026-06-30", "2026-08-12")
    cover = _arm("cover", "6-K", "2026-06-30", "2026-08-13")
    facts = {
        **FACTS,
        "half": [_rev("2026-01-01", "2026-06-30")],
        "euro": [_rev("2026-04-01", "2026-06-30", "EUR")],
        "ifrs": [_rev("2026-04-01", "2026-06-30", concept="ifrs-full:Revenue")],
    }
    store = FilingStore(tmp_path)
    source = Foreign([cover, ifrs, euro, half, Q1, FY], facts)
    assert [m.accession for m in sync_foreign(source, store, "ARM", depth=2)] == ["q1", "fy"]


def test_without_a_tagged_quarter_nothing_is_saved_not_even_the_20f(tmp_path: Path) -> None:
    """MNDY, NBIS and the rest: a 20-F alone would put year-old balances under
    the latest quarter, so the sheet keeps its foreign-company note instead."""
    half = _arm("half", "6-K", "2026-06-30", "2026-08-10")
    store = FilingStore(tmp_path)
    source = Foreign([half, FY], {"half": [_rev("2026-01-01", "2026-06-30")], **FACTS})
    assert sync_foreign(source, store, "ARM", depth=6) == []
    assert store.filings("ARM") == []
    assert source.fetched == ["half"]


def test_a_20f_without_a_dollar_year_is_left_out(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    source = Foreign([Q1, FY], {**FACTS, "fy": [_rev("2025-04-01", "2026-03-31", "EUR")]})
    assert [m.accession for m in sync_foreign(source, store, "ARM", depth=6)] == ["q1"]


def test_saved_reports_are_not_fetched_again_and_still_fill_the_window(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    sync_foreign(Foreign([Q1, FY, Q3], FACTS), store, "ARM", depth=3)
    source = Foreign([Q2, Q1, FY, Q3], FACTS)
    assert [m.accession for m in sync_foreign(source, store, "ARM", depth=3)] == ["q2"]
    assert source.fetched == ["q2"]


def test_a_failed_6k_is_skipped_and_retried_next_run(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    source = Foreign([Q2, Q1, FY], FACTS)
    source.broken.add("q2")
    assert [m.accession for m in sync_foreign(source, store, "ARM", depth=6)] == ["q1", "fy"]
    source.broken.clear()
    assert [m.accession for m in sync_foreign(source, store, "ARM", depth=6)] == ["q2"]


def test_a_company_with_10qs_never_reads_its_6ks(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    tenq = FilingMeta("ARM", 2, "Arm", "10-Q", "10q", "2026-11-04", "2026-09-30", "https://x")
    source = Foreign([tenq, Q2, FY], FACTS)
    assert [m.accession for m in sync_reports(source, store, "ARM", depth=6)] == ["10q"]
    assert source.listed == []


def test_a_foreign_company_keeps_syncing_its_6ks_once_some_are_saved(tmp_path: Path) -> None:
    store = FilingStore(tmp_path)
    assert [m.accession for m in sync_reports(Foreign([Q1, FY], FACTS), store, "ARM", 6)] == [
        "q1",
        "fy",
    ]
    later = Foreign([Q2, Q1, FY], FACTS)
    assert [m.accession for m in sync_reports(later, store, "ARM", 6)] == ["q2"]
    assert later.listed == [12]
