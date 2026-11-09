from dataclasses import dataclass, field
from typing import Any, ClassVar

import edgar
import pytest

from tenq.edgar import EdgarSource, currencies_of, facts_from_frame


@dataclass
class Att:
    document_type: str
    body: str

    def text(self) -> str:
        return self.body


@dataclass
class FakeFiling:
    form: str
    accession_no: str
    items: Any = "2.02,9.01"
    cik: int = 1535527
    company: str = "CrowdStrike Holdings, Inc."
    filing_date: str = "2026-11-03"
    period_of_report: str = "2026-11-03"
    homepage_url: str = "https://www.sec.gov/x"
    acceptance_datetime: Any = None
    attachments: list[Att] = field(default_factory=list)
    header: Any = None

    def text(self) -> str:
        return "FORM 8-K"


class FakeCompany:
    filings: ClassVar[list[FakeFiling]] = []

    def __init__(self, ticker: str) -> None:
        pass

    def get_filings(self, form: list[str], amendments: bool) -> list[FakeFiling]:
        assert amendments is False
        return [f for f in self.filings if f.form in form]


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> EdgarSource:
    monkeypatch.setattr(edgar, "Company", FakeCompany)
    monkeypatch.setattr(edgar, "set_identity", lambda identity: None)
    return EdgarSource("tenq test@example.com")


def test_8ks_list_with_their_items_and_10qs_without(source: EdgarSource) -> None:
    FakeCompany.filings = [
        FakeFiling("8-K", "0001535527-26-000040"),
        FakeFiling("10-Q", "0001535527-26-000031", items="2.02"),
        FakeFiling("8-K", "0001535527-26-000039", items=float("nan")),
    ]
    current = source.list_current("crwd", 5)
    assert [(m.ticker, m.form, m.items) for m in current] == [
        ("CRWD", "8-K", "2.02,9.01"),
        ("CRWD", "8-K", ""),
    ]
    (quarterly,) = source.list_filings("CRWD", 5)
    assert quarterly.items == ""


def test_an_8k_text_carries_its_press_release_exhibits_only(source: EdgarSource) -> None:
    filing = FakeFiling(
        "8-K",
        "0001535527-26-000040",
        attachments=[
            Att("8-K", "main again"),
            Att("EX-99.1", "Press release: revenue grew"),
            Att("EX-101.SCH", "<xsd/>"),
            Att("EX-99.2", "Slides"),
            Att("GRAPHIC", ""),
        ],
    )
    FakeCompany.filings = [filing]
    (meta,) = source.list_current("CRWD", 1)
    text = source.fetch_current(meta)
    assert text == (
        "FORM 8-K\n\nEXHIBIT 99.1\n\nPress release: revenue grew\n\nEXHIBIT 99.2\n\nSlides"
    )


def test_an_unreadable_or_empty_exhibit_keeps_the_8k_text(source: EdgarSource) -> None:
    class Broken(Att):
        def text(self) -> str:
            raise ValueError("bad encoding")

    FakeCompany.filings = [
        FakeFiling(
            "8-K",
            "0001535527-26-000040",
            attachments=[Broken("EX-99.1", ""), Att("EX-99.2", ""), Att("EX-99.3", "Letter")],
        )
    ]
    (meta,) = source.list_current("CRWD", 1)
    assert source.fetch_current(meta) == "FORM 8-K\n\nEXHIBIT 99.3\n\nLetter"


@dataclass
class Header:
    acceptance_datetime: Any


def test_the_acceptance_time_comes_from_the_header_in_eastern_time(
    source: EdgarSource,
) -> None:
    from datetime import UTC, datetime

    FakeCompany.filings = [
        # the listing's own value is ignored: it is hours off for some filers
        FakeFiling(
            "8-K",
            "0001535527-26-000040",
            acceptance_datetime=datetime(2026, 11, 4, 2, 5, tzinfo=UTC),
            header=Header(datetime(2026, 11, 3, 16, 5)),
        ),
        FakeFiling("8-K", "0001535527-26-000041", header=Header(datetime(2026, 7, 3, 16, 5))),
        FakeFiling("8-K", "0001535527-26-000042", header=Header("2026-11-03")),
    ]
    metas = source.list_current("CRWD", 5)
    assert [m.accepted for m in metas] == ["", "", ""]
    assert [source.accepted_at(m) for m in metas] == [
        "2026-11-03T16:05:00-05:00",
        "2026-07-03T16:05:00-04:00",
        "",
    ]


def test_a_foreign_filers_6k_lists_and_reads_like_an_8k(source: EdgarSource) -> None:
    """L11: a 6-K is a foreign company's current report. It lists with the 8-Ks
    (no items: a 6-K has none) and reads the same way, text plus EX-99."""
    FakeCompany.filings = [
        FakeFiling(
            "6-K",
            "0000937966-26-000050",
            items="",
            attachments=[Att("EX-99.1", "Q3 results"), Att("EX-101.INS", "<xbrl/>")],
        ),
        FakeFiling("20-F", "0000937966-26-000010", items=""),
    ]
    (meta,) = source.list_current("ASML", 5)
    assert (meta.form, meta.items) == ("6-K", "")
    assert source.fetch_current(meta) == "FORM 8-K\n\nEXHIBIT 99.1\n\nQ3 results"


def test_a_facts_currency_comes_from_the_filings_unit_definition() -> None:
    """A filer names its own unit ids, so "usd" or "U_EUR" says nothing for sure;
    the unit's iso4217 measure does. Shares and per-share units get none."""
    units = {
        "usd": {"type": "simple", "measure": "iso4217:USD"},
        "U1": {"type": "simple", "measure": "iso4217:EUR"},
        "shares": {"type": "simple", "measure": "xbrli:shares"},
        "usdPerShare": {
            "type": "divide",
            "numerator": ["iso4217:USD"],
            "denominator": ["xbrli:shares"],
        },
    }
    assert currencies_of(units) == {"usd": "USD", "U1": "EUR"}
    row = {"concept": "us-gaap:Revenues", "period_start": "2026-07-01", "period_end": "2026-09-30"}
    rows = [
        row | {"numeric_value": 10.0, "unit_ref": "usd"},
        row | {"numeric_value": 9.0, "unit_ref": "U1"},
        row | {"concept": "us-gaap:SharesOutstanding", "numeric_value": 5.0, "unit_ref": "shares"},
    ]
    got = {(f.unit, f.currency) for f in facts_from_frame(iter(rows), currencies_of(units))}
    assert got == {("usd", "USD"), ("U1", "EUR"), ("shares", None)}
    assert {f.currency for f in facts_from_frame(iter(rows))} == {None}
