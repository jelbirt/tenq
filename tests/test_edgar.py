from dataclasses import dataclass, field
from typing import Any, ClassVar

import edgar
import pytest

from tenq.edgar import EdgarSource


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
    attachments: list[Att] = field(default_factory=list)

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
