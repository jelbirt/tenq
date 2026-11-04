from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tenq.docstore import DocStore
from tenq.filings import FilingStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def store(tmp_path: Path) -> FilingStore:
    """A writable copy of the saved CRWD filings (six, newest has MD&A text)."""
    root = tmp_path / "filings"
    shutil.copytree(FIXTURES / "filings", root)
    return FilingStore(root)


QA_DOC = "3ACME_2020_10K"  # starts with a digit, as 3M_2018_10K does
QA_PAGES = [
    "Acme Widgets annual report\nMade up for tests",
    "",
    "Consolidated statement of cash flows\nPurchases of widget presses 1,234",
    "Risk factors\nWidget demand may fall",
]


@pytest.fixture
def docs(tmp_path: Path) -> DocStore:
    """A saved made-up document: four pages, page 1 empty, no PDF needed."""
    s = DocStore(tmp_path / "docs")
    s.save(QA_DOC, "a" * 64, QA_PAGES, 0.1)
    return s
