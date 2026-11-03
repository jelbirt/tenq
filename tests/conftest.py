from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tenq.filings import FilingStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def store(tmp_path: Path) -> FilingStore:
    """A writable copy of the saved CRWD filings (six, newest has MD&A text)."""
    root = tmp_path / "filings"
    shutil.copytree(FIXTURES / "filings", root)
    return FilingStore(root)
