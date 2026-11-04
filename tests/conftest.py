from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tenq.filings import FilingStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_real_claude(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shadow the real CLI: a test that reaches it by mistake fails, not calls a model."""
    bin_dir = tmp_path_factory.mktemp("no-claude")
    stub = bin_dir / "claude"
    stub.write_text("#!/bin/sh\necho 'the real claude is not for tests' >&2\nexit 97\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def store(tmp_path: Path) -> FilingStore:
    """A writable copy of the saved CRWD filings (six, newest has MD&A text)."""
    root = tmp_path / "filings"
    shutil.copytree(FIXTURES / "filings", root)
    return FilingStore(root)
