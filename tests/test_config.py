from pathlib import Path

import pytest

from tenq.config import load_watchlist


def test_repo_watchlist_has_twenty_names_and_one_active() -> None:
    wl = load_watchlist()
    assert len(wl.tickers) == 20
    assert wl.active == ("CRWD",)


def test_active_must_be_on_watchlist(tmp_path: Path) -> None:
    p = tmp_path / "w.yaml"
    p.write_text("active: [ZZZZ]\nwatchlist: [CRWD]\n")
    with pytest.raises(ValueError, match="not on the watchlist"):
        load_watchlist(p)


def test_duplicates_rejected(tmp_path: Path) -> None:
    p = tmp_path / "w.yaml"
    p.write_text("watchlist: [CRWD, crwd]\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_watchlist(p)


@pytest.mark.parametrize(
    "body",
    [
        "watchlist: [CRWD, ON]\n",  # ON parses as a YAML boolean
        "watchlist: CRWD\n",  # scalar, not a list
        "",  # empty file
        "watchlist: []\n",
    ],
)
def test_malformed_watchlist_rejected(tmp_path: Path, body: str) -> None:
    p = tmp_path / "w.yaml"
    p.write_text(body)
    with pytest.raises(ValueError):
        load_watchlist(p)


def test_repo_watchlist_names_every_company() -> None:
    wl = load_watchlist()
    assert set(wl.names) == set(wl.tickers)
    assert wl.names["CRWD"] == "CrowdStrike"


@pytest.mark.parametrize(
    "names",
    [
        "names: {ZZZZ: Nobody}\n",  # not on the watchlist
        "names: [CrowdStrike]\n",  # a list, not a mapping
        "names: {CRWD: ''}\n",  # blank name
        "names: {CRWD: 7}\n",
    ],
)
def test_malformed_names_rejected(tmp_path: Path, names: str) -> None:
    p = tmp_path / "w.yaml"
    p.write_text("watchlist: [CRWD]\n" + names)
    with pytest.raises(ValueError):
        load_watchlist(p)
