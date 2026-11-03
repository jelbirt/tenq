"""Watchlist config. Recipients and credentials never live here: they are env vars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WATCHLIST = REPO_ROOT / "config" / "watchlist.yaml"


@dataclass(frozen=True)
class Watchlist:
    tickers: tuple[str, ...]
    active: tuple[str, ...]


def _ticker_list(raw: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    # YAML 1.1 reads ON/YES/NO as booleans and a bare scalar as a string, so a
    # loose str() conversion would turn ON Semiconductor into "TRUE" silently.
    val = raw.get(key)
    if val is None:
        return ()
    if not isinstance(val, list) or not all(isinstance(t, str) for t in val):
        raise ValueError(f"{path}: '{key}' must be a list of strings (quote tickers like 'ON')")
    return tuple(t.upper() for t in val)


def load_watchlist(path: Path = DEFAULT_WATCHLIST) -> Watchlist:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")
    tickers = _ticker_list(raw, "watchlist", path)
    active = _ticker_list(raw, "active", path)
    if not tickers:
        raise ValueError(f"{path}: 'watchlist' is empty")
    if len(set(tickers)) != len(tickers):
        raise ValueError("duplicate tickers on the watchlist")
    unknown = [t for t in active if t not in tickers]
    if unknown:
        raise ValueError(f"active tickers not on the watchlist: {unknown}")
    return Watchlist(tickers=tickers, active=active)
