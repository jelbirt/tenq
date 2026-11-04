"""Watchlist config. Recipients and credentials never live here: they are env vars."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WATCHLIST = REPO_ROOT / "config" / "watchlist.yaml"


@dataclass(frozen=True)
class Watchlist:
    tickers: tuple[str, ...]
    active: tuple[str, ...]
    names: dict[str, str] = field(default_factory=dict)  # ticker to company name


def _ticker_list(raw: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    # YAML 1.1 reads ON/YES/NO as booleans and a bare scalar as a string, so a
    # loose str() conversion would turn ON Semiconductor into "TRUE" silently.
    val = raw.get(key)
    if val is None:
        return ()
    if not isinstance(val, list) or not all(isinstance(t, str) for t in val):
        raise ValueError(f"{path}: '{key}' must be a list of strings (quote tickers like 'ON')")
    return tuple(t.upper() for t in val)


def _names(raw: dict[str, Any], path: Path) -> dict[str, str]:
    val = raw.get("names")
    if val is None:
        return {}
    if not isinstance(val, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in val.items()
    ):
        raise ValueError(f"{path}: 'names' must map tickers to company names")
    return {k.upper(): v.strip() for k, v in val.items()}


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
    names = _names(raw, path)
    stray = [t for t in names if t not in tickers]
    if stray:
        raise ValueError(f"names for tickers not on the watchlist: {stray}")
    return Watchlist(tickers=tickers, active=active, names=names)
