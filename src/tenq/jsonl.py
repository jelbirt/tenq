"""Append-only JSONL records, one item per line, keyed by "key".

Eval records are written one at a time as each item finishes, so a run that
is killed loses at most the item in flight; on load the latest line per key
wins, and a line a crash cut short is skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def append(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line, first ending a line a crash left open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.tell():
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((json.dumps(record) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """The latest record per key; a line cut short by a crash is skipped."""
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("key"), str):
            out[rec["key"]] = rec
    return out


def load_lines(path: Path) -> list[dict[str, Any]]:
    """Every readable object line, in order (for logs with no key, such as nights)."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out
