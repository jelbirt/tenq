"""Current reports: what a company told the SEC between its periodic filings.

A US company files an 8-K; a foreign company files a 6-K instead (results,
news, notices). Both are saved in their own store (default ./data/current) with the same layout and
passage ids as 10-Q/10-K filings, so a citation to an 8-K is checked the same way.
The store's search indexes carry their own tag ("c"), so a ticker's 8-K indexes
never mix with its 10-Q/10-K ones.
They carry no facts: the numbers tenq checks come from 10-Q/10-K tagged data only.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from tenq.config import REPO_ROOT
from tenq.edgar import CurrentSource, stamped
from tenq.filings import FilingMeta, FilingStore

log = logging.getLogger(__name__)

DEFAULT_ROOT = REPO_ROOT / "data" / "current"
LIST_DEPTH = 10
CURRENT_TAG = "c"

# Plain names for the 8-K items a daily call cares about (SEC Form 8-K instructions).
ITEM_NAMES = {
    "1.01": "entered a material agreement",
    "1.02": "ended a material agreement",
    "1.05": "material cybersecurity incident",
    "2.01": "completed an acquisition or sale of assets",
    "2.02": "results of operations",
    "2.03": "took on a material debt obligation",
    "2.05": "exit or restructuring costs",
    "2.06": "material impairment",
    "3.01": "notice of delisting",
    "4.01": "changed auditor",
    "4.02": "earlier financial statements should no longer be relied on",
    "5.02": "director or officer change",
    "5.03": "amended articles or bylaws",
    "5.07": "shareholder vote results",
    "7.01": "Regulation FD disclosure",
    "8.01": "other events",
    "9.01": "financial statements and exhibits",
}


@dataclass(frozen=True)
class Synced:
    reports: list[FilingMeta]  # reports filed on or after `since`, newest first, all saved
    new: list[FilingMeta]
    notes: list[str]


def describe(items: str) -> list[str]:
    """ "2.02,9.01" to ["Item 2.02 results of operations", "Item 9.01 ..."]."""
    out = []
    for raw in items.split(","):
        item = raw.strip()
        if item:
            out.append(f"Item {item} {ITEM_NAMES.get(item, 'other item')}")
    return out


def sync_current(
    source: CurrentSource,
    store: FilingStore,
    ticker: str,
    since: date,
    depth: int = LIST_DEPTH,
) -> Synced:
    """Save every 8-K or 6-K filed on or after `since`. Never raises: a failure is a note."""
    try:
        listed = source.list_current(ticker, depth)
    except Exception as exc:
        log.warning("current report listing failed for %s (%s)", ticker, type(exc).__name__)
        return Synced([], [], [f"current reports (8-K, 6-K) unavailable for {ticker}"])
    reports: list[FilingMeta] = []
    new: list[FilingMeta] = []
    notes: list[str] = []
    recent: list[FilingMeta] = []
    for m in listed:
        try:
            filed = date.fromisoformat(m.filed)
        except ValueError:
            notes.append(f"{m.form} {m.accession} has no filing date")
            continue
        if filed >= since:
            recent.append(m)
    if listed and len(listed) == depth and len(recent) == depth:
        notes.append(
            f"more than {depth} current reports since {since} for {ticker}; older ones not checked"
        )
    try:  # report a saved report as saved, with its acceptance time
        saved = {m.accession: m for m in store.filings(ticker)}
    except Exception as exc:
        log.warning("current report store unreadable for %s (%s)", ticker, type(exc).__name__)
        saved = {}
    for meta in recent:
        if meta.accession in saved:
            meta = saved[meta.accession]
        elif not store.has(meta.ticker, meta.accession):
            try:
                text = source.fetch_current(meta)
                meta = stamped(source, meta)
                store.save(meta, text, [])
            except Exception as exc:
                kind = type(exc).__name__
                log.warning("%s fetch failed %s %s (%s)", meta.form, ticker, meta.accession, kind)
                notes.append(f"{meta.form} {meta.accession} unavailable for {ticker}")
                continue
            new.append(meta)
        reports.append(meta)
    return Synced(reports, new, notes)


def main(argv: list[str] | None = None) -> int:
    from tenq.edgar import EdgarSource

    ap = argparse.ArgumentParser(description="Save a ticker's 8-Ks or 6-Ks filed since a date")
    ap.add_argument("ticker")
    ap.add_argument("--since", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    store = FilingStore(args.root, tag=CURRENT_TAG)
    synced = sync_current(EdgarSource(), store, args.ticker.upper(), args.since)
    for note in synced.notes:
        print(f"note: {note}")
    for meta in synced.reports:
        print(f"{meta.filed} {meta.accession} {'; '.join(describe(meta.items))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
