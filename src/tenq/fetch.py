"""Find a ticker's recent 10-Q/10-K filings and save any not already on disk."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tenq.edgar import FilingSource
from tenq.filings import FilingMeta, FilingStore

log = logging.getLogger(__name__)

# The newest filing plus the four before it. Two back is enough to derive any
# quarter from year-to-date facts (a Q1 10-Q's prior quarter is Q4: the 10-K's
# full year minus the Q3 10-Q's nine months), but a 10-Q's balance sheet has no
# year-ago column, so the year-ago balance comes from the filing four back.
DEFAULT_DEPTH = 5


def sync(source: FilingSource, store: FilingStore, ticker: str, depth: int) -> list[FilingMeta]:
    """Save the latest `depth` filings; return the ones that were new this run."""
    new: list[FilingMeta] = []
    for meta in source.list_filings(ticker, depth):
        if store.has(meta.ticker, meta.accession):
            continue
        try:
            text, facts = source.fetch(meta)
        except Exception:  # one bad filing must not stop the rest; next run retries it
            log.exception("fetch failed %s %s", meta.ticker, meta.accession)
            continue
        store.save(meta, text, facts)
        log.info("saved %s %s %s (%s facts)", meta.ticker, meta.form, meta.accession, len(facts))
        new.append(meta)
    return new


def main(argv: list[str] | None = None) -> None:
    from tenq.edgar import EdgarSource

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--root", type=Path, default=Path("data/filings"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    new = sync(EdgarSource(), FilingStore(args.root), args.ticker.upper(), args.depth)
    print(f"{len(new)} new filing(s) for {args.ticker.upper()}")


if __name__ == "__main__":
    main()
