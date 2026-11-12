"""Find a ticker's recent 10-Q/10-K filings and save any not already on disk.

A foreign company files no 10-Q or 10-K. Where it tags its quarterly results
in 6-Ks (ARM does; the other foreign stocks the screen can pick put their
quarter in report text only, which the sheet never reads for numbers), those
6-Ks count like 10-Qs and its yearly 20-F or 40-F like a 10-K: `sync_foreign`.
Either way the listing tells the sheet the company is foreign, so its note can
say why the figures are missing even with no recent 6-K: `Reports.foreign`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from tenq.edgar import FORMS, YEARLY_FOREIGN, FilingSource, ForeignSource, stamped
from tenq.filings import Fact, FilingMeta, FilingStore
from tenq.metrics import QUARTER_DAYS, YEAR_DAYS, tags_period

log = logging.getLogger(__name__)

# The newest filing plus the five before it. Two back is enough to derive any
# quarter from year-to-date facts (a Q1 10-Q's prior quarter is Q4: the 10-K's
# full year minus the Q3 10-Q's nine months), but a 10-Q's balance sheet has no
# year-ago column, so the year-ago balance comes from the filing four back, and
# the balance a year before the prior quarter from the filing five back.
DEFAULT_DEPTH = 6
# Foreign candidates listed per sync: room for a few tagged 6-Ks that turn out
# to hold no quarter (MNDY's half-year report) without missing the window.
FOREIGN_LIST = 2 * DEFAULT_DEPTH


@dataclass(frozen=True)
class Reports:
    """What `sync_reports` saved this run, and whether the company's EDGAR
    listing shows a yearly 20-F or 40-F, so it is a foreign filer. False for a
    company that lists or has saved a 10-Q or 10-K, and when its foreign listing
    could not be read."""

    new: list[FilingMeta]
    foreign: bool = False


def sync(source: FilingSource, store: FilingStore, ticker: str, depth: int) -> list[FilingMeta]:
    """Save the latest `depth` filings; return the ones that were new this run."""
    return _save(source, store, source.list_filings(ticker, depth))


def _save(source: FilingSource, store: FilingStore, listed: list[FilingMeta]) -> list[FilingMeta]:
    """`sync` on a listing already made."""
    new: list[FilingMeta] = []
    for meta in listed:
        if store.has(meta.ticker, meta.accession):
            continue
        try:
            text, facts = source.fetch(meta)
        except Exception:  # one bad filing must not stop the rest; next run retries it
            log.exception("fetch failed %s %s", meta.ticker, meta.accession)
            continue
        meta = stamped(source, meta)
        store.save(meta, text, facts)
        log.info("saved %s %s %s (%s facts)", meta.ticker, meta.form, meta.accession, len(facts))
        new.append(meta)
    return new


def sync_reports(source: ForeignSource, store: FilingStore, ticker: str, depth: int) -> Reports:
    """The 10-Q/10-Ks, or for a company that lists none, its foreign periodic
    reports. Judged by the forms listed and saved, not by an empty store: a
    company that lists 10-Qs is never read as foreign even when every fetch
    fails (a former foreign filer's old 20-Fs stay listed for years), and a
    foreign company's later 6-Ks are still synced where the store persists. A
    failure in the foreign reports costs only the sheet, never the stock's pack.

    Returns what was new and whether its foreign listing shows a 20-F or 40-F."""
    listed = source.list_filings(ticker, depth)
    new = _save(source, store, listed)
    if listed or any(m.form in FORMS for m in store.filings(ticker)):
        return Reports(new)
    foreign = False
    try:
        found = source.list_foreign(ticker, FOREIGN_LIST)
        foreign = any(m.form in YEARLY_FOREIGN for m in found)
        return Reports(new + _save_foreign(source, store, found, depth), foreign)
    except Exception:  # failures degrade: the pack still has its prices, news and reports
        log.exception("%s: foreign periodic reports could not be synced", ticker)
        return Reports(new, foreign)


def sync_foreign(
    source: ForeignSource, store: FilingStore, ticker: str, depth: int
) -> list[FilingMeta]:
    """Save a foreign company's newest `depth` periodic reports, if it tags its
    quarters; return the ones that were new this run.

    A 6-K counts only when its tagged facts hold a quarter in dollars ending at
    its report date. Without one such 6-K nothing is saved, not even the yearly
    report: a sheet built on a 20-F alone would show year-old balances as the
    latest quarter's. With one, the 20-F or 40-F counts like a 10-K, so the
    fourth quarter is the year less the nine months, as for a US company.
    """
    return _save_foreign(source, store, source.list_foreign(ticker, FOREIGN_LIST), depth)


def _save_foreign(
    source: ForeignSource, store: FilingStore, listed: list[FilingMeta], depth: int
) -> list[FilingMeta]:
    """`sync_foreign` on a listing already made."""
    fetched: dict[str, tuple[str, list[Fact]]] = {}
    quarters: list[FilingMeta] = []
    slots = 0  # the window's reports so far: a yearly one takes a slot unfetched
    for meta in listed:
        if slots == depth:
            break
        if meta.form != "6-K":
            slots += 1
            continue
        if store.has(meta.ticker, meta.accession):
            quarters.append(meta)
            slots += 1
            continue
        got = _fetch(source, meta)
        if got is not None and tags_period(got[1], meta.period, QUARTER_DAYS):
            fetched[meta.accession] = got
            quarters.append(meta)
            slots += 1
        elif got is not None:
            log.info("%s 6-K %s holds no tagged quarter in dollars", meta.ticker, meta.accession)
    if not quarters:
        return []
    window = [m for m in listed if m in quarters or m.form in YEARLY_FOREIGN][:depth]
    new: list[FilingMeta] = []
    for meta in window:
        if store.has(meta.ticker, meta.accession):
            continue
        got = fetched.get(meta.accession) or _fetch(source, meta)
        if got is None:
            continue
        if meta.form in YEARLY_FOREIGN and not tags_period(got[1], meta.period, YEAR_DAYS):
            log.info("%s %s %s holds no tagged dollar year", meta.ticker, meta.form, meta.accession)
            continue
        meta = stamped(source, meta)
        store.save(meta, *got)
        log.info("saved %s %s %s (%s facts)", meta.ticker, meta.form, meta.accession, len(got[1]))
        new.append(meta)
    return new


def _fetch(source: FilingSource, meta: FilingMeta) -> tuple[str, list[Fact]] | None:
    try:
        return source.fetch(meta)
    except Exception:  # one bad filing must not stop the rest; next run retries it
        log.exception("fetch failed %s %s", meta.ticker, meta.accession)
        return None


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
