"""SEC EDGAR access through edgartools. The only module here that talks to EDGAR.

Fair access: edgartools throttles to 9 requests a second (SEC allows 10) and
sends the User-Agent set from EDGAR_IDENTITY, which must name a contact email.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from typing import Any, Protocol

from tenq.filings import Fact, FilingMeta

FORMS = ("10-Q", "10-K")


class FilingSource(Protocol):
    def list_filings(self, ticker: str, limit: int) -> list[FilingMeta]:
        """Most recent 10-Q/10-K filings for a ticker, newest first."""
        ...

    def fetch(self, meta: FilingMeta) -> tuple[str, list[Fact]]:
        """The filing's main-document text and its undimensioned numeric facts."""
        ...


def _opt_str(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v)
    return s or None


def _more_precise(new: int | None, old: int | None) -> bool:
    """decimals=None means INF (exact), the most precise a fact can be."""
    if old is None:
        return False
    return new is None or new > old


def facts_from_frame(rows: Iterator[dict[str, Any]]) -> list[Fact]:
    """Keep undimensioned numeric facts: the face-statement numbers a brief can cite.

    A filing can tag the same concept and period more than once: once per
    statement it appears on, and sometimes again rounded in the MD&A ("$1.1
    billion" at decimals -8). The most precise copy wins, so later number checks
    compare against the statement value, not a rounded one.
    """
    best: dict[tuple[str, str, str | None, str | None, str | None], Fact] = {}
    for r in rows:
        if r.get("is_dimensioned"):
            continue
        value = r.get("numeric_value")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        concept = str(r["concept"])
        if concept.startswith("dei:"):
            continue
        start, end = _opt_str(r.get("period_start")), _opt_str(r.get("period_end"))
        instant = _opt_str(r.get("period_instant"))
        dec = _opt_str(r.get("decimals"))
        fact = Fact(
            concept=concept,
            value=float(value),
            unit=str(r.get("unit_ref") or ""),
            start=start,
            end=end,
            instant=instant,
            decimals=int(dec) if dec and dec.lstrip("-").isdigit() else None,
        )
        key = (concept, fact.unit, start, end, instant)
        prev = best.get(key)
        if prev is None or _more_precise(fact.decimals, prev.decimals):
            best[key] = fact
    return list(best.values())


class EdgarSource:
    def __init__(self, identity: str | None = None) -> None:
        identity = identity or os.environ.get("EDGAR_IDENTITY")
        if not identity or "@" not in identity:
            raise RuntimeError("EDGAR_IDENTITY must be set to a name and contact email")
        from edgar import set_identity

        # set_identity logs the identity (a contact email) at INFO; keep it out
        # of public Actions logs whatever logging the caller configured
        logging.getLogger("edgar").setLevel(logging.WARNING)
        set_identity(identity)
        self._by_accession: dict[str, Any] = {}

    def list_filings(self, ticker: str, limit: int) -> list[FilingMeta]:
        from edgar import Company

        company = Company(ticker)
        # amendments (10-Q/A) are skipped in slice 1
        filings = company.get_filings(form=list(FORMS), amendments=False)
        out: list[FilingMeta] = []
        for f in filings:
            if f.form not in FORMS:
                continue
            self._by_accession[str(f.accession_no)] = f
            out.append(
                FilingMeta(
                    ticker=ticker.upper(),
                    cik=int(f.cik),
                    company=str(f.company),
                    form=str(f.form),
                    accession=str(f.accession_no),
                    filed=str(f.filing_date),
                    period=str(f.period_of_report),
                    url=str(f.homepage_url),
                )
            )
            if len(out) == limit:
                break
        return out

    def _filing(self, meta: FilingMeta) -> Any:
        """The edgartools Filing behind a listing. Reuses the listed object: a
        fresh lookup by accession can scan full quarterly indexes for a filing
        too new for EDGAR's search, which is exactly the case tenq exists for."""
        if meta.accession in self._by_accession:
            return self._by_accession[meta.accession]
        from edgar import Company

        found = Company(meta.ticker).get_filings(accession_number=meta.accession)
        if not found:
            raise LookupError(f"filing {meta.accession} not found for {meta.ticker}")
        return found[0]

    def fetch(self, meta: FilingMeta) -> tuple[str, list[Fact]]:
        filing = self._filing(meta)
        text = str(filing.text())
        xbrl = filing.xbrl()
        if xbrl is None:  # no XBRL or it failed to parse: text only, no numbers
            return text, []
        frame = xbrl.facts.to_dataframe()
        facts = facts_from_frame(r for r in frame.to_dict("records"))
        return text, facts
