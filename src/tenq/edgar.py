"""SEC EDGAR access through edgartools. The only module here that talks to EDGAR.

Fair access: edgartools throttles to 9 requests a second (SEC allows 10) and
sends the User-Agent set from EDGAR_IDENTITY, which must name a contact email.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from tenq.filings import Fact, FilingMeta

log = logging.getLogger(__name__)
EASTERN = ZoneInfo("America/New_York")  # where EDGAR stamps acceptance times

FORMS = ("10-Q", "10-K")
CURRENT = "8-K"
# A foreign company files a 6-K where a US one files an 8-K (results, news,
# notices). A 6-K lists no items; its text and EX-99 exhibits are read the same way.
CURRENT_FORMS = (CURRENT, "6-K")
EXHIBIT_PREFIX = "EX-99"  # press releases and other material a current report furnishes


class CurrentSource(Protocol):
    def list_current(self, ticker: str, limit: int) -> list[FilingMeta]:
        """Most recent current reports (8-K or 6-K) for a ticker, newest first."""
        ...

    def fetch_current(self, meta: FilingMeta) -> str:
        """The report's text followed by its EX-99 exhibits' text."""
        ...

    def accepted_at(self, meta: FilingMeta) -> str:
        """EDGAR's acceptance time, ISO with offset; "" when not known."""
        ...


class FilingSource(Protocol):
    def list_filings(self, ticker: str, limit: int) -> list[FilingMeta]:
        """Most recent 10-Q/10-K filings for a ticker, newest first."""
        ...

    def fetch(self, meta: FilingMeta) -> tuple[str, list[Fact]]:
        """The filing's main-document text and its undimensioned numeric facts."""
        ...

    def accepted_at(self, meta: FilingMeta) -> str:
        """EDGAR's acceptance time, ISO with offset; "" when not known."""
        ...


class _Stamps(Protocol):
    def accepted_at(self, meta: FilingMeta) -> str: ...


def stamped(source: _Stamps, meta: FilingMeta) -> FilingMeta:
    """`meta` with its acceptance time. Never raises: unknown stays "" (the
    filing date is then the fallback), which costs only replay precision."""
    try:
        return replace(meta, accepted=source.accepted_at(meta))
    except Exception as exc:
        log.warning("acceptance time unknown for %s (%s)", meta.accession, type(exc).__name__)
        return meta


def _eastern(value: Any) -> str:
    """An SGML header's acceptance time, naive Eastern, as ISO text with its offset."""
    if not isinstance(value, datetime):
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=EASTERN)
    return str(value.isoformat())


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


def currencies_of(units: dict[str, Any]) -> dict[str, str]:
    """Map each plain money unit id to its ISO 4217 code. A filer names its own
    unit ids ("usd", "U_USD", "EUR"), so the id alone does not say the currency;
    the unit's iso4217 measure does."""
    out: dict[str, str] = {}
    for uid, info in units.items():
        if isinstance(info, dict) and info.get("type") == "simple":
            measure = str(info.get("measure") or "")
            if measure.startswith("iso4217:"):
                out[str(uid)] = measure.removeprefix("iso4217:")
    return out


def facts_from_frame(
    rows: Iterator[dict[str, Any]], currencies: dict[str, str] | None = None
) -> list[Fact]:
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
        unit = str(r.get("unit_ref") or "")
        fact = Fact(
            concept=concept,
            value=float(value),
            unit=unit,
            start=start,
            end=end,
            instant=instant,
            decimals=int(dec) if dec and dec.lstrip("-").isdigit() else None,
            currency=(currencies or {}).get(unit),
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
        # amendments (10-Q/A) are skipped in slice 1
        return self._list(ticker, FORMS, limit)

    def list_current(self, ticker: str, limit: int) -> list[FilingMeta]:
        # 8-K/A and 6-K/A amendments are skipped like 10-Q/A: a correction worth
        # a call is usually re-reported in a new report or the next filing
        return self._list(ticker, CURRENT_FORMS, limit)

    def list_recent(self, ticker: str, forms: tuple[str, ...], limit: int) -> list[FilingMeta]:
        """The newest filings of any of `forms` together, in one company lookup."""
        return self._list(ticker, forms, limit)

    def _list(self, ticker: str, forms: tuple[str, ...], limit: int) -> list[FilingMeta]:
        from edgar import Company

        company = Company(ticker)
        filings = company.get_filings(form=list(forms), amendments=False)
        out: list[FilingMeta] = []
        for f in filings:
            if f.form not in forms:
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
                    items=(_opt_str(getattr(f, "items", None)) or "") if f.form == CURRENT else "",
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
        units = getattr(xbrl, "units", None) or {}
        facts = facts_from_frame((r for r in frame.to_dict("records")), currencies_of(units))
        return text, facts

    def accepted_at(self, meta: FilingMeta) -> str:
        # The listing's acceptanceDateTime (submissions JSON) is labelled UTC
        # but is hours off for some filers, so a replayed pack could see a
        # filing before it existed. The filing's own SGML header holds the
        # time SEC's index page shows, in Eastern time, for one request more.
        return _eastern(self._filing(meta).header.acceptance_datetime)

    def fetch_current(self, meta: FilingMeta) -> str:
        filing = self._filing(meta)
        parts = [str(filing.text())]
        for att in filing.attachments:
            kind = str(getattr(att, "document_type", "") or "")
            if not kind.startswith(EXHIBIT_PREFIX):
                continue
            try:  # one unreadable exhibit must not cost the report's own text
                body = att.text()
            except Exception as exc:
                log.warning("%s exhibit %s unreadable (%s)", meta.form, kind, type(exc).__name__)
                continue
            if body:  # a PDF or binary exhibit has no text
                parts.append(f"EXHIBIT {kind.removeprefix('EX-')}\n\n{body}")
        return "\n\n".join(parts)
