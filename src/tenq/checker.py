"""Check every brief claim before it can be sent. A claim that fails is dropped and logged.

The model never types a figure. Where a claim uses one of the filing's numbers,
it writes that number's id from get_metrics in curly brackets,
"{fin:CRWD:revenue:yoy_pct}", and code writes that part of the sentence from
the tagged XBRL data ("CRWD's revenue for the quarter ended 2026-07-31 rose
25.83% from a year earlier"). The rules are the daily calls' rules
(tenq.reasons.fill), with the filing's numbers as the facts.

A claim is sent only if:
1. its passage_id resolves to a saved passage of this ticker's briefed filing
   or an earlier one;
2. every placeholder is a number of the briefed filing that has a value, and
   each one is a clause of its own: it opens one, and what follows it is the
   end, a punctuation mark, or "and", "but", "while", "whereas" or "yet" and
   another placeholder;
3. the words the model wrote itself hold no figure (no digit, "%", currency
   sign, number word, month or quarter name);
4. it is at most MAX_REASON_CHARS long (a longer one is dropped, never cut),
   and it is not a repeat of a kept claim.

What this cannot catch: a real passage that does not support the claim, or the
model's own words disagreeing with or recasting the clause code wrote
("{...}, a record", "On an adjusted basis, {...}"). Those words hold no
figure, the true number is in the same sentence and its passage is quoted
under it. Reading the model's prose for meaning is not attempted (it is the
regex trap this design left); that is left to Jake's review of the briefs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tenq.brief import Claim
from tenq.filings import FilingMeta, FilingStore
from tenq.metrics import MetricsTable
from tenq.reasons import Facts, fill, phrase, table_facts


@dataclass(frozen=True)
class Checked:
    """A claim that may be sent."""

    claim: str  # code's sentence, with every number written by code
    passage_id: str
    numbers: tuple[str, ...]  # the number ids code wrote from
    draft: str  # the claim as the model wrote it, placeholders and all


@dataclass(frozen=True)
class Dropped:
    claim: Claim
    reason: str


@dataclass
class CheckResult:
    kept: list[Checked] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)


def metrics_view(table: MetricsTable, period: str | None = None) -> dict[str, Any]:
    """What get_metrics shows the model: each usable number's id, and the exact
    words code will write for it. A number code cannot write is not listed."""
    facts = table_facts(table, period)
    numbers = []
    for number_id, n in facts.numbers.items():
        words = phrase(number_id, facts)
        if words is not None:
            numbers.append({"id": number_id, "label": n.label, "code_writes": words})
    return {
        "ticker": table.ticker,
        "company": table.company,
        "form": table.form,
        "accession": table.accession,
        "period": table.period,
        "prior_period": table.prior_period,
        "year_ago_period": table.year_ago_period,
        "numbers": numbers,
    }


def check_claim(claim: Claim, facts: Facts, store: FilingStore, visible: set[str]) -> Checked | str:
    """The claim as it may be sent, or the reason it is dropped."""
    p = store.passage(claim.passage_id)
    if p is None:
        return f"passage {claim.passage_id!r} does not exist"
    if p.doc not in visible:
        return f"passage {claim.passage_id!r} is not from this filing or an earlier one"
    got = fill(claim.claim, facts)
    if isinstance(got, str):
        return got
    text, numbers = got
    return Checked(text, claim.passage_id, numbers, claim.claim)


def visible_docs(store: FilingStore, meta: FilingMeta) -> set[str]:
    return {m.doc for m in store.filings(meta.ticker) if m.period <= meta.period}


def check(
    claims: list[Claim], table: MetricsTable, store: FilingStore, meta: FilingMeta
) -> CheckResult:
    facts = table_facts(table, meta.period)
    visible = visible_docs(store, meta)
    result = CheckResult()
    seen: set[str] = set()
    for c in claims:
        got = check_claim(c, facts, store, visible)
        if isinstance(got, Checked) and got.claim.lower() in seen:
            got = "duplicate claim"
        if isinstance(got, str):
            result.dropped.append(Dropped(c, got))
            continue
        seen.add(got.claim.lower())
        result.kept.append(got)
    return result
