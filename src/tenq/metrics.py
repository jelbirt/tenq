"""The numbers tool: quarterly metrics from tagged XBRL facts, changes computed in code.

Every number a brief may state comes from here. For the newest saved filing
of a ticker (or any older one, for backfill) it reports each metric for the
filing's quarter, the prior quarter and the year-ago quarter, plus the
quarter-over-quarter and year-over-year changes.

Flow metrics (revenue, cash flow) are often tagged only year-to-date: a Q2
10-Q tags six months of operating cash flow, and a 10-K tags the full year.
The quarter is then derived as YTD minus the previous quarter's YTD with the
same start date (Q4 = full year minus nine months). Facts are pooled across
the newest few filings; where two filings tag the same period, the newer
filing's value wins, so a restated comparative replaces the original. A
quarter is never derived from a restated YTD and an unrestated one: that
would book the whole restatement in one quarter, so the value is left out.

Wrong ground truth is worse than none, since the checker drops any number in a
brief that this table does not hold. So a comparative is only filled when its
period really is one quarter (or one year) back, and a percent change is only
given when it means something: from a positive base without a sign change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from tenq.filings import Fact, FilingMeta, FilingStore

QUARTER_DAYS = range(80, 101)  # a fiscal quarter, allowing 13/14-week quarters
YEAR_DAYS = range(357, 379)  # a fiscal year, allowing 52/53-week years
PERIOD_TOLERANCE = 4  # report date vs fact end date in a 52/53-week calendar
POOL_DEPTH = 5  # the filing and four before it: the year-ago balance sheet is 4 back


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    kind: Literal["flow", "stock"]
    concepts: tuple[str, ...]  # aliases in priority order; first one tagged wins


BASE_METRICS: tuple[Metric, ...] = (
    Metric(
        "revenue",
        "Revenue",
        "flow",
        (
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
            "us-gaap:Revenues",
        ),
    ),
    Metric("gross_profit", "Gross profit", "flow", ("us-gaap:GrossProfit",)),
    Metric("operating_income", "Operating income", "flow", ("us-gaap:OperatingIncomeLoss",)),
    Metric("net_income", "Net income", "flow", ("us-gaap:NetIncomeLoss",)),
    Metric(
        "operating_cash_flow",
        "Operating cash flow",
        "flow",
        ("us-gaap:NetCashProvidedByUsedInOperatingActivities",),
    ),
    Metric(
        "capex",
        "Capital expenditures",
        "flow",
        ("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",),
    ),
    Metric(
        "deferred_revenue_current",
        "Deferred revenue, current",
        "stock",
        ("us-gaap:ContractWithCustomerLiabilityCurrent", "us-gaap:DeferredRevenueCurrent"),
    ),
    Metric(
        "deferred_revenue_noncurrent",
        "Deferred revenue, noncurrent",
        "stock",
        ("us-gaap:ContractWithCustomerLiabilityNoncurrent", "us-gaap:DeferredRevenueNoncurrent"),
    ),
    Metric(
        "cash",
        "Cash and cash equivalents",
        "stock",
        ("us-gaap:CashAndCashEquivalentsAtCarryingValue",),
    ),
)

# (name, label, unit) of metrics computed from base values in build_table; a
# derived value is None when any input is missing, never built from a partial set
DERIVED: tuple[tuple[str, str, Literal["usd", "ratio"]], ...] = (
    ("free_cash_flow", "Free cash flow (operating cash flow less capex)", "usd"),
    ("deferred_revenue", "Deferred revenue, total", "usd"),
    ("gross_margin", "Gross margin", "ratio"),
    ("operating_margin", "Operating margin", "ratio"),
)

PERIOD_FIELDS = ("current", "prior_quarter", "year_ago")
CHANGE_FIELDS = ("qoq_change", "qoq_pct", "yoy_change", "yoy_pct")
FIELDS = PERIOD_FIELDS + CHANGE_FIELDS


@dataclass(frozen=True)
class Source:
    """Where a value came from: the facts behind it, so a reader can audit it."""

    accession: str
    concept: str
    start: str | None
    end: str | None
    instant: str | None
    value: float


@dataclass(frozen=True)
class Value:
    amount: float
    sources: tuple[Source, ...]
    derivation: str  # "tagged", "ytd difference", or a formula name


@dataclass
class MetricRow:
    name: str
    label: str
    unit: Literal["usd", "ratio"]
    current: Value | None = None
    prior_quarter: Value | None = None
    year_ago: Value | None = None

    def _change(self, other: Value | None, pct: bool) -> float | None:
        if self.current is None or other is None:
            return None
        diff = self.current.amount - other.amount
        if not pct:
            return diff
        if self.unit == "ratio":
            return None  # a margin's change is in points, not percent
        if other.amount <= 0 or self.current.amount < 0:
            return None  # from zero, from a loss, or into a loss: no meaningful %
        return diff / other.amount

    @property
    def qoq_change(self) -> float | None:
        return self._change(self.prior_quarter, pct=False)

    @property
    def qoq_pct(self) -> float | None:
        return self._change(self.prior_quarter, pct=True)

    @property
    def yoy_change(self) -> float | None:
        return self._change(self.year_ago, pct=False)

    @property
    def yoy_pct(self) -> float | None:
        return self._change(self.year_ago, pct=True)

    def number(self, field_name: str) -> float | None:
        if field_name in PERIOD_FIELDS:
            v: Value | None = getattr(self, field_name)
            return None if v is None else v.amount
        if field_name in CHANGE_FIELDS:
            n: float | None = getattr(self, field_name)
            return n
        raise KeyError(field_name)


@dataclass
class MetricsTable:
    ticker: str
    company: str
    form: str
    accession: str
    period: str
    prior_period: str | None
    year_ago_period: str | None
    rows: dict[str, MetricRow] = field(default_factory=dict)

    def number(self, metric: str, field_name: str) -> float | None:
        row = self.rows.get(metric)
        return None if row is None else row.number(field_name)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready view for the MCP tool and the run record."""
        return {
            "ticker": self.ticker,
            "company": self.company,
            "form": self.form,
            "accession": self.accession,
            "period": self.period,
            "prior_period": self.prior_period,
            "year_ago_period": self.year_ago_period,
            "metrics": {
                name: {"label": row.label, "unit": row.unit, **{f: row.number(f) for f in FIELDS}}
                for name, row in self.rows.items()
            },
        }


def _d(s: str) -> date:
    return date.fromisoformat(s)


class _Pool:
    """Facts from several filings, newest filing first, keyed by period."""

    def __init__(self, filings: Sequence[tuple[FilingMeta, list[Fact]]]) -> None:
        self.flows: dict[tuple[str, str, str], Source] = {}
        self.stocks: dict[tuple[str, str], Source] = {}
        self.restated: set[tuple[str, str, str]] = set()  # flows an older filing tagged differently
        for meta, facts in filings:
            for f in facts:
                src = Source(meta.accession, f.concept, f.start, f.end, f.instant, f.value)
                if f.start and f.end:
                    key = (f.concept, f.start, f.end)
                    kept = self.flows.setdefault(key, src)
                    if kept.value != f.value:
                        self.restated.add(key)
                elif f.instant:
                    self.stocks.setdefault((f.concept, f.instant), src)

    def flow_quarter(self, concept: str, end: str) -> Value | None:
        tagged = [
            s
            for (c, start, e), s in self.flows.items()
            if c == concept and e == end and (_d(e) - _d(start)).days in QUARTER_DAYS
        ]
        if tagged:
            return Value(tagged[0].value, (tagged[0],), "tagged")
        for key, ytd in self.flows.items():
            c, start, e = key
            if c != concept or e != end:
                continue
            for key2, earlier in self.flows.items():
                c2, start2, e2 = key2
                if c2 != concept or start2 != start or (_d(e) - _d(e2)).days not in QUARTER_DAYS:
                    continue
                mixed = ytd.accession != earlier.accession
                if mixed and (key in self.restated or key2 in self.restated):
                    return None  # restated YTD less an unrestated one
                return Value(ytd.value - earlier.value, (ytd, earlier), "ytd difference")
        return None

    def stock_at(self, concept: str, instant: str) -> Value | None:
        s = self.stocks.get((concept, instant))
        return None if s is None else Value(s.value, (s,), "tagged")

    def ends(self, concepts: set[str], accession: str | None = None) -> set[str]:
        """Period ends of the metrics' own facts (optionally one filing's): the
        only dates a comparative may snap to, so an unrelated fact's date
        (a subsequent event, a debt issue) can never be mistaken for one."""
        flows = {e for (c, _, e), s in self.flows.items() if c in concepts
                 and accession in (None, s.accession)}  # fmt: skip
        stocks = {i for (c, i), s in self.stocks.items() if c in concepts
                  and accession in (None, s.accession)}  # fmt: skip
        return flows | stocks


def _snap(target: date, candidates: set[str], tolerance: int = 10) -> str | None:
    """The period end nearest a target date, for 52/53-week fiscal calendars."""
    best = min(candidates, key=lambda c: abs((_d(c) - target).days), default=None)
    if best is None or abs((_d(best) - target).days) > tolerance:
        return None
    return best


def _base_value(pool: _Pool, metric: Metric, end: str | None) -> Value | None:
    if end is None:
        return None
    for concept in metric.concepts:
        v = (
            pool.flow_quarter(concept, end)
            if metric.kind == "flow"
            else pool.stock_at(concept, end)
        )
        if v is not None:
            return v
    return None


def _combine(name: str, a: Value | None, b: Value | None, op: str) -> Value | None:
    if a is None or b is None:
        return None
    if op == "-":
        amount = a.amount - b.amount
    elif op == "+":
        amount = a.amount + b.amount
    else:  # "/"
        if b.amount == 0:
            return None
        amount = a.amount / b.amount
    return Value(amount, a.sources + b.sources, name)


def build_table(filings: Sequence[tuple[FilingMeta, list[Fact]]]) -> MetricsTable:
    """Metrics for filings[0], using the rest (newest first) for comparatives."""
    meta = filings[0][0]
    pool = _Pool(filings)
    concepts = {c for m in BASE_METRICS for c in m.concepts}
    ends = pool.ends(concepts)
    # the report date can sit a few days off the facts' own end date
    own = pool.ends(concepts, meta.accession)
    period = _snap(_d(meta.period), own, PERIOD_TOLERANCE) or meta.period

    def back(candidate: str | None, days: range) -> str | None:
        ok = candidate is not None and (_d(period) - _d(candidate)).days in days
        return candidate if ok else None

    # the previous filing is the prior quarter only if it ends one quarter back:
    # with a filing missing from the store, QoQ must not become a half-year change
    prior_target = filings[1][0].period if len(filings) > 1 else None
    prior = back(_snap(_d(prior_target), ends), QUARTER_DAYS) if prior_target else None
    if prior is None:
        prior = back(_snap(_d(period) - timedelta(days=91), ends), QUARTER_DAYS)
    year_ago = back(_snap(_d(period) - timedelta(days=364), ends), YEAR_DAYS)
    table = MetricsTable(
        meta.ticker, meta.company, meta.form, meta.accession, period, prior, year_ago
    )

    by_end: dict[str, dict[str, Value | None]] = {}
    for end_name, end in (("current", period), ("prior_quarter", prior), ("year_ago", year_ago)):
        vals: dict[str, Value | None] = {m.name: _base_value(pool, m, end) for m in BASE_METRICS}
        vals["free_cash_flow"] = _combine(
            "operating cash flow - capex", vals["operating_cash_flow"], vals["capex"], "-"
        )
        vals["deferred_revenue"] = _combine(
            "current + noncurrent",
            vals["deferred_revenue_current"],
            vals["deferred_revenue_noncurrent"],
            "+",
        )
        vals["gross_margin"] = _combine(
            "gross profit / revenue", vals["gross_profit"], vals["revenue"], "/"
        )
        vals["operating_margin"] = _combine(
            "operating income / revenue", vals["operating_income"], vals["revenue"], "/"
        )
        by_end[end_name] = vals

    specs: list[tuple[str, str, Literal["usd", "ratio"]]] = [
        (m.name, m.label, "usd") for m in BASE_METRICS
    ]
    for name, label, unit in specs + list(DERIVED):
        row = MetricRow(name, label, unit)
        row.current = by_end["current"][name]
        row.prior_quarter = by_end["prior_quarter"][name]
        row.year_ago = by_end["year_ago"][name]
        table.rows[name] = row
    return table


def metrics_for(store: FilingStore, ticker: str, accession: str | None = None) -> MetricsTable:
    """The table for a saved filing (default: the newest) of a ticker."""
    metas = store.filings(ticker)
    if not metas:
        raise LookupError(f"no saved filings for {ticker}")
    idx = 0 if accession is None else next(
        (i for i, m in enumerate(metas) if m.accession == accession), -1
    )  # fmt: skip
    if idx < 0:
        raise LookupError(f"no saved filing {accession} for {ticker}")
    window = metas[idx : idx + POOL_DEPTH]
    return build_table([(m, store.facts(m)) for m in window])
