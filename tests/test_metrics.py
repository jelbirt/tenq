from __future__ import annotations

import pytest

from tenq.filings import Fact, FilingMeta, FilingStore
from tenq.metrics import build_table, metrics_for

REV = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
OCF = "us-gaap:NetCashProvidedByUsedInOperatingActivities"


def test_latest_10q_tagged_and_derived_values(store: FilingStore) -> None:
    t = metrics_for(store, "CRWD")
    assert (t.form, t.period, t.prior_period, t.year_ago_period) == (
        "10-Q",
        "2026-07-31",
        "2026-04-30",
        "2025-07-31",
    )
    assert t.number("revenue", "current") == 1_470_897_000
    assert t.number("revenue", "year_ago") == 1_168_952_000
    assert t.number("revenue", "yoy_pct") == pytest.approx(1_470_897_000 / 1_168_952_000 - 1)
    # Q2 operating cash flow is tagged only as six months: derived as 6M less Q1
    assert t.number("operating_cash_flow", "current") == 1_121_205_000 - 590_937_000
    assert t.rows["operating_cash_flow"].current is not None
    assert t.rows["operating_cash_flow"].current.derivation == "ytd difference"
    # a 10-Q balance sheet has no year-ago column: that value is four filings back
    assert t.number("deferred_revenue_current", "year_ago") == 2_781_196_000
    assert t.number("free_cash_flow", "current") == pytest.approx(
        t.number("operating_cash_flow", "current") - t.number("capex", "current")  # type: ignore[operator]
    )


def test_the_prior_quarters_growth_is_measured_from_a_year_before_it(
    store: FilingStore,
) -> None:
    t = metrics_for(store, "CRWD")
    assert t.prior_year_ago_period == "2025-04-30"
    assert t.number("revenue", "prior_yoy_pct") == pytest.approx(1_385_629_000 / 1_103_434_000 - 1)
    assert t.to_dict()["prior_year_ago_period"] == "2025-04-30"
    assert t.number("gross_margin", "prior_yoy_pct") is None  # a margin moves in points


def test_prior_growth_needs_both_of_its_quarters() -> None:
    facts = [
        _flow(REV, "2026-04-01", "2026-06-30", 12.0),
        _flow(REV, "2026-01-01", "2026-03-31", 11.0),
        _flow(REV, "2025-01-01", "2025-03-31", 10.0),
    ]
    t = build_table([(_meta("a-26-2", "2026-06-30"), facts)])
    assert (t.prior_period, t.prior_year_ago_period) == ("2026-03-31", "2025-03-31")
    assert t.number("revenue", "prior_yoy_pct") == pytest.approx(0.1)
    assert t.number("revenue", "yoy_pct") is None  # no year-ago quarter for the latest one
    alone = build_table([(_meta("a-26-2", "2026-06-30"), facts[:2])])
    assert alone.prior_year_ago_period is None
    assert alone.number("revenue", "prior_yoy_pct") is None


def test_10k_quarter_is_full_year_less_nine_months(store: FilingStore) -> None:
    t = metrics_for(store, "CRWD", "0001535527-26-000010")
    assert t.form == "10-K"
    assert t.number("revenue", "current") == 4_812_005_000 - 3_506_630_000
    assert t.number("revenue", "prior_quarter") == 1_234_244_000


def test_margins_change_in_points_not_percent(store: FilingStore) -> None:
    t = metrics_for(store, "CRWD")
    row = t.rows["gross_margin"]
    assert row.unit == "ratio"
    assert row.qoq_pct is None
    assert row.qoq_change == pytest.approx(
        t.number("gross_margin", "current") - t.number("gross_margin", "prior_quarter")  # type: ignore[operator]
    )


def test_unknown_filing_raises(store: FilingStore) -> None:
    with pytest.raises(LookupError):
        metrics_for(store, "CRWD", "0000000000-00-000000")
    with pytest.raises(LookupError):
        metrics_for(store, "NOPE")


def _meta(accession: str, period: str, form: str = "10-Q") -> FilingMeta:
    return FilingMeta("T", 1, "T Co", form, accession, period, period, "https://x")


def _flow(concept: str, start: str, end: str, value: float) -> Fact:
    return Fact(concept, value, "usd", start, end, None, -3)


def test_newer_filing_wins_for_a_restated_period() -> None:
    newest = (_meta("a-26-2", "2026-06-30"), [_flow(REV, "2025-04-01", "2025-06-30", 90.0)])
    older = (_meta("a-25-2", "2025-06-30"), [_flow(REV, "2025-04-01", "2025-06-30", 100.0)])
    t = build_table([newest, older])
    assert t.number("revenue", "year_ago") == 90.0


def test_missing_inputs_give_none_not_zero() -> None:
    only = (_meta("a-26-2", "2026-06-30"), [_flow(OCF, "2026-01-01", "2026-06-30", 50.0)])
    t = build_table([only])
    assert t.number("operating_cash_flow", "current") is None  # no Q1 to subtract
    assert t.number("free_cash_flow", "current") is None
    assert t.number("revenue", "yoy_pct") is None


def test_change_from_zero_has_no_percent() -> None:
    facts = [
        _flow(REV, "2026-04-01", "2026-06-30", 10.0),
        _flow(REV, "2025-04-01", "2025-06-30", 0.0),
    ]
    t = build_table([(_meta("a-26-2", "2026-06-30"), facts)])
    assert t.number("revenue", "yoy_change") == 10.0
    assert t.number("revenue", "yoy_pct") is None


def test_year_ago_snaps_to_a_53_week_calendar() -> None:
    # a 53-week year: the year-ago quarter ends 371 days back, not 364
    facts = [
        _flow(REV, "2026-03-29", "2026-06-27", 10.0),
        _flow(REV, "2025-03-23", "2025-06-21", 8.0),
    ]
    t = build_table([(_meta("a-26-2", "2026-06-27"), facts)])
    assert t.year_ago_period == "2025-06-21"
    assert t.number("revenue", "yoy_pct") == pytest.approx(0.25)


def test_year_ago_too_far_off_is_none() -> None:
    facts = [
        _flow(REV, "2026-03-29", "2026-06-27", 10.0),
        _flow(REV, "2025-03-10", "2025-06-10", 8.0),  # 382 days back
    ]
    t = build_table([(_meta("a-26-2", "2026-06-27"), facts)])
    assert t.year_ago_period is None
    assert t.number("revenue", "yoy_pct") is None


def test_unrelated_fact_dates_are_not_snap_targets() -> None:
    facts = [
        _flow(REV, "2026-03-29", "2026-06-27", 10.0),
        _flow(REV, "2025-03-23", "2025-06-21", 8.0),
        Fact("us-gaap:StockRepurchaseProgramAuthorizedAmount1", 1e9, "USD", None, None,
             "2025-06-30", 0),  # nearer to 364 days back than the real quarter end
    ]  # fmt: skip
    t = build_table([(_meta("a-26-2", "2026-06-27"), facts)])
    assert t.year_ago_period == "2025-06-21"


def test_report_date_a_day_off_the_facts_still_finds_the_quarter() -> None:
    facts = [_flow(REV, "2026-04-01", "2026-06-29", 10.0)]
    t = build_table([(_meta("a-26-2", "2026-06-30"), facts)])
    assert t.period == "2026-06-29"
    assert t.number("revenue", "current") == 10.0


def test_a_missing_filing_does_not_turn_qoq_into_two_quarters(store: FilingStore) -> None:
    metas = [m for m in store.filings("CRWD") if m.accession != "0001535527-26-000025"]
    t = build_table([(m, store.facts(m)) for m in metas[:5]])
    assert t.period == "2026-07-31"
    assert t.prior_period is None
    assert t.number("revenue", "qoq_pct") is None
    assert t.number("revenue", "yoy_pct") is not None


def test_no_percent_across_a_sign_change_or_from_a_loss() -> None:
    ni = "us-gaap:NetIncomeLoss"
    facts = [
        _flow(ni, "2026-04-01", "2026-06-30", 5.0),
        _flow(ni, "2026-01-01", "2026-03-31", -70.0),  # loss to profit
        _flow(ni, "2025-04-01", "2025-06-30", 4.0),
    ]
    t = build_table([(_meta("a-26-2", "2026-06-30"), facts)])
    assert t.number("net_income", "qoq_change") == 75.0
    assert t.number("net_income", "qoq_pct") is None
    assert t.number("net_income", "yoy_pct") == pytest.approx(0.25)
    facts[0] = _flow(ni, "2026-04-01", "2026-06-30", -5.0)  # profit to loss
    t = build_table([(_meta("a-26-2", "2026-06-30"), facts)])
    assert t.number("net_income", "yoy_pct") is None


def test_no_quarter_from_a_restated_ytd_less_an_unrestated_one() -> None:
    q2 = [
        _flow(OCF, "2026-01-01", "2026-06-30", 700.0),
        _flow(OCF, "2025-01-01", "2025-06-30", 640.0),  # restated prior-year 6M
    ]
    q1 = [
        _flow(OCF, "2026-01-01", "2026-03-31", 350.0),
        _flow(OCF, "2025-01-01", "2025-03-31", 300.0),  # as originally reported
    ]
    old_q2 = [_flow(OCF, "2025-01-01", "2025-06-30", 600.0)]
    t = build_table([
        (_meta("a-26-2", "2026-06-30"), q2),
        (_meta("a-26-1", "2026-03-31"), q1),
        (_meta("a-25-4", "2025-12-31", "10-K"), []),
        (_meta("a-25-3", "2025-09-30"), []),
        (_meta("a-25-2", "2025-06-30"), old_q2),
    ])  # fmt: skip
    assert t.number("operating_cash_flow", "current") == 350.0  # same vintage: fine
    assert t.number("operating_cash_flow", "year_ago") is None  # 640 restated - 300 not


def test_to_dict_is_json_ready(store: FilingStore) -> None:
    import json

    d = metrics_for(store, "CRWD").to_dict()
    assert json.loads(json.dumps(d))["metrics"]["revenue"]["current"] == 1_470_897_000


def test_the_sixth_filing_cannot_change_the_older_columns() -> None:
    """The sixth filing holds an old quarter as first reported. Pooled with the
    rest, its value disagrees with a later recast, marks that quarter restated,
    and blanks the year-ago flow derived from it. It is read only for the
    quarter a year before the prior quarter."""
    ocf = "us-gaap:NetCashProvidedByUsedInOperatingActivities"
    filings = [
        (
            _meta("a6", "2026-06-30"),
            [_flow(ocf, "2026-01-01", "2026-06-30", 300.0),
             _flow(ocf, "2025-01-01", "2025-06-30", 250.0)],
        ),
        (
            _meta("a5", "2026-03-31"),
            [_flow(ocf, "2026-01-01", "2026-03-31", 140.0),
             _flow(ocf, "2025-01-01", "2025-03-31", 100.0)],
        ),
        (_meta("a4", "2025-12-31", "10-K"), []),
        (_meta("a3", "2025-09-30"), []),
        (_meta("a2", "2025-06-30"), [_flow(ocf, "2025-01-01", "2025-06-30", 250.0)]),
        (_meta("a1", "2025-03-31"), [_flow(ocf, "2025-01-01", "2025-03-31", 90.0)]),
    ]  # fmt: skip
    five = build_table(filings[:5]).rows["operating_cash_flow"]
    six = build_table(filings).rows["operating_cash_flow"]
    for field in ("current", "prior_quarter", "year_ago", "yoy_pct", "qoq_pct"):
        assert six.number(field) == five.number(field), field
    assert six.number("year_ago") == 150.0


def test_growth_change_is_this_quarters_growth_less_the_prior_quarters(
    store: FilingStore,
) -> None:
    row = metrics_for(store, "CRWD").rows["revenue"]
    now, before = row.yoy_pct, row.prior_yoy_pct
    assert now is not None and before is not None
    # from the rates as the sheet shows them, so the shown figures subtract to it
    change = row.growth_change
    assert change is not None and change == pytest.approx(round(now, 4) - round(before, 4))
    assert round(change * 100, 2) == pytest.approx(round(now * 100, 2) - round(before * 100, 2))
    assert metrics_for(store, "CRWD").rows["gross_margin"].growth_change is None


def test_growth_change_needs_both_growth_rates() -> None:
    newest = (_meta("a2", "2026-06-30"), [_flow(REV, "2026-04-01", "2026-06-30", 120.0)])
    older = (_meta("a1", "2025-06-30"), [_flow(REV, "2025-04-01", "2025-06-30", 100.0)])
    row = build_table([newest, older]).rows["revenue"]
    assert row.yoy_pct == pytest.approx(0.2)
    assert row.prior_yoy_pct is None and row.growth_change is None
