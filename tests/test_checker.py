from __future__ import annotations

from dataclasses import replace

import pytest

from tenq.brief import Claim
from tenq.checker import Checked, check, check_claim, metrics_view, visible_docs
from tenq.filings import FilingStore
from tenq.metrics import MetricsTable, metrics_for
from tenq.opus import load_call_prompt, load_why_prompt
from tenq.reasons import MAX_REASON_CHARS, fill, phrase, placeholders, table_facts

LATEST = "CRWD-0001535527-26-000031"
OLDER_ACC = "0001535527-26-000025"
PID = f"{LATEST}:23"  # "Adoption of Our Solutions..."
REV_YOY = "{fin:CRWD:revenue:yoy_pct}"


@pytest.fixture
def table(store: FilingStore) -> MetricsTable:
    return metrics_for(store, "CRWD")


def _check(store: FilingStore, table: MetricsTable, claim: Claim) -> Checked | str:
    meta = store.filings("CRWD")[0]
    return check_claim(claim, table_facts(table), store, visible_docs(store, meta))


def test_code_writes_the_number_and_keeps_the_draft(
    store: FilingStore, table: MetricsTable
) -> None:
    draft = f"{REV_YOY}, as customers adopted more modules."
    got = _check(store, table, Claim(draft, PID))
    assert isinstance(got, Checked)
    assert got.claim == (
        "CRWD's revenue rose 25.83% from a year earlier, as customers adopted more modules."
    )
    assert got.numbers == ("fin:CRWD:revenue:yoy_pct",) and got.draft == draft
    assert got.passage_id == PID


def test_a_loss_and_a_margin_change_read_as_code_writes_them(
    store: FilingStore, table: MetricsTable
) -> None:
    got = _check(
        store,
        table,
        Claim(
            "{fin:CRWD:operating_income:current}, while {fin:CRWD:gross_margin:yoy_change}.", PID
        ),
    )
    assert isinstance(got, Checked)
    assert "operating income was a loss of $33.23 million in the quarter ended" in got.claim
    assert "gross margin rose 0.97 points from a year earlier" in got.claim


@pytest.mark.parametrize(
    ("number_id", "start"),
    [
        ("fin:CRWD:deferred_revenue:yoy_pct", "CRWD's total deferred revenue rose"),
        ("fin:CRWD:deferred_revenue_current:yoy_pct", "CRWD's current deferred revenue rose"),
        ("fin:CRWD:deferred_revenue_noncurrent:yoy_pct", "CRWD's noncurrent deferred revenue"),
        ("fin:CRWD:capex:current", "CRWD's capital expenditures were"),
        ("fin:CRWD:cash:current", "CRWD's cash and cash equivalents were"),
    ],
)
def test_a_metric_reads_as_a_phrase(table: MetricsTable, number_id: str, start: str) -> None:
    got = phrase(number_id, table_facts(table))
    assert got is not None and got.startswith(start), got


@pytest.mark.parametrize(
    "metric",
    ["cash", "deferred_revenue", "deferred_revenue_current", "deferred_revenue_noncurrent"],
)
def test_a_balance_is_at_the_end_of_the_quarter_not_in_it(table: MetricsTable, metric: str) -> None:
    for fld in ("current", "prior_quarter"):
        got = phrase(f"fin:CRWD:{metric}:{fld}", table_facts(table))
        assert got is not None and " at the end of " in got and " in the " not in got, got


@pytest.mark.parametrize("metric", ["revenue", "free_cash_flow", "operating_income"])
def test_a_flow_is_in_the_quarter_not_at_its_end(table: MetricsTable, metric: str) -> None:
    got = phrase(f"fin:CRWD:{metric}:current", table_facts(table))
    assert got is not None and " in the quarter ended " in got and " at the end " not in got, got


@pytest.mark.parametrize(
    ("number_id", "want"),
    [
        ("fin:CRWD:cash:yoy_pct", "CRWD's cash and cash equivalents were unchanged"),
        ("fin:CRWD:capex:qoq_change", "CRWD's capital expenditures were unchanged"),
        ("fin:CRWD:revenue:yoy_pct", "CRWD's revenue was unchanged"),
    ],
)
def test_no_change_takes_the_metric_s_own_verb(
    table: MetricsTable, number_id: str, want: str
) -> None:
    facts = table_facts(table)
    n = facts.number(number_id)
    assert n is not None
    flat = replace(facts, numbers={**facts.numbers, number_id: replace(n, value=0.0)})
    got = phrase(number_id, flat)
    assert got is not None and got.startswith(want), got


def test_no_metric_s_words_hold_a_table_label_s_comma(table: MetricsTable) -> None:
    facts = table_facts(table)
    for number_id in facts.ids():
        got = phrase(number_id, facts)
        if got is None:
            continue
        subject = got.split(" rose ")[0].split(" fell ")[0].split(" was ")[0].split(" were ")[0]
        assert "," not in subject, got


def test_a_claim_with_no_number_needs_only_its_passage(
    store: FilingStore, table: MetricsTable
) -> None:
    got = _check(store, table, Claim("Customers adopted more modules.", f"{LATEST}:15"))
    assert isinstance(got, Checked) and got.numbers == ()


def test_a_made_up_passage_id_is_dropped(store: FilingStore, table: MetricsTable) -> None:
    for pid in (f"{LATEST}:2", f"{LATEST}:99999", "CRWD-0001535527-99-000001:1", "nonsense"):
        why = _check(store, table, Claim(f"{REV_YOY}.", pid))
        assert isinstance(why, str) and "does not exist" in why


def test_a_passage_from_a_later_filing_is_dropped(store: FilingStore) -> None:
    older = next(m for m in store.filings("CRWD") if m.accession == OLDER_ACC)
    facts = table_facts(metrics_for(store, "CRWD", OLDER_ACC))
    why = check_claim(Claim("Demand grew.", PID), facts, store, visible_docs(store, older))
    assert isinstance(why, str) and "earlier one" in why


@pytest.mark.parametrize(
    "text",
    [
        "Revenue grew 26% from a year earlier.",
        f"{REV_YOY}, up from $1.2 billion.",
        f"{REV_YOY}, across three segments.",
        f"{REV_YOY}, and revenue doubled.",
        f"{REV_YOY}, as the 10-Q describes.",
        f"{REV_YOY}, as it did in fiscal 2026.",
        f"{REV_YOY}; over the six-month period cash came from customers.",
        f"{REV_YOY}, after the outage in July.",
    ],
)
def test_a_figure_in_the_models_own_words_drops_the_claim(
    store: FilingStore, table: MetricsTable, text: str
) -> None:
    why = _check(store, table, Claim(text, PID))
    assert isinstance(why, str) and "a figure typed in its own words" in why


@pytest.mark.parametrize(
    "number_id",
    [
        "fin:CRWD:bookings:current",  # no such metric
        "fin:CRWD:revenue:next_year",  # no such field
        "fin:ACME:revenue:current",  # another company
        "fin:CRWD:operating_income:yoy_pct",  # a loss has no percent change
        "px:CRWD:ret_5d",  # a daily price is not a filing number
    ],
)
def test_an_id_that_is_not_a_filing_number_drops_the_claim(
    store: FilingStore, table: MetricsTable, number_id: str
) -> None:
    why = _check(store, table, Claim("{" + number_id + "}.", PID))
    assert why == f"placeholder {number_id!r} is not a number on the facts sheet"


def test_a_placeholder_must_open_a_clause(store: FilingStore, table: MetricsTable) -> None:
    why = _check(store, table, Claim("Subscription " + REV_YOY + ".", PID))
    assert isinstance(why, str) and "does not open a clause" in why


def test_a_stray_curly_bracket_drops_the_claim(store: FilingStore, table: MetricsTable) -> None:
    why = _check(store, table, Claim(f"{REV_YOY}, and {{more.", PID))
    assert why == "a curly bracket that is not a placeholder"


def test_a_long_claim_is_dropped_whole_never_cut(store: FilingStore, table: MetricsTable) -> None:
    text = f"{REV_YOY}, " + "as customers adopted modules " * 30
    assert len(text) > MAX_REASON_CHARS
    why = _check(store, table, Claim(text, PID))
    assert why == f"longer than {MAX_REASON_CHARS} characters"


def test_check_keeps_good_claims_and_logs_each_drop(
    store: FilingStore, table: MetricsTable
) -> None:
    meta = store.filings("CRWD")[0]
    claims = [
        Claim(f"{REV_YOY}.", PID),
        Claim(f"{REV_YOY}.", PID),  # duplicate
        Claim("Revenue grew 40%.", PID),
        Claim("Customers adopted more modules.", f"{LATEST}:15"),
        Claim("Invented.", f"{LATEST}:4"),
    ]
    result = check(claims, table, store, meta)
    assert [c.draft for c in result.kept] == [claims[0].claim, claims[3].claim]
    assert [d.claim for d in result.dropped] == [claims[1], claims[2], claims[4]]
    assert [d.reason.split()[0] for d in result.dropped] == ["duplicate", "a", "passage"]


def test_get_metrics_lists_only_numbers_code_can_write(table: MetricsTable) -> None:
    view = metrics_view(table)
    assert view["period"] == table.period and view["accession"] == table.accession
    ids = [n["id"] for n in view["numbers"]]
    assert "fin:CRWD:revenue:yoy_pct" in ids
    assert "fin:CRWD:operating_income:yoy_pct" not in ids  # no value, so not offered
    facts = table_facts(table)
    for n in view["numbers"]:
        assert fill("{" + n["id"] + "}.", facts) == (n["code_writes"] + ".", (n["id"],))
        assert set(n) == {"id", "label", "code_writes"}  # no raw value to copy


def test_every_filing_id_the_opus_prompts_show_has_code_words(table: MetricsTable) -> None:
    """The prompts teach by example; an example id code cannot write would teach a drop."""
    facts = table_facts(table)
    shown = {
        i.replace("TICKER", "CRWD")
        for prompt in (load_call_prompt(), load_why_prompt())
        for i in placeholders(prompt.text)
        if i.startswith("fin:")
    }
    assert "fin:CRWD:deferred_revenue:yoy_pct" in shown
    for number_id in sorted(shown):
        assert phrase(number_id, facts) is not None, number_id
