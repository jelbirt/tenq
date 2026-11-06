from __future__ import annotations

from dataclasses import replace

import pytest

from tenq.brief import Claim, load_prompt
from tenq.checker import Checked, check, check_claim, metrics_view, visible_docs
from tenq.filings import FilingStore
from tenq.metrics import MetricsTable, growth_reading, metrics_for
from tenq.opus import load_call_prompt, load_why_prompt
from tenq.pack import _shown
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
        "CRWD's revenue for the quarter ended 2026-07-31 rose 25.83% from a year earlier, as customers adopted more modules."
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
    assert (
        "gross margin for the quarter ended 2026-07-31 rose 0.97 points from a year earlier"
        in got.claim
    )


@pytest.mark.parametrize(
    ("number_id", "start"),
    [
        (
            "fin:CRWD:deferred_revenue:yoy_pct",
            "CRWD's total deferred revenue at the end of the quarter ended 2026-07-31 rose",
        ),
        (
            "fin:CRWD:deferred_revenue_current:yoy_pct",
            "CRWD's current deferred revenue at the end of the quarter ended 2026-07-31 rose",
        ),
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
    ("number_id", "words"),
    [
        (
            "fin:CRWD:revenue:prior_quarter",
            "CRWD's revenue was $1.386 billion in the quarter ended 2026-04-30",
        ),
        (
            "fin:CRWD:net_income:year_ago",
            "CRWD's net income was a loss of $70.15 million in the quarter ended 2025-07-31",
        ),
        (
            "fin:CRWD:cash:prior_quarter",
            "CRWD's cash and cash equivalents were $4.553 billion"
            " at the end of the quarter ended 2026-04-30",
        ),
        (
            "fin:CRWD:revenue:prior_yoy_pct",
            "CRWD's revenue for the quarter ended 2026-04-30 rose 25.57% from a year earlier",
        ),
    ],
)
def test_an_earlier_quarters_amount_names_that_quarter(
    table: MetricsTable, number_id: str, words: str
) -> None:
    assert phrase(number_id, table_facts(table)) == words


@pytest.mark.parametrize(
    ("number_id", "want"),
    [
        (
            "fin:CRWD:cash:yoy_pct",
            "CRWD's cash and cash equivalents at the end of the quarter ended 2026-07-31 were unchanged",
        ),
        (
            "fin:CRWD:capex:qoq_change",
            "CRWD's capital expenditures for the quarter ended 2026-07-31 were unchanged",
        ),
        (
            "fin:CRWD:revenue:yoy_pct",
            "CRWD's revenue for the quarter ended 2026-07-31 was unchanged",
        ),
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


def test_the_brief_prompts_example_is_what_code_writes(table: MetricsTable) -> None:
    """brief.md quotes CRWD's real revenue change under the name ACME."""
    words = phrase("fin:CRWD:revenue:yoy_pct", table_facts(table))
    assert words is not None
    text = " ".join(load_prompt().text.split())
    assert f'"{words.replace("CRWD", "ACME")}, driven by' in text


def test_growth_change_reads_as_steady_faster_slower_or_the_same(store: FilingStore) -> None:
    facts = table_facts(metrics_for(store, "CRWD"))
    assert phrase("fin:CRWD:revenue:growth_change", facts) == (
        "CRWD's revenue growth from a year earlier was steady for the quarter ended "
        "2026-07-31 against the quarter ended 2026-04-30 (0.26 points faster, under 2 points "
        "either way)"
    )
    assert phrase("fin:CRWD:gross_profit:growth_change", facts) == (
        "CRWD's gross profit growth from a year earlier was steady for the quarter ended "
        "2026-07-31 against the quarter ended 2026-04-30 (0.44 points slower, under 2 points "
        "either way)"
    )
    assert phrase("fin:CRWD:deferred_revenue:growth_change", facts) == (
        "CRWD's total deferred revenue growth from a year earlier was steady at the end of the "
        "quarter ended 2026-07-31 against the end of the quarter ended 2026-04-30 (1.06 points "
        "faster, under 2 points either way)"
    )
    # past the line, code writes the size and the direction, the sheet's own reading
    capex = facts.numbers["fin:CRWD:capex:growth_change"]
    assert capex.value is not None and capex.value > 0.02
    assert phrase(capex.id, facts) == (
        "CRWD's capital expenditures growth from a year earlier was 294.10 points faster for "
        "the quarter ended 2026-07-31 than for the quarter ended 2026-04-30"
    )
    fcf = facts.numbers["fin:CRWD:free_cash_flow:growth_change"]
    assert fcf.value is not None and fcf.value < -0.02
    assert "points slower for the quarter ended 2026-07-31 than for" in str(phrase(fcf.id, facts))
    same = replace(facts.numbers["fin:CRWD:revenue:growth_change"], value=0.00001)
    flat = replace(facts, numbers={**facts.numbers, same.id: same})
    assert phrase(same.id, flat) == (
        "CRWD's revenue growth from a year earlier was the same for the quarter ended "
        "2026-07-31 as for the quarter ended 2026-04-30"
    )
    undated = replace(facts, prior_period=None)
    assert phrase("fin:CRWD:revenue:growth_change", undated) == (
        "CRWD's revenue growth from a year earlier was steady for the quarter ended "
        "2026-07-31 against the prior quarter (0.26 points faster, under 2 points either way)"
    )
    cash = replace(facts.numbers["fin:CRWD:cash:growth_change"], value=0.0)
    still = replace(facts, numbers={**facts.numbers, cash.id: cash})
    assert phrase(cash.id, still) == (
        "CRWD's cash and cash equivalents growth from a year earlier was the same at the end "
        "of the quarter ended 2026-07-31 as at the end of the quarter ended 2026-04-30"
    )


@pytest.mark.parametrize(
    ("value", "steady", "pace"),
    [
        (0.0199, True, "1.99 points faster"),
        (0.02, False, "2.00 points faster"),
        (-0.0199, True, "1.99 points slower"),
        (-0.02, False, "2.00 points slower"),
    ],
)
def test_the_growth_wording_turns_at_exactly_two_points(
    store: FilingStore, value: float, steady: bool, pace: str
) -> None:
    facts = table_facts(metrics_for(store, "CRWD"))
    edge = replace(facts.numbers["fin:CRWD:revenue:growth_change"], value=value)
    at = replace(facts, numbers={**facts.numbers, edge.id: edge})
    said = str(phrase(edge.id, at))
    assert pace in said and ("was steady" in said) is steady
    # the sheet's bracket and the rules read the same value the same way
    word = "steady" if steady else pace.rsplit(" ", 1)[1]
    assert f"({word}" in _shown(edge) and growth_reading(value) == word
    balance = replace(facts.numbers["fin:CRWD:cash:growth_change"], value=0.01)
    undated = replace(facts, prior_period=None, numbers={**facts.numbers, balance.id: balance})
    assert str(phrase(balance.id, undated)).startswith(
        "CRWD's cash and cash equivalents growth from a year earlier was steady at the end of "
        "the quarter ended 2026-07-31 against the end of the prior quarter (1.00 points faster"
    )
