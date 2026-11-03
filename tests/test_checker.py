from __future__ import annotations

import pytest

from tenq.brief import Claim, NumberRef
from tenq.checker import check, check_claim, visible_docs, written_numbers
from tenq.filings import FilingStore
from tenq.metrics import MetricsTable, metrics_for

LATEST = "CRWD-0001535527-26-000031"
OLDER_ACC = "0001535527-26-000025"
PID = f"{LATEST}:23"  # "Adoption of Our Solutions..."
REV_YOY = 0.25830401932671315
REV = 1470897000.0


@pytest.fixture
def table(store: FilingStore) -> MetricsTable:
    return metrics_for(store, "CRWD")


def _check(store: FilingStore, table: MetricsTable, claim: Claim) -> str | None:
    meta = store.filings("CRWD")[0]
    return check_claim(claim, table, store, visible_docs(store, meta))


def _rev(text: str, *refs: NumberRef, pid: str = PID) -> Claim:
    return Claim(text, pid, refs or (NumberRef("revenue", "yoy_pct", REV_YOY),))


@pytest.mark.parametrize(
    "text",
    [
        "Revenue grew 26% from a year earlier as customers adopted more modules.",
        "Revenue grew 25.8% from a year earlier.",
        "Revenue grew 25.83 percent year over year.",
        "Revenue rose about 26% in fiscal 2027, as the 10-Q describes.",
        # period labels are wording, not figures (the first live brief lost a claim to this)
        "Revenue grew 26%; over the six-month period cash came from customers.",
        "Revenue grew 26%, as it did over the nine months ended last quarter.",
    ],
)
def test_a_true_rounded_claim_passes(store: FilingStore, table: MetricsTable, text: str) -> None:
    assert _check(store, table, _rev(text)) is None


@pytest.mark.parametrize(
    ("text", "ref"),
    [
        ("Revenue reached $1.47 billion.", NumberRef("revenue", "current", REV)),
        ("Revenue reached $1.5 billion.", NumberRef("revenue", "current", REV)),
        ("Revenue reached $1,470.9 million.", NumberRef("revenue", "current", REV)),
        ("Revenue reached $1,470,897,000.", NumberRef("revenue", "current", REV)),
        ("Revenue rose $302 million.", NumberRef("revenue", "yoy_change", 301945000.0)),
        ("Operating loss narrowed to $33.2 million.", NumberRef("operating_income", "current", -33232000.0)),
        ("Gross margin rose 1.0 percentage point.", NumberRef("gross_margin", "yoy_change", 0.009687261220382326)),
        ("Gross margin rose 97 bps.", NumberRef("gross_margin", "yoy_change", 0.009687261220382326)),
        ("Gross margin was 74.6%.", NumberRef("gross_margin", "current", 0.7457320261037993)),
        ("Net income fell 81% from the prior quarter.", NumberRef("net_income", "qoq_pct", -0.8089580182904875)),
    ],
)  # fmt: skip
def test_display_forms_of_a_tagged_number_pass(
    store: FilingStore, table: MetricsTable, text: str, ref: NumberRef
) -> None:
    assert _check(store, table, Claim(text, PID, (ref,))) is None


def test_a_made_up_passage_id_is_dropped(store: FilingStore, table: MetricsTable) -> None:
    for pid in (f"{LATEST}:2", f"{LATEST}:99999", "CRWD-0001535527-99-000001:1", "nonsense"):
        why = _check(store, table, _rev("Revenue grew 26%.", pid=pid))
        assert why is not None and "does not exist" in why


def test_a_passage_from_a_later_filing_is_dropped(store: FilingStore) -> None:
    older = next(m for m in store.filings("CRWD") if m.accession == OLDER_ACC)
    table = metrics_for(store, "CRWD", OLDER_ACC)
    claim = Claim("Demand grew.", f"{LATEST}:23")
    why = check_claim(claim, table, store, visible_docs(store, older))
    assert why is not None and "earlier one" in why


@pytest.mark.parametrize(
    ("ref", "reason"),
    [
        (NumberRef("revenue", "yoy_pct", 0.30), "claim says"),
        (NumberRef("revenue", "yoy_pct", 0.26), "claim says"),  # rounded in the data, not the text
        (NumberRef("revenue", "current", REV * 1.01), "claim says"),
        (NumberRef("bookings", "current", 1.0), "unknown number"),
        (NumberRef("revenue", "next_year", 1.0), "unknown number"),
        (NumberRef("operating_income", "yoy_pct", 0.5), "no value"),  # loss: no percent
    ],
)
def test_a_wrong_declared_number_is_dropped(
    store: FilingStore, table: MetricsTable, ref: NumberRef, reason: str
) -> None:
    why = _check(store, table, Claim("Revenue grew.", PID, (ref,)))
    assert why is not None and reason in why


def test_a_copied_float_with_dropped_digits_passes(store: FilingStore, table: MetricsTable) -> None:
    assert (
        _check(store, table, _rev("Revenue grew 26%.", NumberRef("revenue", "yoy_pct", 0.2583)))
        is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "Revenue grew 27% from a year earlier.",  # wrong rounding
        "Revenue grew 26% and customers grew 22%.",  # a number not from tagged data
        "Revenue grew 26% to $1.6 billion.",  # undeclared amount
        "Revenue grew 26 million.",  # a percent shown as dollars
        "Revenue grew 26% across 3 segments.",
    ],
)
def test_a_wrong_or_undeclared_number_in_the_text_is_dropped(
    store: FilingStore, table: MetricsTable, text: str
) -> None:
    why = _check(store, table, _rev(text))
    assert why is not None and "in the text" in why


def test_forms_are_not_figures_but_quarter_labels_are() -> None:
    assert [w.text for w in written_numbers("In Q2 of fiscal 2027 the 10-Q and 10-K say")] == [
        "2",
        "2027",
    ]


@pytest.mark.parametrize(
    ("text", "ref"),
    [
        # the text must show the tagged value, not a declared value 0.09% off it
        ("Revenue reached $1,472 million.", NumberRef("revenue", "current", REV * 1.0009)),
        ("Revenue grew 25.85%.", NumberRef("revenue", "yoy_pct", REV_YOY * 1.0009)),
        # figures glued to letters or a leading decimal point
        ("Revenue reached USD1.6 billion.", None),
        ("Revenue reached EUR9 billion and rose x99%.", None),
        ("Gross margin rose .5 percentage points.", None),
        ("In Q4 revenue grew 26%.", None),
        # a bare number, or a number that does not say what it is
        ("Revenue grew 26x.", None),
        ("Revenue grew 26% and the company added 26 new customers.", None),
        ("Revenue grew 26% to 1.47 billion.", None),
        # numbers in words or non-ASCII digit characters
        ("Revenue grew twenty-six percent; it doubled.", None),
        ("Revenue tripled, a threefold rise.", None),
        ("Revenue grew 26%, six times the market's rate.", None),
        ("Revenue doubled in six months.", None),
        ("Revenue grew 26%; cash covers eighteen months of spending.", None),
        ("Revenue grew 26% after six month-over-month gains.", None),
        ("Revenue grew 26% and \u00bd of it came from \u2460\u2461.", None),
        ("Revenue grew 26% \uff12\uff16.", None),
        ("Revenue grew 26% and topped a billion dollars.", None),
        # sign and loss wording against a positive value
        ("Net income was -$5.3 million.", NumberRef("net_income", "current", 5306000.0)),
        (
            "Revenue fell, a net loss of $5.3 million.",
            NumberRef("net_income", "current", 5306000.0),
        ),
        ("Revenue grew -26% from a year earlier.", None),
        # bare years only near the filing's own year
        ("Revenue grew 26% and headcount reached 2,050.", None),
        ("Revenue grew 26%, a pace last seen in 2099.", None),
        # coarse roundings and the wrong unit for the kind
        ("Revenue reached $0 trillion.", NumberRef("revenue", "current", REV)),
        ("Revenue reached $1 billion.", NumberRef("revenue", "current", REV)),
        (
            "Gross margin rose 1.0%.",
            NumberRef("gross_margin", "yoy_change", 0.009687261220382326),
        ),
    ],
)
def test_a_number_the_tagged_data_does_not_show_is_dropped(
    store: FilingStore, table: MetricsTable, text: str, ref: NumberRef | None
) -> None:
    why = _check(store, table, _rev(text) if ref is None else Claim(text, PID, (ref,)))
    assert why is not None and "in the text" in why


def test_a_nan_declared_value_is_dropped(store: FilingStore, table: MetricsTable) -> None:
    why = _check(store, table, _rev("Revenue grew.", NumberRef("revenue", "yoy_pct", float("nan"))))
    assert why is not None and "claim says" in why


def test_an_operating_loss_change_may_say_loss(store: FilingStore, table: MetricsTable) -> None:
    ref = NumberRef("operating_income", "current", -33232000.0)
    assert _check(store, table, Claim("Operating loss was -$33.2 million.", PID, (ref,))) is None


def test_check_keeps_good_claims_and_logs_each_drop(
    store: FilingStore, table: MetricsTable
) -> None:
    meta = store.filings("CRWD")[0]
    good = _rev("Revenue grew 26%.")
    claims = [
        good,
        _rev("Revenue grew 26%."),  # duplicate
        _rev("Revenue grew 40%."),
        Claim("Customers adopted more modules.", f"{LATEST}:15"),
        Claim("Invented.", f"{LATEST}:4"),
    ]
    result = check(claims, table, store, meta)
    assert result.kept == [good, claims[3]]
    assert [d.reason.split()[0] for d in result.dropped] == ["duplicate", "number", "passage"]
