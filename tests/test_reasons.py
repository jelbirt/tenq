"""The daily calls' reasons: placeholders that code fills in from the facts pack."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from tenq.pack import Excerpt, Filing, Headline, Number, Pack, Report, money
from tenq.reasons import (
    MAX_REASON_CHARS,
    Reason,
    Rendered,
    check_reasons,
    figure_in,
    phrase,
    render,
)

NEWS = "news:ACME:0123456789"
EXCERPT = "ACME-0000000000-26-000001:4"


def _n(key: str, label: str, value: float, kind: str) -> Number:
    return Number(key, label, value, kind)  # type: ignore[arg-type]


PACK = Pack(
    "ACME",
    "Acme",
    date(2026, 11, 3),
    datetime(2026, 11, 4, 1, 0, tzinfo=UTC),
    numbers=[
        _n("px:ACME:close", "Close on 2026-11-03", 220.52, "usd"),
        _n("px:ACME:ret_1d", "ACME return over 1 session", 0.0, "percent"),
        _n("px:ACME:ret_5d", "ACME return over 5 sessions", -0.2147, "percent"),
        _n("px:ACME:bench_5d", "QQQ return over 5 sessions", 0.0213, "percent"),
        _n("px:ACME:excess_5d", "ACME return less QQQ over 5 sessions", -0.1649, "points"),
        _n("px:ACME:excess_20d", "ACME return less QQQ over 20 sessions", 0.0412, "points"),
        _n("px:ACME:vol_20d", "Volatility, annualised, over 20 sessions", 0.8157, "percent"),
        _n("fin:ACME:revenue:current", "Revenue, latest quarter", 1_121_000_000.0, "usd"),
        _n(
            "fin:ACME:revenue:yoy_pct",
            "Revenue, change from a year ago, percent",
            0.3564,
            "percent",
        ),
        _n(
            "fin:ACME:operating_income:current",
            "Operating income, latest quarter",
            -4_000_000.0,
            "usd",
        ),
        _n(
            "fin:ACME:operating_income:year_ago",
            "Operating income, same quarter a year ago",
            -118_700_000.0,
            "usd",
        ),
        _n(
            "fin:ACME:operating_income:yoy_change",
            "Operating income, change from a year ago",
            114_700_000.0,
            "usd",
        ),
        _n(
            "fin:ACME:free_cash_flow:prior_quarter",
            "Free cash flow (operating cash flow less capex), prior quarter",
            -5_000_000.0,
            "usd",
        ),
        _n("fin:ACME:gross_margin:current", "Gross margin, latest quarter", 0.42, "percent"),
        _n(
            "fin:ACME:gross_margin:qoq_change",
            "Gross margin, change from the prior quarter",
            -0.012,
            "points",
        ),
    ],
    filing=Filing("ACME-0000000000-26-000002", "10-Q", "2026-07-31", "2026-08-27", 68),
    reports=[
        Report("ACME-0000000000-26-000001", "2026-11-02", ("Item 2.02",), (Excerpt(EXCERPT, "x"),))
    ],
    news=[Headline(NEWS, "Acme stock drops 18% on outlook", "reuters.com", "2026-11-03", "")],
)


@pytest.mark.parametrize(
    ("number_id", "words"),
    [
        ("px:ACME:close", "ACME closed at $220.52 on 2026-11-03"),
        ("px:ACME:ret_1d", "ACME was unchanged over the last session"),
        ("px:ACME:ret_5d", "ACME fell 21.47% over the last 5 sessions"),
        ("px:ACME:bench_5d", "QQQ rose 2.13% over the last 5 sessions"),
        ("px:ACME:excess_5d", "ACME trailed QQQ by 16.49 points over the last 5 sessions"),
        ("px:ACME:excess_20d", "ACME beat QQQ by 4.12 points over the last 20 sessions"),
        ("px:ACME:vol_20d", "ACME's annualised volatility over the last 20 sessions was 81.57%"),
        (
            "fin:ACME:revenue:current",
            "ACME's revenue was $1.121 billion in the quarter ended 2026-07-31",
        ),
        ("fin:ACME:revenue:yoy_pct", "ACME's revenue rose 35.64% from a year earlier"),
        (
            "fin:ACME:operating_income:current",
            "ACME's operating income was a loss of $4.000 million in the quarter ended 2026-07-31",
        ),
        (
            "fin:ACME:operating_income:year_ago",
            "ACME's operating income was a loss of $118.7 million in the same quarter a year earlier",
        ),
        # a loss that shrank is income that rose: the sign says so, not a word
        (
            "fin:ACME:operating_income:yoy_change",
            "ACME's operating income rose by $114.7 million from a year earlier",
        ),
        (
            "fin:ACME:free_cash_flow:prior_quarter",
            "ACME's free cash flow (operating cash flow less capex) was negative $5.000 million in the prior quarter",
        ),
        (
            "fin:ACME:gross_margin:current",
            "ACME's gross margin was 42.00% in the quarter ended 2026-07-31",
        ),
        (
            "fin:ACME:gross_margin:qoq_change",
            "ACME's gross margin fell 1.20 points from the prior quarter",
        ),
    ],
)
def test_code_writes_each_number_with_its_sign_direction_and_period(
    number_id: str, words: str
) -> None:
    assert phrase(number_id, PACK) == words


def test_the_money_in_a_phrase_is_the_packs_own_formatting() -> None:
    assert money(1_121_000_000.0) in phrase("fin:ACME:revenue:current", PACK)  # type: ignore[operator]


def test_without_a_filing_the_latest_quarter_is_named_in_words() -> None:
    bare = replace(PACK, filing=None)
    assert (
        phrase("fin:ACME:revenue:current", bare)
        == "ACME's revenue was $1.121 billion in the latest quarter"
    )


@pytest.mark.parametrize(
    "number_id",
    [
        "px:ACME:nope",  # not on the pack
        "news:ACME:0123456789",  # on the pack, but not a number
        "px:OTHER:ret_5d",
        "",
    ],
)
def test_only_the_packs_numbers_have_words(number_id: str) -> None:
    assert phrase(number_id, PACK) is None


def test_a_number_of_a_shape_code_has_no_words_for_is_none() -> None:
    odd = replace(
        PACK,
        numbers=[
            _n("px:ACME:ret_weird", "?", 0.1, "percent"),
            _n("fin:ACME:revenue:odd", "Revenue, odd", 1.0, "usd"),
        ],
    )
    assert phrase("px:ACME:ret_weird", odd) is None
    assert phrase("fin:ACME:revenue:odd", odd) is None


def test_a_reason_is_sent_in_codes_words() -> None:
    got = render(Reason("{px:ACME:ret_5d}, after a rival cut its outlook.", (NEWS,)), PACK)
    assert got == Rendered(
        "ACME fell 21.47% over the last 5 sessions, after a rival cut its outlook.",
        (NEWS, "px:ACME:ret_5d"),
    )


def test_several_placeholders_and_a_capital_letter() -> None:
    got = render(Reason("{fin:ACME:revenue:yoy_pct} and {px:ACME:excess_5d}.", ()), PACK)
    assert isinstance(got, Rendered)
    assert got.text == (
        "ACME's revenue rose 35.64% from a year earlier and "
        "ACME trailed QQQ by 16.49 points over the last 5 sessions."
    )
    assert got.cites == ("fin:ACME:revenue:yoy_pct", "px:ACME:excess_5d")


def test_a_reason_on_news_alone_is_sent_as_written() -> None:
    got = render(
        Reason("A rival cut its outlook, which weighs on the whole group.", (NEWS, EXCERPT)), PACK
    )
    assert isinstance(got, Rendered) and got.text.startswith("A rival cut")


@pytest.mark.parametrize(
    ("text", "cites", "why"),
    [
        ("The outlook disappointed.", (), "cites nothing"),
        ("The outlook disappointed.", ("news:ACME:ffffffffff",), "not on the facts sheet"),
        ("{px:ACME:ret_20d}, this month.", (), "not a number on the facts sheet"),
        ("{news:ACME:0123456789}, hurt.", (), "not a number on the facts sheet"),
        ("{px:ACME:ret_5d, a weak week.", (), "curly bracket"),
        ("{px:ACME:ret_5d}} on the outlook.", (), "curly bracket"),
        # a figure in the model's own words, however it is written
        ("{px:ACME:ret_5d}, its worst week in 3 years.", (), "'3'"),
        ("{px:ACME:ret_5d}, about 21% in all.", (), "'2'"),
        ("{px:ACME:ret_5d}, nearly %.", (), "'%'"),
        ("{px:ACME:ret_5d}, as $ flowed out.", (), "'$'"),
        ("{px:ACME:ret_5d}, as \u20ac flowed out.", (), "'\u20ac'"),
        ("{px:ACME:ret_5d}, its worst week in \uff13 years.", (), "\uff13"),  # fullwidth digit
        ("{px:ACME:ret_5d}, down \u00bd from the peak.", (), "\u00bd"),  # a fraction sign
        ("{px:ACME:ret_5d}, its worst week in five years.", (), "five"),
        ("{px:ACME:ret_5d}, after losses doubled.", (), "doubled"),
        ("{px:ACME:ret_5d}, giving back a third of its gains.", (), "third"),
        ("{px:ACME:ret_5d}, giving back half its gains.", (), "half"),
        ("{px:ACME:ret_5d}, as millions of shares changed hands.", (), "millions"),
        ("{px:ACME:ret_5d}, several percent below its peak.", (), "percent"),
        ("{px:ACME:ret_5d}, after a double-digit drop.", (), "double-digit"),
        ("{px:ACME:ret_5d}, in a tenfold slide.", (), "tenfold"),
        ("{px:ACME:ret_5d}, the #1 laggard.", (), "'#'"),
    ],
)
def test_a_reason_that_fails_is_dropped_with_why(
    text: str, cites: tuple[str, ...], why: str
) -> None:
    got = render(Reason(text, cites), PACK)
    assert isinstance(got, str) and why in got, got


@pytest.mark.parametrize(
    "words",
    [
        "One of the largest deals in the group.",
        "The first report since the outage.",
        "Its second product line is new.",
        "A zero-day flaw and zero trust products drew attention.",
        "Attackers used double extortion; the company will double down.",
        "Third-party risk tools and third parties were named.",
        "Troops march on; the deal may close.",
        "Monday.com cut its outlook.",
        "A manifold of issues began to unfold.",
    ],
)
def test_plain_english_is_not_a_figure(words: str) -> None:
    assert figure_in(words) is None


def test_check_reasons_keeps_the_good_drops_the_bad_and_the_repeated() -> None:
    reasons = [
        Reason("{px:ACME:ret_5d}, on the outlook.", (NEWS,)),
        Reason("{px:ACME:ret_5d},  on the outlook.", ()),  # the same sentence once rendered
        Reason("{px:ACME:ret_5d}, the worst in 3 years.", ()),
    ]
    got = check_reasons(reasons, PACK)
    assert [k.text for k in got.kept] == [
        "ACME fell 21.47% over the last 5 sessions, on the outlook."
    ]
    assert [d.why for d in got.dropped] == [
        "duplicate reason",
        "a figure typed in its own words ('3')",
    ]


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Revenue rose in the third quarter.", "third"),
        ("A fifth straight quarter of growth.", "fifth"),
        ("Guidance for the second half disappointed.", "second half"),
        ("Its first-half results were strong.", "first-half"),
        ("The first time since the outage.", "first time"),
        ("A first straight gain.", "first straight"),
        ("Shares are down since Monday.", "Monday"),
        ("Halving since October.", "Halving"),
        ("Shares fell by a quarter.", "a quarter"),
        ("Shares rose a quarter on the news.", "a quarter"),
        ("Sales from Sept. to Dec. were weak.", "Sept."),
        ("Chapter XV of the plan.", "XV"),
        ("Down a tenth.", "tenth"),
        ("A twentieth of revenue.", "twentieth"),
        ("One hundredth of a point.", "hundredth"),
        ("Slowing to the low single digits.", "single digits"),
        ("A mid-single digit drop.", "single digit"),
        ("Growth in the mid-teens.", "teens"),
        ("The stock trades in the forties.", "forties"),
        ("Volume quintupled.", "quintupled"),
        ("Sales quadrupling.", "quadrupling"),
        ("A triple-digit gain.", "triple-digit"),
        ("The lowest since IV years.", "IV"),
        ("Up a few pct.", "pct"),
        ("Spreads widened a bp.", "bp"),
        ("A two-thirds share.", "two"),
        ("Dozens of deals.", "Dozens"),
        ("Hundreds of customers.", "Hundreds"),
        ("Zero growth.", "Zero"),
        ("A fourfold rise.", "fourfold"),
        ("Revenue doubled.", "doubled"),
        ("Down \u066a.", "\u066a"),
        ("Up \ufe6a.", "\ufe6a"),
        ("Up \uff05.", "\uff05"),
        ("A \u2030 change.", "\u2030"),
    ],
)
def test_figure_in_finds_every_kind_of_figure(text: str, found: str) -> None:
    assert figure_in(text) == found


@pytest.mark.parametrize(
    "text",
    [
        "Subscription {fin:ACME:revenue:yoy_pct}.",
        "Non-GAAP {fin:ACME:operating_income:current}.",
        "A rival's {fin:ACME:revenue:yoy_pct}.",
        "Shares {px:ACME:ret_5d}.",
        "It is not the case that {px:ACME:ret_5d}.",
        "The outlook was weak, not {px:ACME:ret_5d}.",
        "More than {px:ACME:ret_5d}.",
    ],
)
def test_a_placeholder_must_open_a_clause(text: str) -> None:
    got = render(Reason(text, ()), PACK)
    assert isinstance(got, str) and "does not open a clause" in got, got


@pytest.mark.parametrize(
    "text",
    [
        # words that attach to code's clause change what its number measures
        "{fin:ACME:revenue:yoy_pct} excluding acquisitions.",
        "{px:ACME:ret_5d} after adjusting for the split.",
        "{px:ACME:ret_5d} on the outlook.",
        "{px:ACME:ret_5d} and then some.",
        "{px:ACME:ret_5d} (excluding the split).",
        "{px:ACME:ret_5d}{px:ACME:bench_5d}.",
    ],
)
def test_a_placeholder_must_close_a_clause(text: str) -> None:
    got = render(Reason(text, ()), PACK)
    assert isinstance(got, str) and "does not close a clause" in got, got


@pytest.mark.parametrize(
    "text",
    [
        "{px:ACME:ret_5d}",
        "{px:ACME:ret_5d}.",
        "{px:ACME:ret_5d}; the outlook was cut.",
        "{px:ACME:ret_5d} and {px:ACME:bench_5d}.",
        "{px:ACME:ret_5d} while {px:ACME:bench_5d}.",
        "{px:ACME:ret_5d}. {px:ACME:bench_5d}.",
    ],
)
def test_a_placeholder_closed_by_a_mark_or_a_joined_placeholder_is_kept(text: str) -> None:
    assert isinstance(render(Reason(text, ()), PACK), Rendered)


@pytest.mark.parametrize(
    "text",
    [
        "{px:ACME:ret_5d}.",
        "The outlook disappointed, {px:ACME:ret_5d}.",
        "The outlook disappointed; {px:ACME:ret_5d}.",
        "One thing stood out: {px:ACME:ret_5d}.",
        "The outlook disappointed ({px:ACME:ret_5d}).",
        "The outlook disappointed. {px:ACME:ret_5d}.",
        "Demand held up and {px:ACME:ret_5d}.",
        "Demand held up, but {px:ACME:ret_5d}.",
        "Investors sold while {px:ACME:ret_5d}.",
        "Demand held up even as {px:ACME:ret_5d}.",
        "Because {px:ACME:ret_5d}, the call is cautious.",
    ],
)
def test_a_placeholder_after_a_mark_or_joining_word_is_kept(text: str) -> None:
    assert isinstance(render(Reason(text, ()), PACK), Rendered)


def test_a_number_is_shown_never_only_cited() -> None:
    got = render(Reason("The stock slumped this week.", ("px:ACME:bench_5d",)), PACK)
    assert isinstance(got, str) and "without showing it" in got
    shown = render(Reason("{px:ACME:bench_5d}.", ("px:ACME:bench_5d",)), PACK)
    assert isinstance(shown, Rendered)


def test_a_too_long_reason_is_dropped_with_why() -> None:
    got = render(Reason("x" * (MAX_REASON_CHARS + 1), (NEWS,)), PACK)
    assert got == f"longer than {MAX_REASON_CHARS} characters"


@pytest.mark.parametrize(
    ("number_id", "value", "kind", "words"),
    [
        ("px:ACME:ret_5d", -0.00004, "percent", "ACME was unchanged over the last 5 sessions"),
        ("px:ACME:excess_5d", 1e-17, "points", "ACME matched QQQ over the last 5 sessions"),
        ("px:ACME:excess_5d", 0.0, "points", "ACME matched QQQ over the last 5 sessions"),
        (
            "fin:ACME:gross_margin:current",
            -0.12,
            "percent",
            "ACME's gross margin was negative 12.00% in the quarter ended 2026-07-31",
        ),
        (
            "fin:ACME:gross_margin:current",
            -0.00001,
            "percent",
            "ACME's gross margin was 0.00% in the quarter ended 2026-07-31",
        ),
        (
            "fin:ACME:operating_income:yoy_change",
            0.001,
            "usd",
            "ACME's operating income was unchanged from a year earlier",
        ),
    ],
)
def test_a_value_that_shows_as_zero_claims_no_direction(
    number_id: str, value: float, kind: str, words: str
) -> None:
    label = next(n.label for n in PACK.numbers if n.id == number_id)
    pack = replace(PACK, numbers=[_n(number_id, label, value, kind)])
    assert phrase(number_id, pack) == words


def test_spaces_inside_a_placeholder_are_ignored() -> None:
    got = render(Reason("{ px:ACME:ret_5d }.", ()), PACK)
    assert isinstance(got, Rendered) and got.cites == ("px:ACME:ret_5d",)


def test_a_label_without_the_fields_wording_has_no_words() -> None:
    odd = replace(PACK, numbers=[_n("fin:ACME:revenue:current", "Revenue", 1.0, "usd")])
    assert phrase("fin:ACME:revenue:current", odd) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("number_id", "label", "kind"),
    [
        ("px:ACME:ret_20d", "Return over 20 sessions", "percent"),
        ("fin:ACME:revenue:current", "Revenue, latest quarter", "usd"),
    ],
)
def test_a_number_that_is_not_finite_has_no_words(
    number_id: str, label: str, kind: str, value: float
) -> None:
    bad = replace(PACK, numbers=[_n(number_id, label, value, kind)])
    assert phrase(number_id, bad) is None
    assert render(Reason("{" + number_id + "}.", ()), bad) == (
        f"placeholder {number_id!r} has no finite value"
    )


@pytest.mark.parametrize(
    "text",
    [
        "{px:ACME:excess_5d}, and it beat QQQ.",
        "{px:ACME:ret_5d}; shares actually rose this week.",
        "{px:ACME:ret_5d}, and so did its rival.",
    ],
)
def test_words_that_argue_with_codes_clause_are_left_to_the_reader(text: str) -> None:
    # Deliberate: code writes the number clause and never judges the model's
    # own words for truth. They hold no figure, and the true number is beside them.
    assert isinstance(render(Reason(text, ()), PACK), Rendered)
