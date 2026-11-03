"""Check every claim before it can be sent. A claim that fails is dropped and logged.

A claim passes only if:
1. its passage_id resolves to a saved passage of this ticker's briefed filing
   or an earlier one;
2. every number it declares names a real metric and field, and its value is
   the tagged-data value (allowing only the digits a model drops copying a
   float);
3. every number written in its text displays one of those tagged values, in a
   form that says what it is: "$1.47 billion", "26%", "1.2 percentage points",
   "97 bps". A bare number, a number in words ("doubled", "twenty"), a digit
   character outside 0-9, a minus sign on a positive value and "loss" next to
   a positive amount all fail. The only bare number allowed is a year within
   two of the filing's.

The written-number rules lean strict on purpose: a false drop costs one
sentence of the brief, a false accept puts a wrong number in an email.

What this cannot catch: a real passage that does not support the claim. That
is measured by evals (slice 2), not asserted here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from tenq.brief import Claim, NumberRef
from tenq.filings import FilingMeta, FilingStore
from tenq.metrics import CHANGE_FIELDS, FIELDS, MetricsTable

Kind = Literal["usd", "percent", "points"]

# a model copying 0.25832795 may write 0.2583; anything looser is a different number
VALUE_REL_TOL = 1e-3
VALUE_ABS_TOL = {"usd": 0.5, "percent": 5e-5, "points": 5e-5}
SCALES = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12,
}  # fmt: skip
NUMBER = re.compile(
    r"(?<![0-9])(?P<sign>(?<![0-9])[-\u2212\u2013]\s?)?(?P<dollar>\$\s?)?"
    r"(?:(?P<int>[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.(?P<dec>[0-9]+))?|\.(?P<lead>[0-9]+))"
    r"(?:\s?(?P<unit>%|percent(?:age points?)?|pp|points?|bps|basis points|"
    r"thousand|million|billion|trillion|bn|mn|[kmb])(?![a-z]))?",
    re.IGNORECASE,
)
# amounts and ratios a code check cannot see. Scale words ("billion") only
# matter after a written figure, which NUMBER reads. "one" and "quarter" are left out
# because ordinary period wording ("the quarter") needs them
NUMBER_WORD = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|"
    r"dozen|half|halved|double[ds]?|doubling|twice|triple[ds]?|tripling|"
    r"quadrupled?|\w+fold)\b",
    re.IGNORECASE,
)
# a scale word with no figure in front of it ("topped a billion")
BARE_SCALE = re.compile(
    r"(?<![0-9])(?<![0-9] )\b(?:thousand|million|billion|trillion)\b", re.IGNORECASE
)
LOSS_WORD = re.compile(r"\b(?:loss|losses|negative|deficit)\b", re.IGNORECASE)
LOSS_WINDOW = 30  # characters before an amount searched for a loss word
YEAR_SPAN = 2  # bare years allowed within this many of the filing's year
# a rounding this coarse ("$1 billion" for $1.47B) must still be close
COARSE_REL_TOL = 0.05
FORM_NAME = re.compile(r"\b10-[QK]\b", re.IGNORECASE)
# a filing's own period wording ("the six-month period", "nine months ended",
# "the first half of the year") is not a figure; other month counts ("a runway
# of eighteen months") and halves ("half of revenue") still are
PERIOD_LABEL = re.compile(
    r"\b(?:three|six|nine|twelve)(?:-month|\s+months)\s+(?:periods?|ended|ending)\b"
    r"|\b(?:first|second)\s+half\s+of\s+(?:the\s+)?(?:fiscal\s+)?(?:year\b(?!-)|20[0-9]{2}\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Dropped:
    claim: Claim
    reason: str


@dataclass
class CheckResult:
    kept: list[Claim] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)


def _kind(table: MetricsTable, ref: NumberRef) -> Kind:
    if ref.field.endswith("_pct"):
        return "percent"
    if table.rows[ref.metric].unit == "ratio":
        return "points" if ref.field in CHANGE_FIELDS else "percent"
    return "usd"


def check_declared(table: MetricsTable, ref: NumberRef) -> str | None:
    """None if the declared number is the tagged-data value, else why not."""
    if ref.metric not in table.rows or ref.field not in FIELDS:
        return f"unknown number {ref.metric}.{ref.field}"
    truth = table.number(ref.metric, ref.field)
    if truth is None:
        return f"{ref.metric}.{ref.field} has no value in the tagged data"
    tol = max(VALUE_REL_TOL * abs(truth), VALUE_ABS_TOL[_kind(table, ref)])
    if not abs(ref.value - truth) <= tol:  # written so that NaN fails
        return f"{ref.metric}.{ref.field} is {truth!r}, claim says {ref.value!r}"
    return None


@dataclass(frozen=True)
class Written:
    text: str
    value: float  # as written, before any scale word
    half_unit: float  # rounding allowance for the digits shown
    sig_figs: int
    dollar: bool
    unit: str
    negative: bool  # written with a minus sign
    loss_word: bool  # "loss", "negative" or "deficit" just before it


def written_numbers(text: str) -> list[Written]:
    out = []
    text = FORM_NAME.sub(" ", text)  # "the 10-Q" names a form, it is not a figure
    for m in NUMBER.finditer(text):
        digits = (m.group("int") or "0").replace(",", "")
        dec = m.group("dec") or m.group("lead") or ""
        shown = (digits + dec).lstrip("0")
        out.append(
            Written(
                text=m.group(0).strip(),
                value=float(f"{digits}.{dec}" if dec else digits),
                half_unit=0.5 * 10 ** -len(dec),
                sig_figs=len(shown) if dec or not shown else len(shown.rstrip("0")) or 1,
                dollar=bool(m.group("dollar")),
                unit=(m.group("unit") or "").lower(),
                negative=bool(m.group("sign")),
                loss_word=bool(LOSS_WORD.search(text[max(0, m.start() - LOSS_WINDOW) : m.start()])),
            )
        )
    return out


def _uncheckable(text: str) -> str | None:
    """A number the scanner cannot read: a number word or a non-ASCII digit."""
    m = NUMBER_WORD.search(text) or BARE_SCALE.search(text)
    if m:
        return m.group(0)
    for ch in text:
        if ch not in "0123456789" and unicodedata.numeric(ch, None) is not None:
            return ch
    return None


def _explains(w: Written, kind: Kind, truth: float, field: str) -> bool:
    """Does the written number display `truth` in a form that names its kind?"""
    if w.negative and truth >= 0:
        return False
    if w.loss_word and kind == "usd" and field not in CHANGE_FIELDS and truth >= 0:
        return False
    t = abs(truth)  # "fell 3%" is fine without a sign
    if w.unit in ("%", "percent"):
        shown = t * 100 if kind == "percent" else None
    elif w.unit in ("percentage point", "percentage points", "pp", "point", "points"):
        shown = t * 100 if kind == "points" else None
    elif w.unit in ("bps", "basis points"):
        shown = t * 10_000 if kind == "points" else None
    elif w.dollar and kind == "usd":
        shown = t / SCALES[w.unit] if w.unit in SCALES else t
    else:  # a bare number, or a scale word without "$": it does not say what it is
        shown = None
    if shown is None:
        return False
    if shown == 0:
        return w.value == 0
    eps = 1e-9 * max(1.0, w.value)
    if abs(shown - w.value) > w.half_unit + eps:
        return False
    return w.sig_figs >= 2 or abs(shown - w.value) <= COARSE_REL_TOL * shown


def _is_year(w: Written, year: int) -> bool:
    plain = not (w.dollar or w.unit or w.negative)
    return (
        plain and re.fullmatch(r"[0-9]{4}", w.text) is not None and abs(w.value - year) <= YEAR_SPAN
    )


def check_claim(
    claim: Claim, table: MetricsTable, store: FilingStore, visible: set[str]
) -> str | None:
    """None if the claim may be sent, else the reason it is dropped."""
    p = store.passage(claim.passage_id)
    if p is None:
        return f"passage {claim.passage_id!r} does not exist"
    if p.doc not in visible:
        return f"passage {claim.passage_id!r} is not from this filing or an earlier one"
    for ref in claim.numbers:
        why = check_declared(table, ref)
        if why:
            return why
    odd = _uncheckable(PERIOD_LABEL.sub(" ", FORM_NAME.sub(" ", claim.claim)))
    if odd is not None:
        return f"number {odd!r} in the text cannot be checked"
    # the text is checked against the tagged value, not the declared copy of it
    declared = [
        (_kind(table, r), table.number(r.metric, r.field) or 0.0, r.field) for r in claim.numbers
    ]
    year = int(table.period[:4])
    for w in written_numbers(claim.claim):
        if _is_year(w, year):
            continue
        if not any(_explains(w, kind, truth, fld) for kind, truth, fld in declared):
            return f"number {w.text!r} in the text is not a declared tagged-data number"
    return None


def visible_docs(store: FilingStore, meta: FilingMeta) -> set[str]:
    return {m.doc for m in store.filings(meta.ticker) if m.period <= meta.period}


def check(
    claims: list[Claim], table: MetricsTable, store: FilingStore, meta: FilingMeta
) -> CheckResult:
    visible = visible_docs(store, meta)
    result = CheckResult()
    seen: set[str] = set()
    for c in claims:
        why = check_claim(c, table, store, visible)
        if why is None and c.claim.lower() in seen:
            why = "duplicate claim"
        if why is None:
            result.kept.append(c)
            seen.add(c.claim.lower())
        else:
            result.dropped.append(Dropped(c, why))
    return result
