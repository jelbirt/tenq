"""Reasons for a daily call, written with placeholders that code fills in.

The model never types a figure. Where a reason uses a number from the facts
pack, it writes that number's id in curly brackets, "{px:ACME:ret_5d}", and
code writes that part of the sentence from the pack: the value, its sign, the
direction word and the period ("ACME fell 21.47% over the last 5 sessions").

A reason is sent only when:
- every placeholder is a number on that day's pack, and every cited id is an
  8-K excerpt or headline on the pack (a number is shown, never just cited);
- each placeholder opens a clause: it starts the reason, or follows a
  comma, semicolon, colon or opening bracket, or one of a few joining words
  (CONNECTORS), so no word of the model's can become part of code's subject
  ("Subscription {...}");
- the words the model wrote itself hold no figure at all: no digit or other
  numeric character, no "%" or currency sign, no number word ("doubled",
  "five", "a third", "millions"), no quarter, month or weekday name, and no
  stray curly bracket.

So the clause code writes is always right: its value, sign, direction,
period, unit and company come from the pack. What this does not stop, by
design, is the model's own words disagreeing with that clause ("{...}, and it
beat QQQ", "{...} is not what happened", "{...}, and so did its rival"), or
a weak or beside-the-point reason. Those words hold no figure, and the true
number sits in the same sentence for the reader. Judging them would mean
reading the model's English for truth, which is the design this replaced.
The word "one" and "first" or "second" on their own are allowed, since they
are far more often plain English.

This replaces reading the model's English to judge its numbers, which no set
of rules can do reliably. The quarterly brief's claims are written the same
way (tenq.checker calls fill() on each one), with only filing numbers.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from tenq.metrics import BASE_METRICS, MetricsTable
from tenq.pack import FIELD_LABELS, Number, Pack, filing_numbers, money
from tenq.prices import BENCHMARK

PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_UNITS = r"two|three|four|five|six|seven|eight|nine|ten"
NUMBER_WORD = re.compile(
    # counts and sizes
    r"\b(?:zero(?![\s-](?:day|trust))|" + _UNITS + r"|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"teens|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties|"
    r"hundreds?|thousands?|millions?|billions?|trillions?|dozens?|"
    r"(?:single|double|triple)[\s-]digits?|"
    r"doubled|doubles|doubling|twice|tripl\w*|quadrupl\w*|quintupl\w*|"
    r"(?:" + _UNITS + r"|hundred|thousand)fold|"
    r"hal(?:f|ves|ved|ving)|"
    r"percent|per\s+cent|percentage|pct|basis\s+points?|bps?)\b"
    # ordinals from third up, and first or second as a count of periods
    r"|\b(?:third(?![\s-]part(?:y|ies))|fourth|fifth|sixth|seventh|eighth|ninth|"
    r"tenth|eleventh|twelfth|\w+teenth|\w+tieth|hundredth|thousandth|millionth)\b"
    r"|\b(?:first|second)[\s-]+(?:straight|consecutive|quarter|half|year|time)\b"
    r"|\ba\s+quarter\b(?!\s+(?:earlier|ago|before|later)\b)"
    r"|\b(?:one|two|three)-(?:third|quarter|half|fifth)s?\b",
    re.IGNORECASE,
)
# names that date a claim; capitalised only, so the verbs "march" and "may" pass
DATE_WORD = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Monday(?!\.com)|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday|II|III|IV|VI|VII|VIII|IX|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\b"
    # month abbreviations, only with their full stop so "Mar" and "Dec" as names pass
    r"|\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\."
)
# percent, per mille, per ten thousand, fullwidth, small and Arabic percent, "#1"
MARKS = {"%", "\u2030", "\u2031", "\uff05", "\ufe6a", "\u066a", "#"}
# a placeholder opens a clause: after these marks, or after one of these words
OPENERS = (",", ";", ":", "(", ".", "!", "?")
CONNECTORS = re.compile(
    r"(?:^|\s)(?:and|but|while|as|after|because|though|although|when|yet|since|"
    r"so|then|whereas|even\s+as|even\s+though)$",
    re.IGNORECASE,
)
# and a placeholder closes one: the end, one of these marks, or a joining word and
# another placeholder, so no words of the model's attach to code's clause and change
# what its number measures ("{...} after adjusting for a charge")
CLOSERS = (",", ";", ":", ".", "!", "?", ")")
JOINERS = re.compile(r"(?:and|but|while|whereas|yet)\s+\{", re.IGNORECASE)
MAX_REASON_CHARS = 600  # placeholders are long ids; a longer reason is dropped, never cut
LOSS_METRICS = {"operating_income", "net_income"}  # a negative level reads "a loss of"
# how a metric reads mid-sentence where its table label does not
PHRASE_LABELS = {
    "deferred_revenue": "total deferred revenue",
    "deferred_revenue_current": "current deferred revenue",
    "deferred_revenue_noncurrent": "noncurrent deferred revenue",
}
PLURAL_METRICS = {"capex", "cash"}  # "capital expenditures were", "cash and ... were"
# balances: read "at the end of"; total deferred revenue is the sum of two of them
STOCK_METRICS = frozenset(m.name for m in BASE_METRICS if m.kind == "stock") | {"deferred_revenue"}


@dataclass(frozen=True)
class Facts:
    """What code may write numbers from: one company's numbers by id, and the
    dates its words need. A daily pack and a filing's metrics table both give one."""

    ticker: str
    numbers: dict[str, Number]
    period: str | None = None  # the latest quarter's end, for "the quarter ended ..."
    day: date | None = None  # the price session, for "closed at ... on ..."
    sources: frozenset[str] = frozenset()  # ids that may be cited but hold no number

    def number(self, number_id: str) -> Number | None:
        return self.numbers.get(number_id)

    def ids(self) -> set[str]:
        return set(self.numbers) | self.sources


def _by_id(numbers: Iterable[Number]) -> dict[str, Number]:
    out: dict[str, Number] = {}
    for n in numbers:
        out.setdefault(n.id, n)  # the first wins, as Pack.number does
    return out


def pack_facts(pack: Pack) -> Facts:
    return Facts(
        pack.ticker,
        _by_id(pack.numbers),
        pack.filing.period if pack.filing else None,
        pack.day,
        frozenset(pack.ids() - {n.id for n in pack.numbers}),
    )


def table_facts(table: MetricsTable, period: str | None = None) -> Facts:
    """A filing's numbers, with the same fin: ids the daily pack uses. `period` is
    the filing's own period, the date the email and the daily pack show; the
    table's is the tagged data's, which can be a few days off."""
    numbers = _by_id(filing_numbers(table.ticker, table.to_dict()))
    return Facts(table.ticker, numbers, period or table.period)


def _facts(source: Pack | Facts) -> Facts:
    return pack_facts(source) if isinstance(source, Pack) else source


@dataclass(frozen=True)
class Reason:
    """As the model wrote it: text with placeholders, and the ids it cites."""

    text: str
    cites: tuple[str, ...]


@dataclass(frozen=True)
class Rendered:
    """A reason that may be sent: code's sentence, and every id behind it."""

    text: str
    cites: tuple[str, ...]


@dataclass(frozen=True)
class DroppedReason:
    reason: Reason
    why: str


@dataclass
class ReasonCheck:
    kept: list[Rendered] = field(default_factory=list)
    dropped: list[DroppedReason] = field(default_factory=list)


def _pct(v: float) -> str:
    return f"{abs(v) * 100:.2f}%"


def _pts(v: float) -> str:
    return f"{abs(v) * 100:.2f} points"


def _zero(v: float, kind: str) -> bool:
    """True when the value shows as zero, so no direction may be claimed."""
    return round(abs(v) if kind == "usd" else abs(v) * 100, 2) == 0


def _span(n: int) -> str:
    return "the last session" if n == 1 else f"the last {n} sessions"


def _moved(
    subject: str,
    v: float,
    size: str,
    tail: str,
    *,
    kind: str = "percent",
    by: bool = False,
    was: str = "was",
) -> str:
    if _zero(v, kind):
        return f"{subject} {was} unchanged {tail}"
    verb = "rose" if v > 0 else "fell"
    return f"{subject} {verb} {'by ' if by else ''}{size} {tail}"


def _price(n: Number, facts: Facts, key: str) -> str | None:
    t, v = facts.ticker, n.value
    if key == "close":
        return None if facts.day is None else f"{t} closed at {money(v)} on {facts.day.isoformat()}"
    if key == "vol_20d":
        return f"{t}'s annualised volatility over {_span(20)} was {_pct(v)}"
    m = re.fullmatch(r"(ret|bench|excess)_([0-9]+)d", key)
    if not m:
        return None
    span = _span(int(m.group(2)))
    if m.group(1) == "ret":
        return _moved(t, v, _pct(v), f"over {span}")
    if m.group(1) == "bench":
        return _moved(BENCHMARK, v, _pct(v), f"over {span}")
    if _zero(v, "points"):
        return f"{t} matched {BENCHMARK} over {span}"
    verb = "beat" if v > 0 else "trailed"
    return f"{t} {verb} {BENCHMARK} by {_pts(v)} over {span}"


def _metric_label(n: Number, metric: str, fld: str, ticker: str) -> str | None:
    """'ACME's revenue': the company is named, so nothing put before it reads as the subject."""
    suffix = f", {FIELD_LABELS[fld]}"
    if not n.label.endswith(suffix):
        return None
    label = PHRASE_LABELS.get(metric) or n.label[: -len(suffix)]
    return f"{ticker}'s {label[:1].lower()}{label[1:]}"


def _filing(n: Number, facts: Facts, metric: str, fld: str) -> str | None:
    if fld not in FIELD_LABELS:
        return None
    label = _metric_label(n, metric, fld, facts.ticker)
    if label is None:
        return None
    v = n.value
    was = "were" if metric in PLURAL_METRICS else "was"
    latest = f"the quarter ended {facts.period}" if facts.period else "the latest quarter"
    when = {
        "current": latest,
        "prior_quarter": "the prior quarter",
        "year_ago": "the same quarter a year earlier",
    }
    # a balance is held at the quarter's end; a flow happens in it
    at = "at the end of" if metric in STOCK_METRICS else "in"
    base = "from the prior quarter" if fld.startswith("qoq") else "from a year earlier"
    if fld in when:
        if n.kind == "usd":
            if v < 0 and metric in LOSS_METRICS:
                return f"{label} {was} a loss of {money(abs(v))} {at} {when[fld]}"
            shown = money(abs(v))
        else:
            shown = _pct(v)
        if v < 0 and not _zero(v, n.kind):
            shown = f"negative {shown}"
        return f"{label} {was} {shown} {at} {when[fld]}"
    if n.kind == "usd":
        return _moved(label, v, money(abs(v)), base, kind="usd", by=True, was=was)
    size = _pts(v) if n.kind == "points" else _pct(v)
    return _moved(label, v, size, base, kind=n.kind, was=was)


def phrase(number_id: str, source: Pack | Facts) -> str | None:
    """Code's words for one number, or None if the facts have no such number."""
    facts = _facts(source)
    n = facts.number(number_id)
    if n is None or not math.isfinite(n.value):
        return None
    parts = number_id.split(":")
    if len(parts) == 3 and parts[0] == "px" and parts[1] == facts.ticker:
        return _price(n, facts, parts[2])
    if len(parts) == 4 and parts[0] == "fin" and parts[1] == facts.ticker:
        return _filing(n, facts, parts[2], parts[3])
    return None


def figure_in(words: str) -> str | None:
    """The first figure in the model's own words, if any."""
    for c in words:
        if unicodedata.numeric(c, None) is not None or c in MARKS:
            return c
        if unicodedata.category(c) == "Sc":  # any currency sign
            return c
    m = NUMBER_WORD.search(words) or DATE_WORD.search(words)
    return m.group(0) if m else None


def _opens_clause(before: str) -> bool:
    before = before.rstrip()
    return not before or before.endswith(OPENERS) or CONNECTORS.search(before) is not None


def _closes_clause(after: str) -> bool:
    after = after.lstrip()
    return not after or after.startswith(CLOSERS) or JOINERS.match(after) is not None


def placeholders(text: str) -> list[str]:
    return [m.group(1).strip() for m in PLACEHOLDER.finditer(text)]


def fill(text: str, facts: Facts) -> tuple[str, tuple[str, ...]] | str:
    """Code's sentence for text with placeholders, and the number ids in it;
    or why it cannot be sent. Says nothing about what else the text cites."""
    if len(text) > MAX_REASON_CHARS:
        return f"longer than {MAX_REASON_CHARS} characters"
    found_numbers = list(PLACEHOLDER.finditer(text))
    numbers = placeholders(text)
    own = PLACEHOLDER.sub(" ", text)
    if "{" in own or "}" in own:
        return "a curly bracket that is not a placeholder"
    for m in found_numbers:
        if not _opens_clause(text[: m.start()]):
            return f"placeholder {m.group(1).strip()!r} does not open a clause"
        if not _closes_clause(text[m.end() :]):
            return f"placeholder {m.group(1).strip()!r} does not close a clause"
    found = figure_in(own)
    if found is not None:
        return f"a figure typed in its own words ({found!r})"
    phrases: dict[str, str] = {}
    for number_id in numbers:
        p = phrase(number_id, facts)
        if p is None:
            n = facts.number(number_id)
            if n is not None and not math.isfinite(n.value):
                return f"placeholder {number_id!r} has no finite value"
            return f"placeholder {number_id!r} is not a number on the facts sheet"
        phrases[number_id] = p
    out = PLACEHOLDER.sub(lambda m: phrases[m.group(1).strip()], text)
    out = " ".join(out.split())
    out = out[:1].upper() + out[1:]  # phrases name their company, so only the start needs it
    return out, tuple(dict.fromkeys(numbers))


def render(reason: Reason, source: Pack | Facts) -> Rendered | str:
    """The sentence to send, or why the reason is dropped."""
    facts = _facts(source)
    if len(reason.text) > MAX_REASON_CHARS:
        return f"longer than {MAX_REASON_CHARS} characters"
    numbers = placeholders(reason.text)
    own = PLACEHOLDER.sub(" ", reason.text)
    if "{" in own or "}" in own:
        return "a curly bracket that is not a placeholder"
    if not numbers and not reason.cites:
        return "cites nothing"
    known = facts.ids()
    for c in reason.cites:
        if c not in known:
            return f"cites {c!r}, which is not on the facts sheet"
        if facts.number(c) is not None and c not in numbers:
            return f"cites the number {c!r} without showing it as a placeholder"
    got = fill(reason.text, facts)
    if isinstance(got, str):
        return got
    text, shown = got
    return Rendered(text, tuple(dict.fromkeys([*reason.cites, *shown])))


def check_reasons(reasons: list[Reason], source: Pack | Facts) -> ReasonCheck:
    facts = _facts(source)
    result = ReasonCheck()
    seen: set[str] = set()
    for r in reasons:
        got = render(r, facts)
        if isinstance(got, Rendered) and got.text.lower() in seen:
            got = "duplicate reason"
        if isinstance(got, str):
            result.dropped.append(DroppedReason(r, got))
            continue
        seen.add(got.text.lower())
        result.kept.append(got)
    return result
