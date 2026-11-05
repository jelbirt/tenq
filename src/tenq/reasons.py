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
  ("Subscription {...}") or negate it ("not that {...}");
- the words the model wrote itself hold no figure at all: no digit or other
  numeric character, no "%" or currency sign, no number word ("doubled",
  "five", "a third", "millions"), no quarter, month or weekday name, and no
  stray curly bracket.

So a wrong value, sign, direction, period, unit or subject cannot reach an
email: code wrote all of them, and every phrase names its own company. What
still can is a weak or beside-the-point reason, or words that argue with the
filled-in phrase ("despite the rally, ACME fell 21.47%"); a reader sees
those, next to the true number. The word "one" and "first" or "second" on
their own are allowed, since they are far more often plain English.

This replaces reading the model's English to judge its numbers, which no set
of rules can do reliably. The quarterly brief still uses tenq.checker.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from tenq.pack import FIELD_LABELS, Number, Pack, money
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
    r"|\b(?:by|down|up|off|about|nearly|roughly|almost|over|under)\s+a\s+quarter\b"
    r"|\ba\s+quarter\s+of\b"
    r"|\b(?:one|two|three)-(?:third|quarter|half|fifth)s?\b",
    re.IGNORECASE,
)
# names that date a claim; capitalised only, so the verbs "march" and "may" pass
DATE_WORD = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Monday(?!\.com)|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday|II|III|IV|VI|VII|VIII|IX|XI|XII|XX)\b"
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
MAX_REASON_CHARS = 600  # placeholders are long ids; a longer reason is dropped, never cut
LOSS_METRICS = {"operating_income", "net_income"}  # a negative level reads "a loss of"


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
    subject: str, v: float, size: str, tail: str, *, kind: str = "percent", by: bool = False
) -> str:
    if _zero(v, kind):
        return f"{subject} was unchanged {tail}"
    verb = "rose" if v > 0 else "fell"
    return f"{subject} {verb} {'by ' if by else ''}{size} {tail}"


def _price(n: Number, pack: Pack, key: str) -> str | None:
    t, v = pack.ticker, n.value
    if key == "close":
        return f"{t} closed at {money(v)} on {pack.day.isoformat()}"
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


def _metric_label(n: Number, fld: str, ticker: str) -> str | None:
    """'ACME's revenue': the company is named, so nothing put before it reads as the subject."""
    suffix = f", {FIELD_LABELS[fld]}"
    if not n.label.endswith(suffix):
        return None
    label = n.label[: -len(suffix)]
    return f"{ticker}'s {label[:1].lower()}{label[1:]}"


def _filing(n: Number, pack: Pack, metric: str, fld: str) -> str | None:
    if fld not in FIELD_LABELS:
        return None
    label = _metric_label(n, fld, pack.ticker)
    if label is None:
        return None
    v = n.value
    latest = f"the quarter ended {pack.filing.period}" if pack.filing else "the latest quarter"
    when = {
        "current": latest,
        "prior_quarter": "the prior quarter",
        "year_ago": "the same quarter a year earlier",
    }
    base = "from the prior quarter" if fld.startswith("qoq") else "from a year earlier"
    if fld in when:
        if n.kind == "usd":
            if v < 0 and metric in LOSS_METRICS:
                return f"{label} was a loss of {money(abs(v))} in {when[fld]}"
            shown = money(abs(v))
        else:
            shown = _pct(v)
        if v < 0 and not _zero(v, n.kind):
            shown = f"negative {shown}"
        return f"{label} was {shown} in {when[fld]}"
    if n.kind == "usd":
        return _moved(label, v, money(abs(v)), base, kind="usd", by=True)
    size = _pts(v) if n.kind == "points" else _pct(v)
    return _moved(label, v, size, base, kind=n.kind)


def phrase(number_id: str, pack: Pack) -> str | None:
    """Code's words for one pack number, or None if the pack has no such number."""
    n = pack.number(number_id)
    if n is None:
        return None
    parts = number_id.split(":")
    if len(parts) == 3 and parts[0] == "px" and parts[1] == pack.ticker:
        return _price(n, pack, parts[2])
    if len(parts) == 4 and parts[0] == "fin" and parts[1] == pack.ticker:
        return _filing(n, pack, parts[2], parts[3])
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


def render(reason: Reason, pack: Pack) -> Rendered | str:
    """The sentence to send, or why the reason is dropped."""
    if len(reason.text) > MAX_REASON_CHARS:
        return f"longer than {MAX_REASON_CHARS} characters"
    known = pack.ids()
    found_numbers = list(PLACEHOLDER.finditer(reason.text))
    numbers = [m.group(1).strip() for m in found_numbers]
    own = PLACEHOLDER.sub(" ", reason.text)
    if "{" in own or "}" in own:
        return "a curly bracket that is not a placeholder"
    if not numbers and not reason.cites:
        return "cites nothing"
    for c in reason.cites:
        if c not in known:
            return f"cites {c!r}, which is not on the facts sheet"
        if pack.number(c) is not None and c not in numbers:
            return f"cites the number {c!r} without showing it as a placeholder"
    for m in found_numbers:
        if not _opens_clause(reason.text[: m.start()]):
            return f"placeholder {m.group(1).strip()!r} does not open a clause"
    found = figure_in(own)
    if found is not None:
        return f"a figure typed in its own words ({found!r})"
    phrases: dict[str, str] = {}
    for number_id in numbers:
        p = phrase(number_id, pack)
        if p is None:
            return f"placeholder {number_id!r} is not a number on the facts sheet"
        phrases[number_id] = p
    text = PLACEHOLDER.sub(lambda m: phrases[m.group(1).strip()], reason.text)
    text = " ".join(text.split())
    text = text[:1].upper() + text[1:]  # phrases name their company, so only the start needs it
    return Rendered(text, tuple(dict.fromkeys([*reason.cites, *numbers])))


def check_reasons(reasons: list[Reason], pack: Pack) -> ReasonCheck:
    result = ReasonCheck()
    seen: set[str] = set()
    for r in reasons:
        got = render(r, pack)
        if isinstance(got, Rendered) and got.text.lower() in seen:
            got = "duplicate reason"
        if isinstance(got, str):
            result.dropped.append(DroppedReason(r, got))
            continue
        seen.add(got.text.lower())
        result.kept.append(got)
    return result
