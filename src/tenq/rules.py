"""Rules: Jev's averaged answers and the pack's price move to one call.

config/rules.yaml holds every setting and reads top to bottom: a news part
(direction times materiality times evidence strength), guidance, the filing
trend and recent risk, each from -1 to +1 and weighted. A choice's doubt is
counted once, in its expected value under Jev's averaged probabilities; a
yes/no or a score is also scaled by 1 less its spread across rounds; and every
part is scaled by the share of its planned rounds that came back, so an unsure
or half-answered Jev drifts toward Hold rather than hitting a cut-off. A
score's spread leaves out the reversal lean the rounds measured. Risk counts
only the bad news the news part did not already score (after its certainty).
The weighted sum shrinks with the market's reaction, whichever way the stock
moved against QQQ (with the signal: priced in; against it: doubted), and the
final score picks Buy, Hold, Sell or Short. Every part's
contribution is kept, so the email and the record show why.

Confidence, 0 to 1: for Buy, Sell and Short, the score as a share of the most
the included parts could reach that way; for Hold, how close the score sits to
0 within the Hold band (1 at 0, 0 at its edge) times the evidence coverage:
the weighted certainty that came in, each part also scaled by how decisive its
answers were (how far from a coin flip, and for a choice how many rounds picked
its winner). A Jev that knew nothing, flat answers every round, is a Hold with
no confidence, not full confidence.

A Jev result with no answers is no call (Opus's call goes out alone); a
missing answer, one of the wrong type, or an unreadable one (a NaN, a wrong
field type, round counts out of range, a key that is not its own) counts as
nothing and is noted, and anything else unexpected makes no call, so decide()
never raises on what Jev or a stored record sends.
Every call records the rules version and the file's sha256; a test pins one
hash per version, so an edit without a version bump fails the green bar.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from tenq.config import REPO_ROOT
from tenq.jev import Answer, QuestionSet, Result
from tenq.pack import Pack
from tenq.prices import SESSIONS_PER_YEAR, WINDOWS

log = logging.getLogger(__name__)

DEFAULT_RULES = REPO_ROOT / "config" / "rules.yaml"
MAX_ROUNDS = 64  # far above any planned round count; more is a corrupt record
CallName = Literal["Buy", "Hold", "Sell", "Short"]
PARTS = ("news", "guidance", "fundamentals", "risk")
TOP_LEVEL = {"version", "direction", "evidence", "guidance", "weights", "reaction", "calls"}
# which question each label map belongs to, and the type each used question must have
LABEL_MAPS = {"direction": "direction", "evidence": "source", "guidance": "guidance"}
QUESTION_TYPES = {
    "material": "noul",
    "direction": "choice",
    "source": "choice",
    "guidance": "choice",
    "fundamentals": "score",
    "risk": "noul",
}


@dataclass(frozen=True)
class Rules:
    version: str
    direction: dict[str, float]
    evidence: dict[str, float]
    guidance: dict[str, float]
    weights: dict[str, float]
    sessions: int
    per_move: float
    most: float
    buy: float
    sell: float
    short: float
    sha256: str  # of the file, recorded on every call


@dataclass(frozen=True)
class Part:
    name: str
    value: float  # -1 to +1, before weight and certainty
    weight: float
    certainty: float  # 0 to 1: how far to trust the answers (see certainty())
    detail: str
    # how far the answers sit from a coin flip, 0 to 1 (see decisive()). Not in
    # the contribution (doubt is already in the value and certainty), only in
    # coverage, so flat answers cannot make a confident Hold.
    decisive: float = 1.0

    @property
    def contribution(self) -> float:
        return self.value * self.weight * self.certainty


@dataclass
class Call:
    ticker: str
    day: str
    call: CallName
    score: float  # -1 to +1
    confidence: float  # 0 to 1, see the module docstring
    version: str
    raw: float  # the weighted sum before the market's reaction
    reaction: float  # fraction of the raw score removed, 0 to `most`
    parts: list[Part] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    coverage: float = 0.0  # weighted certainty of the included parts, 0 to 1
    rules_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in vars(self).items() if k != "parts"}
        out["parts"] = [asdict(p) | {"contribution": p.contribution} for p in self.parts]
        return out

    def explain(self) -> list[str]:
        """Plain lines for the email: each part, the market-reaction cut, the call."""
        lines = [
            f"{p.name}: {p.contribution:+.2f} ({p.detail}; weight {p.weight:.2f}, "
            f"certainty {p.certainty:.0%})"
            for p in self.parts
        ]
        if self.reaction:
            lines.append(f"market reaction: score cut by {self.reaction:.0%}")
        lines.append(f"score {self.score:+.2f}: {self.call} (confidence {self.confidence:.0%})")
        return lines


def _number(raw: Any, where: str, low: float, high: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        raise ValueError(f"rules {where}: must be a number")
    if not low <= raw <= high:
        raise ValueError(f"rules {where}: must be from {low} to {high}")
    return float(raw)


def _values(raw: Any, where: str, low: float) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw or not all(isinstance(k, str) for k in raw):
        raise ValueError(f"rules {where}: must map labels to numbers")
    return {k: _number(v, f"{where}.{k}", low, 1.0) for k, v in raw.items()}


class _StrictLoader(yaml.SafeLoader):
    """A YAML loader that refuses a key given twice instead of keeping the last."""


def _no_repeats(loader: _StrictLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    keys = [loader.construct_object(k, deep=True) for k, _ in node.value]
    repeated = sorted({str(k) for k in keys if keys.count(k) > 1})
    if repeated:
        raise ValueError(f"rules file repeats {repeated}")
    return dict(loader.construct_mapping(node, deep=True))


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_repeats)


def load_rules(questions: QuestionSet, path: Path = DEFAULT_RULES) -> Rules:
    """Validate the rules file, and its questions and labels against `questions`."""
    data = path.read_bytes()
    try:
        raw = yaml.load(data.decode("utf-8"), Loader=_StrictLoader)  # a SafeLoader subclass
    except (yaml.YAMLError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError(f"rules file unreadable ({type(exc).__name__})") from exc
    if not isinstance(raw, dict) or set(raw) != TOP_LEVEL:
        raise ValueError(f"rules file must hold exactly {sorted(TOP_LEVEL)}")
    version = raw["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("rules version must be a string")
    weights = _values(raw["weights"], "weights", 0.0)
    if set(weights) != set(PARTS):
        raise ValueError(f"rules weights must be exactly {list(PARTS)}")
    if not 0.0 < sum(weights.values()) <= 1.0 + 1e-9:
        raise ValueError("rules weights must add up to more than 0 and at most 1")
    moved, calls = raw["reaction"], raw["calls"]
    if not isinstance(moved, dict) or set(moved) != {"sessions", "per_move", "most"}:
        raise ValueError("rules reaction must hold sessions, per_move and most")
    if not isinstance(calls, dict) or set(calls) != {"buy", "sell", "short"}:
        raise ValueError("rules calls must hold buy, sell and short")
    sessions = moved["sessions"]
    if isinstance(sessions, bool) or sessions not in WINDOWS:
        raise ValueError(f"rules reaction.sessions must be one of {list(WINDOWS)}")
    rules = Rules(
        version=version,
        direction=_values(raw["direction"], "direction", -1.0),
        evidence=_values(raw["evidence"], "evidence", 0.0),
        guidance=_values(raw["guidance"], "guidance", -1.0),
        weights=weights,
        sessions=int(sessions),
        per_move=_number(moved["per_move"], "reaction.per_move", 0.0, 1.0),
        most=_number(moved["most"], "reaction.most", 0.0, 1.0),
        buy=_number(calls["buy"], "calls.buy", 0.0, 1.0),
        sell=_number(calls["sell"], "calls.sell", -1.0, 0.0),
        short=_number(calls["short"], "calls.short", -1.0, 0.0),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    if not rules.short < rules.sell < 0.0 < rules.buy:
        raise ValueError("rules calls must order short < sell < 0 < buy")
    up, down = _reach(weights, with_fundamentals=True)
    if rules.buy > up + 1e-9 or rules.short < -down - 1e-9:
        raise ValueError("rules calls must be reachable: buy at most the parts that can")
    check_against(rules, questions)
    return rules


def _reach(weights: dict[str, float], *, with_fundamentals: bool) -> tuple[float, float]:
    """The highest and the lowest (as a size) score the parts can reach.

    Down, news and risk never both count in full: fully bad news switches the
    risk part off, so the lower bound takes the larger of the two.
    """
    shared = weights["guidance"] + (weights["fundamentals"] if with_fundamentals else 0.0)
    return shared + weights["news"], shared + max(weights["news"], weights["risk"])


def check_against(rules: Rules, questions: QuestionSet) -> None:
    """Every question the rules read exists with its type, and every label matches."""
    by_key = {q.key: q for q in questions.questions}
    for key, kind in QUESTION_TYPES.items():
        q = by_key.get(key)
        if q is None or q.type != kind:
            raise ValueError(f"rules need a {kind} question {key!r} in {questions.version}")
    for name, key in LABEL_MAPS.items():
        labels = set(by_key[key].labels())
        if set(getattr(rules, name)) != labels:
            raise ValueError(f"rules {name} labels must be exactly {sorted(labels)}")


def certainty(answer: Answer, levels: int, rounds: int) -> float:
    """How far to trust an answer, 0 to 1.

    Every type: the share of its planned rounds that came back, since a lone
    round neither cancels the position lean nor shows any agreement. A choice
    adds nothing more: its doubt is already in its expected value, and its
    spread moves with the rotation's lean on purpose. A yes/no or a score is
    also scaled by 1 less its spread across rounds, a score's on a 0 to 1 scale
    and less the reversal lean (reversed rounds sit about twice the measured
    lean from file-order ones, which is not doubt).
    """
    done = min(1.0, answer.repeats / rounds) if rounds > 0 else 0.0
    if answer.type == "choice":
        return done
    spread = answer.spread
    if answer.type == "score":
        lift = answer.first_lift
        if answer.balanced and lift is not None and _finite(lift):
            spread = max(0.0, spread - 2.0 * abs(float(lift)))
        spread /= levels - 1
    return min(1.0, max(0.0, 1.0 - spread)) * done


def _finite(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def usable(answer: Any, key: str) -> bool:
    """An answer to `key` whose fields the rules can read; anything else is missing."""
    if not isinstance(answer, Answer) or answer.key != key:
        return False
    if not _finite(answer.spread) or answer.spread < 0:
        return False
    if any(isinstance(n, bool) or not isinstance(n, int) for n in (answer.repeats, answer.agreed)):
        return False
    if not 0 < answer.repeats <= MAX_ROUNDS or not 0 <= answer.agreed <= answer.repeats:
        return False
    probs = answer.probabilities
    if not isinstance(probs, dict) or not all(
        isinstance(k, str) and _finite(v) and 0.0 <= v <= 1.0 for k, v in probs.items()
    ):
        return False
    if answer.type == "choice":
        return isinstance(answer.value, str)
    return _finite(answer.value)


def expected(answer: Answer, values: dict[str, float]) -> float:
    """The answer's value under Jev's averaged probabilities, made to add up to 1.

    Jev's probabilities are each checked to lie from 0 to 1 but need not add up
    to 1; without the division, 1.0 on every label would count several times.
    """
    total = sum(answer.probabilities.values())
    if total > 0:
        return sum(p * values.get(k, 0.0) for k, p in answer.probabilities.items()) / total
    return values.get(str(answer.value), 0.0)


def _unit(x: float) -> float:
    return min(1.0, max(-1.0, x))


def _agreement(answer: Answer) -> float:
    return min(1.0, answer.agreed / answer.repeats) if answer.repeats > 0 else 0.0


def decisive(answer: Answer, labels: list[str]) -> float:
    """How far an answer sits from a coin flip, 0 (flat) to 1 (all on one side).

    The averaged probabilities' distance from an even spread over the labels,
    scaled so one label at 1.0 is 1; a yes/no is |2p - 1|, the same measure on
    two sides. A choice is also scaled by its share of agreeing rounds. With
    no probabilities to read, a choice or score counts as decisive and its
    spread and agreement carry the doubt.
    """
    if answer.type == "noul":
        return min(1.0, abs(2.0 * float(answer.value) - 1.0))
    probs = answer.probabilities
    total = sum(probs.values())
    n = len(labels)
    d = 1.0
    if total > 0 and n > 1:
        even = 1.0 / n
        gap = sum(abs(probs.get(label, 0.0) / total - even) for label in labels) / 2.0
        d = min(1.0, gap / (1.0 - even))
    if answer.type == "choice":
        d *= _agreement(answer)
    return d


def _missing(name: str, weight: float, keys: list[str]) -> Part:
    return Part(name, 0.0, weight, 0.0, f"no answer for {', '.join(keys)}")


def _parts(pack: Pack, result: Result, rules: Rules, questions: QuestionSet) -> list[Part]:
    a = {
        k: v for k, v in result.answers.items() if usable(v, k) and QUESTION_TYPES.get(k) == v.type
    }
    by_key = {q.key: q for q in questions.questions}

    def sure(x: Answer) -> float:
        q = by_key[x.key]
        return certainty(x, len(q.labels()), q.rounds())

    def firm(*xs: Answer) -> float:
        return min(decisive(x, by_key[x.key].labels()) for x in xs)

    w = rules.weights
    news_bad = 0.0  # how much bad news the news part scored, 0 to 1
    out: list[Part] = []

    need = [k for k in ("direction", "material", "source") if k not in a]
    if need:
        out.append(_missing("news", w["news"], need))
    else:
        d, m, s = a["direction"], a["material"], a["source"]
        good = expected(d, rules.direction)
        material = min(1.0, max(0.0, float(m.value)))
        strength = expected(s, rules.evidence)
        news_sure = min(sure(x) for x in (d, m, s))
        out.append(
            Part(
                "news",
                _unit(good * material * strength),
                w["news"],
                news_sure,
                f"good {good:+.2f}, material {material:.0%}, evidence {strength:.2f}",
                firm(d, m, s),
            )
        )
        # only the bad news the news part actually scored, after its certainty
        bad = expected(d, {k: max(0.0, -v) for k, v in rules.direction.items()})
        news_bad = min(1.0, max(0.0, bad * material * strength * news_sure))

    if "guidance" not in a:
        out.append(_missing("guidance", w["guidance"], ["guidance"]))
    else:
        g = a["guidance"]
        out.append(
            Part(
                "guidance",
                _unit(expected(g, rules.guidance)),
                w["guidance"],
                sure(g),
                f"most likely {g.value}",
                firm(g),
            )
        )

    if not any(n.id.startswith("fin:") for n in pack.numbers):
        out.append(Part("fundamentals", 0.0, 0.0, 0.0, "no 10-Q/10-K numbers"))
    elif "fundamentals" not in a:
        out.append(_missing("fundamentals", w["fundamentals"], ["fundamentals"]))
    else:
        f = a["fundamentals"]
        top = len(by_key["fundamentals"].labels()) - 1
        level = min(float(top), max(0.0, float(f.value)))
        out.append(
            Part(
                "fundamentals",
                2.0 * level / top - 1.0,
                w["fundamentals"],
                sure(f),
                f"trend level {level:.1f} of {top}",
                firm(f),
            )
        )

    if "risk" not in a:
        out.append(_missing("risk", w["risk"], ["risk"]))
    else:
        r = a["risk"]
        risk = min(1.0, max(0.0, float(r.value)))
        # the bad news the news part already scored is not counted again here
        new = risk * (1.0 - news_bad)
        out.append(
            Part(
                "risk",
                -new,
                w["risk"],
                sure(r),
                f"risk event {risk:.0%}, {1.0 - news_bad:.0%} of it not already in news",
                firm(r),
            )
        )
    return out


def reaction(pack: Pack, raw: float, rules: Rules) -> tuple[float, str | None]:
    """The fraction of `raw` the market's reaction removes, and a note when it can't be told.

    A move with the signal (priced in) and a move against it (doubted) cut the
    same way: by the size of the move, never past `most`, so never past Hold.
    """
    t = pack.ticker
    excess = pack.number(f"px:{t}:excess_{rules.sessions}d")
    vol = pack.number(f"px:{t}:vol_20d")
    if (
        excess is None
        or vol is None
        or not _finite(excess.value)
        or not _finite(vol.value)
        or not vol.value > 0
    ):
        return 0.0, "no price move to weigh the market's reaction"
    if raw == 0:
        return 0.0, None
    usual = vol.value * math.sqrt(rules.sessions / SESSIONS_PER_YEAR)
    moves = excess.value / usual
    if not math.isfinite(moves):
        return 0.0, "price move too large to read against its volatility"
    return min(rules.most, rules.per_move * abs(moves)), None


def name_call(score: float, rules: Rules) -> CallName:
    """The call a score falls in; each line belongs to the call beyond it."""
    if score >= rules.buy:
        return "Buy"
    if score <= rules.short:
        return "Short"
    if score <= rules.sell:
        return "Sell"
    return "Hold"


def coverage(parts: list[Part]) -> float:
    """The weighted certainty times decisiveness of the included parts, 0 to 1."""
    total = sum(p.weight for p in parts)
    sure = sum(p.weight * p.certainty * p.decisive for p in parts)
    return sure / total if total > 0 else 0.0


def confidence(score: float, call: CallName, rules: Rules, parts: list[Part]) -> float:
    """How strongly the evidence backs the call (see the module docstring)."""
    included = any(p.name == "fundamentals" and p.weight > 0 for p in parts)
    up, down = _reach(rules.weights, with_fundamentals=included)
    if call == "Buy":
        return min(1.0, score / up)
    if call in ("Sell", "Short"):
        return min(1.0, -score / down)
    edge = rules.buy if score >= 0 else -rules.sell
    return max(0.0, 1.0 - abs(score) / edge) * coverage(parts)


def decide(pack: Pack, result: Result, rules: Rules, questions: QuestionSet) -> Call | None:
    """Jev's call on one pack, or None when Jev gave no answers or could not be read."""
    try:
        return _decide(pack, result, rules, questions)
    except Exception as exc:  # a backstop: no call beats a crash or a wrong call
        log.warning("no Jev call for %s: %s", pack.ticker, type(exc).__name__)
        return None


def _decide(pack: Pack, result: Result, rules: Rules, questions: QuestionSet) -> Call | None:
    if not result.answers:
        return None
    parts = _parts(pack, result, rules, questions)
    raw = sum(p.contribution for p in parts)
    cut, note = reaction(pack, raw, rules)
    score = min(1.0, max(-1.0, raw * (1.0 - cut)))
    call = name_call(score, rules)
    notes = [note] if note else []
    for k, ans in result.answers.items():
        if k not in QUESTION_TYPES:
            continue
        if not usable(ans, k):
            notes.append(f"{k} answer unreadable")
        elif ans.type != QUESTION_TYPES[k]:
            notes.append(f"{k} answer is a {ans.type}, the rules read a {QUESTION_TYPES[k]}")
        elif not ans.balanced:
            notes.append(f"{k} answer unbalanced ({ans.repeats} rounds)")
    if result.version != questions.version:
        notes.append(f"answers are {result.version}, questions are {questions.version}")
    return Call(
        pack.ticker,
        pack.day.isoformat(),
        call,
        score,
        confidence(score, call, rules, parts),
        rules.version,
        raw,
        cut,
        parts,
        notes,
        coverage=coverage(parts),
        rules_sha=rules.sha256,
    )
