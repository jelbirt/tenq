"""Rules: Jev's averaged answers and the pack's price move to one call.

config/rules.yaml holds every setting and reads top to bottom: a news part
(direction times materiality times evidence strength), guidance, the filing
trend and recent risk, each from -1 to +1, weighted, and each scaled by how
well Jev's rounds agreed, so an unsure Jev drifts toward Hold rather than
hitting a cut-off. The weighted sum shrinks when the stock has already moved
that way against QQQ (priced in), and the final score picks Buy, Hold, Sell or
Short. Every part's contribution is kept, so the email and the record show why.

Confidence: for Buy, Sell and Short, the size of the score (0 to 1); for Hold,
how close the score sits to 0 within the Hold band (1 at 0, 0 at its edge).

A Jev result with no answers is no call (Opus's call goes out alone); a
missing answer counts as nothing and is noted. Changing the rules file bumps
its version, which every call records.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from tenq.config import REPO_ROOT
from tenq.jev import Answer, QuestionSet, Result
from tenq.pack import Pack
from tenq.prices import SESSIONS_PER_YEAR, WINDOWS

DEFAULT_RULES = REPO_ROOT / "config" / "rules.yaml"
CallName = Literal["Buy", "Hold", "Sell", "Short"]
PARTS = ("news", "guidance", "fundamentals", "risk")
TOP_LEVEL = {"version", "direction", "evidence", "guidance", "weights", "priced_in", "calls"}
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


@dataclass(frozen=True)
class Part:
    name: str
    value: float  # -1 to +1, before weight and certainty
    weight: float
    certainty: float  # 0 to 1: 1 less Jev's spread across rounds
    detail: str

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
    raw: float  # the weighted sum before priced-in
    priced_in: float  # fraction of the raw score removed, 0 to `most`
    parts: list[Part] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in vars(self).items() if k != "parts"}
        out["parts"] = [asdict(p) | {"contribution": p.contribution} for p in self.parts]
        return out

    def explain(self) -> list[str]:
        """Plain lines for the email: each part, the priced-in cut, the call."""
        lines = [
            f"{p.name}: {p.contribution:+.2f} ({p.detail}; weight {p.weight:.2f}, "
            f"certainty {p.certainty:.0%})"
            for p in self.parts
        ]
        if self.priced_in:
            lines.append(f"priced in: score cut by {self.priced_in:.0%}")
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


def load_rules(path: Path = DEFAULT_RULES, questions: QuestionSet | None = None) -> Rules:
    """Validate the rules file, and its labels against `questions` when given."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != TOP_LEVEL:
        raise ValueError(f"rules file must hold exactly {sorted(TOP_LEVEL)}")
    version = raw["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("rules version must be a string")
    weights = _values(raw["weights"], "weights", 0.0)
    if set(weights) != set(PARTS):
        raise ValueError(f"rules weights must be exactly {list(PARTS)}")
    if sum(weights.values()) > 1.0 + 1e-9:
        raise ValueError("rules weights must add up to 1 or less")
    priced, calls = raw["priced_in"], raw["calls"]
    if not isinstance(priced, dict) or set(priced) != {"sessions", "per_move", "most"}:
        raise ValueError("rules priced_in must hold sessions, per_move and most")
    if not isinstance(calls, dict) or set(calls) != {"buy", "sell", "short"}:
        raise ValueError("rules calls must hold buy, sell and short")
    sessions = priced["sessions"]
    if isinstance(sessions, bool) or sessions not in WINDOWS:
        raise ValueError(f"rules priced_in.sessions must be one of {list(WINDOWS)}")
    rules = Rules(
        version=version,
        direction=_values(raw["direction"], "direction", -1.0),
        evidence=_values(raw["evidence"], "evidence", 0.0),
        guidance=_values(raw["guidance"], "guidance", -1.0),
        weights=weights,
        sessions=int(sessions),
        per_move=_number(priced["per_move"], "priced_in.per_move", 0.0, 1.0),
        most=_number(priced["most"], "priced_in.most", 0.0, 1.0),
        buy=_number(calls["buy"], "calls.buy", 0.0, 1.0),
        sell=_number(calls["sell"], "calls.sell", -1.0, 0.0),
        short=_number(calls["short"], "calls.short", -1.0, 0.0),
    )
    if not rules.short < rules.sell < 0.0 < rules.buy:
        raise ValueError("rules calls must order short < sell < 0 < buy")
    if questions is not None:
        check_against(rules, questions)
    return rules


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


def certainty(answer: Answer, levels: int) -> float:
    """1 less the answer's spread across rounds, the spread put on a 0 to 1 scale."""
    spread = answer.spread / (levels - 1) if answer.type == "score" else answer.spread
    return min(1.0, max(0.0, 1.0 - spread))


def expected(answer: Answer, values: dict[str, float]) -> float:
    """The answer's value under Jev's averaged probabilities over the labels."""
    if answer.probabilities:
        return sum(p * values.get(label, 0.0) for label, p in answer.probabilities.items())
    return values.get(str(answer.value), 0.0)


def _missing(name: str, weight: float, keys: list[str]) -> Part:
    return Part(name, 0.0, weight, 0.0, f"no answer for {', '.join(keys)}")


def _parts(pack: Pack, result: Result, rules: Rules, questions: QuestionSet) -> list[Part]:
    a = result.answers
    levels = {q.key: len(q.criteria) for q in questions.questions}
    w = rules.weights
    out: list[Part] = []

    need = [k for k in ("direction", "material", "source") if k not in a]
    if need:
        out.append(_missing("news", w["news"], need))
    else:
        d, m, s = a["direction"], a["material"], a["source"]
        good = expected(d, rules.direction)
        material = float(m.value)
        strength = expected(s, rules.evidence)
        out.append(
            Part(
                "news",
                good * material * strength,
                w["news"],
                min(certainty(x, levels[x.key]) for x in (d, m, s)),
                f"good {good:+.2f}, material {material:.0%}, evidence {strength:.2f}",
            )
        )

    if "guidance" not in a:
        out.append(_missing("guidance", w["guidance"], ["guidance"]))
    else:
        g = a["guidance"]
        out.append(
            Part(
                "guidance",
                expected(g, rules.guidance),
                w["guidance"],
                certainty(g, levels["guidance"]),
                f"most likely {g.value}",
            )
        )

    if not any(n.id.startswith("fin:") for n in pack.numbers):
        out.append(Part("fundamentals", 0.0, 0.0, 0.0, "no 10-Q/10-K numbers"))
    elif "fundamentals" not in a:
        out.append(_missing("fundamentals", w["fundamentals"], ["fundamentals"]))
    else:
        f = a["fundamentals"]
        top = levels["fundamentals"] - 1
        level = min(float(top), max(0.0, float(f.value)))
        out.append(
            Part(
                "fundamentals",
                2.0 * level / top - 1.0,
                w["fundamentals"],
                certainty(f, levels["fundamentals"]),
                f"trend level {level:.1f} of {top}",
            )
        )

    if "risk" not in a:
        out.append(_missing("risk", w["risk"], ["risk"]))
    else:
        r = a["risk"]
        risk = float(r.value)
        out.append(
            Part("risk", -risk, w["risk"], certainty(r, levels["risk"]), f"risk event {risk:.0%}")
        )
    return out


def priced_in(pack: Pack, raw: float, rules: Rules) -> tuple[float, str | None]:
    """The fraction of `raw` already priced in, and a note when it can't be told."""
    t = pack.ticker
    excess = pack.number(f"px:{t}:excess_{rules.sessions}d")
    vol = pack.number(f"px:{t}:vol_20d")
    if excess is None or vol is None or not vol.value > 0:
        return 0.0, "no price move to check whether the news is priced in"
    usual = vol.value * math.sqrt(rules.sessions / SESSIONS_PER_YEAR)
    moves = excess.value / usual
    if raw * moves <= 0:
        return 0.0, None
    return min(rules.most, rules.per_move * abs(moves)), None


def name_call(score: float, rules: Rules) -> tuple[CallName, float]:
    """The call for a score and its confidence (see the module docstring)."""
    if score >= rules.buy:
        return "Buy", min(1.0, abs(score))
    if score <= rules.short:
        return "Short", min(1.0, abs(score))
    if score <= rules.sell:
        return "Sell", min(1.0, abs(score))
    edge = rules.buy if score >= 0 else -rules.sell
    return "Hold", max(0.0, 1.0 - abs(score) / edge)


def decide(pack: Pack, result: Result, rules: Rules, questions: QuestionSet) -> Call | None:
    """Jev's call on one pack, or None when Jev gave no answers."""
    if not result.answers:
        return None
    parts = _parts(pack, result, rules, questions)
    raw = sum(p.contribution for p in parts)
    cut, note = priced_in(pack, raw, rules)
    score = min(1.0, max(-1.0, raw * (1.0 - cut)))
    call, confidence = name_call(score, rules)
    notes = [note] if note else []
    notes += [
        f"{k} answer unbalanced ({ans.repeats} rounds)"
        for k, ans in result.answers.items()
        if k in QUESTION_TYPES and not ans.balanced
    ]
    if result.version != questions.version:
        notes.append(f"answers are {result.version}, questions are {questions.version}")
    return Call(
        pack.ticker,
        pack.day.isoformat(),
        call,
        score,
        confidence,
        rules.version,
        raw,
        cut,
        parts,
        notes,
    )
