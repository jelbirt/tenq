"""Rules: Jev's answers to a call, worked by hand against the committed files."""

from __future__ import annotations

import itertools
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from tenq.jev import Answer, QuestionSet, Result, load_questions
from tenq.pack import Number, Pack
from tenq.rules import (
    DEFAULT_RULES,
    Rules,
    certainty,
    decide,
    load_rules,
    name_call,
)

QUESTIONS = load_questions()
RULES = load_rules(questions=QUESTIONS)
FIN = Number("fin:ACME:revenue:yoy_pct", "Revenue change from a year ago", 0.2, "percent")


def _choice(key: str, probs: dict[str, float], spread: float = 0.0) -> Answer:
    (q,) = [q for q in QUESTIONS.questions if q.key == key]
    full = {label: probs.get(label, 0.0) for label in q.labels()}
    top = max(full, key=lambda k: full[k])
    return Answer(key, "choice", top, full, spread, len(full), balanced=True)


def _noul(key: str, p: float, spread: float = 0.0) -> Answer:
    return Answer(key, "noul", p, {}, spread, 3, balanced=True)


def _score(level: float, spread: float = 0.0) -> Answer:
    return Answer("fundamentals", "score", level, {}, spread, 4, balanced=True)


def _good(**over: Answer) -> Result:
    answers = {
        "material": _noul("material", 0.8),
        "direction": _choice("direction", {"good": 0.9, "mixed": 0.1}),
        "source": _choice("source", {"reported": 1.0}),
        "guidance": _choice("guidance", {"raised": 1.0}),
        "fundamentals": _score(3.0),
        "risk": _noul("risk", 0.1),
    }
    answers.update(over)
    return Result(QUESTIONS.version, "jev-latest", answers=answers)


def _pack(*numbers: Number, fin: bool = True) -> Pack:
    day = date(2026, 11, 3)
    as_of = datetime(2026, 11, 4, 1, 0, tzinfo=UTC)
    return Pack("ACME", "Acme", day, as_of, numbers=[*numbers, *([FIN] if fin else [])])


def _move(excess_5d: float, vol: float = 0.40) -> tuple[Number, Number]:
    return (
        Number("px:ACME:excess_5d", "ACME return less QQQ over 5 sessions", excess_5d, "points"),
        Number("px:ACME:vol_20d", "Volatility, annualised, over 20 sessions", vol, "percent"),
    )


USUAL_5D = 0.40 * math.sqrt(5 / 252)  # one usual 5-session move at 40% volatility


def test_the_committed_rules_fit_the_committed_questions() -> None:
    assert RULES.version == "rules-v1"
    assert sum(RULES.weights.values()) == pytest.approx(1.0)


def test_good_news_is_a_buy_worked_by_hand() -> None:
    call = decide(_pack(), _good(), RULES, QUESTIONS)
    assert call is not None
    parts = {p.name: p.contribution for p in call.parts}
    # news: good 0.9 x material 0.8 x evidence 1.0 = 0.72, weight 0.45
    assert parts["news"] == pytest.approx(0.324)
    assert parts["guidance"] == pytest.approx(0.25)  # raised, weight 0.25
    assert parts["fundamentals"] == pytest.approx(0.075)  # level 3 of 4 is +0.5
    assert parts["risk"] == pytest.approx(-0.015)  # 10% risk, weight 0.15
    assert call.raw == pytest.approx(0.634) and call.score == pytest.approx(0.634)
    assert call.call == "Buy" and call.confidence == pytest.approx(0.634)
    assert call.version == "rules-v1" and call.day == "2026-11-03"
    assert call.notes == ["no price move to check whether the news is priced in"]


def test_bad_news_with_a_risk_event_is_a_short() -> None:
    result = _good(
        direction=_choice("direction", {"bad": 1.0}),
        guidance=_choice("guidance", {"lowered": 1.0}),
        fundamentals=_score(1.0),
        risk=_noul("risk", 0.9),
    )
    call = decide(_pack(), result, RULES, QUESTIONS)
    assert call is not None
    # -0.36 - 0.25 - 0.075 - 0.135
    assert call.score == pytest.approx(-0.82) and call.call == "Short"


def test_mildly_bad_news_is_a_sell_and_quiet_news_a_hold() -> None:
    mild = _good(
        direction=_choice("direction", {"bad": 0.8, "neutral": 0.2}),
        source=_choice("source", {"reporting": 1.0}),
        guidance=_choice("guidance", {"none": 1.0}),
        fundamentals=_score(2.0),
    )
    call = decide(_pack(), mild, RULES, QUESTIONS)
    assert call is not None
    # news -0.8 x 0.8 x 0.6 x 0.45 = -0.1728; risk -0.015
    assert call.score == pytest.approx(-0.1878) and call.call == "Hold"
    worse = _good(
        direction=_choice("direction", {"bad": 1.0}),
        source=_choice("source", {"reported": 1.0}),
        guidance=_choice("guidance", {"none": 1.0}),
        fundamentals=_score(2.0),
    )
    call = decide(_pack(), worse, RULES, QUESTIONS)
    assert call is not None
    assert call.score == pytest.approx(-0.375) and call.call == "Sell"


def test_an_unsure_jev_drifts_toward_hold_without_a_cut_off() -> None:
    scores = []
    for spread in (0.0, 0.3, 0.6, 0.9):
        result = _good(
            direction=_choice("direction", {"good": 0.9, "mixed": 0.1}, spread),
            guidance=_choice("guidance", {"raised": 1.0}, spread),
        )
        call = decide(_pack(), result, RULES, QUESTIONS)
        assert call is not None
        scores.append(call.score)
    assert scores == sorted(scores, reverse=True)
    steps = [a - b for a, b in itertools.pairwise(scores)]
    assert steps == pytest.approx([steps[0]] * 3)  # a straight line, no jump
    assert scores[-1] < RULES.buy  # unsure enough, the same news is a Hold


def test_certainty_reads_a_score_spread_on_its_own_scale() -> None:
    assert certainty(_score(2.0, spread=2.0), 5) == pytest.approx(0.5)
    assert certainty(_noul("risk", 0.5, spread=0.3), 2) == pytest.approx(0.7)
    assert certainty(_noul("risk", 0.5, spread=1.5), 2) == 0.0


def test_a_missing_answer_counts_as_nothing_and_is_named() -> None:
    result = _good()
    del result.answers["source"], result.answers["guidance"]
    call = decide(_pack(), result, RULES, QUESTIONS)
    assert call is not None
    parts = {p.name: p for p in call.parts}
    assert parts["news"].contribution == 0.0 and parts["news"].detail == "no answer for source"
    assert parts["guidance"].detail == "no answer for guidance"
    assert call.score == pytest.approx(0.06) and call.call == "Hold"


def test_no_filing_numbers_leaves_fundamentals_out() -> None:
    call = decide(_pack(fin=False), _good(), RULES, QUESTIONS)
    assert call is not None
    (part,) = [p for p in call.parts if p.name == "fundamentals"]
    assert part.weight == 0.0 and part.detail == "no 10-Q/10-K numbers"
    assert call.score == pytest.approx(0.559)


def test_no_answers_is_no_call() -> None:
    assert decide(_pack(), Result(QUESTIONS.version, "jev-latest"), RULES, QUESTIONS) is None


@pytest.mark.parametrize(
    ("moves", "cut"),
    [
        (2.0, 0.5),  # already beat QQQ by two usual moves: half priced in
        (1.0, 0.25),
        (10.0, 0.75),  # never more than `most`
        (-2.0, 0.0),  # moved against good news: nothing priced in
        (0.0, 0.0),
    ],
)
def test_a_move_already_made_shrinks_the_score(moves: float, cut: float) -> None:
    call = decide(_pack(*_move(moves * USUAL_5D)), _good(), RULES, QUESTIONS)
    assert call is not None
    assert call.priced_in == pytest.approx(cut)
    assert call.score == pytest.approx(0.634 * (1 - cut))
    assert call.notes == []


def test_bad_news_after_a_fall_is_partly_priced_in() -> None:
    result = _good(direction=_choice("direction", {"bad": 1.0}))
    call = decide(_pack(*_move(-2.0 * USUAL_5D)), result, RULES, QUESTIONS)
    assert call is not None and call.raw < 0
    assert call.priced_in == pytest.approx(0.5)


def test_zero_volatility_cannot_price_anything_in() -> None:
    call = decide(_pack(*_move(0.05, vol=0.0)), _good(), RULES, QUESTIONS)
    assert call is not None and call.priced_in == 0.0
    assert call.notes == ["no price move to check whether the news is priced in"]


@pytest.mark.parametrize(
    ("score", "name", "confidence"),
    [
        (0.20, "Buy", 0.20),  # at the line is the call
        (0.19, "Hold", 0.05),
        (0.0, "Hold", 1.0),
        (-0.10, "Hold", 0.5),
        (-0.20, "Sell", 0.20),
        (-0.45, "Short", 0.45),
        (-1.0, "Short", 1.0),
    ],
)
def test_each_score_names_one_call(score: float, name: str, confidence: float) -> None:
    got, conf = name_call(score, RULES)
    assert got == name and conf == pytest.approx(confidence)


def test_an_unbalanced_answer_and_a_version_mismatch_are_noted() -> None:
    result = _good(risk=Answer("risk", "noul", 0.1, {}, 0.0, 2, balanced=False))
    result.version = "jev-q1"
    call = decide(_pack(*_move(0.0)), result, RULES, QUESTIONS)
    assert call is not None
    assert call.notes == [
        "risk answer unbalanced (2 rounds)",
        f"answers are jev-q1, questions are {QUESTIONS.version}",
    ]


def test_the_record_and_the_email_lines_show_every_part() -> None:
    call = decide(_pack(*_move(2.0 * USUAL_5D)), _good(), RULES, QUESTIONS)
    assert call is not None
    d = call.to_dict()
    assert [p["name"] for p in d["parts"]] == ["news", "guidance", "fundamentals", "risk"]
    assert d["parts"][0]["contribution"] == pytest.approx(0.324)
    lines = call.explain()
    assert lines[0] == (
        "news: +0.32 (good +0.90, material 80%, evidence 1.00; weight 0.45, certainty 100%)"
    )
    assert lines[-2:] == ["priced in: score cut by 50%", "score +0.32: Buy (confidence 32%)"]


def _rules_file(tmp_path: Path, change: dict[str, Any]) -> Path:
    raw = yaml.safe_load(DEFAULT_RULES.read_text(encoding="utf-8"))
    for dotted, value in change.items():
        *path, last = dotted.split(".")
        node = raw
        for key in path:
            node = node[key]
        if value is ...:
            del node[last]
        else:
            node[last] = value
    out = tmp_path / "rules.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


@pytest.mark.parametrize(
    "change",
    [
        {"version": ""},
        {"extra": 1},
        {"weights.news": 0.9},  # adds up past 1
        {"weights.risk": ...},
        {"weights.news": -0.1},
        {"weights.news": True},
        {"direction.good": 1.5},
        {"direction.good": float("nan")},
        {"evidence.none": -0.5},
        {"calls.buy": -0.1},
        {"calls.short": -0.1},  # short must sit below sell
        {"calls.extra": 0.3},
        {"priced_in.sessions": 3},
        {"priced_in.sessions": True},
        {"priced_in.most": 1.5},
        {"priced_in.per_move": ...},
    ],
)
def test_a_malformed_rules_file_is_refused(tmp_path: Path, change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="rules"):
        load_rules(_rules_file(tmp_path, change), QUESTIONS)


@pytest.mark.parametrize(
    "change",
    [
        {"guidance.issued": ...},  # a label the question has, missing here
        {"direction.great": 1.0},  # a label the question does not have
        {"evidence.none": ...},
    ],
)
def test_labels_must_match_the_questions(tmp_path: Path, change: dict[str, Any]) -> None:
    path = _rules_file(tmp_path, change)
    assert isinstance(load_rules(path), Rules)  # fine on its own
    with pytest.raises(ValueError, match="labels must be exactly"):
        load_rules(path, QUESTIONS)


def test_the_rules_need_each_question_they_read(tmp_path: Path) -> None:
    text = tmp_path / "q.yaml"
    raw = yaml.safe_load(Path("prompts/jev.yaml").read_text(encoding="utf-8"))
    del raw["questions"]["risk"]
    text.write_text(yaml.safe_dump(raw), encoding="utf-8")
    fewer: QuestionSet = load_questions(text)
    with pytest.raises(ValueError, match="noul question 'risk'"):
        load_rules(questions=fewer)
