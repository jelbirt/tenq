"""Rules: Jev's answers to a call, worked by hand against the committed files."""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

import tenq.rules as rules_mod
from tenq import jev
from tenq.jev import QUESTIONS_PATH, Answer, Result, load_questions
from tenq.pack import Number, Pack
from tenq.rules import (
    DEFAULT_RULES,
    Part,
    certainty,
    confidence,
    decide,
    decisive,
    expected,
    load_rules,
    name_call,
    reaction,
)

QUESTIONS = load_questions()
RULES = load_rules(QUESTIONS)
TREND_LABEL = "Revenue, growth from a year ago less the prior quarter's growth, points"


def _growth(change: float, metric: str = "revenue") -> Number:
    return Number(f"fin:ACME:{metric}:growth_change", TREND_LABEL, change, "points")


# One hash per rules version. Editing config/rules.yaml without bumping its
# version fails here; after a bump, add the new version and its file's hash.
KNOWN_RULES = {
    "rules-v2": "ac1b87982703da0f24f361deb9b3e6214c5e5afbda6818365f96a1368e726b3a",
    "rules-v3": "4ed506bcd451cf06d5f3eaeae41adffdf0e12b0c37323c2bf0d39c18e53a91e7",
    "rules-v4": "21d1ba4bf231fabbd4dfe12e2c5112ae6fca22b4b562a65271aa3a47f67e6df5",
    "rules-v5": "a0231854f80cd0a4c2093cdfa1191b252f6df3fe274d34da6629290af7475d73",
    "rules-v6": "32e85d14721d79fd3fdf52e785c0787e7cd319b4a3c20e3028acfda7be4dc21a",
    "rules-v7": "abd8f7f459135a8d171faef67f43c0d261774046c69e3606edb9c1fe2db68da4",
    # v8: the same settings as v7; only comments changed (Short defined against QQQ)
    "rules-v8": "e5c58c81f3a964dddf9810a5a0c13d4006622c4380efcfaf49a26c7de3a86a2a",
    # v9: the filing-trend part dropped (no edge on history); its 0.15 is not
    # handed to the other parts
    "rules-v9": "526d4d0deff65450b3f9aada9d67100e1ee1ebfa297de9a1737cff424a08d004",
}


def _choice(key: str, probs: dict[str, float], agreed: int | None = None) -> Answer:
    (q,) = [q for q in QUESTIONS.questions if q.key == key]
    full = {label: probs.get(label, 0.0) for label in q.labels()}
    top = max(full, key=lambda k: full[k])
    n = len(full)
    agree = n if agreed is None else agreed
    return Answer(key, "choice", top, full, 0.0, n, agreed=agree, balanced=True)


def _noul(key: str, p: float, spread: float = 0.0) -> Answer:
    return Answer(key, "noul", p, {}, spread, 3, balanced=True)


def _score(level: float, spread: float = 0.0) -> Answer:
    """A score answer; no committed question is one now, but the adapter keeps the type."""
    return Answer("trend", "score", level, {}, spread, 4, balanced=True)


def _good(**over: Answer) -> Result:
    answers = {
        "material": _noul("material", 0.8),
        "direction": _choice("direction", {"good": 0.9, "mixed": 0.1}),
        "source": _choice("source", {"reported": 1.0}),
        "guidance": _choice("guidance", {"raised": 1.0}),
        "risk": _noul("risk", 0.1),
    }
    answers.update(over)
    return Result(QUESTIONS.version, "jev-latest", answers=answers)


def _pack(*numbers: Number, fin: bool = True, growth: float = 0.03) -> Pack:
    """A pack whose filing shows revenue growth `growth` faster (3 points:
    improving), a line rules-v9 no longer reads: tests keep it to show that."""
    day = date(2026, 11, 3)
    as_of = datetime(2026, 11, 4, 1, 0, tzinfo=UTC)
    trend = [_growth(growth)] if fin else []
    return Pack("ACME", "Acme", day, as_of, numbers=[*numbers, *trend])


def _move(excess_5d: float, vol: float = 0.40) -> tuple[Number, Number]:
    return (
        Number("px:ACME:excess_5d", "ACME return less QQQ over 5 sessions", excess_5d, "points"),
        Number("px:ACME:vol_20d", "Volatility, annualised, over 20 sessions", vol, "percent"),
    )


USUAL_5D = 0.40 * math.sqrt(5 / 252)  # one usual 5-session move at 40% volatility


def test_the_committed_rules_fit_the_committed_questions() -> None:
    # the dropped filing-trend part's 0.15 was not handed to the other parts
    assert sum(RULES.weights.values()) == pytest.approx(0.85)


def test_a_rules_edit_needs_a_new_version() -> None:
    assert RULES.version in KNOWN_RULES, "new rules version: add it and its hash above"
    assert RULES.sha256 == KNOWN_RULES[RULES.version], "rules changed: bump the version"


def test_good_news_is_a_buy_worked_by_hand() -> None:
    call = decide(_pack(), _good(), RULES, QUESTIONS)
    assert call is not None
    parts = {p.name: p.contribution for p in call.parts}
    # news: good 0.9 x material 0.8 x evidence 1.0 = 0.72, weight 0.45
    assert parts["news"] == pytest.approx(0.324)
    assert parts["guidance"] == pytest.approx(0.25)  # raised, weight 0.25
    assert parts["risk"] == pytest.approx(-0.015)  # 10% risk, none of it bad news yet
    assert set(parts) == {"news", "guidance", "risk"}  # no filing-trend part since v9
    assert call.raw == pytest.approx(0.559) and call.score == pytest.approx(0.559)
    # the most the parts could reach upward is 0.45 + 0.25
    assert call.call == "Buy" and call.confidence == pytest.approx(0.559 / 0.70)
    # coverage: material 80% sits 0.6 from a coin flip (the news part's least
    # decisive answer), 10% risk sits 0.8 from one, guidance is one-sided
    assert call.coverage == pytest.approx((0.45 * 0.6 + 0.25 + 0.15 * 0.8) / 0.85)
    assert call.version == "rules-v9" and call.rules_sha == RULES.sha256
    assert call.day == "2026-11-03"
    assert call.notes == ["no price move to weigh the market's reaction"]


def test_bad_news_is_not_counted_again_as_risk() -> None:
    result = _good(
        material=_noul("material", 1.0),
        direction=_choice("direction", {"bad": 1.0}),
        guidance=_choice("guidance", {"lowered": 1.0}),
        risk=_noul("risk", 0.9),
    )
    call = decide(_pack(growth=-0.03), result, RULES, QUESTIONS)
    assert call is not None
    (risk,) = [p for p in call.parts if p.name == "risk"]
    assert risk.contribution == 0.0  # the news part already scored it as bad
    # -0.45 - 0.25, with revenue growth slower read by nothing; down, news and
    # risk never both count, so the reach is 0.45 + 0.25 = 0.70
    assert call.score == pytest.approx(-0.70) and call.call == "Short"
    assert call.confidence == pytest.approx(1.0)


def test_risk_counts_what_the_news_part_left_unscored() -> None:
    weak = _good(
        direction=_choice("direction", {"bad": 1.0}),
        source=_choice("source", {"opinion": 1.0}),
        risk=_noul("risk", 1.0),
    )
    call = decide(_pack(), weak, RULES, QUESTIONS)
    assert call is not None
    (risk,) = [p for p in call.parts if p.name == "risk"]
    # the news part scored bad 1.0 x material 0.8 x evidence 0.4 = 0.32 of it
    assert risk.contribution == pytest.approx(-0.15 * 0.68)
    unread = _good(direction=_choice("direction", {"bad": 1.0}), risk=_noul("risk", 1.0))
    del unread.answers["source"]
    call = decide(_pack(), unread, RULES, QUESTIONS)
    assert call is not None
    (risk,) = [p for p in call.parts if p.name == "risk"]
    assert risk.contribution == pytest.approx(-0.15)  # no news part, so none of it scored


def test_a_risk_event_beside_good_news_still_counts() -> None:
    call = decide(_pack(), _good(risk=_noul("risk", 1.0)), RULES, QUESTIONS)
    assert call is not None
    (risk,) = [p for p in call.parts if p.name == "risk"]
    assert risk.contribution == pytest.approx(-0.15)


def test_mildly_bad_news_is_a_hold_and_firmer_bad_news_a_sell() -> None:
    mild = _good(
        direction=_choice("direction", {"bad": 0.8, "neutral": 0.2}),
        source=_choice("source", {"opinion": 1.0}),
        guidance=_choice("guidance", {"none": 1.0}),
    )
    call = decide(_pack(growth=0.0), mild, RULES, QUESTIONS)
    assert call is not None
    # news -0.8 x 0.8 x 0.4 = -0.256, x 0.45; risk 0.1 x (1 - 0.256) x 0.15
    assert call.score == pytest.approx(-0.256 * 0.45 - 0.1 * 0.744 * 0.15)
    assert call.call == "Hold"
    worse = _good(
        direction=_choice("direction", {"bad": 1.0}),
        guidance=_choice("guidance", {"none": 1.0}),
    )
    call = decide(_pack(growth=0.0), worse, RULES, QUESTIONS)
    assert call is not None
    # news -0.8 x 0.45; risk 0.1 x (1 - 0.8) x 0.15
    assert call.score == pytest.approx(-0.363) and call.call == "Sell"


def test_an_unsure_jev_drifts_toward_hold_without_a_cut_off() -> None:
    scores = []
    for spread in (0.0, 0.3, 0.6, 0.9):
        result = _good(
            material=_noul("material", 0.8, spread),
            guidance=_choice("guidance", {"none": 1.0}),
        )
        call = decide(_pack(), result, RULES, QUESTIONS)
        assert call is not None
        scores.append(call.score)
    assert scores == sorted(scores, reverse=True)
    steps = [a - b for a, b in itertools.pairwise(scores)]
    assert steps == pytest.approx([steps[0]] * 3)  # a straight line, no jump
    assert scores[0] >= RULES.buy > scores[-1]  # the same news, unsure enough, is a Hold


def test_the_news_part_is_as_certain_as_its_least_certain_answer() -> None:
    src = dataclasses.replace(_choice("source", {"reported": 1.0}), repeats=2, agreed=2)
    way = dataclasses.replace(
        _choice("direction", {"good": 0.9, "mixed": 0.1}), repeats=2, agreed=2
    )
    overs: list[dict[str, Answer]] = [
        {"material": _noul("material", 0.8, 0.5)},
        {"source": src},  # 2 of its 5 planned rounds came back
        {"direction": way},  # 2 of 4
    ]
    expected_certainty = [0.5, 2 / 5, 0.5]
    for over, want in zip(overs, expected_certainty, strict=True):
        call = decide(_pack(), _good(**over), RULES, QUESTIONS)
        assert call is not None
        assert call.parts[0].certainty == pytest.approx(want), over


def test_certainty_reads_each_answer_on_its_own_scale() -> None:
    assert certainty(_score(2.0, spread=2.0), 5, 4) == pytest.approx(0.5)
    assert certainty(_noul("risk", 0.5, spread=0.3), 0, 3) == pytest.approx(0.7)
    assert certainty(_noul("risk", 0.5, spread=1.5), 0, 3) == 0.0
    # a choice's doubt is in its value: neither a split nor the rotation's lean
    # counts again here, only how many of its rounds came back
    lean = dataclasses.replace(_choice("direction", {"good": 0.7}, agreed=1), spread=0.25)
    assert certainty(lean, 4, 4) == 1.0
    assert certainty(dataclasses.replace(lean, repeats=1), 4, 4) == pytest.approx(0.25)
    assert certainty(dataclasses.replace(lean, repeats=0), 4, 4) == 0.0
    # one round of three: no agreement to read, and the position lean uncancelled
    assert certainty(dataclasses.replace(_noul("risk", 0.5), repeats=1), 0, 3) == pytest.approx(
        1 / 3
    )


def test_probabilities_are_made_to_add_up_to_one() -> None:
    every = Answer("source", "choice", "none", dict.fromkeys(RULES.evidence, 1.0), 0.0, 5)
    assert expected(every, RULES.evidence) == pytest.approx(3.05 / 5)
    half = Answer("direction", "choice", "good", {"good": 0.5, "bad": 0.0}, 0.0, 4)
    assert expected(half, RULES.direction) == pytest.approx(1.0)
    empty = Answer("direction", "choice", "bad", {"good": 0.0}, 0.0, 4)
    assert expected(empty, RULES.direction) == -1.0  # falls back to the pick
    both = {"good": 1.0, "neutral": 1.0}
    loud = Answer("direction", "choice", "good", both, 0.0, 4, agreed=4, balanced=True)
    call = decide(_pack(), _good(direction=loud), RULES, QUESTIONS)
    assert call is not None
    assert all(-1.0 <= p.value <= 1.0 for p in call.parts)
    assert call.parts[0].value == pytest.approx(0.5 * 0.8)


def test_a_missing_answer_counts_as_nothing_and_is_named() -> None:
    result = _good()
    del result.answers["source"], result.answers["guidance"]
    call = decide(_pack(), result, RULES, QUESTIONS)
    assert call is not None
    parts = {p.name: p for p in call.parts}
    assert parts["news"].contribution == 0.0 and parts["news"].detail == "no answer for source"
    assert parts["guidance"].detail == "no answer for guidance"
    assert call.score == pytest.approx(-0.015) and call.call == "Hold"  # risk alone
    assert call.coverage == pytest.approx(0.15 * 0.8 / 0.85)  # risk only


def test_an_answer_of_the_wrong_type_is_missing_and_noted() -> None:
    wrong = Answer("risk", "choice", "good", {"good": 1.0}, 0.0, 1, agreed=1)
    call = decide(_pack(), _good(risk=wrong), RULES, QUESTIONS)
    assert call is not None
    (part,) = [p for p in call.parts if p.name == "risk"]
    assert part.detail == "no answer for risk"
    assert "risk answer is a choice, the rules read a noul" in call.notes


@pytest.mark.parametrize("growth", [0.10, -0.10, 0.0])
def test_the_filing_trend_no_longer_moves_jevs_call(growth: float) -> None:
    """rules-v9: revenue growth speeding up or slowing changes nothing."""
    plain = decide(_pack(fin=False), _good(), RULES, QUESTIONS)
    trend = decide(
        _pack(_growth(growth), _growth(growth, "deferred_revenue"), fin=False),
        _good(),
        RULES,
        QUESTIONS,
    )
    assert plain is not None and trend is not None
    assert trend.score == plain.score == pytest.approx(0.559)
    assert trend.confidence == plain.confidence == pytest.approx(0.559 / 0.70)
    assert [p.name for p in trend.parts] == ["news", "guidance", "risk"]


def test_no_answers_is_no_call() -> None:
    assert decide(_pack(), Result(QUESTIONS.version, "jev-latest"), RULES, QUESTIONS) is None


def test_kept_guidance_counts_for_nothing_either_way() -> None:
    call = decide(_pack(), _good(guidance=_choice("guidance", {"kept": 1.0})), RULES, QUESTIONS)
    assert call is not None
    parts = {p.name: p.contribution for p in call.parts}
    assert parts["guidance"] == pytest.approx(0.0)


def test_a_jev_that_knew_nothing_is_a_hold_without_confidence() -> None:
    flat_way = dict.fromkeys(RULES.direction, 0.25)
    flat_guide = dict.fromkeys(RULES.guidance, 1 / 6)
    result = _good(
        material=_noul("material", 0.5, 1.0),
        direction=_choice("direction", flat_way, agreed=0),
        guidance=_choice("guidance", flat_guide, agreed=0),
        risk=_noul("risk", 0.5, 1.0),
    )
    call = decide(_pack(fin=False), result, RULES, QUESTIONS)
    assert call is not None
    assert call.call == "Hold" and call.score == pytest.approx(0.0)
    assert call.coverage == 0.0 and call.confidence == 0.0
    # a growth line in the filing is no evidence since rules-v9 (it once was)
    for growth in (0.0, 0.05):
        call = decide(_pack(growth=growth), result, RULES, QUESTIONS)
        assert call is not None
        assert call.call == "Hold" and call.score == pytest.approx(0.0)
        assert call.coverage == 0.0 and call.confidence == 0.0


@pytest.mark.parametrize(
    ("moves", "cut"),
    [
        (2.0, 0.5),  # already beat QQQ by two usual moves: half priced in
        (1.0, 0.25),
        (10.0, 0.75),  # never more than `most`
        (-2.0, 0.5),  # lagged QQQ despite good news: the market doubts it, same cut
        (-10.0, 0.75),
        (0.0, 0.0),
    ],
)
def test_a_move_either_way_shrinks_the_score(moves: float, cut: float) -> None:
    call = decide(_pack(*_move(moves * USUAL_5D)), _good(), RULES, QUESTIONS)
    assert call is not None
    assert call.reaction == pytest.approx(cut)
    assert call.score == pytest.approx(0.559 * (1 - cut))
    assert call.notes == []


def test_bad_news_after_a_fall_is_partly_priced_in() -> None:
    result = _good(direction=_choice("direction", {"bad": 1.0}))
    call = decide(_pack(*_move(-2.0 * USUAL_5D), growth=0.0), result, RULES, QUESTIONS)
    assert call is not None and call.raw < 0
    assert call.reaction == pytest.approx(0.5)


def test_bad_news_after_a_rise_is_doubted_by_the_same_amount() -> None:
    result = _good(direction=_choice("direction", {"bad": 1.0}))
    call = decide(_pack(*_move(2.0 * USUAL_5D), growth=0.0), result, RULES, QUESTIONS)
    assert call is not None and call.raw < 0
    assert call.reaction == pytest.approx(0.5) and call.reaction_way == "against"
    assert call.score == pytest.approx(call.raw * 0.5)


def test_the_reaction_never_turns_a_call_past_hold() -> None:
    call = decide(_pack(*_move(-50.0 * USUAL_5D)), _good(), RULES, QUESTIONS)
    assert call is not None and call.raw > 0
    assert call.reaction == pytest.approx(RULES.most) and call.score > 0
    assert call.call == "Hold" and call.reaction_way == "against"
    assert call.explain()[-2] == (
        "market reaction: score cut by 75% (moved against the signal over the last 5 sessions, the market doubts it)"
    )


def test_zero_volatility_reads_no_reaction() -> None:
    call = decide(_pack(*_move(0.05, vol=0.0)), _good(), RULES, QUESTIONS)
    assert call is not None and call.reaction == 0.0
    assert call.notes == ["no price move to weigh the market's reaction"]


@pytest.mark.parametrize(
    ("score", "name"),
    [
        (0.20, "Buy"),  # each line belongs to the call beyond it
        (0.19, "Hold"),
        (0.0, "Hold"),
        (-0.19, "Hold"),
        (-0.20, "Sell"),
        (-0.44, "Sell"),
        (-0.45, "Short"),
        (-1.0, "Short"),
    ],
)
def test_each_score_names_one_call(score: float, name: str) -> None:
    assert name_call(score, RULES) == name


def test_hold_confidence_is_measured_against_its_own_side_of_the_band() -> None:
    lopsided = dataclasses.replace(RULES, buy=0.30)
    call = decide(_pack(), _good(), RULES, QUESTIONS)
    assert call is not None
    parts = [dataclasses.replace(p, decisive=1.0) for p in call.parts]  # full coverage
    assert confidence(0.15, "Hold", lopsided, parts) == pytest.approx(0.5)  # half of 0.30
    assert confidence(-0.10, "Hold", lopsided, parts) == pytest.approx(0.5)  # half of 0.20
    assert confidence(0.0, "Hold", lopsided, parts) == pytest.approx(1.0)


def test_an_unbalanced_answer_and_a_version_mismatch_are_noted() -> None:
    result = _good(risk=Answer("risk", "noul", 0.1, {}, 0.0, 2, balanced=False))
    result.answers["unread"] = Answer("unread", "noul", 0.5, {}, 0.0, 1, balanced=False)
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
    assert [p["name"] for p in d["parts"]] == ["news", "guidance", "risk"]
    assert d["parts"][0]["contribution"] == pytest.approx(0.324)
    assert d["rules_sha"] == RULES.sha256 and d["coverage"] == pytest.approx(0.64 / 0.85)
    lines = call.explain()
    assert lines[0] == (
        "news: +0.32 (good +0.90, material 80%, evidence 1.00; weight 0.45, certainty 100%)"
    )
    assert lines[-2:] == [
        "market reaction: score cut by 50% (moved with the signal over the last 5 sessions, partly priced in)",
        "score +0.28: Buy (confidence 40%)",  # 0.559 halved, of a reach of 0.70
    ]


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
        {"weights.news": 0, "weights.guidance": 0, "weights.risk": 0},
        {"weights.fundamentals": 0.15},  # the part rules-v9 dropped
        {"weights.risk": ...},
        {"weights.news": -0.1},
        {"weights.news": True},
        {"direction.good": 1.5},
        {"direction.good": float("nan")},
        {"evidence.none": -0.5},
        {"calls.buy": -0.1},
        {"calls.buy": 0.0},
        {"calls.sell": 0.0},
        {"calls.buy": 0.75},  # more than news and guidance can reach (0.70)
        {"weights.risk": 0.05, "calls.short": -0.95},  # more than all parts can reach
        {"calls.short": -0.1},  # short must sit below sell
        {"calls.extra": 0.3},
        {"reaction.sessions": 3},
        {"reaction.sessions": True},
        {"reaction.most": 1.5},
        {"reaction.per_move": ...},
    ],
)
def test_a_malformed_rules_file_is_refused(tmp_path: Path, change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="rules"):
        load_rules(QUESTIONS, _rules_file(tmp_path, change))


def test_a_key_given_twice_is_refused(tmp_path: Path) -> None:
    text = DEFAULT_RULES.read_text(encoding="utf-8").replace(
        "  good: 1.0\n", "  good: 1.0\n  good: -1.0\n", 1
    )
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"repeats \['good'\]"):
        load_rules(QUESTIONS, path)


@pytest.mark.parametrize(
    "change",
    [
        {"guidance.issued": ...},  # a label the question has, missing here
        {"direction.great": 1.0},  # a label the question does not have
        {"evidence.none": ...},
    ],
)
def test_labels_must_match_the_questions(tmp_path: Path, change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="labels must be exactly"):
        load_rules(QUESTIONS, _rules_file(tmp_path, change))


def _drop_risk(q: dict[str, Any]) -> None:
    del q["risk"]


def _material_as_choice(q: dict[str, Any]) -> None:
    q["material"] = {"type": "choice", "instructions": "x", "criteria": {"a": "x", "b": "y"}}


@pytest.mark.parametrize(
    ("edit", "message"),
    [(_drop_risk, "noul question 'risk'"), (_material_as_choice, "noul question 'material'")],
)
def test_the_rules_need_each_question_they_read_with_its_type(
    tmp_path: Path, edit: Any, message: str
) -> None:
    raw = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    edit(raw["questions"])
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_rules(load_questions(path))


def _flat_rounds() -> dict[str, Answer]:
    """What jev.average makes of a Jev that knew nothing: flat answers every round."""
    responses = []
    for _ in range(QUESTIONS.rounds()):
        answers: dict[str, object] = {}
        for q in QUESTIONS.questions:
            labels = q.labels()
            flat = {label: 1.0 / len(labels) for label in labels} if labels else {}
            if q.type == "noul":
                answers[q.key] = {"noul": 0.5}
            elif q.type == "choice":
                answers[q.key] = {"choice": labels[0], "probabilities": flat}
            else:
                answers[q.key] = {"score": (len(labels) - 1) / 2, "probabilities": flat}
        responses.append(answers)
    return jev.average(QUESTIONS, responses)


@pytest.mark.parametrize("risk", [0.0, 0.5])
def test_a_jev_that_knew_nothing_every_round_is_a_hold_with_no_confidence(risk: float) -> None:
    answers = _flat_rounds()
    answers["risk"] = dataclasses.replace(answers["risk"], value=risk)
    result = Result(QUESTIONS.version, "jev-latest", answers)
    call = decide(_pack(fin=False), result, RULES, QUESTIONS)
    assert call is not None and call.call == "Hold"
    # flat ties pick the same label every round, so agreement alone looked sure
    assert all(answers[k].agreed == answers[k].repeats for k in ("direction", "source"))
    # a confident "no risk" is the only decisive answer, so it is all the coverage
    # (of news, guidance and risk)
    assert call.coverage == pytest.approx(0.15 * abs(2 * risk - 1) / 0.85)
    assert call.confidence <= 0.15 / 0.85


def test_decisive_measures_the_distance_from_a_coin_flip() -> None:
    labels = QUESTIONS.questions[1].labels()
    flat = _choice("direction", {k: 0.25 for k in labels})
    assert decisive(flat, labels) == pytest.approx(0.0)
    assert decisive(_choice("direction", {"good": 1.0}), labels) == pytest.approx(1.0)
    half = _choice("direction", {"good": 0.5, "bad": 0.5})
    assert decisive(half, labels) == pytest.approx(0.5 / 0.75)
    split = dataclasses.replace(_choice("direction", {"good": 1.0}), agreed=2)
    assert decisive(split, labels) == pytest.approx(0.5)  # two of four rounds agreed
    assert decisive(_noul("risk", 0.5), []) == 0.0
    assert decisive(_noul("risk", 0.9), []) == pytest.approx(0.8)
    assert decisive(_score(2.0), []) == 1.0  # no probabilities: spread carries the doubt


def test_unsure_bad_news_leaves_the_risk_event_counted() -> None:
    def run(spread: float) -> dict[str, float]:
        call = decide(
            _pack(fin=False),
            _good(
                direction=_choice("direction", {"bad": 1.0}),
                material=_noul("material", 1.0, spread),
                guidance=_choice("guidance", {"none": 1.0}),
                risk=_noul("risk", 1.0),
            ),
            RULES,
            QUESTIONS,
        )
        assert call is not None
        return {p.name: p.contribution for p in call.parts}

    sure = run(0.0)
    assert sure["news"] == pytest.approx(-0.45) and sure["risk"] == pytest.approx(0.0)
    noisy = run(0.8)  # material certainty 0.2: the news part scored a fifth of it
    assert noisy["news"] == pytest.approx(-0.45 * 0.2)
    assert noisy["risk"] == pytest.approx(-0.15 * 0.8)
    gone = run(1.0)  # nothing scored as news, so the lawsuit counts in full
    assert gone["news"] == 0.0 and gone["risk"] == pytest.approx(-0.15)


def test_risk_is_reduced_by_the_chance_of_bad_news_not_the_net() -> None:
    call = decide(
        _pack(fin=False),
        _good(
            direction=_choice("direction", {"good": 0.5, "bad": 0.5}),
            material=_noul("material", 1.0),
            risk=_noul("risk", 1.0),
        ),
        RULES,
        QUESTIONS,
    )
    assert call is not None
    parts = {p.name: p.contribution for p in call.parts}
    assert parts["news"] == pytest.approx(0.0)  # good and bad cancel in the news part
    assert parts["risk"] == pytest.approx(-0.15 * 0.5)  # half of it was scored bad news


@pytest.mark.parametrize(
    "broken",
    [
        {"key": "zzz"},
        {"repeats": 10**400},
        {"repeats": -3},
        {"repeats": 0},
        {"repeats": True},
        {"agreed": -1},
        {"agreed": 99},
        {"spread": math.nan},
        {"spread": -0.1},
        {"value": math.inf},
        {"probabilities": {"good": 1.5}},
        {"probabilities": {"good": math.nan}},
    ],
)
def test_an_unreadable_answer_counts_as_nothing_and_is_noted(broken: dict[str, object]) -> None:
    bad = dataclasses.replace(_choice("direction", {"good": 1.0}), **broken)  # type: ignore[arg-type]
    if "value" in broken:
        bad = dataclasses.replace(_noul("material", 0.8), **broken)  # type: ignore[arg-type]
    key = "material" if "value" in broken else "direction"
    call = decide(_pack(), _good(**{key: bad}), RULES, QUESTIONS)
    assert call is not None
    news = call.parts[0]
    assert news.contribution == 0.0 and news.detail == f"no answer for {key}"
    assert f"{key} answer unreadable" in call.notes
    assert -1.0 <= call.score <= 1.0 and 0.0 <= call.confidence <= 1.0


def test_out_of_range_values_are_clamped() -> None:
    call = decide(
        _pack(),
        _good(
            material=_noul("material", 1.0),
            risk=_noul("risk", 0.0),
        ),
        RULES,
        QUESTIONS,
    )
    high = decide(_pack(), _good(material=_noul("material", 7.0)), RULES, QUESTIONS)
    assert call is not None and high is not None
    assert high.parts[0].value == pytest.approx(call.parts[0].value)


def test_an_unexpected_failure_is_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: object) -> list[Part]:
        raise RuntimeError("bug")

    monkeypatch.setattr(rules_mod, "_parts", boom)
    assert decide(_pack(), _good(), RULES, QUESTIONS) is None


def test_sell_and_short_confidence_use_the_downward_reach() -> None:
    tilted = dataclasses.replace(RULES, weights={"news": 0.2, "guidance": 0.4, "risk": 0.4})
    call = decide(_pack(), _good(), RULES, QUESTIONS)
    assert call is not None
    # up: 0.4 + 0.2 = 0.6; down: 0.4 + max(0.2, 0.4) = 0.8
    assert confidence(0.3, "Buy", tilted, call.parts) == pytest.approx(0.5)
    assert confidence(-0.4, "Sell", tilted, call.parts) == pytest.approx(0.5)
    assert confidence(-0.6, "Short", tilted, call.parts) == pytest.approx(0.75)


def test_a_scores_reversal_lean_is_not_doubt() -> None:
    leaned = Answer("trend", "score", 2.0, {}, 1.0, 4, balanced=True, first_lift=0.5)
    assert certainty(leaned, 5, 4) == pytest.approx(1.0)
    partial = dataclasses.replace(leaned, first_lift=0.25)
    assert certainty(partial, 5, 4) == pytest.approx(1.0 - 0.5 / 4)
    unbalanced = dataclasses.replace(leaned, balanced=False)
    assert certainty(unbalanced, 5, 4) == pytest.approx(1.0 - 1.0 / 4)


def test_the_reaction_reads_nothing_into_a_zero_score_or_an_unreadable_move() -> None:
    assert reaction(_pack(*_move(2.0 * USUAL_5D)), 0.0, RULES) == (0.0, None)
    # a zero score with a vanishing volatility is 0 times infinity: still no cut
    assert reaction(_pack(*_move(0.05, vol=1e-320)), 0.0, RULES) == (0.0, None)
    cut, note = reaction(_pack(*_move(0.05, vol=1e-320)), 0.3, RULES)
    assert cut == 0.0 and note == "price move too large to read against its volatility"


def test_a_one_session_reaction_reads_in_the_singular() -> None:
    call = decide(_pack(*_move(2.0 * USUAL_5D)), _good(), RULES, QUESTIONS)
    assert call is not None
    one = replace(call, reaction_id="px:ACME:excess_1d")
    line = next(x for x in one.explain() if x.startswith("market reaction:"))
    assert "over the last session," in line and "1 sessions" not in line


def test_a_jev_fundamentals_answer_is_ignored() -> None:
    """Even a stray one in a stored record: Jev is never asked about the
    filing's figures, and no part reads such an answer."""
    stray = {"fundamentals": Answer("fundamentals", "score", 1.0, {}, 0.0, 4, balanced=True)}
    assert "fundamentals" not in {q.key for q in QUESTIONS.questions}
    call = decide(_pack(), _good(**stray), RULES, QUESTIONS)
    plain = decide(_pack(), _good(), RULES, QUESTIONS)
    assert call is not None and plain is not None
    assert [p.name for p in call.parts] == ["news", "guidance", "risk"]
    assert call.score == plain.score
