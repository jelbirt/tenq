"""Jev: typed answers to narrow questions about one stock's facts pack.

Jev (TypeSafe's decision model, served through OpenRouter's System One API)
returns probabilities, not text. The questions live in prompts/jev.yaml; the
state is the pack's rendered text, whose numbers were all computed in code.
Each question is asked over several rounds and the answers averaged, because
Jev's answers vary between identical calls and lean toward options listed
earlier. The rounds are balanced so that lean cancels: a choice is asked once
per label, with each label shown first exactly once; a score is asked as
often with its levels reversed as in file order; a true/false question has no
order and is asked REPEATS times against noise. Each answer keeps its spread,
how many rounds agreed, and the measured lean toward whichever option was
listed first (`first_lift`), so the rules and the scorecard can see how
settled an answer was and how strong the lean is.

Never raises: a missing key, an HTTP error or an unreadable answer is an error
on the result, and the brief goes out with Opus's call alone. A request that
fails for good (a bad key, no credit, a service still down after one retry)
ends the rounds, so an outage costs about one minute per stock, not six. Logs carry only
status codes and exception types, never the state or an answer.

Env: OPENROUTER_API_KEY (the only secret), JEV_MODEL (default jev-latest; the
resolved model, e.g. typesafe/jev-1.13-20260917, is recorded per run).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import yaml

from tenq.config import REPO_ROOT

log = logging.getLogger(__name__)

URL = "https://openrouter.ai/api/v1/systemone"
DEFAULT_MODEL = "jev-latest"
QUESTIONS_PATH = REPO_ROOT / "prompts" / "jev.yaml"
REPEATS = 3  # true/false questions: asked this often, averaged against noise
SCORE_ROUNDS = 4  # scores: twice in file order, twice reversed
MAX_LABELS = 8  # a choice is asked once per label, so its size sets the request count
TIMEOUT_S = 30
ATTEMPTS = 2  # per round: one retry on a rate limit, an upstream failure or a timeout
RETRY_CODES = {429, 500, 502, 503, 524, 529}
FINAL_CODES = {400, 401, 402, 403, 404, 413}  # the same request would fail the same way
MODEL_NAME = re.compile(r"[A-Za-z0-9._:/~-]{1,100}")
MAX_STATE_CHARS = 60_000  # well under Jev's 32k-token state limit
KEY = r"[a-z][a-z0-9_]{0,39}"

QType = Literal["noul", "choice", "score"]
Post = Callable[[str, dict[str, str], bytes], tuple[int, str]]


@dataclass(frozen=True)
class Question:
    key: str
    type: QType
    instructions: str
    criteria: Any  # noul: {"true", "false"} or None; choice: {label: text}; score: [level...]

    def labels(self) -> list[str]:
        """The answer space: choice labels, or score levels as "0".."n-1"."""
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return []

    def rounds(self) -> int:
        """How many rounds ask this question: enough to show every option first equally."""
        if self.type == "choice":
            return len(self.criteria)
        if self.type == "score":
            return SCORE_ROUNDS
        return REPEATS

    def first(self, repeat: int) -> str:
        """The choice label shown first in a round ("" for other types)."""
        if self.type == "choice":
            return self.labels()[repeat % len(self.criteria)]
        return ""

    def to_request(self, repeat: int = 0) -> dict[str, Any]:
        """The question as sent in one round, its options moved to cancel position bias.

        Round r starts a choice's labels at label r, so over `rounds()` rounds
        each label sits in each position exactly once; odd rounds list a
        score's levels in reverse (`unflip` maps those answers back).
        """
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        criteria = self.criteria
        if self.type == "choice":
            items = list(criteria.items())
            shift = repeat % len(items)
            criteria = dict(items[shift:] + items[:shift])
        elif self.type == "score" and repeat % 2:
            criteria = list(reversed(criteria))
        if criteria is not None:
            body["criteria"] = criteria
        return body

    def unflip(self, raw: Any, repeat: int) -> Any:
        """A score answer from a reversed repeat, mapped back to the file's level order."""
        if self.type != "score" or not repeat % 2 or not isinstance(raw, dict):
            return raw
        top = len(self.criteria) - 1
        out = dict(raw)
        level = _finite(raw.get("score"))
        out["score"] = top - level if level is not None else raw.get("score")
        probs = raw.get("probabilities")
        if isinstance(probs, dict):
            # matched against the known levels, never parsed: a 5000-digit key
            # would make int() raise
            levels = {str(i) for i in range(top + 1)}
            flipped: dict[str, Any] = {}
            for k, v in probs.items():
                if k in levels:
                    flipped[str(top - int(k))] = v
            out["probabilities"] = flipped
        return out


@dataclass(frozen=True)
class QuestionSet:
    version: str
    questions: tuple[Question, ...]
    sha256: str  # of the file's bytes as read, so a test can pin the questions to their version

    def rounds(self) -> int:
        return max(q.rounds() for q in self.questions)

    def to_request(self, repeat: int = 0) -> dict[str, Any]:
        """The questions still asked in this round (a 4-label choice sits out rounds 4 and 5)."""
        return {q.key: q.to_request(repeat) for q in self.questions if repeat < q.rounds()}

    def unflip(self, answers: Mapping[str, Any], repeat: int) -> dict[str, Any]:
        by_key = {q.key: q for q in self.questions}
        return {k: by_key[k].unflip(v, repeat) if k in by_key else v for k, v in answers.items()}


def load_questions(path: Path = QUESTIONS_PATH) -> QuestionSet:
    """Validate the question file; a malformed one is a ValueError, never a guess."""
    data = path.read_bytes()
    raw = yaml.safe_load(data.decode("utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"version", "questions"}:
        raise ValueError("question file must hold exactly version and questions")
    version, items = raw["version"], raw["questions"]
    if not isinstance(version, str) or not version:
        raise ValueError("question file version must be a string")
    if not isinstance(items, dict) or not items:
        raise ValueError("question file must list questions")
    out: list[Question] = []
    for key, q in items.items():
        if not isinstance(key, str) or not re.fullmatch(KEY, key):
            raise ValueError(f"bad question key: {key!r}")
        if not isinstance(q, dict) or not set(q) <= {"type", "instructions", "criteria"}:
            raise ValueError(f"question {key}: unknown fields")
        kind, text, criteria = q.get("type"), q.get("instructions"), q.get("criteria")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"question {key}: instructions must be text")
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict)
                or set(criteria) != {"true", "false"}
                or not all(isinstance(v, str) for v in criteria.values())
            ):
                raise ValueError(f"question {key}: noul criteria must be true and false")
        elif kind == "choice":
            if (
                not isinstance(criteria, dict)
                or len(criteria) < 2
                or not all(isinstance(k, str) and re.fullmatch(KEY, k) for k in criteria)
                or not all(isinstance(v, str) for v in criteria.values())
            ):
                raise ValueError(f"question {key}: choice needs two or more labels")
            if len(criteria) > MAX_LABELS:
                raise ValueError(f"question {key}: more than {MAX_LABELS} labels")
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or len(criteria) < 2
                or not all(isinstance(v, str) and v for v in criteria)
            ):
                raise ValueError(f"question {key}: score needs two or more levels")
        else:
            raise ValueError(f"question {key}: unknown type {kind!r}")
        out.append(Question(key, kind, text.strip(), criteria))
    return QuestionSet(version, tuple(out), hashlib.sha256(data).hexdigest())


@dataclass
class Answer:
    """One question's answer averaged over the rounds that returned it.

    value: noul, the mean probability; choice, the label with the highest mean
    probability; score, the mean level (0 is the first criterion).
    probabilities: choice and score, the mean per label (levels as "0".."n-1").
    spread: max minus min across rounds of the headline number (noul
    probability, choice top-label probability, score level); 0 for one repeat.
    margin: choice only, the winning label's mean probability less the
    runner-up's (how decisive the average is); None for noul and score.
    agreed: rounds whose own answer matches `value` (choice: the same top
    label; score: the same level when rounded half up; noul: the same side
    of 0.5).
    balanced: every round was answered, so each option was shown first
    equally often and the position lean cancels in `value`.
    first_lift: the measured lean, set only when balanced. Choice: the mean
    probability Jev gave whichever label was listed first, less the mean of
    all labels' probabilities (0 means no lean; 0.25 on a four-way choice
    means first place alone is worth as much as an option's fair share).
    Score: half the gap in mean level between reversed and file-order rounds
    (positive means answers drift toward whichever end is listed first).
    None for true/false questions, which have no order.
    """

    key: str
    type: QType
    value: float | str
    probabilities: dict[str, float]
    spread: float
    repeats: int  # rounds that returned this question
    margin: float | None = None
    agreed: int = 0
    balanced: bool = False
    first_lift: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass
class Result:
    version: str
    model: str  # requested
    answers: dict[str, Answer] = field(default_factory=dict)
    resolved_model: str = ""
    rounds: int = 0  # requests that returned usable answers
    answered: list[int] = field(default_factory=list)  # which rounds those were
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    duration_ms: int = 0
    errors: list[str] = field(default_factory=list)  # one per failed request
    notes: list[str] = field(default_factory=list)  # e.g. the state was cut
    down: bool = False  # the service or the key failed: skip Jev for the rest of the run
    error: str | None = None  # set when no usable answer came back

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in vars(self).items() if k != "answers"}
        out["answers"] = {k: a.to_dict() for k, a in self.answers.items()}
        return out


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect comes back as its status: urllib would resend the key to any host."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
            return int(resp.status), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _finite(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    try:
        f = float(v)
    except OverflowError:  # a 400-digit integer literal
        return None
    return f if math.isfinite(f) else None


def _one(q: Question, raw: Any) -> tuple[float, dict[str, float]] | None:
    """(headline number, probabilities) from one request's answer, or None."""
    if not isinstance(raw, dict):
        return None
    if q.type == "noul":
        p = _finite(raw.get("noul"))
        return None if p is None or not 0 <= p <= 1 else (p, {})
    labels = q.labels()
    probs: dict[str, float] = {}
    given = raw.get("probabilities")
    if isinstance(given, dict):
        for k, v in given.items():
            f = _finite(v)
            if str(k) in labels and f is not None and 0 <= f <= 1:
                probs[str(k)] = f
    if q.type == "choice":
        pick = raw.get("choice")
        if pick not in labels:
            return None
        if not probs:  # no distribution given: the pick alone
            probs = {pick: 1.0}
        probs = {label: probs.get(label, 0.0) for label in labels}
        return probs[pick], probs
    level = _finite(raw.get("score"))
    if level is None or not 0 <= level <= len(labels) - 1:
        return None
    return level, {label: probs.get(label, 0.0) for label in labels} if probs else {}


def average(
    questions: QuestionSet,
    responses: Sequence[Mapping[str, Any]],
    rounds: Sequence[int] | None = None,
) -> dict[str, Answer]:
    """Each question's answers across the rounds' `answers` maps, averaged.

    `rounds` names the round each response came from (default 0, 1, 2...),
    which says which option was listed first. A question missing or unreadable
    in one round is averaged over the rest, and its answer is not balanced; a
    question no round answered is left out.
    """
    order = rounds if rounds is not None else list(range(len(responses)))
    out: dict[str, Answer] = {}
    for q in questions.questions:
        pairs = [(i, _one(q, r.get(q.key))) for i, r in zip(order, responses, strict=True)]
        answered = [(i, a) for i, a in pairs if a is not None]
        got = [a for _, a in answered]
        if not got:
            continue
        heads = [h for h, _ in got]
        if q.type == "noul":
            value: float | str = fmean(heads)
            probs: dict[str, float] = {}
        else:
            dists = [p for _, p in got if p]
            probs = (
                {label: fmean(d.get(label, 0.0) for d in dists) for label in q.labels()}
                if dists
                else {}
            )
            if q.type == "choice":
                # max keeps the first of equal labels: the file lists "nothing here" first
                value = max(q.labels(), key=lambda label: probs[label])
                heads = [p.get(str(value), 0.0) for _, p in got]
            else:
                value = fmean(heads)
        margin: float | None = None
        if q.type == "noul":
            agreed = sum((h >= 0.5) == (float(value) >= 0.5) for h in heads)
        elif q.type == "choice":
            ranked = sorted(probs.values(), reverse=True)
            margin = ranked[0] - ranked[1]
            agreed = sum(max(q.labels(), key=lambda label: p[label]) == value for _, p in got)
        else:
            # half up, not Python's half-to-even, so 0.5 and 1.5 round alike
            agreed = sum(math.floor(h + 0.5) == math.floor(float(value) + 0.5) for h in heads)
        balanced = {i for i, _ in answered} == set(range(q.rounds()))
        out[q.key] = Answer(
            q.key,
            q.type,
            value,
            probs,
            max(heads) - min(heads),
            len(got),
            margin,
            agreed,
            balanced,
            _first_lift(q, answered) if balanced else None,
        )
    return out


def _first_lift(
    q: Question, answered: list[tuple[int, tuple[float, dict[str, float]]]]
) -> float | None:
    """How far answers lean toward whichever option was listed first (see Answer)."""
    if q.type == "choice":
        # with no probabilities given, a round counts its pick as 1.0, so this
        # is the rate at which the first-listed label was picked, less 1/n
        shown_first = fmean(p[q.first(i)] for i, (_, p) in answered)
        fair = fmean(fmean(p.values()) for _, (_, p) in answered)
        return shown_first - fair
    if q.type == "score":
        reversed_rounds = [h for i, (h, _) in answered if i % 2]
        file_order = [h for i, (h, _) in answered if not i % 2]
        return (fmean(reversed_rounds) - fmean(file_order)) / 2
    return None


def ask(
    state: str,
    questions: QuestionSet,
    *,
    model: str | None = None,
    rounds: int | None = None,
    session_id: str = "",
    post: Post = _post,
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    """Ask every question about `state` over balanced rounds and average. Never raises.

    `rounds` caps the round count (default: as many as the largest question
    needs); questions needing more rounds than the cap come back unbalanced.
    """
    model = model or os.environ.get("JEV_MODEL") or DEFAULT_MODEL
    result = Result(questions.version, model)
    key = os.environ.get("OPENROUTER_API_KEY") or ""
    if not key:
        result.error = "Jev skipped: OPENROUTER_API_KEY not set"
        return result
    if len(state) > MAX_STATE_CHARS:
        state = state[:MAX_STATE_CHARS]
        result.notes.append(f"state cut to {MAX_STATE_CHARS} characters")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    responses: list[Mapping[str, Any]] = []
    answered: list[int] = []
    started = time.monotonic()
    planned = questions.rounds() if rounds is None else min(questions.rounds(), rounds)
    for i in range(planned):
        body: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": questions.to_request(i),
        }
        if session_id:
            body["session_id"] = session_id[:256]
        data = json.dumps(body).encode()
        for attempt in range(ATTEMPTS):
            status, text = _request(post, headers, data)
            if (status in RETRY_CODES or status == 0) and attempt + 1 < ATTEMPTS:
                sleep(2.0 * (attempt + 1))
                continue
            break
        parsed = _parse(status, text)
        if isinstance(parsed, str):
            log.warning("Jev request %d failed (%s)", i + 1, parsed)
            result.errors.append(parsed)
            # a bad request, key or credit fails the same way again, and a
            # service still down after its retry would only spend the run's time
            if status in FINAL_CODES or status == 0 or status >= 500:
                result.down = status in {0, 401, 402, 403} or status >= 500
                break
            continue
        # an answer to a question this round did not ask would unbalance it
        asked = {k: v for k, v in parsed["answers"].items() if k in body["questions"]}
        responses.append(questions.unflip(asked, i))
        answered.append(i)
        model_name = parsed.get("model")
        if isinstance(model_name, str) and MODEL_NAME.fullmatch(model_name):
            result.resolved_model = model_name
        usage = parsed.get("usage")
        if isinstance(usage, dict):
            result.input_tokens += int(_finite(usage.get("input_tokens")) or 0)
            result.output_tokens += int(_finite(usage.get("output_tokens")) or 0)
            result.cost += _finite(usage.get("cost")) or 0.0
    result.duration_ms = int((time.monotonic() - started) * 1000)
    result.rounds = len(responses)
    result.answered = answered
    try:
        result.answers = average(questions, responses, answered)
    except Exception as exc:  # the never-raises backstop: a reply nothing above foresaw
        result.errors.append(f"unreadable response ({type(exc).__name__})")
        result.answers = {}
    if not result.answers:
        result.error = "Jev gave no usable answer" + (
            f" ({result.errors[-1]})" if result.errors else ""
        )
    return result


def _request(post: Post, headers: dict[str, str], data: bytes) -> tuple[int, str]:
    try:
        return post(URL, headers, data)
    except Exception as exc:  # timeout, DNS, reset: a failed request, not a crash
        return 0, type(exc).__name__


def _parse(status: int, text: str) -> dict[str, Any] | str:
    """The response, or a short error naming only the status (never the body)."""
    if status == 0:
        return f"request error {text}"
    if status != 200:
        return f"HTTP {status}"
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        return "unreadable response"
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), dict):
        return "response without answers"
    return parsed
