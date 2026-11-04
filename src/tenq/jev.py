"""Jev: typed answers to narrow questions about one stock's facts pack.

Jev (TypeSafe's decision model, served through OpenRouter's System One API)
returns probabilities, not text. The questions live in prompts/jev.yaml; the
state is the pack's rendered text, whose numbers were all computed in code.
Each request is repeated (REPEATS) and the answers averaged, because Jev's
answers vary between identical calls; the spread across repeats is kept, so
the rules and the scorecard can see how settled an answer was.

Never raises: a missing key, an HTTP error or an unreadable answer is an error
on the result, and the brief goes out with Opus's call alone. A request that
fails for good (a bad key, no credit, a service still down after one retry)
ends the repeats, so an outage costs about one minute per stock, not three. Logs carry only
status codes and exception types, never the state or an answer.

Env: OPENROUTER_API_KEY (the only secret), JEV_MODEL (default jev-latest; the
resolved model, e.g. typesafe/jev-1.13-20260917, is recorded per run).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
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
REPEATS = 3
TIMEOUT_S = 30
ATTEMPTS = 2  # per repeat: one retry on a rate limit, an upstream failure or a timeout
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

    def to_request(self, repeat: int = 0) -> dict[str, Any]:
        """The question as sent on one repeat, with its options moved to cancel position bias.

        Jev leans toward earlier options, so each repeat starts a choice's
        labels at a different place, and odd repeats list a score's levels in
        reverse (`unflip` maps those answers back).
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
            flipped: dict[str, Any] = {}
            for k, v in probs.items():
                if str(k).isascii() and str(k).isdigit() and int(str(k)) <= top:
                    flipped[str(top - int(str(k)))] = v
            out["probabilities"] = flipped
        return out


@dataclass(frozen=True)
class QuestionSet:
    version: str
    questions: tuple[Question, ...]

    def to_request(self, repeat: int = 0) -> dict[str, Any]:
        return {q.key: q.to_request(repeat) for q in self.questions}

    def unflip(self, answers: Mapping[str, Any], repeat: int) -> dict[str, Any]:
        by_key = {q.key: q for q in self.questions}
        return {k: by_key[k].unflip(v, repeat) if k in by_key else v for k, v in answers.items()}


def load_questions(path: Path = QUESTIONS_PATH) -> QuestionSet:
    """Validate the question file; a malformed one is a ValueError, never a guess."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    return QuestionSet(version, tuple(out))


@dataclass
class Answer:
    """One question's answer averaged over the repeats that returned it.

    value: noul, the mean probability; choice, the label with the highest mean
    probability; score, the mean level (0 is the first criterion).
    probabilities: choice and score, the mean per label (levels as "0".."n-1").
    spread: max minus min across repeats of the headline number (noul
    probability, choice top-label probability, score level); 0 for one repeat.
    margin: choice only, the winning label's mean probability less the
    runner-up's (how decisive the average is); None for noul and score.
    agreed: repeats whose own answer matches `value` (choice: the same top
    label; score: the same level when rounded half up; noul: the same side
    of 0.5).
    """

    key: str
    type: QType
    value: float | str
    probabilities: dict[str, float]
    spread: float
    repeats: int
    margin: float | None = None
    agreed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass
class Result:
    version: str
    model: str  # requested
    answers: dict[str, Answer] = field(default_factory=dict)
    resolved_model: str = ""
    repeats: int = 0  # requests that returned usable answers
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


def average(questions: QuestionSet, responses: list[Mapping[str, Any]]) -> dict[str, Answer]:
    """Each question's answers across the repeats' `answers` maps, averaged.

    A question missing or unreadable in one repeat is averaged over the rest;
    a question no repeat answered is left out.
    """
    out: dict[str, Answer] = {}
    for q in questions.questions:
        got = [a for a in (_one(q, r.get(q.key)) for r in responses) if a is not None]
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
        out[q.key] = Answer(
            q.key, q.type, value, probs, max(heads) - min(heads), len(got), margin, agreed
        )
    return out


def ask(
    state: str,
    questions: QuestionSet,
    *,
    model: str | None = None,
    repeats: int = REPEATS,
    session_id: str = "",
    post: Post = _post,
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    """Ask every question about `state`, `repeats` times, and average. Never raises."""
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
    started = time.monotonic()
    for i in range(repeats):
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
        responses.append(questions.unflip(parsed["answers"], i))
        model_name = parsed.get("model")
        if isinstance(model_name, str) and MODEL_NAME.fullmatch(model_name):
            result.resolved_model = model_name
        usage = parsed.get("usage")
        if isinstance(usage, dict):
            result.input_tokens += int(_finite(usage.get("input_tokens")) or 0)
            result.output_tokens += int(_finite(usage.get("output_tokens")) or 0)
            result.cost += _finite(usage.get("cost")) or 0.0
    result.duration_ms = int((time.monotonic() - started) * 1000)
    result.repeats = len(responses)
    try:
        result.answers = average(questions, responses)
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
