"""Grader calibration: tenq's grader against the paper's human grades.

The calibration set is the paper's three singleStore result files: 450 answers
at their natural mix (200 Correct, 114 Incorrect, 136 Refusal). Grading them
also gives the singleStore comparison, grader against grader. If the grader
is revised, it is gated again on a held-out sample of the same mix drawn from
the other 13 files.

Gate: Cohen's kappa at least 0.8 over every item, and agreement at least 90%
on the items people did not grade Refusal. A grader failure counts as a
disagreement, and the gate is decided only once every item is finished.
Intervals are 95%: Wilson for agreement, a percentile bootstrap for kappa.

Records (one JSON line per item, with the grader's reason) are private and
belong in the evals repo, never the public one. The report holds counts and
rates only. The command prints how many items were graded, never how they
agreed, so a public log says nothing about the result.

    uv run python -m tenq.calibrate [--set singlestore|heldout] [--limit N]
        [--records PATH] [--report PATH] [--model opus]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from tenq.brief import Prompt, load_prompt
from tenq.financebench import (
    DEFAULT_ROOT,
    GradedAnswer,
    Question,
    download,
    download_results,
)
from tenq.grade import GRADED, PROMPT_PATH, Grade, grade
from tenq.llm import DEFAULT_MODEL, AgentRun
from tenq.qa import INFRA_FAILURE, MODEL_FAILURE

log = logging.getLogger(__name__)

SINGLE_STORE = ("gpt-4-1106-preview_singleStore", "gpt-4_singleStore", "llama2_singleStore")
SETS = ("singlestore", "heldout")
KAPPA_GATE = 0.8
NON_REFUSAL_GATE = 0.9
Z = 1.959964
BOOT_ROUNDS = 2000
SEED = 13
FAILED = "grader failed"
REFUSAL = "Refusal"
CORRECT = "Correct Answer"
DEFAULT_RECORDS = DEFAULT_ROOT.parent / "evals"

Grader = Callable[..., tuple[AgentRun | None, Grade]]


def calibration_set(results: dict[str, list[GradedAnswer]]) -> list[GradedAnswer]:
    return [a for source in SINGLE_STORE for a in results[source]]


def held_out_set(
    results: dict[str, list[GradedAnswer]], mix: Counter[str] | None = None, seed: int = SEED
) -> list[GradedAnswer]:
    """A sample of the calibration set's label mix from the files it does not use."""
    mix = mix or Counter(a.label for a in calibration_set(results))
    pool: dict[str, list[GradedAnswer]] = {}
    for source in sorted(set(results) - set(SINGLE_STORE)):
        for a in results[source]:
            pool.setdefault(a.label, []).append(a)
    rng = random.Random(seed)
    out = []
    for label in sorted(mix):
        have = sorted(pool.get(label, []), key=lambda a: a.key)
        if len(have) < mix[label]:
            raise ValueError(f"only {len(have)} held-out {label} answers, {mix[label]} wanted")
        out += rng.sample(have, mix[label])
    return out


def run_order(items: Iterable[GradedAnswer]) -> list[GradedAnswer]:
    """A fixed shuffle, so a partial run is a fair sample of the whole set."""
    return sorted(items, key=lambda a: hashlib.sha256(a.key.encode()).hexdigest())


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa; None when chance agreement is total and kappa is undefined."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(h == g for h, g in pairs) / n
    humans, graders = Counter(h for h, _ in pairs), Counter(g for _, g in pairs)
    chance = sum(humans[c] * graders[c] for c in humans) / (n * n)
    if chance >= 1:
        return None
    return (observed - chance) / (1 - chance)


def kappa_interval(
    pairs: list[tuple[str, str]], rounds: int = BOOT_ROUNDS, seed: int = SEED
) -> tuple[float, float] | None:
    if not pairs:
        return None
    rng = random.Random(seed)
    draws = sorted(
        k for k in (kappa(rng.choices(pairs, k=len(pairs))) for _ in range(rounds)) if k is not None
    )
    if not draws:
        return None
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def finished(record: dict[str, Any], prompt_version: str, model: str) -> bool:
    """Graded, or failed by the grader itself, by this prompt version and model.

    A record from another version or model is stale: the item is graded again,
    so a gate is never decided over a mix of graders.
    """
    return (
        record.get("status") in (GRADED, MODEL_FAILURE)
        and record.get("prompt_version") == prompt_version
        and record.get("model") == model
    )


def _append(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line, first ending a line a crash left open."""
    with path.open("a+b") as f:
        if f.tell():
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((json.dumps(record) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """The latest record per item; a line cut short by a crash is skipped."""
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("key"), str):
            out[rec["key"]] = rec
    return out


def _record(
    item: GradedAnswer, run: AgentRun | None, g: Grade, prompt: Prompt, model: str
) -> dict[str, Any]:
    return {
        "key": item.key,
        "source": item.source,
        "question_id": item.question_id,
        "human": item.label,
        **g.to_dict(),
        "prompt_version": prompt.version,
        "model": model,
        "resolved_model": run.resolved_model if run else None,
        "usage": run.usage if run else {},
        "duration_ms": run.duration_ms if run else 0,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def grade_items(
    items: list[GradedAnswer],
    questions: dict[str, Question],
    records: Path,
    prompt: Prompt,
    *,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
    grader: Grader = grade,
) -> tuple[int, bool]:
    """Grade unfinished items in run order, appending a record for each.

    Returns (graded this run, stopped on an infrastructure failure). An
    infrastructure failure is not recorded: that item is graded again later.
    """
    done = {k for k, r in load_records(records).items() if finished(r, prompt.version, model)}
    records.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in run_order(items):
        if item.key in done:
            continue
        if limit is not None and count >= limit:
            break
        # grade against the gold the people saw: the questions file has since
        # reformatted some golds and reversed one
        q = replace(questions[item.question_id], answer=item.gold_answer)
        run, g = grader(q, item.model_answer, prompt, model=model)
        if g.status == INFRA_FAILURE:
            return count, True
        _append(records, _record(item, run, g, prompt, model))
        count += 1
    return count, False


def _rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"agree": k, "n": n, "rate": k / n if n else None, "ci95": list(ci) if ci else None}


def report(
    items: list[GradedAnswer],
    records: dict[str, dict[str, Any]],
    *,
    prompt_version: str,
    model: str,
) -> dict[str, Any]:
    """Agreement with the human grades, counts and rates only, and the gate.

    Only records by this prompt version and model count.
    """
    done = [
        (a, records[a.key])
        for a in items
        if a.key in records and finished(records[a.key], prompt_version, model)
    ]
    pairs = [(a.label, r.get("label") or FAILED) for a, r in done]
    matrix: dict[str, dict[str, int]] = {}
    for h, g in pairs:
        row = matrix.setdefault(h, {})
        row[g] = row.get(g, 0) + 1
    non_refusal = [(h, g) for h, g in pairs if h != REFUSAL]
    per_source: dict[str, dict[str, int]] = {}
    for a, r in done:
        s = per_source.setdefault(a.source, {"n": 0, "human_correct": 0, "grader_correct": 0})
        s["n"] += 1
        s["human_correct"] += a.label == CORRECT
        s["grader_correct"] += r.get("label") == CORRECT
    k = kappa(pairs)
    nr = _rate(sum(h == g for h, g in non_refusal), len(non_refusal))
    complete = len(done) == len(items)
    passed = (
        complete
        and k is not None
        and k >= KAPPA_GATE
        and nr["rate"] is not None
        and nr["rate"] >= NON_REFUSAL_GATE
    )
    ci = kappa_interval(pairs)
    return {
        "items": len(items),
        "finished": len(done),
        "grader_failed": sum(g == FAILED for _, g in pairs),
        "prompt_version": prompt_version,
        "model": model,
        "stale": sum(a.key in records for a in items) - len(done),
        "kappa": k,
        "kappa_ci95": list(ci) if ci else None,
        "agreement": _rate(sum(h == g for h, g in pairs), len(pairs)),
        "non_refusal_agreement": nr,
        "matrix": matrix,
        "per_source": per_source,
        "gate": {
            "kappa_at_least": KAPPA_GATE,
            "non_refusal_agreement_at_least": NON_REFUSAL_GATE,
            "decided": complete,
            "passed": passed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", choices=SETS, default="singlestore")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        records = args.records or DEFAULT_RECORDS / f"calibration-{args.set}.jsonl"
        ds = download(args.root)
        results = download_results(args.root)
        items = calibration_set(results) if args.set == "singlestore" else held_out_set(results)
        questions = {q.id: q for q in ds.questions}
        prompt = load_prompt(PROMPT_PATH)
        graded, stopped = grade_items(
            items,
            questions,
            records,
            prompt,
            model=args.model,
            limit=args.limit,
            grader=grade,  # looked up here, so a test can stand in for the model
        )
        rep = report(items, load_records(records), prompt_version=prompt.version, model=args.model)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    except Exception as e:  # the message may quote model or dataset text
        log.error("calibration failed: %s", type(e).__name__)
        return 1
    print(
        f"calibration {args.set}: {graded} graded this run, {rep['finished']} of "
        f"{rep['items']} finished" + (", stopped on an infrastructure failure" if stopped else "")
    )
    return 2 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
