"""One pipeline run: find new filings, brief each one, check, email, trace, record.

    python -m tenq.run                      # every active ticker: brief any new filing
    python -m tenq.run --backfill 4         # also brief the last 4 filings without a record
    python -m tenq.run --accession <acc>    # brief one filing (again)

A filing counts as briefed once runs/<TICKER>/<accession>.json records a sent
email; a re-run that sends nothing writes <accession>.rerun-<run_id>.json
beside it instead. The daily workflow commits those records, so they are both
the permanent run log and the state that stops a filing being emailed twice.

Every step degrades instead of crashing: no xerj or no Claude gives a
numbers-only email; a failed email or trace is recorded and the exit code is
non-zero only after the record is written.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tenq.brief import Draft, Prompt, load_prompt, user_message, write_brief
from tenq.checker import CheckResult, check
from tenq.config import load_watchlist
from tenq.deliver import Email, render, send
from tenq.edgar import FilingSource
from tenq.fetch import DEFAULT_DEPTH, sync
from tenq.filings import FilingMeta, FilingStore
from tenq.llm import DEFAULT_MODEL, AgentRun
from tenq.metrics import POOL_DEPTH, metrics_for
from tenq.search import Xerj, xerj_node
from tenq.trace import send_trace

log = logging.getLogger(__name__)

Briefer = Callable[[Path, FilingMeta, str, Prompt, str], tuple[AgentRun, Draft]]
NodeFactory = Callable[[], AbstractContextManager[str]]


class Steps:
    """Timed pipeline steps for the trace and the run record."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, start: float, output: Any = None, error: str | None = None) -> None:
        self.items.append(
            {"name": name, "start": start, "end": time.time(), "output": output, "error": error}
        )


def record_path(runs: Path, meta: FilingMeta) -> Path:
    return runs / meta.ticker / f"{meta.accession}.json"


def _claim_dict(c: Any) -> dict[str, Any]:
    return asdict(c)


def brief_filing(
    meta: FilingMeta,
    store: FilingStore,
    runs: Path,
    xerj_url: str | None,
    *,
    prompt: Prompt,
    model: str = DEFAULT_MODEL,
    email: bool = True,
    briefer: Briefer = write_brief,
    sender: Callable[[Email], tuple[bool | None, str]] = send,
    tracer: Callable[[dict[str, Any]], tuple[bool, str]] = send_trace,
    steps: Steps | None = None,
) -> dict[str, Any]:
    """Brief one saved filing; write and return its run record."""
    steps = steps or Steps()
    run_id = str(uuid.uuid4())
    started = time.time()
    table = metrics_for(store, meta.ticker, meta.accession)

    agent: AgentRun | None = None
    draft = Draft(error="search unavailable")
    t = time.time()
    if xerj_url is not None:
        agent, draft = briefer(store.root, meta, xerj_url, prompt, model)
    steps.add("claude", t, {"claims": len(draft.claims)}, draft.error)

    t = time.time()
    result = check(draft.claims, table, store, meta) if draft.error is None else CheckResult()
    steps.add(
        "check",
        t,
        {"kept": len(result.kept), "dropped": [d.reason for d in result.dropped]},
    )
    mode = "brief" if result.kept else "numbers-only"

    message = render(
        meta,
        table,
        result,
        store,
        summary_error=draft.error,
        run_id=run_id,
        prompt_version=prompt.version,
    )
    t = time.time()
    sent, status = sender(message) if email else (False, "email off for this run")
    steps.add("email", t, status, None if sent or not email else status)

    record: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started,
        "ended_at": time.time(),
        "filing": asdict(meta),
        "mode": mode,
        "model": model,
        "prompt_version": prompt.version,
        # the commit checked out, set by the workflow from git right after checkout
        "code": os.environ.get("TENQ_CODE_SHA", ""),
        "prompt_input": user_message(meta),
        "metrics": table.to_dict(),
        "agent": None if agent is None else agent.to_dict(),
        "draft_error": draft.error,
        "check": {
            "kept": [_claim_dict(c) for c in result.kept],
            "dropped": [{**_claim_dict(d.claim), "reason": d.reason} for d in result.dropped],
        },
        "email": {"sent": sent, "status": status, "subject": message.subject},
        "steps": steps.items,
    }
    # written before tracing too: once the email is out, a cancelled job must
    # still leave the record that stops the next run sending it again
    record["trace"] = {"sent": False, "status": "trace pending"}
    path = record_path(runs, meta)
    if not record["email"]["sent"] and briefed(path):
        # a re-run that sent nothing (email off, or SMTP down) must not replace the
        # record of an email that went out, or the next run would send it again
        path = path.with_name(f"{meta.accession}.rerun-{run_id}.json")
    _write(path, record)
    traced, trace_status = tracer(record)
    record["trace"] = {"sent": traced, "status": trace_status}
    _write(path, record)
    log.info("%s %s: %s, %s, %s", meta.ticker, meta.accession, mode, status, trace_status)
    return record


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")


def briefed(path: Path) -> bool:
    """A filing is done once its email went out; a failed or email-off run is retried."""
    try:
        return bool(json.loads(path.read_text(encoding="utf-8"))["email"]["sent"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_ticker(
    ticker: str,
    source: FilingSource,
    store: FilingStore,
    runs: Path,
    *,
    backfill: int = 0,
    accession: str | None = None,
    node: NodeFactory = xerj_node,
    **kw: Any,
) -> list[dict[str, Any]]:
    """Sync a ticker's filings and brief the ones that need it, oldest first."""
    steps = Steps()
    t = time.time()
    # enough history behind the oldest filing briefed for its year-ago numbers
    depth = max(DEFAULT_DEPTH, backfill + POOL_DEPTH - 1)
    new = sync(source, store, ticker, depth)
    steps.add("fetch", t, {"new": [m.accession for m in new]})
    saved = store.filings(ticker)
    if accession is not None:
        targets = [m for m in saved if m.accession == accession]
        if not targets:
            raise LookupError(f"{ticker} {accession} is not among the saved filings")
    else:
        window = saved[: max(1, backfill)]
        targets = [m for m in window if not briefed(record_path(runs, m))]
    if not targets:
        log.info("%s: nothing new", ticker)
        return []

    records = []
    with ExitStack() as stack:
        url: str | None = None
        t = time.time()
        try:
            url = stack.enter_context(node())
            state = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            Xerj(url, store).index(ticker, state)
            steps.add("index", t, {"url": url})
        except Exception as exc:  # no search: briefs go out numbers-only
            log.warning("xerj unavailable: %s", exc)
            steps.add("index", t, None, f"{type(exc).__name__}: {exc}")
            url = None
        for meta in reversed(targets):
            filing_steps = Steps()
            filing_steps.items = list(steps.items)
            records.append(brief_filing(meta, store, runs, url, steps=filing_steps, **kw))
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Brief new watchlist filings.")
    ap.add_argument("--ticker", action="append", help="default: the watchlist's active names")
    ap.add_argument("--backfill", type=int, default=0, help="brief the last N filings too")
    ap.add_argument("--accession", help="brief this one filing, even if it has a record")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--root", type=Path, default=Path("data/filings"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from tenq.edgar import EdgarSource

    tickers = [t.upper() for t in args.ticker] if args.ticker else list(load_watchlist().active)
    if args.accession and len(tickers) != 1:
        ap.error("--accession needs exactly one --ticker")
    source, store, prompt = EdgarSource(), FilingStore(args.root), load_prompt()
    failed = False
    for ticker in tickers:
        try:
            records = run_ticker(
                ticker,
                source,
                store,
                args.runs,
                backfill=args.backfill,
                accession=args.accession,
                prompt=prompt,
                model=args.model,
                email=not args.no_email,
            )
        except Exception:
            log.exception("%s: run failed", ticker)
            failed = True
            continue
        failed |= any(not r["email"]["sent"] for r in records if not args.no_email)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
