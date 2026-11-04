"""Text search over saved text with xerj: lexical (BM25) passages with file:line.

xerj runs as a local node on loopback for the length of a job. `xerj autoindex`
indexes one text directory (a ticker's filings, or one FinanceBench document);
a query is a BM25 `match` on the chunk text. xerj picks its own schema per
corpus (the text field is `body` for some, `text` for others), so the field is
read from the index mapping. xerj chunks files its own way,
so each hit's file and start line is mapped back to the tenq passage that
contains it: the ids a brief or an answer cites are always tenq's, never xerj's.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tenq.env import MODEL_ENV, PRIVATE_ENV, scrubbed
from tenq.filings import FilingStore, Passage

START_TIMEOUT_S = 30
INDEX_TIMEOUT_S = 600
WORD = re.compile(r"[a-z0-9]+")
# autoindex exit codes that mean "indexed": 3 is done with some junk records skipped
INDEX_OK = {0, 3}
STEM_CHARS = 5
TEXT_FIELDS = ("body", "text")  # xerj names the chunk text either way
# xerj asks agents to file usage reports; tenq never sends any
XERJ_ENV = {"XERJ_DISABLE_FEEDBACK": "true"}


@dataclass(frozen=True)
class SearchHit:
    passage_id: str
    score: float
    text: str


class Corpus(Protocol):
    """Saved text that xerj can index under one key: a ticker or a document."""

    def text_dir(self, key: str) -> Path: ...
    def index_prefix(self, key: str) -> str: ...
    def owns(self, key: str, doc: str) -> bool: ...
    def passages_between(self, doc: str, first: int, last: int) -> list[Passage]: ...


def _request(url: str, body: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # loopback only
        return json.loads(resp.read())


def _stems(text: str) -> set[str]:
    # crude stand-in for the index's stemming: "revised" and "revision" agree
    return {w[:STEM_CHARS] for w in WORD.findall(text.lower()) if len(w) >= 3}


def hits_from_response(
    resp: dict[str, Any], store: Corpus, query: str, k: int, key: str | None = None
) -> list[SearchHit]:
    """Map raw xerj hits to tenq passages, best first, one hit per passage.

    A xerj chunk can span several tenq passages (a heading and the paragraph
    under it), so of the passages the chunk covers, the one sharing the most
    query words wins; ties go to the earlier one. With `key`, hits from text
    the store does not file under that key (another company's filings) are dropped.
    """
    want = _stems(query)
    out: list[SearchHit] = []
    seen: set[str] = set()
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        path, line, end = src.get("ax_path"), src.get("start_line"), src.get("end_line")
        body = next((src[f] for f in TEXT_FIELDS if isinstance(src.get(f), str)), "")
        if not isinstance(path, str) or line is None:
            continue
        first = int(line)
        last = first + body.count("\n")
        if isinstance(end, int) and end >= first:
            last = end
        doc = Path(path).stem
        if key is not None and not store.owns(key, doc):
            continue
        covered = store.passages_between(doc, first, last)
        if not covered:
            continue
        # pick first, dedupe after: a chunk whose best passage was already
        # returned is a repeat, not a reason to return a neighbour without the words
        best = max(covered, key=lambda p: (len(want & _stems(p.text)), -p.line))
        if best.id in seen:
            continue
        seen.add(best.id)
        out.append(SearchHit(best.id, float(h.get("_score") or 0.0), best.text))
        if len(out) == k:
            break
    return out


def text_field(mapping: dict[str, Any]) -> str:
    """The searchable text field of an index pattern's mapping response."""
    found = {
        f
        for index in mapping.values()
        if isinstance(index, dict)
        for f in index.get("mappings", {}).get("properties", {})
    }
    for f in TEXT_FIELDS:
        if f in found:
            return f
    raise RuntimeError(f"no text field among {sorted(found)}")


class Xerj:
    """Client for a running xerj node."""

    def __init__(self, url: str, store: Corpus) -> None:
        self.url = url.rstrip("/")
        self.store = store
        self._fields: dict[str, str] = {}

    def index(self, key: str, state_dir: Path) -> None:
        cmd = [
            "xerj", "autoindex", str(self.store.text_dir(key)),
            "--prefix", self.store.index_prefix(key),
            "--url", self.url,
            "--state-dir", str(state_dir),
            # pure BM25, no link graph, and never stop to ask about a time estimate
            "--no-semantic", "--no-graph", "--max-minutes", "0",
        ]  # fmt: skip
        done = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=INDEX_TIMEOUT_S,
            env={**scrubbed(PRIVATE_ENV + MODEL_ENV), **XERJ_ENV},
        )
        if done.returncode not in INDEX_OK:
            tail = (done.stderr or done.stdout).strip().splitlines()[-5:]
            raise RuntimeError(f"xerj autoindex failed ({done.returncode}): " + " | ".join(tail))

    def search(self, key: str, query: str, k: int = 5) -> list[SearchHit]:
        # over-fetch: several xerj chunks can fall in one tenq passage
        pattern = f"{self.url}/{self.store.index_prefix(key)}-*"
        if key not in self._fields:
            try:
                mapping = _request(f"{pattern}/_mapping")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return []  # nothing indexed under this key (yet): no hits, nothing cached
                raise
            self._fields[key] = text_field(mapping)
        body = {"size": k * 4, "query": {"match": {self._fields[key]: query}}}
        resp = _request(f"{pattern}/_search", body)
        return hits_from_response(resp, self.store, query, k, key)


def _taken(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port() -> int:
    """A loopback port with the two after it also free (xerj claims P, P+1, P+2)."""
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])
        if port < 65000 and not any(_taken(port + i) for i in range(3)):
            return port
    raise RuntimeError("no free port for xerj")


@contextmanager
def xerj_node(port: int | None = None) -> Iterator[str]:
    """Start a throwaway xerj node on loopback; yield its URL; stop it after.

    Refuses a port something else already answers on, so a run never indexes
    into or searches a node it did not start.
    """
    if shutil.which("xerj") is None:
        raise RuntimeError("xerj not found on PATH (scripts/install-xerj.sh installs it)")
    if port is None:
        port = free_port()
    elif _taken(port):
        raise RuntimeError(f"port {port} already in use")
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="tenq-xerj-") as data_dir:
        proc = subprocess.Popen(
            ["xerj", "--insecure", "--data-dir", data_dir, "--bind", "127.0.0.1",
             "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**scrubbed(PRIVATE_ENV + MODEL_ENV), **XERJ_ENV},
        )  # fmt: skip
        try:
            deadline = time.monotonic() + START_TIMEOUT_S
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"xerj exited with {proc.returncode} on start")
                try:
                    _request(url, timeout=2)
                    if proc.poll() is not None:
                        raise RuntimeError(f"xerj exited with {proc.returncode} on start")
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("xerj did not start in time") from None
                    time.sleep(0.25)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Index a ticker's saved filings and search them.")
    ap.add_argument("ticker")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--root", type=Path, default=Path("data/filings"))
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    store = FilingStore(args.root)
    ticker = args.ticker.upper()
    with xerj_node(args.port) as url, tempfile.TemporaryDirectory() as state:
        xerj = Xerj(url, store)
        xerj.index(ticker, Path(state))
        for hit in xerj.search(ticker, args.query, args.k):
            first = hit.text.split("\n", 1)[0][:100]
            print(f"{hit.passage_id}  {hit.score:.2f}  {first}")


if __name__ == "__main__":
    main()
