"""FinanceBench, pinned: the questions, document information and the PDFs they need.

Everything comes from the dataset repo at one commit and is checked against
config/financebench.json, which holds a sha256 for each file and nothing from
the dataset itself (it is CC BY-NC 4.0, per the Hugging Face card
PatronusAI/financebench; the GitHub repo carries no licence file). Files land
in the gitignored data/ and are never committed. A file already on disk with
the right hash is not fetched again.

    uv run python -m tenq.financebench [--root <dir>]   (default: data/financebench)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tenq.config import REPO_ROOT

log = logging.getLogger(__name__)

REPO = "patronus-ai/financebench"
MANIFEST = REPO_ROOT / "config" / "financebench.json"
# tied to the repo root, not the working directory: .gitignore covers /data/ only there
DEFAULT_ROOT = REPO_ROOT / "data" / "financebench"
QUESTIONS = "data/financebench_open_source.jsonl"
DOCUMENTS = "data/financebench_document_information.jsonl"
# names such as 3M_2018_10K or AMCOR_2023Q4_EARNINGS; nothing that could leave pdfs/
DOC_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,79}")
SHA256 = re.compile(r"[0-9a-f]{64}")
FETCH_TIMEOUT_S = 120

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Evidence:
    doc_name: str
    page: int  # zero-indexed, as the dataset gives it
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    company: str
    doc_name: str
    question_type: str
    question_reasoning: str | None
    question: str
    answer: str
    justification: str  # empty for about a third of the questions
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Document:
    doc_name: str
    company: str
    doc_type: str
    doc_period: int


@dataclass(frozen=True)
class Manifest:
    commit: str
    files: dict[str, str]  # path in the dataset repo -> sha256

    @property
    def pdfs(self) -> set[str]:
        return {pdf_doc(p) for p in self.files if p.startswith("pdfs/")}


def pdf_path(doc_name: str) -> str:
    if not DOC_NAME.fullmatch(doc_name):
        raise ValueError(f"not a FinanceBench document name: {doc_name!r}")
    return f"pdfs/{doc_name}.pdf"


def pdf_doc(path: str) -> str:
    name = path.removeprefix("pdfs/").removesuffix(".pdf")
    if pdf_path(name) != path:
        raise ValueError(f"not a FinanceBench pdf path: {path!r}")
    return name


def load_manifest(path: Path = MANIFEST) -> Manifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # only a commit and hashes: the dataset's own text must never be committed
    if not isinstance(raw, dict) or set(raw) != {"commit", "files"}:
        raise ValueError("manifest must hold exactly commit and files")
    commit, files = raw["commit"], raw["files"]
    if not isinstance(commit, str) or not isinstance(files, dict):
        raise ValueError("manifest is malformed")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"manifest commit is not a full sha: {commit!r}")
    for name, digest in files.items():
        if name not in (QUESTIONS, DOCUMENTS):
            pdf_doc(name)
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError(f"manifest sha256 for {name} is malformed")
    if QUESTIONS not in files or DOCUMENTS not in files:
        raise ValueError("manifest must pin the questions and document information")
    return Manifest(commit, dict(files))


def raw_url(commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}"


def http_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"user-agent": "tenq"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        body: bytes = resp.read()
    return body


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _on_disk(path: Path, digest: str) -> bool:
    return path.is_file() and _sha256(path.read_bytes()) == digest


def ensure(root: Path, manifest: Manifest, name: str, fetch: Fetch) -> tuple[Path, bool]:
    """The file at `root/name`, fetched if absent or wrong; True if it was fetched.
    A download whose hash does not match is never written."""
    digest = manifest.files[name]
    path = root / name
    if _on_disk(path, digest):
        return path, False
    data = fetch(raw_url(manifest.commit, name))
    if _sha256(data) != digest:
        raise ValueError(f"{name}: sha256 does not match the manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path, True


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def parse_questions(rows: list[dict[str, Any]]) -> list[Question]:
    questions = []
    for r in rows:
        evidence = tuple(
            Evidence(e["doc_name"], int(e["evidence_page_num"]), e["evidence_text"])
            for e in r["evidence"]
        )
        for name in {r["doc_name"], *(e.doc_name for e in evidence)}:
            pdf_path(name)
        questions.append(
            Question(
                id=r["financebench_id"],
                company=r["company"],
                doc_name=r["doc_name"],
                question_type=r["question_type"],
                question_reasoning=r.get("question_reasoning"),
                question=r["question"],
                answer=r["answer"],
                justification=r.get("justification") or "",
                evidence=evidence,
            )
        )
    return questions


def parse_documents(rows: list[dict[str, Any]]) -> dict[str, Document]:
    return {
        r["doc_name"]: Document(r["doc_name"], r["company"], r["doc_type"], int(r["doc_period"]))
        for r in rows
    }


def needed_docs(questions: list[Question]) -> set[str]:
    return {q.doc_name for q in questions} | {e.doc_name for q in questions for e in q.evidence}


@dataclass(frozen=True)
class Dataset:
    root: Path
    questions: list[Question]
    documents: dict[str, Document]  # only the ones a question uses

    def pdf(self, doc_name: str) -> Path:
        return self.root / pdf_path(doc_name)


def download(root: Path, manifest: Manifest | None = None, fetch: Fetch = http_fetch) -> Dataset:
    """Fetch what is missing and check every file; raise on any mismatch."""
    manifest = manifest or load_manifest()
    fetched = 0
    paths = {}
    for name in (QUESTIONS, DOCUMENTS):
        paths[name], new = ensure(root, manifest, name, fetch)
        fetched += new
    questions = parse_questions(_jsonl(paths[QUESTIONS]))
    docs = needed_docs(questions)
    if docs != manifest.pdfs:
        missing, extra = sorted(docs - manifest.pdfs), sorted(manifest.pdfs - docs)
        raise ValueError(
            f"manifest pdfs differ from the questions' (missing {missing}, extra {extra})"
        )
    all_docs = parse_documents(_jsonl(paths[DOCUMENTS]))
    unknown = sorted(docs - all_docs.keys())
    if unknown:
        raise ValueError(f"questions use documents with no information row: {unknown}")
    for doc in sorted(docs):
        fetched += ensure(root, manifest, pdf_path(doc), fetch)[1]
    total = len(docs) + 2
    log.info("financebench: %s file(s) fetched, %s already on disk", fetched, total - fetched)
    return Dataset(root, questions, {d: all_docs[d] for d in sorted(docs)})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ds = download(args.root)
    print(f"{len(ds.questions)} questions on {len(ds.documents)} documents under {args.root}")


if __name__ == "__main__":
    main()
