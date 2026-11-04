"""FinanceBench documents as text: page markers, passage ids "<doc_name>:<line>".

Layout under a root (default data/financebench/docs, gitignored with the PDFs):

    <doc_name>/text/<doc_name>.txt   page-marked text; one folder per document,
                                     so each document gets its own xerj index
    <doc_name>/meta.json             the PDF's sha256, the extractor, page count,
                                     each page marker's line, pages with no text

Each page starts with a marker line "[[page N]]", N zero-indexed as FinanceBench
numbers its evidence pages, with a blank line on both sides. Passages break at
blank lines, so no passage crosses a page, and a passage's page is the last
marker above it. Marker lines are not passages. Text depends on the extractor,
so a document is extracted again when its PDF or the extractor version changes.

    uv run python -m tenq.docstore     extract every document a question uses
                                       and report pages, time and evidence pages
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tenq.filings import Passage, normalise_text, split_passages, xerj_prefix
from tenq.financebench import DEFAULT_ROOT, DOC_NAME, Dataset, Question, download

PAGE_MARKER = "[[page {n}]]"
DOC_PASSAGE_ID = re.compile(rf"^(?P<doc>{DOC_NAME.pattern}):(?P<line>[0-9]+)$", re.ASCII)
DEFAULT_DOCS = DEFAULT_ROOT / "docs"
EVIDENCE_GRAM = 4
EVIDENCE_HIT = 0.5  # share of the evidence text's word 4-grams (or figures) found on the page
FIGURE = re.compile(r"[0-9][0-9,.]*[0-9]")
MIN_FIGURES = 5  # below this a passage is prose, and its words decide
TOKEN = re.compile(r"[a-z0-9]+")


def extractor_version() -> str:
    import pdfplumber

    return f"pdfplumber {pdfplumber.__version__}"


@dataclass(frozen=True)
class DocMeta:
    doc_name: str
    pdf_sha256: str
    extractor: str
    page_lines: list[int]  # 1-based line of each page's marker, page 0 first
    empty_pages: list[int]
    seconds: float

    @property
    def pages(self) -> int:
        return len(self.page_lines)


def page_marked_text(pages: list[str]) -> tuple[str, list[int], list[int]]:
    """The document text, each page marker's line, and the pages with no text."""
    lines: list[str] = []
    starts: list[int] = []
    empty: list[int] = []
    for n, raw in enumerate(pages):
        if lines:
            lines.append("")
        starts.append(len(lines) + 1)
        lines.append(PAGE_MARKER.format(n=n))
        body = normalise_text(raw).strip("\n")
        if body:
            lines += ["", *body.split("\n")]
        else:
            empty.append(n)
    return "\n".join(lines) + "\n", starts, empty


def extract_pages(pdf: Path) -> tuple[list[str], float]:
    """Each page's text, in order, and how long extraction took."""
    import pdfplumber

    start = time.monotonic()
    out: list[str] = []
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            out.append(page.extract_text() or "")
            page.close()  # pdfplumber caches every parsed page otherwise
    return out, time.monotonic() - start


class DocStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text_dir(self, doc: str) -> Path:
        if not DOC_NAME.fullmatch(doc):
            raise ValueError(f"not a FinanceBench document name: {doc!r}")
        return self.root / doc / "text"

    def index_prefix(self, doc: str) -> str:
        if not DOC_NAME.fullmatch(doc):
            raise ValueError(f"not a FinanceBench document name: {doc!r}")
        return xerj_prefix(doc, "d")

    def owns(self, key: str, doc: str) -> bool:
        return doc == key  # one index per document: a hit belongs to it or to nothing

    def meta(self, doc: str) -> DocMeta | None:
        if not DOC_NAME.fullmatch(doc):
            return None
        path = self.root / doc / "meta.json"
        if not path.is_file():
            return None
        return DocMeta(**json.loads(path.read_text(encoding="utf-8")))

    def has(self, doc: str, pdf_sha256: str) -> bool:
        """Saved from this exact PDF by this extractor version."""
        m = self.meta(doc)
        return m is not None and m.pdf_sha256 == pdf_sha256 and m.extractor == extractor_version()

    def save(self, doc: str, pdf_sha256: str, pages: list[str], seconds: float) -> DocMeta:
        text, starts, empty = page_marked_text(pages)
        meta = DocMeta(doc, pdf_sha256, extractor_version(), starts, empty, round(seconds, 2))
        self.text_dir(doc).mkdir(parents=True, exist_ok=True)
        (self.text_dir(doc) / f"{doc}.txt").write_text(text, encoding="utf-8")
        # meta last: has() keys on it, so a crash mid-save leaves no half document
        (self.root / doc / "meta.json").write_text(json.dumps(asdict(meta), indent=1) + "\n")
        return meta

    def _text(self, doc: str) -> tuple[DocMeta, str] | None:
        """A saved document's meta and text, or None. `doc` comes from the model
        or a search hit, so the resolved path must stay inside the store."""
        m = self.meta(doc)
        if m is None:
            return None
        path = (self.text_dir(doc) / f"{doc}.txt").resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            return None
        return m, path.read_text(encoding="utf-8")

    def passages(self, doc: str) -> list[Passage]:
        saved = self._text(doc)
        if saved is None:
            return []
        m, text = saved
        markers = set(m.page_lines)
        return [p for p in split_passages(doc, text) if p.line not in markers]

    def page_of(self, doc: str, line: int) -> int | None:
        """The zero-indexed page a line falls on."""
        m = self.meta(doc)
        if m is None or not m.page_lines or line < m.page_lines[0]:
            return None
        return bisect.bisect_right(m.page_lines, line) - 1

    def page_text(self, doc: str, page: int) -> str:
        return "\n".join(p.text for p in self.passages(doc) if self.page_of(doc, p.line) == page)

    def passage(self, passage_id: str) -> Passage | None:
        """The passage with this exact id, or None; ids come from the model."""
        m = DOC_PASSAGE_ID.match(passage_id.strip())
        if not m:
            return None
        return next((p for p in self.passages(m["doc"]) if p.id == passage_id.strip()), None)

    def passages_between(self, doc: str, first: int, last: int) -> list[Passage]:
        """Passages overlapping lines first..last (inclusive), in file order."""
        return [
            p
            for p in self.passages(doc)
            if p.line <= last and first < p.line + p.text.count("\n") + 1
        ]


def _grams(text: str) -> set[tuple[str, ...]]:
    words = TOKEN.findall(text.lower())
    return {tuple(words[i : i + EVIDENCE_GRAM]) for i in range(len(words) - EVIDENCE_GRAM + 1)}


def _figures(text: str) -> set[str]:
    return {f.replace(",", "") for f in FIGURE.findall(text) if sum(c.isdigit() for c in f) >= 3}


def evidence_share(evidence: str, page: str) -> float:
    """Share of the evidence text's word 4-grams that appear on the page, or of
    its figures when it has at least MIN_FIGURES, whichever is higher. Word
    grams, not exact text: FinanceBench's own extraction splits lines and words
    differently ("Cash Flow s"), and in tables runs words together
    ("Cashandcashequivalents"), where only the figures still line up."""
    want = _grams(evidence)
    words = len(want & _grams(page)) / len(want) if want else 0.0
    figures = _figures(evidence)
    if len(figures) < MIN_FIGURES:
        return words
    return max(words, len(figures & _figures(page)) / len(figures))


@dataclass(frozen=True)
class EvidenceCheck:
    question_id: str
    doc_name: str
    page: int
    share: float
    best_page: int
    best_share: float

    @property
    def hit(self) -> bool:
        return self.share >= EVIDENCE_HIT


def check_evidence(store: DocStore, questions: Iterable[Question]) -> list[EvidenceCheck]:
    """For every gold evidence item, how much of it the extracted gold page holds,
    and which page holds the most (a miss on a neighbouring page is an offset)."""
    pages: dict[str, list[str]] = {}
    out = []
    for q in questions:
        for e in q.evidence:
            if e.doc_name not in pages:
                m = store.meta(e.doc_name)
                n = m.pages if m else 0
                pages[e.doc_name] = [store.page_text(e.doc_name, i) for i in range(n)]
            texts = pages[e.doc_name]
            shares = [evidence_share(e.text, t) for t in texts]
            share = shares[e.page] if 0 <= e.page < len(shares) else 0.0
            best = max(range(len(shares)), key=lambda i: shares[i], default=-1)
            top = shares[best] if shares else 0.0
            out.append(EvidenceCheck(q.id, e.doc_name, e.page, share, best, top))
    return out


def _extract(pdf: Path) -> tuple[list[str], float]:
    return extract_pages(pdf)


def extract_all(ds: Dataset, store: DocStore, shas: dict[str, str], jobs: int) -> list[DocMeta]:
    """Extract every document not already saved from the same PDF; return all metas."""
    todo = [d for d in ds.documents if not store.has(d, shas[d])]
    if todo:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for doc, (pages, seconds) in zip(
                todo, pool.map(_extract, [ds.pdf(d) for d in todo]), strict=True
            ):
                meta = store.save(doc, shas[doc], pages, seconds)
                print(f"extracted {doc}: {meta.pages} pages in {seconds:.1f}s", flush=True)
    metas = [store.meta(d) for d in ds.documents]
    return [m for m in metas if m is not None]


def report(metas: list[DocMeta], checks: list[EvidenceCheck]) -> dict[str, Any]:
    """Counts, page numbers and document names only: no dataset text, so the
    report can be pasted into a public PR."""
    misses = [c for c in checks if not c.hit]
    return {
        "documents": len(metas),
        "pages": sum(m.pages for m in metas),
        "extract_seconds": round(sum(m.seconds for m in metas), 1),
        "per_document": [
            {"doc": m.doc_name, "pages": m.pages, "seconds": m.seconds, "empty": m.empty_pages}
            for m in metas
        ],
        "evidence_items": len(checks),
        "evidence_on_gold_page": len(checks) - len(misses),
        "evidence_misses": [
            {
                "question": c.question_id,
                "doc": c.doc_name,
                "gold_page": c.page,
                "share": round(c.share, 2),
                "best_page": c.best_page,
                "best_share": round(c.best_share, 2),
            }
            for c in misses
        ],
    }


def main(argv: list[str] | None = None) -> None:
    from tenq.financebench import load_manifest, pdf_path

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="FinanceBench download")
    ap.add_argument("--docs", type=Path, default=None, help="default: <root>/docs")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args(argv)
    manifest = load_manifest()
    ds = download(args.root, manifest)
    store = DocStore(args.docs or args.root / "docs")
    shas = {d: manifest.files[pdf_path(d)] for d in ds.documents}
    metas = extract_all(ds, store, shas, args.jobs)
    rep = report(metas, check_evidence(store, ds.questions))
    if args.report:
        args.report.write_text(json.dumps(rep, indent=1) + "\n")
    no_text = sum(len(m.empty_pages) for m in metas)
    print(
        f"{rep['documents']} documents, {rep['pages']} pages ({no_text} with no text), "
        f"{rep['extract_seconds']}s extracting; gold evidence found on its page for "
        f"{rep['evidence_on_gold_page']} of {rep['evidence_items']} items"
    )
    for miss in rep["evidence_misses"]:
        print("miss", json.dumps(miss))


if __name__ == "__main__":
    main()
