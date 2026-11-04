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
import os
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
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
YEAR = re.compile(r"(?:19|20)[0-9]{2}")  # column years are on every page: not evidence
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
        try:
            return DocMeta(**json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError):
            return None  # unreadable: treated as not saved, so it is extracted again

    def has(self, doc: str, pdf_sha256: str) -> bool:
        """Saved from this exact PDF by this extractor version."""
        m = self.meta(doc)
        return m is not None and m.pdf_sha256 == pdf_sha256 and m.extractor == extractor_version()

    def save(self, doc: str, pdf_sha256: str, pages: list[str], seconds: float) -> DocMeta:
        text, starts, empty = page_marked_text(pages)
        meta = DocMeta(doc, pdf_sha256, extractor_version(), starts, empty, round(seconds, 2))
        self.text_dir(doc).mkdir(parents=True, exist_ok=True)
        meta_path = self.root / doc / "meta.json"
        # has() keys on meta: drop the old one first and write the new one last,
        # each file whole or not at all, so a crash never pairs old meta with new text
        meta_path.unlink(missing_ok=True)
        _write_whole(self.text_dir(doc) / f"{doc}.txt", text)
        _write_whole(meta_path, json.dumps(asdict(meta), indent=1) + "\n")
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
        return None if m is None else _page(m, line)

    def page_texts(self, doc: str) -> list[str]:
        """Every page's passage text, page 0 first; one read of the document."""
        saved = self._text(doc)
        if saved is None:
            return []
        m, text = saved
        out: list[list[str]] = [[] for _ in m.page_lines]
        markers = set(m.page_lines)
        for p in split_passages(doc, text):
            page = _page(m, p.line)
            if page is not None and p.line not in markers:
                out[page].append(p.text)
        return ["\n".join(t) for t in out]

    def page_text(self, doc: str, page: int) -> str:
        texts = self.page_texts(doc)
        return texts[page] if 0 <= page < len(texts) else ""

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


def _write_whole(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _page(m: DocMeta, line: int) -> int | None:
    if not m.page_lines or line < m.page_lines[0]:
        return None
    return bisect.bisect_right(m.page_lines, line) - 1


def _grams(text: str) -> set[tuple[str, ...]]:
    words = TOKEN.findall(text.lower())
    return {tuple(words[i : i + EVIDENCE_GRAM]) for i in range(len(words) - EVIDENCE_GRAM + 1)}


def _figures(text: str) -> set[str]:
    return {
        f.replace(",", "")
        for f in FIGURE.findall(text)
        if sum(c.isdigit() for c in f) >= 3 and not YEAR.fullmatch(f)
    }


def evidence_shares(evidence: str, page: str) -> tuple[float, float]:
    """(words, figures): the share of the evidence text's word 4-grams that
    appear on the page, and of its figures (years aside) when it has at least
    MIN_FIGURES, else 0. Word grams, not exact text: FinanceBench's own
    extraction splits lines and words differently ("Cash Flow s"), and in
    tables runs words together ("Cashandcashequivalents"), where only the
    figures still line up."""
    want = _grams(evidence)
    words = len(want & _grams(page)) / len(want) if want else 0.0
    figures = _figures(evidence)
    if len(figures) < MIN_FIGURES:
        return words, 0.0
    return words, len(figures & _figures(page)) / len(figures)


def evidence_share(evidence: str, page: str) -> float:
    return max(evidence_shares(evidence, page))


@dataclass(frozen=True)
class EvidenceCheck:
    question_id: str
    doc_name: str
    page: int
    words: float  # on the gold page
    figures: float
    best_page: int
    best_share: float

    @property
    def share(self) -> float:
        return max(self.words, self.figures)

    @property
    def hit(self) -> bool:
        return self.share >= EVIDENCE_HIT

    @property
    def figures_only(self) -> bool:
        """A hit the words alone would not make: worth a look by hand."""
        return self.hit and self.words < EVIDENCE_HIT


def check_evidence(store: DocStore, questions: Iterable[Question]) -> list[EvidenceCheck]:
    """For every gold evidence item, how much of it the extracted gold page holds,
    and which page holds the most (a miss on a neighbouring page is an offset)."""
    pages: dict[str, list[str]] = {}
    out = []
    for q in questions:
        for e in q.evidence:
            if e.doc_name not in pages:
                pages[e.doc_name] = store.page_texts(e.doc_name)
            each = [evidence_shares(e.text, t) for t in pages[e.doc_name]]
            words, figures = each[e.page] if 0 <= e.page < len(each) else (0.0, 0.0)
            best = max(range(len(each)), key=lambda i: max(each[i]), default=-1)
            top = max(each[best]) if each else 0.0
            out.append(EvidenceCheck(q.id, e.doc_name, e.page, words, figures, best, top))
    return out


def _extract(pdf: Path) -> tuple[list[str], float]:
    return extract_pages(pdf)


Extractor = Callable[[Path], tuple[list[str], float]]


def extract_all(
    ds: Dataset, store: DocStore, shas: dict[str, str], jobs: int, extract: Extractor = _extract
) -> tuple[list[DocMeta], dict[str, str]]:
    """Extract every document not already saved from the same PDF, saving each
    as it finishes. Returns all saved metas and, for documents that failed, the
    error type: one bad PDF degrades the run, it does not stop it."""
    todo = [d for d in ds.documents if not store.has(d, shas[d])]
    failed: dict[str, str] = {}
    if todo:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(extract, ds.pdf(d)): d for d in todo}
            for f in as_completed(futures):
                doc = futures[f]
                try:
                    pages, seconds = f.result()
                except Exception as e:  # any PDF failure is reported, not raised
                    failed[doc] = type(e).__name__
                    print(f"failed {doc}: {failed[doc]}", flush=True)
                    continue
                meta = store.save(doc, shas[doc], pages, seconds)
                print(f"extracted {doc}: {meta.pages} pages in {seconds:.1f}s", flush=True)
    metas = [store.meta(d) for d in ds.documents]
    return [m for m in metas if m is not None], dict(sorted(failed.items()))


def _item(c: EvidenceCheck) -> dict[str, Any]:
    return {
        "question": c.question_id,
        "doc": c.doc_name,
        "gold_page": c.page,
        "words": round(c.words, 2),
        "figures": round(c.figures, 2),
        "best_page": c.best_page,
        "best_share": round(c.best_share, 2),
    }


def report(
    metas: list[DocMeta], checks: list[EvidenceCheck], failed: dict[str, str] | None = None
) -> dict[str, Any]:
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
        "failed": failed or {},
        "evidence_items": len(checks),
        "evidence_on_gold_page": len(checks) - len(misses),
        "evidence_misses": [_item(c) for c in misses],
        "evidence_on_figures_only": [_item(c) for c in checks if c.figures_only],
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
    metas, failed = extract_all(ds, store, shas, args.jobs)
    rep = report(metas, check_evidence(store, ds.questions), failed)
    if args.report:
        args.report.write_text(json.dumps(rep, indent=1) + "\n")
    no_text = sum(len(m.empty_pages) for m in metas)
    print(
        f"{rep['documents']} documents, {rep['pages']} pages ({no_text} with no text), "
        f"{rep['extract_seconds']}s extracting; gold evidence found on its page for "
        f"{rep['evidence_on_gold_page']} of {rep['evidence_items']} items "
        f"({len(rep['evidence_on_figures_only'])} on figures only)"
    )
    for doc, err in rep["failed"].items():
        print("failed", doc, err)
    for miss in rep["evidence_misses"]:
        print("miss", json.dumps(miss))
    for item in rep["evidence_on_figures_only"]:
        print("figures only", json.dumps(item))


if __name__ == "__main__":
    main()
