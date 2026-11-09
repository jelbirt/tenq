"""Saved filings on disk: text with stable passage ids, tagged facts, metadata.

Layout under a data root (default ./data/filings):

    <TICKER>/<accession>/meta.json   form, dates, company
    <TICKER>/<accession>/facts.json  undimensioned numeric XBRL facts
    <TICKER>/text/<doc>.txt          normalised filing text, one file per filing

Text files sit together in one folder per ticker so a search index over that
folder returns paths that name the filing. A passage is a run of non-blank
lines (capped at MAX_PASSAGE_LINES); its id is "<doc>:<first line>", 1-based,
so an id is also a file:line a reader can open.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_PASSAGE_LINES = 40
TICKER = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}")
INDEX_WORD = re.compile(r"[a-z0-9]+")
PASSAGE_ID = re.compile(
    r"^(?P<doc>[A-Z][A-Z0-9.\-]*-[0-9]{10}-[0-9]{2}-[0-9]{6}):(?P<line>[0-9]+)$", re.ASCII
)


@dataclass(frozen=True)
class FilingMeta:
    ticker: str
    cik: int
    company: str
    form: str  # "10-Q", "10-K", "8-K" or "6-K"
    accession: str  # "0001535527-26-000031"
    filed: str  # ISO date the filing was accepted
    period: str  # ISO date the reporting period ends (an 8-K: the event date)
    url: str
    items: str = ""  # an 8-K's item numbers as EDGAR lists them, e.g. "2.02,9.01"; a 6-K has none
    accepted: str = ""  # EDGAR's acceptance time, ISO with offset; "" when not known

    @property
    def doc(self) -> str:
        """The text file stem and passage-id prefix, e.g. CRWD-0001535527-26-000031."""
        return f"{self.ticker}-{self.accession}"


@dataclass(frozen=True)
class Fact:
    concept: str  # "us-gaap:NetCashProvidedByUsedInOperatingActivities"
    value: float
    unit: str
    start: str | None  # duration facts carry start and end
    end: str | None
    instant: str | None  # instant (balance sheet) facts carry only this
    decimals: int | None
    # ISO 4217 code of a money unit ("USD", "EUR"), from the filing's own unit
    # definition; a per-share unit carries its numerator's code ("USD" for USD per
    # share). None for shares, ratios, or facts saved before this field existed.
    currency: str | None = None


@dataclass(frozen=True)
class Passage:
    id: str
    doc: str
    line: int  # first line, 1-based
    text: str


def normalise_text(text: str) -> str:
    """Deterministic line layout: trailing spaces stripped, blank runs collapsed.

    The same filing must always produce the same lines, or passage ids saved in
    one run would point at different text in the next.
    """
    out: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.rstrip()
        if not line and (not out or not out[-1]):
            continue
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def xerj_prefix(name: str, tag: str) -> str:
    """xerj's own prefix form ([a-z0-9] runs joined by "-"), ended with "-<tag>"
    so `<prefix>-*` for BRK can never match BRK.B's `tenq-brk-b-t-*` indices,
    and a ticker's indices never match a document's."""
    return "tenq-" + "-".join(INDEX_WORD.findall(name.lower())) + "-" + tag


def index_prefix(ticker: str, tag: str = "t") -> str:
    if not TICKER.fullmatch(ticker):
        raise ValueError(f"not a ticker: {ticker!r}")
    return xerj_prefix(ticker, tag)


def split_passages(doc: str, text: str) -> list[Passage]:
    passages: list[Passage] = []
    block: list[str] = []
    start = 0

    def flush() -> None:
        if block:
            passages.append(Passage(f"{doc}:{start}", doc, start, "\n".join(block)))
            block.clear()

    for n, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            flush()
            continue
        if not block or len(block) >= MAX_PASSAGE_LINES:
            flush()
            start = n
        block.append(line)
    flush()
    return passages


class FilingStore:
    def __init__(self, root: Path, tag: str = "t") -> None:
        self.root = root
        self.tag = tag  # ends the store's xerj index prefix; "t" for 10-Q/10-K

    def text_dir(self, ticker: str) -> Path:
        return self.root / ticker / "text"

    def index_prefix(self, ticker: str) -> str:
        return index_prefix(ticker, self.tag)

    def owns(self, ticker: str, doc: str) -> bool:
        """Whether a text file stem found by search is one of this ticker's filings."""
        return doc.rsplit("-", 3)[0] == ticker

    def _dir(self, ticker: str, accession: str) -> Path:
        return self.root / ticker / accession

    def has(self, ticker: str, accession: str) -> bool:
        return (self._dir(ticker, accession) / "meta.json").exists()

    def save(self, meta: FilingMeta, text: str, facts: list[Fact]) -> None:
        d = self._dir(meta.ticker, meta.accession)
        d.mkdir(parents=True, exist_ok=True)
        self.text_dir(meta.ticker).mkdir(parents=True, exist_ok=True)
        (self.text_dir(meta.ticker) / f"{meta.doc}.txt").write_text(normalise_text(text))
        (d / "facts.json").write_text(json.dumps([asdict(f) for f in facts], indent=1) + "\n")
        # meta last: has() keys on it, so a crash mid-save leaves no half filing
        (d / "meta.json").write_text(json.dumps(asdict(meta), indent=1) + "\n")

    def filings(self, ticker: str) -> list[FilingMeta]:
        """Saved filings for a ticker, newest period first."""
        metas = [
            FilingMeta(**json.loads(p.read_text()))
            for p in (self.root / ticker).glob("*/meta.json")
        ]
        return sorted(metas, key=lambda m: (m.period, m.filed), reverse=True)

    def facts(self, meta: FilingMeta) -> list[Fact]:
        raw = json.loads((self._dir(meta.ticker, meta.accession) / "facts.json").read_text())
        return [Fact(**r) for r in raw]

    def text(self, meta: FilingMeta) -> str:
        return (self.text_dir(meta.ticker) / f"{meta.doc}.txt").read_text()

    def passages(self, meta: FilingMeta) -> list[Passage]:
        return split_passages(meta.doc, self.text(meta))

    def _doc_path(self, doc: str) -> Path | None:
        """A saved filing's text file, or None. `doc` comes from the model or a
        search hit, so the resolved path must stay inside the store."""
        ticker = doc.rsplit("-", 3)[0]
        path = (self.text_dir(ticker) / f"{doc}.txt").resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            return None
        return path

    def passage(self, passage_id: str) -> Passage | None:
        """The passage with this exact id, or None. Ids come from the model, so
        anything that does not parse or does not exist is simply not found."""
        m = PASSAGE_ID.match(passage_id.strip())
        if not m:
            return None
        doc = m["doc"]
        path = self._doc_path(doc)
        if path is None:
            return None
        for p in split_passages(doc, path.read_text()):
            if p.id == passage_id.strip():
                return p
        return None

    def passages_between(self, doc: str, first: int, last: int) -> list[Passage]:
        """Passages overlapping lines first..last (inclusive), in file order."""
        path = self._doc_path(doc)
        if path is None:
            return []
        return [
            p
            for p in split_passages(doc, path.read_text())
            if p.line <= last and first < p.line + p.text.count("\n") + 1
        ]

    def passage_at(self, doc: str, line: int) -> Passage | None:
        """The passage containing a file:line hit, as returned by text search."""
        path = self._doc_path(doc)
        if path is None:
            return None
        for p in split_passages(doc, path.read_text()):
            if p.line <= line < p.line + p.text.count("\n") + 1:
                return p
        return None
