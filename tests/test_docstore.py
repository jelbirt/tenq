"""The FinanceBench document store, on made-up PDFs built here: no dataset text."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import tenq.docstore as docstore
from tenq.docstore import (
    DocStore,
    check_evidence,
    evidence_share,
    evidence_shares,
    extract_all,
    extract_pages,
    page_marked_text,
    report,
)
from tenq.financebench import Dataset, Document, Evidence, Question
from tenq.search import Xerj, hits_from_response, xerj_node

DOC = "3ACME_2020_10K"  # starts with a digit, as 3M_2018_10K does


def make_pdf(pages: list[list[str]]) -> bytes:
    """A minimal valid PDF: one Helvetica text line per entry, top to bottom."""
    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    font = 3 + 2 * len(pages)
    kids = []
    for i, lines in enumerate(pages):
        page_obj, content_obj = 3 + 2 * i, 4 + 2 * i
        kids.append(f"{page_obj} 0 R".encode())
        ops = b"".join(
            b"BT /F1 12 Tf 72 %d Td (%s) Tj ET\n" % (720 - 20 * n, line.encode())
            for n, line in enumerate(lines)
        )
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (font, content_obj)
        )
        objs.append(b"<< /Length %d >>\nstream\n%sendstream" % (len(ops), ops))
    objs[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(kids), len(pages))
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (n, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


PAGES = [
    ["Acme Widgets annual report", "Made up for tests"],
    [],
    ["Consolidated statement of cash flows", "Purchases of widget presses 1,234"],
    ["Risk factors", "Widget demand may fall"],
]


@pytest.fixture
def store(tmp_path: Path) -> DocStore:
    pdf = tmp_path / f"{DOC}.pdf"
    pdf.write_bytes(make_pdf(PAGES))
    pages, seconds = extract_pages(pdf)
    s = DocStore(tmp_path / "docs")
    s.save(DOC, "a" * 64, pages, seconds)
    return s


def test_extraction_keeps_pages_in_order_and_empty_pages_empty(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(make_pdf(PAGES))
    pages, seconds = extract_pages(pdf)
    assert len(pages) == 4 and seconds >= 0
    assert "Acme Widgets annual report" in pages[0]
    assert pages[1].strip() == ""
    assert "Purchases of widget presses 1,234" in pages[2]


def test_page_markers_are_zero_indexed_and_fenced_by_blank_lines() -> None:
    text, starts, empty = page_marked_text(["a\nb", "", "c"])
    assert text == "[[page 0]]\n\na\nb\n\n[[page 1]]\n\n[[page 2]]\n\nc\n"
    assert starts == [1, 6, 8]
    assert empty == [1]


def test_passages_never_cross_a_page_and_markers_are_not_passages(store: DocStore) -> None:
    passages = store.passages(DOC)
    assert passages and all(p.id == f"{DOC}:{p.line}" for p in passages)
    assert not any("[[page" in p.text for p in passages)
    pages = {store.page_of(DOC, p.line) for p in passages}
    assert pages == {0, 2, 3}
    for p in passages:
        last = p.line + p.text.count("\n")
        assert store.page_of(DOC, p.line) == store.page_of(DOC, last)


def test_a_passage_is_found_by_its_id_and_knows_its_page(store: DocStore) -> None:
    (p,) = [p for p in store.passages(DOC) if "widget presses" in p.text]
    assert store.passage(p.id) == p
    assert store.page_of(DOC, p.line) == 2
    assert "widget presses" in store.page_text(DOC, 2)
    assert store.page_text(DOC, 1) == ""


@pytest.mark.parametrize(
    "pid",
    [
        f"{DOC}:1",  # a page marker
        f"{DOC}:999",
        "../docs:1",
        f"{DOC}/../{DOC}:3",
        "OTHER_2020_10K:3",
        "CRWD-0001535527-26-000031:23",  # a slice 1 id
        "",
    ],
)
def test_ids_that_are_not_passages_are_not_found(store: DocStore, pid: str) -> None:
    assert store.passage(pid) is None


def test_a_document_is_re_extracted_when_its_pdf_or_extractor_changes(
    store: DocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert store.has(DOC, "a" * 64)
    assert not store.has(DOC, "b" * 64)
    monkeypatch.setattr("tenq.docstore.extractor_version", lambda: "pdfplumber 99")
    assert not store.has(DOC, "a" * 64)


def test_document_index_prefixes_never_match_a_ticker_or_a_sibling(store: DocStore) -> None:
    assert store.index_prefix("3M_2018_10K") == "tenq-3m-2018-10k-d"
    assert not store.index_prefix("ACME_2020_10K_x").startswith(
        store.index_prefix("ACME_2020_10K") + "-"
    )
    with pytest.raises(ValueError):
        store.index_prefix("../x")


def test_a_table_whose_words_ran_together_still_matches_on_its_figures() -> None:
    page = "Cash and cash equivalents $ 452,030 $ 461,329\nRestricted cash 22,131 13,537 9,446\n"
    evidence = "Cashandcashequivalents | $ | 452,030 $ | 461,329 | Restrictedcash | 22,131 | 13,537 | 9,446"
    assert evidence_share(evidence, page) == 1.0
    assert evidence_share(evidence + " | 1,234", page) == 5 / 6  # the page lacks 1,234
    assert evidence_share(evidence, "Cash 111,111 222,222") == 0.0


def test_a_few_figures_do_not_outvote_the_words() -> None:
    # under MIN_FIGURES the words decide: matching years alone are no evidence
    assert (
        evidence_share("Net sales grew in 2019 and 2020 on demand", "2019 2020 other text") == 0.0
    )


def test_column_years_are_not_figures() -> None:
    # every 10-K page has the column years: they must not make a page a match
    evidence = "Revenue | 2022 | 2021 | 2020 | 81,462 | 53,823 | 31,536 | 24,578"
    page = "Selected data for fiscal 2022 2021 2020 shown in other notes 1,111 2,222"
    assert evidence_shares(evidence, page) == (0.0, 0.0)  # four figures: under MIN_FIGURES


def test_evidence_share_tolerates_line_breaks_but_not_other_text() -> None:
    page = "Purchases of property, plant and equipment (PP&E) (1,577) (1,373)"
    assert evidence_share("Purchases of property, \nplant and \nequipment (PP&E)", page) == 1.0
    assert evidence_share("Proceeds from sale of businesses", page) == 0.0
    assert evidence_share("", page) == 0.0


def _question(page: int, text: str) -> Question:
    return Question("q1", "Acme", DOC, "made-up", None, "q?", "a", "", (Evidence(DOC, page, text),))


def test_evidence_check_finds_the_gold_page_and_names_the_best_page_on_a_miss(
    store: DocStore,
) -> None:
    text = "Consolidated statement of cash flows Purchases of widget presses"
    (hit,) = check_evidence(store, [_question(2, text)])
    assert hit.hit and hit.best_page == 2
    (miss,) = check_evidence(store, [_question(3, text)])
    assert not miss.hit and miss.best_page == 2 and miss.best_share == 1.0
    (gone,) = check_evidence(store, [_question(40, text)])
    assert gone.share == 0.0 and not gone.hit


def test_a_hit_on_figures_alone_is_reported_for_a_look(store: DocStore) -> None:
    # same figures, other words between them: no word 4-gram survives
    pages = ["North 111,111 South 222,222 East 333,333 West 444,444 Other 555,555"]
    s = DocStore(store.root.parent / "fig")
    s.save(DOC, "b" * 64, pages, 0.0)
    q = _question(0, "Totalsrow | 111,111 | 222,222 | 333,333 | 444,444 | 555,555")
    (c,) = check_evidence(s, [q])
    assert c.hit and c.figures_only and c.words < c.figures
    meta = s.meta(DOC)
    assert meta is not None
    rep = report([meta], [c])
    assert rep["evidence_misses"] == []
    assert [i["question"] for i in rep["evidence_on_figures_only"]] == ["q1"]


def test_page_texts_reads_the_document_once_and_matches_page_text(tmp_path: Path) -> None:
    pages = [f"Page {i} heading\n\nBody text of page {i}\n\nMore of page {i}" for i in range(60)]
    s = DocStore(tmp_path / "many")
    meta = s.save(DOC, "c" * 64, pages, 0.0)
    texts = s.page_texts(DOC)
    assert len(texts) == 60 and texts[37] == s.page_text(DOC, 37)
    assert "Body text of page 37" in texts[37] and "page 38" not in texts[37]
    assert s.page_text(DOC, 60) == "" and s.page_text(DOC, -1) == ""
    assert s.page_of(DOC, meta.page_lines[5]) == 5  # a marker line opens its page
    assert s.page_of(DOC, meta.page_lines[0] - 1) is None
    assert s.page_of(DOC, 10**6) == 59


def test_a_re_extract_that_dies_mid_save_leaves_no_stale_meta(
    store: DocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def die(path: Path, text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(docstore, "_write_whole", die)
    with pytest.raises(OSError):
        store.save(DOC, "d" * 64, ["New text"], 0.0)
    assert store.meta(DOC) is None and store.passages(DOC) == []


def test_unreadable_meta_counts_as_not_saved(store: DocStore) -> None:
    (store.root / DOC / "meta.json").write_text('{"doc_name": "3ACME', encoding="utf-8")
    assert store.meta(DOC) is None and not store.has(DOC, "a" * 64)
    (store.root / DOC / "meta.json").write_text(json.dumps({"doc_name": DOC}), encoding="utf-8")
    assert store.meta(DOC) is None


def _fake_extract(pdf: Path) -> tuple[list[str], float]:
    if pdf.stem == "BROKEN_2020_10K":
        raise ValueError("not a PDF")
    return [f"Text of {pdf.stem}"], 0.5


def test_extract_all_skips_saved_documents_and_survives_a_bad_pdf(tmp_path: Path) -> None:
    names = ["GOOD_2020_10K", "BROKEN_2020_10K", "SAVED_2020_10K"]
    ds = Dataset(tmp_path, [], {n: Document(n, "x", "10k", 2020) for n in names})
    s = DocStore(tmp_path / "docs")
    s.save("SAVED_2020_10K", "e" * 64, ["Already here"], 9.0)
    shas = {n: "e" * 64 for n in names}
    metas, failed = extract_all(ds, s, shas, jobs=2, extract=_fake_extract)
    assert failed == {"BROKEN_2020_10K": "ValueError"}
    assert sorted(m.doc_name for m in metas) == ["GOOD_2020_10K", "SAVED_2020_10K"]
    saved = s.meta("SAVED_2020_10K")
    assert saved is not None and saved.seconds == 9.0  # not extracted again
    assert "Text of GOOD_2020_10K" in s.page_text("GOOD_2020_10K", 0)
    assert report(metas, [], failed)["failed"] == failed


def test_the_report_carries_no_document_text(store: DocStore) -> None:
    meta = store.meta(DOC)
    assert meta is not None
    checks = check_evidence(store, [_question(3, "Consolidated statement of cash flows")])
    rep = report([meta], checks)
    assert rep["pages"] == 4 and rep["per_document"][0]["empty"] == [1]
    assert rep["evidence_misses"][0]["best_page"] == 2
    assert "widget" not in str(rep).lower() and "cash" not in str(rep).lower()


def test_search_hits_map_to_this_documents_passages_only(store: DocStore) -> None:
    (p,) = [p for p in store.passages(DOC) if "widget presses" in p.text]
    resp = {
        "hits": {
            "hits": [
                {"_score": 2.0, "_source": {"ax_path": f"OTHER/{DOC}x.txt", "start_line": p.line}},
                {
                    "_score": 1.0,
                    "_source": {"ax_path": f"{DOC}.txt", "start_line": p.line, "body": "x"},
                },
            ]
        }
    }
    (hit,) = hits_from_response(resp, store, "widget presses", k=5, key=DOC)
    assert hit.passage_id == p.id


@pytest.mark.skipif(shutil.which("xerj") is None, reason="xerj not installed")
def test_one_document_indexed_and_searched_with_real_xerj(store: DocStore, tmp_path: Path) -> None:
    with xerj_node() as url:
        xerj = Xerj(url, store)
        xerj.index(DOC, tmp_path / "xerj-state")
        hits = xerj.search(DOC, "purchases of widget presses", k=3)
    assert hits and store.page_of(DOC, int(hits[0].passage_id.split(":")[1])) == 2
