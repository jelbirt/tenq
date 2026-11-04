"""The FinanceBench loader, on made-up files: nothing here is dataset text."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import cast

import pytest

from tenq.financebench import (
    DOCUMENTS,
    MANIFEST,
    QUESTIONS,
    Manifest,
    download,
    download_results,
    load_manifest,
    pdf_doc,
    pdf_path,
    raw_url,
)

COMMIT = "0" * 40


def _question(
    qid: str, doc: str, page: int, justification: str | None = "why"
) -> dict[str, object]:
    return {
        "financebench_id": qid,
        "company": "Acme",
        "doc_name": doc,
        "question_type": "metrics-generated",
        "question_reasoning": None,
        "question": f"made-up question {qid}",
        "answer": "$1.00",
        "justification": justification,
        "dataset_subset_label": "OPEN_SOURCE",
        "evidence": [
            {
                "doc_name": doc,
                "evidence_page_num": page,
                "evidence_text": "made-up evidence",
                "evidence_text_full_page": "made-up page",
            }
        ],
    }


def _doc(name: str, period: int) -> dict[str, object]:
    return {
        "doc_name": name,
        "company": "Acme",
        "gics_sector": "Made Up",
        "doc_type": "10k",
        "doc_period": period,
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


class Remote:
    """The dataset repo at COMMIT, served from memory; records every fetch."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.fetched: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.fetched.append(url)
        for name, data in self.files.items():
            if url == raw_url(COMMIT, name):
                return data
        raise AssertionError(f"unexpected fetch {url}")

    def manifest(self, skip: tuple[str, ...] = ()) -> Manifest:
        return Manifest(
            COMMIT,
            {n: hashlib.sha256(d).hexdigest() for n, d in self.files.items() if n not in skip},
        )


@pytest.fixture
def remote() -> Remote:
    return Remote(
        {
            QUESTIONS: _jsonl(
                [_question("q1", "1ACME_2020_10K", 3), _question("q2", "ACME_2021Q2_10Q", 0, None)]
            ),
            DOCUMENTS: _jsonl(
                [
                    _doc("1ACME_2020_10K", 2020),
                    _doc("ACME_2021Q2_10Q", 2021),
                    _doc("ACME_2019_10K", 2019),
                ]
            ),
            "pdfs/1ACME_2020_10K.pdf": b"%PDF made up one",
            "pdfs/ACME_2021Q2_10Q.pdf": b"%PDF made up two",
            "results/m1_singleStore.jsonl": _jsonl(
                [_result("q1", "Correct Answer"), _result("q2", "Refusal", None)]
            ),
        }
    )


def _result(qid: str, label: str, answer: str | None = "made-up answer") -> dict[str, object]:
    return {
        "financebench_id": qid,
        "model_name": "m1",
        "eval_mode": "singleStore",
        "temp": "0",
        "question": f"made-up question {qid}",
        "gold_answer": "$1.00",
        "model_answer": answer,
        "label": label,
    }


def test_download_fetches_and_checks_every_file(tmp_path: Path, remote: Remote) -> None:
    ds = download(tmp_path, remote.manifest(), remote)
    assert len(remote.fetched) == 4  # result files are fetched only when asked for
    assert [q.id for q in ds.questions] == ["q1", "q2"]
    q1, q2 = ds.questions
    assert (q1.evidence[0].doc_name, q1.evidence[0].page) == ("1ACME_2020_10K", 3)
    assert q2.justification == ""
    assert ds.pdf("1ACME_2020_10K").read_bytes() == b"%PDF made up one"


def test_only_the_documents_a_question_uses_are_kept(tmp_path: Path, remote: Remote) -> None:
    ds = download(tmp_path, remote.manifest(), remote)
    assert sorted(ds.documents) == ["1ACME_2020_10K", "ACME_2021Q2_10Q"]
    assert ds.documents["ACME_2021Q2_10Q"].doc_period == 2021
    assert not any("ACME_2019_10K" in u for u in remote.fetched)


def test_files_already_on_disk_are_not_fetched_again(tmp_path: Path, remote: Remote) -> None:
    download(tmp_path, remote.manifest(), remote)
    remote.fetched.clear()
    download(tmp_path, remote.manifest(), remote)
    assert remote.fetched == []


def test_a_damaged_file_on_disk_is_fetched_again(tmp_path: Path, remote: Remote) -> None:
    download(tmp_path, remote.manifest(), remote)
    (tmp_path / "pdfs/ACME_2021Q2_10Q.pdf").write_bytes(b"truncated")
    remote.fetched.clear()
    download(tmp_path, remote.manifest(), remote)
    assert remote.fetched == [raw_url(COMMIT, "pdfs/ACME_2021Q2_10Q.pdf")]
    assert (tmp_path / "pdfs/ACME_2021Q2_10Q.pdf").read_bytes() == b"%PDF made up two"


def test_a_download_that_does_not_match_is_never_written(tmp_path: Path, remote: Remote) -> None:
    manifest = remote.manifest()
    remote.files["pdfs/ACME_2021Q2_10Q.pdf"] = b"%PDF tampered"
    with pytest.raises(ValueError, match="sha256"):
        download(tmp_path, manifest, remote)
    assert list((tmp_path / "pdfs").iterdir()) == [tmp_path / "pdfs/1ACME_2020_10K.pdf"]


def test_a_manifest_that_misses_a_needed_pdf_is_refused(tmp_path: Path, remote: Remote) -> None:
    with pytest.raises(ValueError, match="missing"):
        download(tmp_path, remote.manifest(skip=("pdfs/ACME_2021Q2_10Q.pdf",)), remote)


def test_a_question_on_a_document_with_no_information_row_is_refused(
    tmp_path: Path, remote: Remote
) -> None:
    remote.files[DOCUMENTS] = _jsonl([_doc("1ACME_2020_10K", 2020)])
    with pytest.raises(ValueError, match="no information row"):
        download(tmp_path, remote.manifest(), remote)


@pytest.mark.parametrize("name", ["../etc", "a/b", "", "_x", "x.pdf", "x" * 81, "\uff13M_2018_10K"])
def test_document_names_cannot_leave_the_pdf_folder(name: str) -> None:
    with pytest.raises(ValueError):
        pdf_path(name)


def test_a_question_naming_a_bad_document_is_refused(tmp_path: Path, remote: Remote) -> None:
    remote.files[QUESTIONS] = _jsonl([_question("q1", "../../x", 0)])
    with pytest.raises(ValueError, match="document name"):
        download(tmp_path, remote.manifest(), remote)


def test_pdf_paths_round_trip_and_odd_ones_are_refused() -> None:
    assert pdf_doc(pdf_path("3M_2018_10K")) == "3M_2018_10K"
    for path in ("pdfs/../x.pdf", "pdfs/x.txt", "other/x.pdf"):
        with pytest.raises(ValueError):
            pdf_doc(path)


@pytest.mark.parametrize(
    "raw",
    [
        {"commit": "cc39aeb", "files": {QUESTIONS: "a" * 64, DOCUMENTS: "a" * 64}},
        {"commit": COMMIT, "files": {QUESTIONS: "A" * 64, DOCUMENTS: "a" * 64}},
        {"commit": COMMIT, "files": {DOCUMENTS: "a" * 64}},
        {"commit": COMMIT, "files": {QUESTIONS: "a" * 64, DOCUMENTS: "a" * 64, "x.pdf": "a" * 64}},
        {"commit": COMMIT, "files": {QUESTIONS: "a" * 64, DOCUMENTS: 7}},
        {"commit": COMMIT, "files": [QUESTIONS, DOCUMENTS]},
        {
            "commit": COMMIT,
            "files": {QUESTIONS: "a" * 64, DOCUMENTS: "a" * 64, "results/../x.jsonl": "a" * 64},
        },
        {
            "commit": COMMIT,
            "files": {QUESTIONS: "a" * 64, DOCUMENTS: "a" * 64, "results/x.json": "a" * 64},
        },
        {"files": {QUESTIONS: "a" * 64, DOCUMENTS: "a" * 64}},
        {"commit": COMMIT, "files": {QUESTIONS: "a" * 64, DOCUMENTS: "a" * 64}, "questions": []},
        [COMMIT],
    ],
)
def test_a_malformed_manifest_is_refused(tmp_path: Path, raw: object) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_the_committed_manifest_pins_cc39aeb_and_the_84_documents() -> None:
    manifest = load_manifest()
    assert manifest.commit == "cc39aeb4afdf33909ee1412188bf89035950c2eb"
    assert len(manifest.pdfs) == 84
    assert "3M_2018_10K" in manifest.pdfs
    assert len(manifest.results) == 16
    assert {"gpt-4-1106-preview_singleStore", "gpt-4_singleStore", "llama2_singleStore"} <= (
        manifest.results
    )


def test_the_committed_manifest_holds_only_names_and_hashes() -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(raw) == {"commit", "files"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in raw["files"].values())


@pytest.mark.parametrize("name", [QUESTIONS, DOCUMENTS])
def test_a_tampered_question_or_document_file_is_refused(
    tmp_path: Path, remote: Remote, name: str
) -> None:
    manifest = remote.manifest()
    remote.files[name] = remote.files[name] + b"\n"
    with pytest.raises(ValueError, match="sha256"):
        download(tmp_path, manifest, remote)
    assert not (tmp_path / name).exists()


def test_a_manifest_pinning_a_pdf_no_question_uses_is_refused(
    tmp_path: Path, remote: Remote
) -> None:
    remote.files["pdfs/ACME_2019_10K.pdf"] = b"%PDF unused"
    with pytest.raises(ValueError, match="extra"):
        download(tmp_path, remote.manifest(), remote)


def test_evidence_on_another_document_fetches_that_document_too(
    tmp_path: Path, remote: Remote
) -> None:
    row = _question("q1", "1ACME_2020_10K", 3)
    other = {"doc_name": "ACME_2021Q2_10Q", "evidence_page_num": 1, "evidence_text": "made-up"}
    row["evidence"] = [*cast(list[object], row["evidence"]), other]
    remote.files[QUESTIONS] = _jsonl([row])
    ds = download(tmp_path, remote.manifest(), remote)
    assert sorted(ds.documents) == ["1ACME_2020_10K", "ACME_2021Q2_10Q"]
    assert [e.doc_name for e in ds.questions[0].evidence] == ["1ACME_2020_10K", "ACME_2021Q2_10Q"]


def test_result_files_are_fetched_checked_and_parsed(tmp_path: Path, remote: Remote) -> None:
    results = download_results(tmp_path, remote.manifest(), remote)
    (a1, a2) = results["m1_singleStore"]
    assert (a1.key, a1.label, a1.model_answer) == (
        "m1_singleStore:q1",
        "Correct Answer",
        "made-up answer",
    )
    assert (a2.label, a2.model_answer) == ("Refusal", "")  # a null answer reads as empty


def test_a_tampered_result_file_is_refused(tmp_path: Path, remote: Remote) -> None:
    manifest = remote.manifest()
    remote.files["results/m1_singleStore.jsonl"] += b"\n"
    with pytest.raises(ValueError):
        download_results(tmp_path, manifest, remote)


@pytest.mark.parametrize(
    "rows",
    [
        [_result("q1", "Partly Right")],
        [_result("q1", "Refusal"), _result("q1", "Correct Answer")],
    ],
)
def test_a_result_file_with_an_unknown_label_or_a_repeat_is_refused(
    tmp_path: Path, remote: Remote, rows: list[dict[str, object]]
) -> None:
    remote.files["results/m1_singleStore.jsonl"] = _jsonl(rows)
    with pytest.raises(ValueError):
        download_results(tmp_path, remote.manifest(), remote)
