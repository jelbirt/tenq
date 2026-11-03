"""Write config/financebench.json: a sha256 for each FinanceBench file tenq uses.

Each download is first checked against the git blob id the dataset repo's tree
gives for it at the pinned commit, so the manifest is tied to that commit and
not to whatever the network returned. Only file names and hashes are written.

    uv run python scripts/make_financebench_manifest.py [root, default data/financebench]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tenq.financebench import (
    DEFAULT_ROOT,
    DOCUMENTS,
    MANIFEST,
    QUESTIONS,
    REPO,
    _jsonl,
    http_fetch,
    needed_docs,
    parse_questions,
    pdf_path,
    raw_url,
)

COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(root: Path) -> None:
    tree = json.loads(
        http_fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1")
    )
    if tree.get("truncated"):
        sys.exit("tree listing truncated")
    blobs = {t["path"]: t["sha"] for t in tree["tree"] if t["type"] == "blob"}

    def get(name: str) -> bytes:
        path = root / name
        data = path.read_bytes() if path.is_file() else http_fetch(raw_url(COMMIT, name))
        if git_blob_id(data) != blobs[name]:
            sys.exit(f"{name}: does not match the git tree at {COMMIT}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    files = {name: hashlib.sha256(get(name)).hexdigest() for name in (QUESTIONS, DOCUMENTS)}
    docs = sorted(needed_docs(parse_questions(_jsonl(root / QUESTIONS))))
    for i, doc in enumerate(docs, 1):
        files[pdf_path(doc)] = hashlib.sha256(get(pdf_path(doc))).hexdigest()
        print(f"{i}/{len(docs)} {doc}", flush=True)
    MANIFEST.write_text(json.dumps({"commit": COMMIT, "files": files}, indent=1) + "\n")
    print(f"wrote {MANIFEST} ({len(files)} files)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT)
