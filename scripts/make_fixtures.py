"""Rebuild tests/fixtures/filings from a store fetched live with `python -m tenq.fetch`.

Fixtures stay small: facts are trimmed to the concepts the numbers tool reads,
and only the newest filing keeps text, cut to the opening of its MD&A. Tests
read these files and never touch the network.

    uv run python scripts/make_fixtures.py <store root> CRWD
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from tenq.filings import FilingStore, normalise_text

KEEP_PREFIXES = (
    "us-gaap:Revenue",
    "us-gaap:GrossProfit",
    "us-gaap:OperatingIncomeLoss",
    "us-gaap:NetIncomeLoss",
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:ContractWithCustomerLiability",
    "us-gaap:DeferredRevenue",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
)
MDNA_LINES = 120
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "filings"


def main(src_root: Path, ticker: str) -> None:
    src = FilingStore(src_root)
    metas = src.filings(ticker)
    if not metas:
        sys.exit(f"no filings for {ticker} under {src_root}")
    # MD&A is Item 2 in a 10-Q and Item 7 in a 10-K; find it before deleting anything
    heading = "ITEM 7. MANAGEMENT" if metas[0].form == "10-K" else "ITEM 2. MANAGEMENT"
    lines = src.text(metas[0]).split("\n")
    start = next((n for n, line in enumerate(lines) if line.startswith(heading)), None)
    if start is None:
        sys.exit(f"no line starting {heading!r} in {metas[0].doc}")
    out = OUT / ticker
    if out.exists():
        shutil.rmtree(out)
    for i, meta in enumerate(metas):
        d = out / meta.accession
        d.mkdir(parents=True)
        shutil.copy(src_root / ticker / meta.accession / "meta.json", d / "meta.json")
        facts = json.loads((src_root / ticker / meta.accession / "facts.json").read_text())
        kept = [f for f in facts if f["concept"].startswith(KEEP_PREFIXES)]
        (d / "facts.json").write_text(json.dumps(kept, indent=1) + "\n")
        if i == 0:
            (out / "text").mkdir()
            excerpt = "\n".join(lines[start : start + MDNA_LINES])
            (out / "text" / f"{meta.doc}.txt").write_text(normalise_text(excerpt))
    print(f"wrote {len(metas)} filings to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
