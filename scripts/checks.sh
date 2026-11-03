#!/usr/bin/env bash
# The single definition of this repo's green bar. CI (.github/workflows/ci.yml)
# runs THIS script, so the local bar and the CI bar cannot drift. If you change
# the commands here, update the Commands section of CLAUDE.md in the same commit.
# Pattern adapted from ticker-sentinel's scripts/checks.sh (MIT, same author).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "checks.sh: uv not found; install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

# One file per invocation: `bash -n a.sh b.sh` parses only a.sh.
for f in scripts/*.sh; do
  bash -n "$f"
done

uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
