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

# Eval answers and grades are private: the eval workflow stays read only, never
# uploads an artifact, and names the private checkout in one path only (its
# checkout step); every cache holds data/financebench and nothing else.
evals=.github/workflows/evals.yml
if [ -f "$evals" ]; then
  if grep -Eq 'upload-artifact|:[[:space:]]*write' "$evals" \
    || ! grep -Eq '^  contents: read$' "$evals" \
    || [ "$(grep -Ec 'path:.*evals-state' "$evals")" != 1 ] \
    || ! awk '/uses: actions\/cache/ { c = 1; next }
              c && /path:/ { if ($0 !~ /path: data\/financebench$/) bad = 1; c = 0 }
              END { exit bad }' "$evals"; then
    echo "checks.sh: $evals must be read only, upload no artifact, cache no evals-state" >&2
    exit 1
  fi
fi

uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
