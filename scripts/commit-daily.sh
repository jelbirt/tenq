#!/usr/bin/env bash
# Commit and push a runs/ folder to main (default runs/daily). Used by daily.yml
# twice: after the claim (the email is sent only if this push worked) and after
# the send; by consistency.yml with runs/consistency; and by the brief with all
# of runs (its records sit in runs/<TICKER>/, plus runs/heartbeat), before each
# email (tenq.run.save_claim) and at the end.
#
# The commit is made in a separate, temporary copy of main's current tip, and
# only the record files this run wrote are copied into it. The job's own
# checkout never moves, so every later step runs the code the record names
# (versions.code), even if a pull request merged during the run.
#
# A file this run writes is saved only if main still has the version this job
# last saw: the one its own last save pushed, or else the one it checked out.
# If anything else changed it on main (another run, a hand fix), the save fails
# and main keeps the other version, so two runs can never both claim a day.
# refs/tenq/saved holds the checkout with only this job's saved files laid over
# it, never other writers' changes, so a file this job saves for the first time
# is still compared with the version it checked out.
#
# Needs GH_TOKEN (TENQ_PUSH_URL, used by the tests, replaces the GitHub URL).
# Exits 1 if the save fails.
set -euo pipefail

dir="${1:-runs/daily}"
case "$dir" in
  runs/daily | runs/consistency | runs) ;;
  *) echo "not a records folder: $dir"; exit 2 ;;
esac
pause="${TENQ_RETRY_SLEEP:-5}"

if [ ! -d "$dir" ]; then
  echo "no record in $dir"
  exit 0
fi
# a write the job was stopped in, at any depth: never committed
find "$dir" -name '.*.tmp' -type f -delete

# the files this run wrote: new or changed against the job's own checkout
files=()
while IFS= read -r -d '' f; do
  [ -f "$f" ] && files+=("$f")
done < <(git ls-files -z --others --modified --exclude-standard -- "$dir")
if [ ${#files[@]} -eq 0 ]; then
  echo "no new record in $dir"
  exit 0
fi

url="${TENQ_PUSH_URL:-https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git}"
copy=""
idx=""
cleanup() {
  [ -n "$idx" ] && rm -f "$idx"
  if [ -n "$copy" ]; then
    git worktree remove --force "$copy" 2>/dev/null || rm -rf "$copy" || true
  fi
  git worktree prune 2>/dev/null || true
}
# cleanup can never change the script's own exit status
trap 'rc=$?; cleanup; exit $rc' EXIT

fetched=""
for attempt in 1 2; do
  if git fetch -q "$url" main; then
    fetched=1
    break
  fi
  sleep "$pause"
done
if [ -z "$fetched" ]; then
  echo "could not fetch main after $attempt tries"
  exit 1
fi

# the version of each file this job last saw: its own last save, else its checkout
base="$(git rev-parse -q --verify refs/tenq/saved || git rev-parse HEAD)"
blob() { git rev-parse -q --verify "$1" 2>/dev/null || true; }
for f in "${files[@]}"; do
  if [ "$(blob "FETCH_HEAD:$f")" != "$(blob "$base:$f")" ]; then
    echo "main changed $f during this run; not saving over it"
    exit 1
  fi
done

copy="$(mktemp -d)"
git worktree add -q --detach "$copy" FETCH_HEAD
for f in "${files[@]}"; do
  mkdir -p "$copy/$(dirname "$f")"
  cp "$f" "$copy/$f"
done
git -C "$copy" add -- "$dir"
if git -C "$copy" diff --cached --quiet; then
  echo "no new record in $dir"
  exit 0
fi
names=""
while IFS= read -r -d '' f; do
  names+="${f##*/} "
done < <(git -C "$copy" diff --cached --name-only -z)
git -C "$copy" -c user.name=tenq-bot -c user.email=bot@users.noreply.github.com \
  commit -q -m "runs: ${dir#runs/} record $names"

# the version this job last saw, for the next save: its base, plus this save's files
note_saved() {
  idx="$(mktemp)" && rm -f "$idx" || return 1
  GIT_INDEX_FILE="$idx" git read-tree "$base" || return 1
  local f b
  for f in "${files[@]}"; do
    b="$(git hash-object -w -- "$f")" || return 1
    GIT_INDEX_FILE="$idx" git update-index --add --cacheinfo "100644,$b,$f" || return 1
  done
  local tree saved
  tree="$(GIT_INDEX_FILE="$idx" git write-tree)" || return 1
  saved="$(git -c user.name=tenq-bot -c user.email=bot@users.noreply.github.com \
    commit-tree "$tree" -m "files this job saved")" || return 1
  git update-ref refs/tenq/saved "$saved"
}

# other jobs also commit runs/ (other files): rebase onto them, retry once. A
# rebase that touches this run's files conflicts, and the save fails.
for attempt in 1 2; do
  if git -C "$copy" -c user.name=tenq-bot -c user.email=bot@users.noreply.github.com \
    pull -q --rebase "$url" main && git -C "$copy" push -q "$url" HEAD:main; then
    note_saved || echo "saved, but could not note it: the next save will refuse"
    exit 0
  fi
  git -C "$copy" rebase --abort 2>/dev/null || true
  sleep "$pause"
done
echo "push failed after $attempt tries"
exit 1
