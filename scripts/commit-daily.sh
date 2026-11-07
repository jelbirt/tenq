#!/usr/bin/env bash
# Commit and push a runs/ folder to main (default runs/daily). Used by daily.yml
# twice: after the claim (the email is sent only if this push worked) and after
# the send; and by consistency.yml with runs/consistency.
#
# The commit is made in a separate, temporary copy of main's current tip, and
# only the record files this run wrote are copied into it. The job's own
# checkout never moves, so every later step runs the code the record names
# (versions.code), even if a pull request merged during the run. If main
# changed a file this run also wrote, this run's copy wins; only these jobs
# write their records, so that is not expected.
#
# Needs GH_TOKEN (TENQ_PUSH_URL, used by the tests, replaces the GitHub URL).
# Exits 1 if the push fails.
set -euo pipefail

dir="${1:-runs/daily}"
case "$dir" in
  runs/daily | runs/consistency) ;;
  *) echo "not a records folder: $dir"; exit 2 ;;
esac

if [ ! -d "$dir" ]; then
  echo "no record in $dir"
  exit 0
fi
rm -f "$dir"/.*.tmp # a write the job was stopped in: never committed

# the files this run wrote: new or changed against the job's own checkout
files=()
while IFS= read -r f; do
  [ -f "$f" ] && files+=("$f")
done < <(git ls-files --others --modified --exclude-standard -- "$dir")
if [ ${#files[@]} -eq 0 ]; then
  echo "no new record in $dir"
  exit 0
fi

url="${TENQ_PUSH_URL:-https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git}"
copy="$(mktemp -d)"
cleanup() {
  git worktree remove --force "$copy" 2>/dev/null || rm -rf "$copy"
  git worktree prune
}
trap cleanup EXIT

git fetch -q "$url" main
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
names="$(git -C "$copy" diff --cached --name-only | xargs -n1 basename | tr '\n' ' ')"
git -C "$copy" -c user.name=tenq-bot -c user.email=bot@users.noreply.github.com \
  commit -q -m "runs: ${dir#runs/} record $names"

# the brief job also commits runs/ (other files): rebase onto it, retry once
for attempt in 1 2; do
  if git -C "$copy" -c user.name=tenq-bot -c user.email=bot@users.noreply.github.com \
    pull -q --rebase "$url" main && git -C "$copy" push -q "$url" HEAD:main; then
    exit 0
  fi
  git -C "$copy" rebase --abort 2>/dev/null || true
  sleep 5
done
echo "push failed after $attempt tries"
exit 1
