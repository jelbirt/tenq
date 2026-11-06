#!/usr/bin/env bash
# Commit and push a runs/ folder to main (default runs/daily). Used by daily.yml
# twice: after the claim (the email is sent only if this push worked) and after
# the send; and by consistency.yml with runs/consistency.
# Needs GH_TOKEN. Exits 1 if the push fails.
set -euo pipefail

dir="${1:-runs/daily}"
case "$dir" in
  runs/daily | runs/consistency) ;;
  *) echo "not a records folder: $dir"; exit 2 ;;
esac

git config user.name tenq-bot
git config user.email bot@users.noreply.github.com
git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if [ ! -d "$dir" ]; then
  echo "no record in $dir"
  exit 0
fi
rm -f "$dir"/.*.tmp # a write the job was stopped in: never committed
git add "$dir"
if git diff --cached --quiet; then
  echo "no new record in $dir"
  exit 0
fi
git commit -m "runs: ${dir#runs/} record $(git diff --cached --name-only | xargs -n1 basename | tr '\n' ' ')"
# the brief job also commits runs/ (other files): rebase onto it, retry once
for attempt in 1 2; do
  if git pull --rebase origin main && git push; then
    exit 0
  fi
  git rebase --abort 2>/dev/null || true
  sleep 5
done
echo "push failed after $attempt tries"
exit 1
