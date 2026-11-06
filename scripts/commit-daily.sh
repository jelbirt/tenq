#!/usr/bin/env bash
# Commit and push runs/daily/ to main. Used by daily.yml twice: after the claim
# (the email is sent only if this push worked) and after the send.
# Needs GH_TOKEN. Exits 1 if the push fails.
set -euo pipefail

git config user.name tenq-bot
git config user.email bot@users.noreply.github.com
git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if [ ! -d runs/daily ]; then
  echo "no daily record"
  exit 0
fi
rm -f runs/daily/.*.tmp # a write the job was stopped in: never committed
git add runs/daily
if git diff --cached --quiet; then
  echo "no new daily record"
  exit 0
fi
git commit -m "runs: daily record $(git diff --cached --name-only | xargs -n1 basename | tr '\n' ' ')"
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
