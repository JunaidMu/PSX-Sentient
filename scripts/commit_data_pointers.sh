#!/usr/bin/env sh
# Record today's data version in git: commit the DVC pointer files (*.dvc + dvc.lock) to the `data`
# branch and push it to GitHub. `main` and the working tree are untouched (git plumbing on a temporary
# index), so this can run on the server alongside deployments.
#
#   git checkout data -- psx_historical_data/daily.dvc && dvc pull psx_historical_data/daily.dvc
# restores the daily files exactly as they were on that day.
#
# Needs GITHUB_DATA_TOKEN in .env: a fine-grained GitHub token with "Contents: read and write" on this
# repository only. Without it the step is skipped (S3 snapshots from src.monitoring.backup still exist).
set -u
cd "${REPO_DIR:-/app}"
git config --global --add safe.directory "$(pwd)" 2>/dev/null

if [ -z "${GITHUB_DATA_TOKEN:-}" ]; then
  echo "data branch: skipped (GITHUB_DATA_TOKEN not set)"
  exit 0
fi

REPO="${GITHUB_REPOSITORY:-JunaidMu/PSX-Sentient}"
REMOTE="https://x-access-token:${GITHUB_DATA_TOKEN}@github.com/${REPO}.git"
DAY="$(date -u +%Y-%m-%d)"
INDEX="$(mktemp)"
export GIT_INDEX_FILE="$INDEX" GIT_AUTHOR_NAME="psx-sentient-bot" GIT_AUTHOR_EMAIL="psx-sentient-bot@users.noreply.github.com"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

PARENT=""
if git fetch -q "$REMOTE" "+refs/heads/data:refs/remotes/data-branch" 2>/dev/null; then
  PARENT="$(git rev-parse refs/remotes/data-branch)"
  git read-tree "$PARENT"
else
  git read-tree --empty
fi

for f in psx_historical_data/*.dvc psx_historical_data/new_batch/*.dvc dvc.lock; do
  [ -f "$f" ] && git update-index --add "$f"
done
TREE="$(git write-tree)"
rm -f "$INDEX"
unset GIT_INDEX_FILE

if [ -n "$PARENT" ] && [ "$(git rev-parse "$PARENT^{tree}")" = "$TREE" ]; then
  echo "data branch: no data changes since the last snapshot"
  exit 0
fi
if [ -n "$PARENT" ]; then
  COMMIT="$(git commit-tree "$TREE" -p "$PARENT" -m "Data snapshot $DAY")"
else
  COMMIT="$(git commit-tree "$TREE" -m "Data snapshot $DAY")"
fi
if git push -q "$REMOTE" "$COMMIT:refs/heads/data" 2>/dev/null; then
  echo "data branch: pushed snapshot $DAY ($COMMIT)"
else
  echo "data branch: push failed (check GITHUB_DATA_TOKEN permissions)"
fi
