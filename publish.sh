#!/usr/bin/env bash
# publish.sh — copy latest figures and push to GitHub
# Usage:  ./publish.sh "commit message"
#         ./publish.sh              (defaults to "update results")

set -euo pipefail
cd "$(dirname "$0")"

MSG="${1:-update results}"

echo "==> Copying figures from code/output/benchmark/ …"
cp -v code/output/benchmark/comparison.png figures/

echo ""
echo "==> Staging changes …"
git add -A
git status --short

echo ""
echo "==> Committing: $MSG"
git commit -m "$MSG" || echo "(nothing to commit)"

echo ""
echo "==> Pushing …"
git push

echo ""
echo "✅  Published. View at:"
git remote get-url origin | sed 's/\.git$//' | sed 's|git@github.com:|https://github.com/|'
