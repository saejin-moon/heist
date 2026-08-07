#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/evaluate.sh -- Production evaluation helper script.
#
# Usage:
#     ./scripts/evaluate.sh [results_dir] [episodes]
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

RUN_DIR="${1:-results/run001}"
EPISODES="${2:-50}"

echo "Running full evaluation suite on ${RUN_DIR} with ${EPISODES} episodes per stage ..."
uv run python tools/evaluate_campaign.py --results-dir "$RUN_DIR" --eval-episodes "$EPISODES" "${@:3}"
