#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# scripts/assess-time.zsh -- Run throughput benchmark on target machine.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

for cmd in uv git; do
    if ! command -v $cmd >/dev/null 2>&1; then
        echo "error: $cmd not found." >&2
        exit 1
    fi
done

echo "Running tools/assess_time.py ..."
uv run python tools/assess_time.py "$@"
