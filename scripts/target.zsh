#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# scripts/target.zsh -- Launch full 5-stage campaign on target machine.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
exec "$SCRIPT_DIR/train.zsh" -j 5 --daemon --steps 1000000 --stages 0,1,2,3,4 --resume "$@"