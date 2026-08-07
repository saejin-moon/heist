#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# scripts/side-tasks.zsh -- Launch side-task ablation campaign on target machine.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
exec "$SCRIPT_DIR/train.zsh" -j 5 --side-tasks --stages 0,1,2,3,4 --from-stage 4 --daemon --steps 1000000 "$@"