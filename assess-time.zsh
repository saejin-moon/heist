#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# assess-time.zsh -- run the throughput benchmark on the target machine.
#
# Usage (on the target machine, from the repo root):
#     ./assess-time.zsh                       # full run (~5-10 min)
#     ./assess-time.zsh --trainer-steps 10240 # faster, less precise
#     ./assess-time.zsh --algos ippo comm     # subset
#
# All flags are forwarded to tools/assess_time.py.
# Requires: uv, git, a CUDA-capable GPU with an NVIDIA driver.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# ── preflight checks ─────────────────────────────────────────────────
for cmd in uv git; do
    if ! command -v $cmd >/dev/null 2>&1; then
        echo "error: $cmd not found." >&2
        if [ "$cmd" = "uv" ]; then
            echo "install uv first:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        fi
        exit 1
    fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found; torch CUDA will likely not work." >&2
    echo "         Install the NVIDIA driver for your GPU." >&2
fi

# ── run the benchmark ────────────────────────────────────────────────
echo "Running tools/assess_time.py ..."
echo "All flags are forwarded: $@"
echo ""

uv run python tools/assess_time.py "$@"
