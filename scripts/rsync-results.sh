#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/rsync-results.sh -- Production rsync script to sync results & checkpoints
# from target machine (bae@forest.local) to local machine.
#
# Usage:
#     ./scripts/rsync-results.sh [local_dest]
# ---------------------------------------------------------------------------
set -euo pipefail

TARGET_USER="${HEIST_TARGET_USER:-bae}"
TARGET_HOST="${HEIST_TARGET_HOST:-forest.local}"
REMOTE_PATH="${HEIST_REMOTE_PATH:-heist}"
LOCAL_DEST="${1:-.}"

echo "Syncing results and checkpoints from ${TARGET_USER}@${TARGET_HOST}:${REMOTE_PATH} ..."

rsync -avzP \
    --include="results/*** \
    --include="checkpoints/*** \
    --exclude="*" \
    "${TARGET_USER}@${TARGET_HOST}:${REMOTE_PATH}/" \
    "${LOCAL_DEST}/"

echo "Sync complete!"
