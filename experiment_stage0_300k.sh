#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage-0 research experiment (FUTURE_PLANS.md §1)
#
# ⚠️⚠️⚠️  DO NOT RUN THIS YET.  ⚠️⚠️⚠️
# The full 300k-step campaign is intentionally NOT scheduled for this
# machine.  Everything below is SETUP only: the campaign will be launched
# later on a different (more powerful / dedicated) machine.
#
# Rules while we are still setting up:
#   * Do NOT execute this script end-to-end.
#   * Only run short smoke tests of the individual trainers (see
#     `src/train_*.py` `--total-steps 2048` style invocations in README).
#   * If a full campaign is ever desired, run it on the target machine
#     following the README "Full research experiment" section, and update
#     FUTURE_PLANS.md + PLAN.md before/after.
#
# Status: see log/experiment_stage0_300k.log (an early partial campaign
# was aborted; its checkpoints were quarantined to deleted/).
# ---------------------------------------------------------------------------
# What this script will do (once approved):
#
# Run IPPO (Phase A floor), MAPPO, QMIX, and train_comm at 300k steps on
# stage-0 (seeds 0-2).  Resolves the deferred G4 claim "comm >= Phase A floor".
#
# Runs sequentially on one GPU.  Each run: ~8-50 min depending on algo.
# Total budget: ~4-8 hours on a typical CUDA GPU.
# (Measured with tools/assess_time.py on an RTX 3000 Ada; note the trainer
#  logs sps in rollout-iterations/s -- multiply by num_envs for env-steps/s.)
#
# Logs:   log/experiment_stage0_300k.log
# Runs:   runs/<algo>_s<seed>/
# Ckpts:  checkpoints/<algo>_s<seed>/
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

LOG="log/experiment_stage0_300k.log"
mkdir -p log

STAGE0='{"map_size": [11, 11], "num_rooms_range": [1, 2], "guard_count": 0, "camera_count": 0, "door_count": 0, "max_steps": 60, "spawn_mode": "role"}'

run() {
    local label="$1"; shift
    echo "========================================================================" | tee -a "$LOG"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] START $label" | tee -a "$LOG"
    echo "========================================================================" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] OK   $label" | tee -a "$LOG"
    else
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] FAIL $label (exit $rc)" | tee -a "$LOG"
    fi
}

# Skip a run whose checkpoint dir already exists (resume support).
# Trainers save to checkpoints/<run_name> (repo root), see train_*.py.
already_done() {
    local dir="$1"; shift
    [ -n "$(ls -A "checkpoints/$dir" 2>/dev/null)" ]
}

SEEDS=(0 1 2)

echo "Experiment started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG"
echo "GPU: $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')" | tee -a "$LOG"

# ── IPPO (Phase A floor) ─────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    if already_done "ippo_s0_s${seed}"; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SKIP ippo s${seed} (checkpoint exists)" | tee -a "$LOG"
        continue
    fi
    run "ippo s${seed}" \
        uv run python src/train_ippo.py \
        --total-timesteps 300000 --num-envs 8 --num-steps 256 \
        --eval-every 20 --eval-episodes 20 --seed "$seed" \
        --env-config "$STAGE0" \
        --exp-name ippo_s0
done

# ── MAPPO ─────────────────────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    if already_done "mappo_s0_s${seed}"; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SKIP mappo s${seed} (checkpoint exists)" | tee -a "$LOG"
        continue
    fi
    run "mappo s${seed}" \
        uv run python src/train_mappo.py \
        --total-timesteps 300000 --num-envs 8 --num-steps 256 \
        --eval-every 20 --eval-episodes 20 --seed "$seed" \
        --env-config "$STAGE0" \
        --exp-name mappo_s0
done

# ── QMIX ──────────────────────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    if already_done "qmix_s0_s${seed}"; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SKIP qmix s${seed} (checkpoint exists)" | tee -a "$LOG"
        continue
    fi
    run "qmix s${seed}" \
        uv run python src/train_qmix.py \
        --total-steps 300000 --eval-every 5000 --eval-episodes 20 --seed "$seed" \
        --env-config "$STAGE0" \
        --exp-name qmix_s0
done

# ── Comm (REV-7) ─────────────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    if already_done "comm_s0_s${seed}"; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SKIP comm s${seed} (checkpoint exists)" | tee -a "$LOG"
        continue
    fi
    run "comm s${seed}" \
        uv run python src/train_comm.py \
        --total-steps 300000 --num-envs 8 --num-steps 256 \
        --eval-every 5000 --eval-episodes 20 --seed "$seed" \
        --env-config "$STAGE0" \
        --exp-name comm_s0
done

echo "" | tee -a "$LOG"
echo "========================================================================" | tee -a "$LOG"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] EXPERIMENT COMPLETE" | tee -a "$LOG"
echo "========================================================================" | tee -a "$LOG"
