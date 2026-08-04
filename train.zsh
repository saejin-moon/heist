#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# train.zsh -- provision a GPU machine, setup dev env, and run HEIST training.
#
# INTENDED FOR TRAINING RUNS (e.g. on target machine or local GPU).
#
# Usage:
#     ./train.zsh                      # default stage 0 campaign
#     ./train.zsh --stages 0,1,2       # run selected stages (0, 1, 2)
#     ./train.zsh --num-stages 5       # run stages 0 through 4
#     ./train.zsh --daemon             # background the campaign with nohup
#     ./train.zsh --eval               # run eval script after campaign
#     ./train.zsh -h, --help           # show this help message
#
# Flags:
#   --stages STAGES   comma-separated curriculum stage indices (default 0)
#   --num-stages N    run first N curriculum stages (0..N-1)
#   --workdir DIR     repo directory (default $HOME/heist)
#   --repo-url URL    clone URL (default the HEIST GitHub repo)
#   --daemon          background the campaign with nohup; logs to log/launch.out
#   --eval            run evaluation scripts after training completes
#   --skip-smoke      skip the 2048-step GPU smoke test before campaign
#   --pull            git pull an existing --workdir clone before running
#   -h, --help        show this help
# ---------------------------------------------------------------------------
set -euo pipefail

WORKDIR="${HEIST_WORKDIR:-$HOME/heist}"
REPO_URL="${HEIST_REPO_URL:-https://github.com/saejin-moon/heist.git}"
CAMPAIGN_STEPS=299008
CIR_COEF=0.5
CAR_COEF=0.5
STAGES="0"
SKIP_SMOKE=0
DO_PULL=0
DAEMON=0
RUN_EVAL=0

usage() {
    cat << 'EOF'
train.zsh -- provision a GPU machine, setup dev env, and run HEIST training.

INTENDED FOR TRAINING RUNS (e.g. on target machine or local GPU).

Usage:
    ./train.zsh                      # default stage 0 campaign (300k steps)
    ./train.zsh --sample             # quick sample test (10k steps across all algos)
    ./train.zsh --steps 10240        # custom step budget for tuning
    ./train.zsh --stages 0,1,2       # run selected stages (0, 1, 2)
    ./train.zsh --cir-coef 0.2       # custom CIR coefficient
    ./train.zsh --car-coef 0.3       # custom CAR coefficient
    ./train.zsh --daemon             # background the campaign with nohup
    ./train.zsh -h, --help           # show this help message

Flags:
  --sample          shortcut for a quick 10,240 step sample test
  --steps STEPS     total training timesteps per algorithm (default 299008)
  --stages STAGES   comma-separated curriculum stage indices (default 0)
  --num-stages N    run first N curriculum stages (0..N-1)
  --cir-coef COEF   CIR coefficient for routing advantages (default 0.5)
  --car-coef COEF   CAR coefficient for intrinsic affordance rewards (default 0.5)
  --workdir DIR     repo directory (default $HOME/heist)
  --repo-url URL    clone URL (default the HEIST GitHub repo)
  --daemon          background the campaign with nohup; logs to log/launch.out
  --eval            run evaluation scripts after training completes
  --skip-smoke      skip the 2048-step GPU smoke test before campaign
  --pull            git pull an existing --workdir clone before running
  -h, --help        show this help
EOF
    exit 0
}

while (( $# )); do
    case "$1" in
        --workdir)    WORKDIR="$2"; shift 2 ;;
        --repo-url)   REPO_URL="$2"; shift 2 ;;
        --stages)     STAGES="$2"; shift 2 ;;
        --steps)      CAMPAIGN_STEPS="$2"; shift 2 ;;
        --sample)     CAMPAIGN_STEPS=10240; STAGES="0"; SKIP_SMOKE=1; shift ;;
        --cir-coef)   CIR_COEF="$2"; shift 2 ;;
        --car-coef)   CAR_COEF="$2"; shift 2 ;;
        --num-stages)
            N="$2"; shift 2
            STAGES=$(python3 -c "print(','.join(str(i) for i in range($N)))")
            ;;
        --daemon)     DAEMON=1; shift ;;
        --eval)       RUN_EVAL=1; shift ;;
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --pull)       DO_PULL=1; shift ;;
        -h|--help)    usage ;;
        *) print -u2 "error: unknown argument: $1"; usage ;;
    esac
done


# ... (system packages, gpu check, uv setup, gates) ...


SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
fi

log() { print -- "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }
step() { print; print "────────────────────────────────────────────────────────"; print "==> $*"; }

# ── 1. System package installation & Dev Env setup ────────────────────
step "1/7  System packages & environment setup"
if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -y
    $SUDO apt-get install -y git curl ca-certificates zsh python3 python3-pip python3-venv build-essential
elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y git curl ca-certificates zsh python3 python3-pip gcc gcc-c++ make
elif command -v pacman >/dev/null 2>&1; then
    $SUDO pacman -Sy --noconfirm git curl ca-certificates zsh python python-pip base-devel
elif command -v brew >/dev/null 2>&1; then
    brew install git curl python zsh
fi

# ── 2. GPU Check ─────────────────────────────────────────────────────
step "2/7  GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    log "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed if running on GPU."
fi

# ── 3. install uv ────────────────────────────────────────────────────
step "3/7  uv package manager"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv package manager"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ── 4. clone/pull repo ────────────────────────────────────────────────
step "4/7  Repository setup"
if [ ! -d "$WORKDIR" ]; then
    log "cloning $REPO_URL into $WORKDIR"
    git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
if [ "$DO_PULL" -eq 1 ]; then
    log "pulling latest changes"
    git pull --ff-only
fi

# ── 5. dev env sync ──────────────────────────────────────────────────
step "5/7  Dev environment sync (uv sync --locked)"
if ! uv python find 3.12 >/dev/null 2>&1; then
    log "installing CPython 3.12"
    uv python install 3.12
fi
uv sync --locked

# ── 6. gates & tests ─────────────────────────────────────────────────
step "6/7  Validation gates & tests"
uv run ruff check
uv run ruff format --check
uv run pytest -q

if ! uv run python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    log "WARNING: PyTorch CUDA is not available. Reinstalling PyTorch with CUDA wheels..."
    uv pip install --reinstall "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121
fi


if [ "$SKIP_SMOKE" -eq 0 ]; then
    log "running smoke test..."
    STAGE0='{"map_size": [11, 11], "num_rooms_range": [1, 2], "guard_count": 0, "camera_count": 0, "door_count": 0, "max_steps": 60, "spawn_mode": "role"}'
    uv run python src/train_ippo.py --total-timesteps 2048 --num-envs 8 --num-steps 256 --no-save-model --env-config "$STAGE0" --exp-name smoke
    rm -rf runs/smoke_s0
fi

# ── 7. Launch campaign for selected stages ───────────────────────────
step "7/7  Launching campaign for stages: $STAGES"
mkdir -p log

run_stage() {
    local s="$1"
    log "Starting training for stage $s"
    local cfg
    cfg=$(uv run python -c "from curriculum import CURRICULUM, env_config_str; print(env_config_str(CURRICULUM[$s]))")
    uv run python src/train_ippo.py --total-timesteps "$CAMPAIGN_STEPS" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "ippo_s${s}" 2>&1 | tee "log/ippo_s${s}.log"
    uv run python src/train_mappo.py --total-timesteps "$CAMPAIGN_STEPS" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "mappo_s${s}" 2>&1 | tee "log/mappo_s${s}.log"
    if (( $(echo "$CAR_COEF > 0" | bc -l 2>/dev/null || echo 1) )); then
        uv run python src/train_mappo.py --total-timesteps "$CAMPAIGN_STEPS" --num-envs 8 --num-steps 256 --car-coef "$CAR_COEF" --env-config "$cfg" --seed 0 --exp-name "mappo_car_s${s}" 2>&1 | tee "log/mappo_car_s${s}.log"
    fi
    uv run python src/train_comm.py --total-steps "$CAMPAIGN_STEPS" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "comm_s${s}" 2>&1 | tee "log/comm_s${s}.log"
    if (( $(echo "$CIR_COEF > 0" | bc -l 2>/dev/null || echo 1) )); then
        uv run python src/train_comm.py --total-steps "$CAMPAIGN_STEPS" --num-envs 8 --num-steps 256 --cir-coef "$CIR_COEF" --env-config "$cfg" --exp-name "comm_cir_s${s}" 2>&1 | tee "log/comm_cir_s${s}.log"
        uv run python src/train_comm.py --total-steps "$CAMPAIGN_STEPS" --num-envs 8 --num-steps 256 --cir-coef "$CIR_COEF" --car-coef "$CAR_COEF" --env-config "$cfg" --exp-name "comm_cir_car_s${s}" 2>&1 | tee "log/comm_cir_car_s${s}.log"
    fi
    uv run python src/train_qmix.py --total-steps "$CAMPAIGN_STEPS" --train-freq 4 --seed 0 --env-config "$cfg" --exp-name "qmix_s${s}" 2>&1 | tee "log/qmix_s${s}.log"
}


IFS=',' read -A STAGE_LIST <<< "$STAGES"

if [ "$DAEMON" -eq 1 ]; then
    log "Launching campaign as background daemon..."
    for stg in "${STAGE_LIST[@]}"; do
        nohup zsh -c "run_stage $stg" > log/launch.out 2>&1 &
    done
    print "Campaign launched. Check log status with: uv run python tools/status.py"
else
    for stg in "${STAGE_LIST[@]}"; do
        run_stage "$stg"
    done
    log "All selected stages finished successfully!"
    if [ "$RUN_EVAL" -eq 1 ]; then
        log "Running evaluation..."
        uv run python src/eval_stage0_300k.py || true
    fi
    print "Check final status with: uv run python tools/status.py"
fi
