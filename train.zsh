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

WORKDIR="${HEIST_WORKDIR:-$(dirname "$(readlink -f "$0")")}"
REPO_URL="${HEIST_REPO_URL:-https://github.com/saejin-moon/heist.git}"
CAMPAIGN_STEPS=299008
CIR_COEF=0.5
CAR_COEF=0.5
STAGES="0"
SKIP_SMOKE=0
DO_PULL=0
DAEMON=0
RUN_EVAL=1
CONCURRENT_JOBS=2
FAST_MODE=0

usage() {
    cat << 'EOF'
train.zsh -- provision a GPU machine, setup dev env, and run HEIST training.

INTENDED FOR TRAINING RUNS (e.g. on target machine or local GPU).

Usage:
    ./train.zsh                      # default stage 0 campaign (300k steps, all models)
    ./train.zsh --fast               # super fast local test (10k steps across all models)
    ./train.zsh --parallel 4         # run up to 4 models concurrently to maximize GPU/CPU
    ./train.zsh --steps 20480        # custom step budget for tuning
    ./train.zsh --stages 0,1,2       # run selected stages (0, 1, 2)
    ./train.zsh --daemon             # background the campaign with nohup
    ./train.zsh -h, --help           # show this help message

Flags:
  --fast, --quick   fast local validation test (10k steps, skips smoke test)
  --parallel, -j N  number of concurrent model training jobs (default 2)
  --steps STEPS     total training timesteps per algorithm (default 299008)
  --stages STAGES   comma-separated curriculum stage indices (default 0)
  --num-stages N    run first N curriculum stages (0..N-1)
  --cir-coef COEF   CIR coefficient for routing advantages (default 0.5)
  --car-coef COEF   CAR coefficient for intrinsic affordance rewards (default 0.5)
  --workdir DIR     repo directory (default $HOME/heist)
  --repo-url URL    clone URL (default the HEIST GitHub repo)
  --daemon          background the campaign with nohup; logs to log/launch.out
  --no-eval         skip evaluation scripts after training completes
  --skip-smoke      skip the 2048-step GPU smoke test before campaign
  --pull            git pull an existing --workdir clone before running
  -h, --help        show this help
EOF
    exit 0
}

RAW_ARGS=("$@")
USE_RND=0
RND_COEF=0.05

while (( $# )); do
    case "$1" in
        --workdir)    WORKDIR="$2"; shift 2 ;;
        --repo-url)   REPO_URL="$2"; shift 2 ;;
        --stages)     STAGES="$2"; shift 2 ;;
        --steps)      CAMPAIGN_STEPS="$2"; shift 2 ;;
        --fast|--quick|--sample) FAST_MODE=1; CAMPAIGN_STEPS=10240; STAGES="0"; SKIP_SMOKE=1; shift ;;
        --parallel|-j) CONCURRENT_JOBS="$2"; shift 2 ;;
        --cir-coef)   CIR_COEF="$2"; shift 2 ;;
        --car-coef)   CAR_COEF="$2"; shift 2 ;;
        --rnd)        USE_RND=1; shift ;;
        --rnd-coef)   USE_RND=1; RND_COEF="$2"; shift 2 ;;
        --num-stages)
            N="$2"; shift 2
            STAGES=$(python3 -c "print(','.join(str(i) for i in range($N)))")
            ;;
        --daemon)     DAEMON=1; shift ;;
        --eval)       RUN_EVAL=1; shift ;;
        --no-eval)    RUN_EVAL=0; shift ;;
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --pull)       DO_PULL=1; shift ;;
        -h|--help)    usage ;;
        *) print -u2 "error: unknown argument: $1"; usage ;;
    esac
done


SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
fi

log() { print -- "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }
step() { print; print "────────────────────────────────────────────────────────"; print "==> $*"; }

# ---------------------------------------------------------------------------
# 1. System packages & environment setup
# ---------------------------------------------------------------------------
step "1/7  System packages & environment setup"
if ! command -v git >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1 || ! command -v zsh >/dev/null 2>&1; then
    log "Installing system dependencies..."
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq git python3 python3-pip python3-venv zsh build-essential curl
else
    log "System dependencies (git, python3, zsh) already satisfied."
fi

# ── 2. GPU check ──────────────────────────────────────────────────────────
step "2/7  GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    log "No nvidia-smi found. Will use CPU or default PyTorch device."
fi

# ── 3. uv package manager ────────────────────────────────────────────────
step "3/7  uv package manager"
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
uv --version

# ── 4. Repository setup ─────────────────────────────────────────────────
step "4/7  Repository setup"
if [ ! -d "$WORKDIR/.git" ]; then
    log "Cloning HEIST repository to $WORKDIR..."
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
else
    log "Repository already exists at $WORKDIR."
    cd "$WORKDIR"
    if [ "$DO_PULL" -eq 1 ]; then
        log "Pulling latest changes..."
        git pull origin main || true
    fi
fi

# ── 5. Dev environment sync ─────────────────────────────────────────────
step "5/7  Dev environment sync (uv sync --locked)"
uv sync --locked

# ── 6. Validation gates & tests ─────────────────────────────────────────
step "6/7  Validation gates & tests"
uv run ruff check --fix
uv run ruff format
uv run pytest -q
log "All checks passed!"

# Smoke test (optional)
if [ "$SKIP_SMOKE" -eq 0 ]; then
    log "running smoke test..."
    STAGE0='{"map_size": [11, 11], "num_rooms_range": [1, 2], "guard_count": 0, "camera_count": 0, "door_count": 0, "max_steps": 60, "spawn_mode": "role"}'
    uv run python src/train_ippo.py --total-timesteps 2048 --num-envs 8 --num-steps 256 --no-save-model --env-config "$STAGE0" --exp-name smoke
    rm -rf runs/smoke_s0
fi

# ── 7. Launch campaign for selected stages ───────────────────────────
step "7/7  Launching campaign for stages: $STAGES"
mkdir -p log

trigger_eval() {
    local name="$1"
    local s="$2"
    local run_id="$3"
    local steps="$4"
    log "-> Launching background evaluation for $name (stage $s) ..."
    .venv/bin/python -u src/eval_stage.py --stage "$s" --algo "$name" --run-id "$run_id" --steps "$steps" > "log/eval_${name}_s${s}.log" 2>&1 &
}

run_stage() {
    local s="$1"
    log "Starting training for stage $s"
    local cfg
    cfg=$(uv run python -c "import sys; sys.path.insert(0, 'src'); from curriculum import CURRICULUM, env_config_str; print(env_config_str(CURRICULUM[$s]))")
    
    local -a model_names=("ippo" "mappo" "mappo_car" "comm" "comm_cir" "comm_cir_car" "qmix")
    local -a pids=()
    local -a status_map=()
    local -a eval_pids=()

    for ((i=1; i<=${#model_names[@]}; i++)); do
        pids[$i]=0
        status_map[$i]="queued"
    done

    local steps_for_stage=$CAMPAIGN_STEPS
    if [ "$FAST_MODE" -eq 0 ]; then
        steps_for_stage=$(( CAMPAIGN_STEPS * (2 + s) / 2 ))
    fi

    if [ "$FAST_MODE" -eq 1 ]; then
        log "[FAST MODE] High-verbosity validation active ($steps_for_stage steps/model)"
        log "[FAST MODE] Launching ${#model_names[@]} models concurrently (max $CONCURRENT_JOBS parallel jobs)..."
    else
        log "Starting training campaign for stage $s with adaptive steps: $steps_for_stage"
        log "Training models concurrently (max $CONCURRENT_JOBS parallel jobs)..."
    fi

    for ((i=1; i<=${#model_names[@]}; i++)); do
        local name="${model_names[$i]}"
        
        # Wait if we hit the concurrency limit
        local active_count=999
        while [ "$active_count" -ge "$CONCURRENT_JOBS" ]; do
            active_count=0
            for ((j=1; j<=${#model_names[@]}; j++)); do
                if [ "${status_map[$j]}" = "running" ]; then
                    local pid="${pids[$j]}"
                    if kill -0 "$pid" 2>/dev/null; then
                        ((++active_count))
                    else
                        status_map[$j]="done"
                        log "✓ Model '${model_names[$j]}' completed."
                        trigger_eval "${model_names[$j]}" "$s" "$EVAL_RUN_ID" "$steps_for_stage"
                        eval_pids+=($!)
                    fi
                fi
            done
            if [ "$active_count" -ge "$CONCURRENT_JOBS" ]; then
                sleep 1.5
            fi
        done

        local -a rnd_flags=()
        if [ "$USE_RND" -eq 1 ]; then
            rnd_flags=("--use-rnd" "--rnd-coef" "$RND_COEF")
        fi

        log "-> Launching $name ..."
        case "$name" in
            ippo)
                nohup .venv/bin/python -u src/train_ippo.py --total-timesteps "$steps_for_stage" --num-envs 8 --num-steps 256 --no-cuda --seed 0 --env-config "$cfg" --exp-name "ippo_s${s}" "${rnd_flags[@]}" > "log/${name}_s${s}.log" 2>&1 &
                ;;
            mappo)
                nohup .venv/bin/python -u src/train_mappo.py --total-timesteps "$steps_for_stage" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "mappo_s${s}" "${rnd_flags[@]}" > "log/${name}_s${s}.log" 2>&1 &
                ;;
            mappo_car)
                nohup .venv/bin/python -u src/train_mappo.py --total-timesteps "$steps_for_stage" --num-envs 8 --num-steps 256 --car-coef "$CAR_COEF" --seed 0 --env-config "$cfg" --exp-name "mappo_car_s${s}" "${rnd_flags[@]}" > "log/${name}_s${s}.log" 2>&1 &
                ;;
            comm)
                nohup .venv/bin/python -u src/train_comm.py --total-steps "$steps_for_stage" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "comm_s${s}" --save-model "${rnd_flags[@]}" > "log/${name}_s${s}.log" 2>&1 &
                ;;
            comm_cir)
                nohup .venv/bin/python -u src/train_comm.py --total-steps "$steps_for_stage" --num-envs 8 --num-steps 256 --cir-coef "$CIR_COEF" --env-config "$cfg" --exp-name "comm_cir_s${s}" --save-model "${rnd_flags[@]}" > "log/${name}_s${s}.log" 2>&1 &
                ;;
            comm_cir_car)
                nohup .venv/bin/python -u src/train_comm.py --total-steps "$steps_for_stage" --num-envs 8 --num-steps 256 --cir-coef "$CIR_COEF" --car-coef "$CAR_COEF" --env-config "$cfg" --exp-name "comm_cir_car_s${s}" --save-model "${rnd_flags[@]}" > "log/${name}_s${s}.log" 2>&1 &
                ;;
            qmix)
                nohup .venv/bin/python -u src/train_qmix.py --total-steps "$steps_for_stage" --train-freq 4 --seed 0 --env-config "$cfg" --exp-name "qmix_s${s}" "${rnd_flags[@]}" > "log/${name}_s${s}.log" 2>&1 &
                ;;
        esac
        
        local last_pid=$!
        pids[$i]=$last_pid
        log "-> Launched $name with PID: ${pids[$i]}"
        status_map[$i]="running"
    done

    # Wait for all remaining jobs and monitor progress if FAST_MODE
    local running=1
    while [ "$running" -eq 1 ]; do
        sleep 2
        running=0
        local status_str=""
        for ((j=1; j<=${#model_names[@]}; j++)); do
            local name="${model_names[$j]}"
            local stat="${status_map[$j]}"
            if [ "$stat" = "running" ]; then
                local pid="${pids[$j]}"
                if kill -0 "$pid" 2>/dev/null; then
                    running=1
                    # Read latest progress from log
                    local log_file="log/${name}_s${s}.log"
                    local prog="starting"
                    if [ -f "$log_file" ]; then
                        local last_line
                        last_line=$(grep -E "update=|step=" "$log_file" | tail -n 1 || true)
                        if [ -n "$last_line" ]; then
                            prog=$(echo "$last_line" | awk '{print $1" "$2}')
                        fi
                    fi
                    status_str="${status_str} | ${name}: ${prog}"
                else
                    status_map[$j]="done"
                    log "✓ Model '$name' completed."
                    status_str="${status_str} | ${name}: done"
                    trigger_eval "$name" "$s" "$EVAL_RUN_ID" "$steps_for_stage"
                    eval_pids+=($!)
                fi
            else
                status_str="${status_str} | ${name}: ${stat}"
            fi
        done
        if [ "$FAST_MODE" -eq 1 ] && [ "$running" -eq 1 ]; then
            printf "\r[FAST MODE] Progress:%s" "$status_str"
        fi
    done
    if [ "$FAST_MODE" -eq 1 ]; then
        printf "\n[FAST MODE] All models completed successfully.\n"
    fi

    log "Waiting for background evaluations to complete..."
    for pid in "${eval_pids[@]}"; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 0.5
        done
    done

    log "Merging evaluation results..."
    .venv/bin/python src/eval_stage.py --stage "$s" --merge --run-id "$EVAL_RUN_ID" --steps "$steps_for_stage"
}


IFS=',' read -A STAGE_LIST <<< "$STAGES"
EVAL_RUN_ID=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from eval_stage import get_next_run_id; print(get_next_run_id())")
log "Campaign Evaluation Run ID: $EVAL_RUN_ID"

if [ "$DAEMON" -eq 1 ]; then
    log "Launching campaign as background daemon..."
    local -a pass_args=()
    for arg in "${RAW_ARGS[@]}"; do
        if [ "$arg" != "--daemon" ]; then
            pass_args+=("$arg")
        fi
    done
    nohup zsh "$0" "${pass_args[@]}" > log/launch.out 2>&1 &
    print "Campaign launched as daemon (PID $!). Check log status with: uv run python tools/status.py"
else
    for stg in "${STAGE_LIST[@]}"; do
        run_stage "$stg"
    done
    log "All selected stages finished successfully!"
    print "Check final status with: uv run python tools/status.py"
fi
