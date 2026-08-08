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
#   --models MODELS   comma-separated model names to train (e.g. coma or ippo,mappo)
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

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WORKDIR="${HEIST_WORKDIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_URL="${HEIST_REPO_URL:-https://github.com/saejin-moon/heist.git}"
CAMPAIGN_STEPS=299008
CIR_COEF=0.5
CAR_COEF=0.5
STAGES="0"
MODELS=""
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
    ./train.zsh --models coma        # train only specific model(s) (e.g. coma, ippo, mappo)
    ./train.zsh --fast               # super fast local test (10k steps across all models)
    ./train.zsh --parallel 4         # run up to 4 models concurrently to maximize GPU/CPU
    ./train.zsh --steps 20480        # custom step budget for tuning
    ./train.zsh --stages 0,1,2       # run selected stages (0, 1, 2)
    ./train.zsh --stages 0,1 --resume # skip completed stage 0 and warm-start stage 1 from stage 0
    ./train.zsh --daemon             # background the campaign with nohup
    ./train.zsh -h, --help           # show this help message

Flags:
  --models, --model MODELS  comma-separated model names to train (e.g. coma or ippo,mappo)
  --fast, --quick   fast local validation test (10k steps, skips smoke test)
  --parallel, -j N  number of concurrent model training jobs (default 2)
  --steps STEPS     total training timesteps per algorithm (default 299008)
  --stages STAGES   comma-separated curriculum stage indices (default 0)
  --num-stages N    run first N curriculum stages (0..N-1)
  --resume, --use-ckpt  reuse completed model checkpoints and skip already trained stages
  --from-stage N    initialize model weights from completed stage N checkpoints
  --cir-coef COEF   CIR coefficient for routing advantages (default 0.5)
  --car-coef COEF   CAR coefficient for intrinsic affordance rewards (default 0.5)
  --no-rnd          disable Random Network Distillation (RND is enabled by default)
  --side-tasks      enable dynamic side-tasks (decoy ping, door override, wall breach, beacon calibration)
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
USE_RND=1
RND_COEF=0.05
ENABLE_SIDE_TASKS=0
CUSTOM_PREFIX=""
USE_CKPT=0
FROM_STAGE=""
MAX_CPU_TEMP=""
MAX_GPU_TEMP=""

while (( $# )); do
    case "$1" in
        --workdir)    WORKDIR="$2"; shift 2 ;;
        --repo-url)   REPO_URL="$2"; shift 2 ;;
        --models|--model) MODELS="$2"; shift 2 ;;
        --stages)     STAGES="$2"; shift 2 ;;
        --steps)      CAMPAIGN_STEPS="$2"; shift 2 ;;

        --fast|--quick|--sample) FAST_MODE=1; CAMPAIGN_STEPS=10240; STAGES="0"; SKIP_SMOKE=1; shift ;;
        --turbo) FAST_MODE=1; CAMPAIGN_STEPS=1024; STAGES="0"; SKIP_SMOKE=1; shift ;;
        --parallel|-j) CONCURRENT_JOBS="$2"; shift 2 ;;
        --resume|--use-ckpt|--reuse-checkpoints|--skip-completed) USE_CKPT=1; shift ;;
        --from-stage|--init-stage) FROM_STAGE="$2"; shift 2 ;;
        --cir-coef)   CIR_COEF="$2"; shift 2 ;;
        --car-coef)   CAR_COEF="$2"; shift 2 ;;
        --rnd)        USE_RND=1; shift ;;
        --no-rnd)     USE_RND=0; shift ;;
        --rnd-coef)   USE_RND=1; RND_COEF="$2"; shift 2 ;;
        --side-tasks) ENABLE_SIDE_TASKS=1; shift ;;
        --run-prefix|--prefix) CUSTOM_PREFIX="$2"; shift 2 ;;
        --num-stages)
            N="$2"; shift 2
            STAGES=$(python3 -c "print(','.join(str(i) for i in range($N)))")
            ;;
        --daemon)     DAEMON=1; shift ;;
        --eval)       RUN_EVAL=1; shift ;;
        --no-eval)    RUN_EVAL=0; shift ;;
        --max-cpu-temp) MAX_CPU_TEMP="$2"; shift 2 ;;
        --max-gpu-temp) MAX_GPU_TEMP="$2"; shift 2 ;;
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
    local st_suffix=""
    if [ "$ENABLE_SIDE_TASKS" -eq 1 ]; then
        st_suffix="_st"
    fi
    mkdir -p "log/${run_id}"
    .venv/bin/python -u src/eval_stage.py --stage "$s" --algo "$name" --run-id "$run_id" --steps "$steps" > "log/${run_id}/eval_${name}${st_suffix}_s${s}.log" 2>&1 &
    EVAL_PID=$!
}

run_campaign() {
    local st_py="False"
    local st_suffix=""
    if [ "$ENABLE_SIDE_TASKS" -eq 1 ]; then
        st_py="True"
        st_suffix="_st"
        log "[SIDE-TASKS ENABLED] Running with dynamic side-task environment features!"
    fi

    local -a model_names=()
    if [ -n "$MODELS" ]; then
        IFS=',' read -A model_names <<< "$MODELS"
    else
        model_names=("ippo" "mappo" "coma" "comm" "mappo_car" "mappo_cir" "loo" "ate" "macca" "marc")
    fi

    local -a rnd_flags=()
    if [ "$USE_RND" -eq 1 ]; then
        rnd_flags=("--use-rnd" "--rnd-coef" "$RND_COEF")
    fi

    # Build queue of all tasks across selected stages
    local -a task_stages=()
    local -a task_models=()
    local -a task_steps=()
    local -a task_cfgs=()
    local -a task_pids=()
    local -a task_status=()
    local -a task_start_times=()
    local -a task_eval_pids=()
    local -a task_eval_queued=()
    local -a task_eval_done=()

    # Per-stage metrics tracking
    typeset -A stage_start_times
    typeset -A stage_merged
    typeset -A stage_cfgs

    for stg in "${STAGE_LIST[@]}"; do
        stage_start_times[$stg]=0
        stage_merged[$stg]=0

        local steps_for_stage=$CAMPAIGN_STEPS
        if [ "$FAST_MODE" -eq 0 ]; then
            steps_for_stage=$(uv run python -c "import sys; sys.path.insert(0, 'src'); from curriculum import CURRICULUM; c = CURRICULUM[$stg]; area = c['map_size'][0] * c['map_size'][1]; entities = c['guard_count'] + c['camera_count'] + c['door_count']; mu = 1.0 + 0.10 * entities; print(int(round($CAMPAIGN_STEPS * (area / 121.0) * mu)))")
        fi

        local cfg
        cfg=$(uv run python -c "import sys; sys.path.insert(0, 'src'); from curriculum import CURRICULUM, env_config_str; c = dict(CURRICULUM[$stg]); c['enable_side_tasks'] = $st_py; print(env_config_str(c))")
        stage_cfgs[$stg]="$cfg"

        for name in "${model_names[@]}"; do
            task_stages+=("$stg")
            task_models+=("$name")
            task_steps+=("$steps_for_stage")
            task_cfgs+=("$cfg")
            task_pids+=(0)
            task_status+=("queued")
            task_start_times+=(0)
            task_eval_pids+=(0)
            task_eval_queued+=(0)
            task_eval_done+=(0)
        done
    done

    local total_tasks=${#task_stages[@]}
    log "Starting training campaign for stages: ${STAGE_LIST[*]}"
    log "Total campaign tasks: $total_tasks models across ${#STAGE_LIST[@]} stages (max $CONCURRENT_JOBS parallel jobs)"

    local all_done=0
    while [ "$all_done" -eq 0 ]; do
        local now=$(date +%s)

        # Thermal protection check & hardware temp logging
        local t_args=("--log")
        if [ -n "$MAX_CPU_TEMP" ]; then t_args+=("--max-cpu-temp" "$MAX_CPU_TEMP"); fi
        if [ -n "$MAX_GPU_TEMP" ]; then t_args+=("--max-gpu-temp" "$MAX_GPU_TEMP"); fi
        if ! uv run python tools/thermal_guard.py "${t_args[@]}"; then
            log "[EMERGENCY STOP] High hardware temperature detected! Stopping training campaign for safety."
            pkill -f "src/train_" 2>/dev/null || true
            exit 1
        fi

        # 1. Check running training tasks
        local active_count=0
        for ((t=1; t<=total_tasks; t++)); do
            if [ "${task_status[$t]}" = "running" ]; then
                local pid="${task_pids[$t]}"
                if kill -0 "$pid" 2>/dev/null; then
                    ((++active_count))
                else
                    task_status[$t]="done"
                    local elapsed=$(( now - task_start_times[$t] ))
                    local mins=$(( elapsed / 60 ))
                    local secs=$(( elapsed % 60 ))
                    local s="${task_stages[$t]}"
                    local name="${task_models[$t]}"
                    log "[DONE] Model '${name}${st_suffix}' (stage $s) completed in ${mins}m ${secs}s (${elapsed}s total)."

                    if [ "$RUN_EVAL" -eq 1 ]; then
                        task_eval_queued[$t]=1
                    else
                        task_eval_done[$t]=1
                    fi
                fi
            fi
        done

        # 2. Fill available concurrency slots from queued tasks (enforcing Stage Gating)
        while [ "$active_count" -lt "$CONCURRENT_JOBS" ]; do
            local next_t=0
            for ((t=1; t<=total_tasks; t++)); do
                if [ "${task_status[$t]}" = "queued" ]; then
                    local s="${task_stages[$t]}"
                    local can_launch=1
                    local stage_idx=0
                    for ((idx=1; idx<=${#STAGE_LIST[@]}; idx++)); do
                        if [ "${STAGE_LIST[$idx]}" = "$s" ]; then
                            stage_idx=$idx
                            break
                        fi
                    done
                    if [ "$stage_idx" -gt 1 ]; then
                        local prev_stg="${STAGE_LIST[$((stage_idx - 1))]}"
                        local prev_task_done=0
                        if [ "${stage_merged[$prev_stg]}" -eq 1 ]; then
                            prev_task_done=1
                        else
                            local model_name="${task_models[$t]}"
                            for ((pt=1; pt<=total_tasks; pt++)); do
                                if [ "${task_stages[$pt]}" = "$prev_stg" ] && [ "${task_models[$pt]}" = "$model_name" ]; then
                                    if [ "${task_status[$pt]}" = "done" ] || [ -f "checkpoints/${model_name}${st_suffix}_s${prev_stg}/complete.json" ]; then
                                        prev_task_done=1
                                    fi
                                    break
                                fi
                            done
                        fi
                        if [ "$prev_task_done" -ne 1 ]; then
                            can_launch=0
                        fi
                    fi
                    if [ "$can_launch" -eq 1 ]; then
                        next_t=$t
                        break
                    fi
                fi
            done

            if [ "$next_t" -eq 0 ]; then
                break
            fi

            local s="${task_stages[$next_t]}"
            local name="${task_models[$next_t]}"
            local steps="${task_steps[$next_t]}"
            local cfg="${task_cfgs[$next_t]}"
            local exp_name_tag="${name}${st_suffix}_s${s}"
            local log_name_tag="${EVAL_RUN_ID}/${name}${st_suffix}_s${s}.log"
            mkdir -p "log/${EVAL_RUN_ID}"

            # Skip completed models if --resume / --use-ckpt is active
            if [ "$USE_CKPT" -eq 1 ] && [ -f "checkpoints/${exp_name_tag}/complete.json" ]; then
                log "[SKIP] Model '${name}${st_suffix}' (stage $s) completed checkpoint found at checkpoints/${exp_name_tag}. Skipping training."
                task_status[$next_t]="done"
                task_start_times[$next_t]=$now
                if [ "$RUN_EVAL" -eq 1 ]; then
                    task_eval_queued[$next_t]=1
                else
                    task_eval_done[$next_t]=1
                fi
                continue
            fi

            if [ "${stage_start_times[$s]}" -eq 0 ]; then
                stage_start_times[$s]=$now
                log "Starting training for stage $s at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
            fi

            local -a load_ckpt_flag=()
            if [ -n "$FROM_STAGE" ]; then
                local src_ckpt="checkpoints/${name}${st_suffix}_s${FROM_STAGE}"
                if [ ! -d "$src_ckpt" ]; then
                    for ((s_idx=FROM_STAGE; s_idx>=0; s_idx--)); do
                        if [ -d "checkpoints/${name}_s${s_idx}" ]; then
                            src_ckpt="checkpoints/${name}_s${s_idx}"
                            break
                        fi
                    done
                fi
                if [ -d "$src_ckpt" ]; then
                    load_ckpt_flag=("--load-checkpoint" "$src_ckpt")
                    log "-> Warm-starting ${name}${st_suffix} (stage $s) from checkpoint: ${src_ckpt}"
                fi
            fi

            log "-> Launching ${name}${st_suffix} (stage $s, RND=${USE_RND}) ..."
            rm -f "checkpoints/${exp_name_tag}/complete.json"
            task_start_times[$next_t]=$now

            case "$name" in
                ippo)
                    nohup .venv/bin/python -u src/train_ippo.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --no-cuda --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                mappo)
                    nohup .venv/bin/python -u src/train_mappo.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                mappo_car)
                    nohup .venv/bin/python -u src/train_mappo.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --car-coef "$CAR_COEF" --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                mappo_cir)
                    nohup .venv/bin/python -u src/train_mappo.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --cir-coef "$CIR_COEF" --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                comm)
                    nohup .venv/bin/python -u src/train_comm.py --total-steps "$steps" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" --save-model "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                comm_cir)
                    nohup .venv/bin/python -u src/train_comm.py --total-steps "$steps" --num-envs 8 --num-steps 256 --cir-coef "$CIR_COEF" --env-config "$cfg" --exp-name "$exp_name_tag" --save-model "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                comm_cir_car)
                    nohup .venv/bin/python -u src/train_comm.py --total-steps "$steps" --num-envs 8 --num-steps 256 --cir-coef "$CIR_COEF" --car-coef "$CAR_COEF" --env-config "$cfg" --exp-name "$exp_name_tag" --save-model "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                qmix)
                    nohup .venv/bin/python -u src/train_qmix.py --total-steps "$steps" --train-freq 4 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                coma)
                    nohup .venv/bin/python -u src/train_coma.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                coma_cir)
                    nohup .venv/bin/python -u src/train_coma.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --cir-coef "$CIR_COEF" --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                loo)
                    nohup .venv/bin/python -u src/train_loo.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                ate)
                    nohup .venv/bin/python -u src/train_ate.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                macca)
                    nohup .venv/bin/python -u src/train_macca.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                marc)
                    nohup .venv/bin/python -u src/train_marc.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                marc_no_shielding)
                    nohup .venv/bin/python -u src/train_marc.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --no-shielding --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                marc_no_macro)
                    nohup .venv/bin/python -u src/train_marc.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --no-macro --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
                marc_no_affordance)
                    nohup .venv/bin/python -u src/train_marc.py --total-timesteps "$steps" --num-envs 8 --num-steps 256 --affordance-coef 0.0 --seed 0 --env-config "$cfg" --exp-name "$exp_name_tag" "${rnd_flags[@]}" "${load_ckpt_flag[@]}" > "log/${log_name_tag}" 2>&1 &
                    local launched_pid=$!
                    task_pids[$next_t]=$launched_pid
                    ;;
            esac

            local launched_pid="${task_pids[$next_t]}"
            task_status[$next_t]="running"
            log "-> Launched ${name}${st_suffix} (stage $s) with PID: $launched_pid"
            ((++active_count))
        done

        # 3. Check and launch background evaluation tasks (max 2 concurrent)
        local MAX_EVAL_JOBS=2
        local active_eval_count=0

        for ((t=1; t<=total_tasks; t++)); do
            local eval_pid="${task_eval_pids[$t]}"
            if [[ "$eval_pid" =~ ^[0-9]+$ ]] && [ "$eval_pid" -gt 0 ] && [ "${task_eval_done[$t]}" -eq 0 ]; then
                if kill -0 "$eval_pid" 2>/dev/null; then
                    ((++active_eval_count))
                else
                    task_eval_done[$t]=1
                    log "Background evaluation for ${task_models[$t]}${st_suffix} (stage ${task_stages[$t]}) completed."
                fi
            fi
        done

        while [ "$active_eval_count" -lt "$MAX_EVAL_JOBS" ]; do
            local next_eval_t=0
            for ((t=1; t<=total_tasks; t++)); do
                local eval_pid="${task_eval_pids[$t]}"
                if [ "${task_eval_queued[$t]}" -eq 1 ] && [ "${task_eval_done[$t]}" -eq 0 ] && { [ -z "$eval_pid" ] || [ "$eval_pid" = "0" ]; }; then
                    next_eval_t=$t
                    break
                fi
            done

            if [ "$next_eval_t" -eq 0 ]; then
                break
            fi

            local s="${task_stages[$next_eval_t]}"
            local name="${task_models[$next_eval_t]}"
            local steps="${task_steps[$next_eval_t]}"
            task_eval_queued[$next_eval_t]=0
            trigger_eval "$name" "$s" "$EVAL_RUN_ID" "$steps"
            task_eval_pids[$next_eval_t]=$EVAL_PID
            log "-> Launched background evaluation for ${name}${st_suffix} (stage $s) with PID: $EVAL_PID"
            ((++active_eval_count))
        done

        # 4. Check stage completion and merge per stage
        for stg in "${STAGE_LIST[@]}"; do
            if [ "${stage_merged[$stg]}" -eq 0 ]; then
                local stg_finished=1
                local stg_steps=0
                for ((t=1; t<=total_tasks; t++)); do
                    if [ "${task_stages[$t]}" = "$stg" ]; then
                        stg_steps="${task_steps[$t]}"
                        if [ "${task_status[$t]}" != "done" ] || [ "${task_eval_done[$t]}" -ne 1 ]; then
                            stg_finished=0
                            break
                        fi
                    fi
                done
                if [ "$stg_finished" -eq 1 ]; then
                    if [ "$RUN_EVAL" -eq 1 ]; then
                        log "Merging evaluation results for stage $stg..."
                        .venv/bin/python src/eval_stage.py --stage "$stg" --merge --run-id "$EVAL_RUN_ID" --steps "$stg_steps"
                    fi
                    stage_merged[$stg]=1
                    local stage_elapsed=$(( now - stage_start_times[$stg] ))
                    log "Stage $stg completed in $(( stage_elapsed / 60 ))m $(( stage_elapsed % 60 ))s (${stage_elapsed}s total)."
                fi
            fi
        done

        # 5. Check if all tasks and merges across all stages are complete
        all_done=1
        for ((t=1; t<=total_tasks; t++)); do
            if [ "${task_status[$t]}" != "done" ] || [ "${task_eval_done[$t]}" -ne 1 ]; then
                all_done=0
                break
            fi
        done
        for stg in "${STAGE_LIST[@]}"; do
            if [ "${stage_merged[$stg]}" -eq 0 ]; then
                all_done=0
                break
            fi
        done

        if [ "$FAST_MODE" -eq 1 ] && [ "$all_done" -eq 0 ]; then
            local status_str=""
            for ((t=1; t<=total_tasks; t++)); do
                if [ "${task_status[$t]}" = "running" ]; then
                    local s="${task_stages[$t]}"
                    local name="${task_models[$t]}"
                    local log_file="log/${EVAL_RUN_ID}/${name}${st_suffix}_s${s}.log"
                    local prog="starting"
                    if [ -f "$log_file" ]; then
                        local last_line
                        last_line=$(grep -E "update=|step=" "$log_file" | tail -n 1 || true)
                        if [ -n "$last_line" ]; then
                            prog=$(echo "$last_line" | awk '{print $1" "$2}')
                        fi
                    fi
                    status_str="${status_str} | ${name}_s${s}: ${prog}"
                fi
            done
            printf "\r[FAST MODE] Active:%s" "$status_str"
        fi

        if [ "$all_done" -eq 0 ]; then
            sleep 1.5
        fi
    done

    if [ "$FAST_MODE" -eq 1 ]; then
        printf "\n[FAST MODE] All models completed successfully.\n"
    fi
}


IFS=',' read -A STAGE_LIST <<< "$STAGES"
EVAL_PREFIX="run"
if [ "$ENABLE_SIDE_TASKS" -eq 1 ]; then
    EVAL_PREFIX="st"
fi
if [ -n "$CUSTOM_PREFIX" ]; then
    EVAL_PREFIX="$CUSTOM_PREFIX"
fi
EVAL_RUN_ID=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from eval_stage import get_next_run_id; print(get_next_run_id(prefix='$EVAL_PREFIX'))")
log "Campaign Evaluation Run ID: $EVAL_RUN_ID"

if [ "$DAEMON" -eq 1 ]; then
    log "Launching campaign as background daemon (Run ID: $EVAL_RUN_ID)..."
    mkdir -p "log/${EVAL_RUN_ID}"
    local -a pass_args=()
    for arg in "${RAW_ARGS[@]}"; do
        if [ "$arg" != "--daemon" ]; then
            pass_args+=("$arg")
        fi
    done
    nohup zsh "$0" "${pass_args[@]}" > "log/${EVAL_RUN_ID}/launch.out" 2>&1 &
    ln -sf "${EVAL_RUN_ID}/launch.out" log/launch.out 2>/dev/null || true
    print "Campaign launched as daemon (PID $!, Run ID: $EVAL_RUN_ID). Check log status with: uv run python tools/status.py"
else
    run_campaign
    log "All selected stages finished successfully!"
    print "Check final status with: uv run python tools/status.py"
fi

