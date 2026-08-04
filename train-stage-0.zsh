#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# train-stage-0.zsh -- provision a fresh GPU machine and run the 300k
# stage-0 campaign.
#
# INTENDED FOR THE TARGET MACHINE (e.g. a 3080 Ti box), not this laptop.
# The campaign itself (experiment_stage0_300k.sh) is bash; this script only
# handles provisioning, dev-env setup, gates, an optional smoke test, and
# launching it.  Run this BEFORE launching the campaign to also get a
# wall-clock estimate:
#
#     ./assess-time.zsh            # benchmark + estimate (~5-10 min)
#     ./train-stage-0.zsh          # provision + gates + smoke + run
#     ./train-stage-0.zsh --daemon # same, but backgrounded (nohup)
#     ./train-stage-0.zsh --eval   # also run eval_stage0_300k.py afterwards
#
# If zsh itself is missing on the target box:
#     sudo apt-get install -y zsh && zsh train-stage-0.zsh
#
# Flags:
#   --workdir DIR     repo directory (default $HOME/heist)
#   --repo-url URL    clone URL (default the HEIST GitHub repo)
#   --daemon          background the campaign with nohup; logs to log/launch.out
#   --eval            run src/eval_stage0_300k.py after the campaign finishes
#   --skip-smoke      skip the 2048-step GPU smoke before the real run
#   --pull            git pull an existing --workdir clone before running
#   -h, --help        show this help
# ---------------------------------------------------------------------------
set -euo pipefail

WORKDIR="${HEIST_WORKDIR:-$HOME/heist}"
REPO_URL="${HEIST_REPO_URL:-https://github.com/saejin-moon/heist.git}"
DAEMON=0
RUN_EVAL=0
SKIP_SMOKE=0
DO_PULL=0

usage() {
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while (( $# )); do
    case "$1" in
        --workdir)   WORKDIR="$2"; shift 2 ;;
        --repo-url)  REPO_URL="$2"; shift 2 ;;
        --daemon)    DAEMON=1; shift ;;
        --eval)      RUN_EVAL=1; shift ;;
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --pull)      DO_PULL=1; shift ;;
        -h|--help)   usage ;;
        *) print -u2 "error: unknown argument: $1"; usage ;;
    esac
done

STAGE0='{"map_size": [11, 11], "num_rooms_range": [1, 2], "guard_count": 0, "camera_count": 0, "door_count": 0, "max_steps": 60, "spawn_mode": "role"}'

# ── helpers ──────────────────────────────────────────────────────────
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        print -u2 "error: not root and no sudo available"
        exit 1
    fi
fi

log() { print -- "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

step() { print; print "────────────────────────────────────────────────────────"; print "==> $*"; }

# ── 1. system packages ───────────────────────────────────────────────
step "1/8  System packages"
if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -y
    $SUDO apt-get install -y git curl ca-certificates zsh python3 python3-pip python3-venv build-essential
elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y git curl ca-certificates zsh python3 python3-pip gcc gcc-c++ make
elif command -v pacman >/dev/null 2>&1; then
    $SUDO pacman -Sy --noconfirm git curl ca-certificates zsh python python-pip base-devel
elif command -v apk >/dev/null 2>&1; then
    $SUDO apk add --no-cache git curl ca-certificates zsh python3 py3-pip build-base
elif command -v brew >/dev/null 2>&1; then
    brew install git curl python zsh
else
    print -u2 "error: no supported package manager found; install git, curl, python3, zsh manually"
    exit 1
fi

# ── 2. NVIDIA driver check (driver install stays manual: too invasive) ─
step "2/8  GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    print -u2 "error: nvidia-smi not found. Install the NVIDIA driver for this GPU"
    print -u2 "       (e.g. 'sudo apt-get install nvidia-driver-535' + reboot on Ubuntu)."
    exit 1
fi

# ── 3. uv (python package manager) ───────────────────────────────────
step "3/8  uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv via the official installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ── 4. clone the repo ────────────────────────────────────────────────
step "4/8  Repo"
if [ ! -d "$WORKDIR" ]; then
    log "cloning $REPO_URL into $WORKDIR"
    git clone "$REPO_URL" "$WORKDIR"
elif [ ! -f "$WORKDIR/experiment_stage0_300k.sh" ]; then
    print -u2 "error: $WORKDIR exists but does not look like the HEIST repo"
    exit 1
else
    log "using existing repo at $WORKDIR"
    if [ "$DO_PULL" -eq 1 ]; then
        log "git pull"
        git -C "$WORKDIR" pull --ff-only
    fi
fi
cd "$WORKDIR"

# ── 5. python env + pinned deps ──────────────────────────────────────
step "5/8  Dev env (uv sync --locked)"
if ! uv python find 3.12 >/dev/null 2>&1; then
    log "installing managed CPython 3.12 for uv"
    uv python install 3.12
fi
uv sync --locked

# ── 6. validation gates (workflow rule) ──────────────────────────────
step "6/8  Gates"
uv run ruff check
uv run ruff format --check
uv run pytest -q
uv run python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
if ! uv run python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    print -u2 "error: torch cannot see CUDA. Reinstall torch with the CUDA build:"
    print -u2 "       uv pip install --reinstall 'torch>=2.2' --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

# ── 7. smoke test at the real batch shape ────────────────────────────
if [ "$SKIP_SMOKE" -eq 0 ]; then
    step "7/8  GPU smoke (2048 steps, 8 envs x 256, no checkpoint)"
    uv run python src/train_ippo.py \
        --total-timesteps 2048 --num-envs 8 --num-steps 256 \
        --eval-every 1000000000 --no-save-model --seed 0 \
        --env-config "$STAGE0" --exp-name smoke_assess
    rm -rf runs/smoke_assess_s0
    log "smoke OK"
else
    step "7/8  GPU smoke (skipped)"
fi

# ── 8. launch the 300k stage-0 campaign ──────────────────────────────
step "8/8  Campaign (300k stage-0: 4 algos x 3 seeds)"
mkdir -p log
CAMPAIGN="bash experiment_stage0_300k.sh"
if [ "$RUN_EVAL" -eq 1 ]; then
    CAMPAIGN="$CAMPAIGN && uv run python src/eval_stage0_300k.py"
fi
if [ "$DAEMON" -eq 1 ]; then
    nohup sh -c "$CAMPAIGN" > log/launch.out 2>&1 &
    print "campaign launched in background (pid $!)"
    print "log:  tail -f $WORKDIR/log/experiment_stage0_300k.log"
    print "      tail -f $WORKDIR/log/launch.out"
    print "tb:   uv run tensorboard --logdir runs --port 6006"
else
    print "campaign running in the foreground; Ctrl-C aborts it"
    print "log:  tail -f $WORKDIR/log/experiment_stage0_300k.log"
    sh -c "$CAMPAIGN"
    print "campaign finished (exit $?)"
fi
