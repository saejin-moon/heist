#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# eval.zsh -- standalone evaluation script
#
# Usage:
#     ./scripts/eval.zsh --models roma,lrs,mahiro,charm --stages 0,1,2,3,4 --run-id run020
# ---------------------------------------------------------------------------
set -euo pipefail

MODELS=""
STAGES="0"
RUN_ID=""
STEPS=299008

usage() {
    cat << 'EOF'
eval.zsh -- run evaluation independently of training

Usage:
    ./scripts/eval.zsh --models roma,lrs,mahiro,charm --stages 0,1,2 --run-id run020

Flags:
  --models, --model MODELS  comma-separated model names to evaluate (e.g. roma,lrs)
  --stages STAGES           comma-separated curriculum stage indices (default 0)
  --run-id ID               the run ID to save logs and results into (e.g., run020)
  --steps STEPS             total training timesteps to log in results (default 299008)
  -h, --help                show this help
EOF
    exit 0
}

while (( $# )); do
    case "$1" in
        --models|--model) MODELS="$2"; shift 2 ;;
        --stages)     STAGES="$2"; shift 2 ;;
        --run-id)     RUN_ID="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) print -u2 "error: unknown argument: $1"; usage ;;
    esac
done

if [ -z "$MODELS" ]; then
    echo "Error: --models must be provided (e.g. --models charm,mahiro)"
    exit 1
fi

if [ -z "$RUN_ID" ]; then
    # Generate the next available run ID
    RUN_ID=$(uv run python -c "import sys; sys.path.insert(0, 'src'); from eval_stage import get_next_run_id; print(get_next_run_id(prefix='run'))")
fi

echo "========================================================="
echo "Starting standalone evaluation"
echo "Run ID: $RUN_ID"
echo "Models: $MODELS"
echo "Stages: $STAGES"
echo "========================================================="

mkdir -p "log/${RUN_ID}"
mkdir -p "results/${RUN_ID}"

IFS=',' read -r -A model_array <<< "$MODELS"
IFS=',' read -r -A stage_array <<< "$STAGES"

for s in "${stage_array[@]}"; do
    echo "Evaluating stage $s..."
    
    pids=()
    for name in "${model_array[@]}"; do
        echo "  -> Launching eval for ${name} (stage ${s})..."
        uv run python -u src/eval_stage.py --stage "$s" --algo "$name" --run-id "$RUN_ID" --steps "$STEPS" > "log/${RUN_ID}/eval_${name}_s${s}.log" 2>&1 &
        pids+=($!)
    done
    
    # Wait for all evals in this stage to finish
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    echo "  -> Merging results for stage $s..."
    uv run python src/eval_stage.py --stage "$s" --merge --run-id "$RUN_ID" --steps "$STEPS"
done

echo "========================================================="
echo "Evaluation complete! Check results/${RUN_ID}/summary.json"
