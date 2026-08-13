#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B-Instruct}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
LIMIT="${LIMIT:-128}"
SAMPLES="${SAMPLES:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/vanilla_vs_subproblem_parallel_$(date +%Y%m%d_%H%M%S)}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1
mkdir -p "$RUN_ROOT/logs"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_vanilla_vs_subproblem_parallel.txt"

for path in "$MODEL_PATH/config.json" "$TEACHER_MODEL/config.json" "$DATA_FILE"; do
  [[ -s "$path" ]] || { echo "Missing required offline asset: $path" >&2; exit 2; }
done

echo "[1/2] Build and calibrate root-derived verifiable subproblems on GPU 0."
RUN_ROOT="$RUN_ROOT/data" LIMIT="$LIMIT" SAMPLES="$SAMPLES" \
  MODEL_PATH="$MODEL_PATH" TEACHER_MODEL="$TEACHER_MODEL" DATA_FILE="$DATA_FILE" \
  ENV_NAME="$ENV_NAME" GPU=0 \
  bash "$PROJECT_ROOT/scripts/run_subproblem_feasibility_2h100.sh" \
  2>&1 | tee "$RUN_ROOT/logs/data.log"

ROOT_DATA="$RUN_ROOT/data/root_train.parquet"
MIXED_DATA="$RUN_ROOT/data/mixed_train.parquet"
[[ -s "$ROOT_DATA" && -s "$MIXED_DATA" ]] || {
  echo "Matched training datasets were not produced; RL will not start." >&2
  exit 3
}

echo "[2/2] Launch Vanilla on GPU 0 and subproblem GRPO on GPU 1 in parallel."
run_train() {
  local gpu="$1" data="$2" output="$3" name="$4"
  mkdir -p "$RUN_ROOT/ray_$name"
  CUDA_VISIBLE_DEVICES="$gpu" \
  RAY_TMPDIR="$RUN_ROOT/ray_$name" RAY_ADDRESS="" \
  METHOD=vanilla MODE=smoke N_GPUS=1 TP_SIZE=1 \
  OVERRIDE_TOTAL_STEPS="$TRAIN_STEPS" DATALOADER_NUM_WORKERS=0 \
  SKIP_ASSET_VALIDATION=1 PROJECT_ROOT_OVERRIDE="$PROJECT_ROOT" \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$data" \
  OUTPUT_DIR="$output" \
  conda run --no-capture-output -n "$ENV_NAME" \
    bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_train_2h100.sh" \
    >"$RUN_ROOT/logs/$name.log" 2>&1
}

run_train 0 "$ROOT_DATA" "$RUN_ROOT/vanilla" vanilla &
vanilla_pid=$!
run_train 1 "$MIXED_DATA" "$RUN_ROOT/subproblem" subproblem &
subproblem_pid=$!
printf '%s\n' "$vanilla_pid" > "$RUN_ROOT/vanilla.pid"
printf '%s\n' "$subproblem_pid" > "$RUN_ROOT/subproblem.pid"

cleanup() {
  kill "$vanilla_pid" "$subproblem_pid" 2>/dev/null || true
}
trap cleanup INT TERM

status=0
wait "$vanilla_pid" || { echo "Vanilla branch failed." >&2; status=1; }
wait "$subproblem_pid" || { echo "Subproblem branch failed." >&2; status=1; }
trap - INT TERM

if [[ "$status" -ne 0 ]]; then
  echo "One or both branches failed. Inspect $RUN_ROOT/logs/*.log" >&2
  exit "$status"
fi

echo "Both parallel training branches completed successfully."
echo "Run root: $RUN_ROOT"
echo "Vanilla log: $RUN_ROOT/logs/vanilla.log"
echo "Subproblem log: $RUN_ROOT/logs/subproblem.log"
