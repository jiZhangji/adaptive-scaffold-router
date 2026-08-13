#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
TEACHER_MODEL="${TEACHER_MODEL:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B-Instruct}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
LIMIT="${LIMIT:-64}"
SAMPLES="${SAMPLES:-4}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/vanilla_vs_subproblem_smoke_$TIMESTAMP}"

mkdir -p "$RUN_ROOT/logs"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_vanilla_vs_subproblem.txt"

echo "[1/3] Calibrating verifiable subproblems by the policy success rate q..."
RUN_ROOT="$RUN_ROOT/data" LIMIT="$LIMIT" SAMPLES="$SAMPLES" \
  MODEL_PATH="$MODEL_PATH" TEACHER_MODEL="$TEACHER_MODEL" DATA_FILE="$DATA_FILE" \
  ENV_NAME="$ENV_NAME" GPU=0 \
  bash "$PROJECT_ROOT/scripts/run_subproblem_feasibility_2h100.sh" \
  2>&1 | tee "$RUN_ROOT/logs/data.log"

MIXED_DATA="$RUN_ROOT/data/mixed_train.parquet"
ROOT_DATA="$RUN_ROOT/data/root_train.parquet"
if [[ ! -s "$MIXED_DATA" ]]; then
  echo "No trainable subproblem data was produced; stop before RL." >&2
  exit 2
fi

echo "[2/3] Running the same one-step Vanilla GRPO baseline..."
conda run --no-capture-output -n "$ENV_NAME" env \
  METHOD=vanilla MODE=smoke CUDA_VISIBLE_DEVICES=0,1 \
  PROJECT_ROOT_OVERRIDE="$PROJECT_ROOT" SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" \
  TRAIN_DATA="$ROOT_DATA" OUTPUT_DIR="$RUN_ROOT/vanilla" \
  DATALOADER_NUM_WORKERS=0 SKIP_ASSET_VALIDATION=1 \
  bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_train_2h100.sh" \
  2>&1 | tee "$RUN_ROOT/logs/vanilla.log"

echo "[3/3] Running subproblem mixed GRPO (1:1 root/subproblem, same optimizer)..."
conda run --no-capture-output -n "$ENV_NAME" env \
  METHOD=vanilla MODE=smoke CUDA_VISIBLE_DEVICES=0,1 \
  PROJECT_ROOT_OVERRIDE="$PROJECT_ROOT" SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" \
  TRAIN_DATA="$MIXED_DATA" OUTPUT_DIR="$RUN_ROOT/subproblem" \
  DATALOADER_NUM_WORKERS=0 SKIP_ASSET_VALIDATION=1 \
  bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_train_2h100.sh" \
  2>&1 | tee "$RUN_ROOT/logs/subproblem.log"

echo "Both smoke runs completed: $RUN_ROOT"
echo "This validates mechanics only; it is not a final paper result."
