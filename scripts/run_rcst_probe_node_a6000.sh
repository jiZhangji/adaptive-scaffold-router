#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/powerleader/project/adaptive-scaffold-router}"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
CANDIDATES="${CANDIDATES:?Set CANDIDATES}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
OUTPUT="${OUTPUT:?Set OUTPUT}"
EXIT_FILE="${EXIT_FILE:?Set EXIT_FILE}"
SEED="${SEED:?Set SEED}"
SELECTION_SEED="${SELECTION_SEED:-42}"
ROOT_LIMIT="${ROOT_LIMIT:-212}"
ROOT_SAMPLES="${ROOT_SAMPLES:-8}"
PROBE_STEPS="${PROBE_STEPS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$EXIT_FILE")"
trap 'code=$?; printf "%s\n" "$code" > "$EXIT_FILE"' EXIT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0

"$PYTHON_BIN" "$PROJECT_ROOT/probe_subproblem_transfer.py" \
  --candidates "$CANDIDATES" --model "$MODEL_PATH" --output "$OUTPUT" \
  --root-limit "$ROOT_LIMIT" --root-samples "$ROOT_SAMPLES" \
  --probe-steps "$PROBE_STEPS" --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED" --selection-seed "$SELECTION_SEED" --device cuda:0
