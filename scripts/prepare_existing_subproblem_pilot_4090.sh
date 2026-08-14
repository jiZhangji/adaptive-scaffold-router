#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
ROOT_LIMIT="${ROOT_LIMIT:-256}"
SAMPLES="${SAMPLES:-4}"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/existing_subproblem_pilot_n${ROOT_LIMIT}}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_existing_subproblem_pilot.txt"

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/calibrate_deepseek_subproblems.py" \
  --candidates "$CANDIDATES" \
  --model "$MODEL_PATH" \
  --output-dir "$RUN_ROOT/calibration" \
  --root-limit "$ROOT_LIMIT" \
  --samples-per-candidate "$SAMPLES" \
  --q-low 0.25 --q-high 0.60 \
  --device cuda:0 --dtype bfloat16 \
  --batch-size 8 --job-chunk-size 64 \
  --max-input-tokens 2048 --max-new-tokens 512 \
  --temperature 1.0 --top-p 1.0 --stop-after-boxed \
  2>&1 | tee "$RUN_ROOT/calibration.log"

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/build_subproblem_train_data.py" \
  --source-data "$SOURCE_DATA" \
  --candidates "$RUN_ROOT/calibration/training_candidates.jsonl" \
  --output "$RUN_ROOT/mixed_train.parquet" \
  --root-output "$RUN_ROOT/root_train.parquet"

echo "Pilot summary: $RUN_ROOT/calibration/summary.json"
echo "Matched Vanilla data: $RUN_ROOT/root_train.parquet"
echo "Subproblem data: $RUN_ROOT/mixed_train.parquet"
