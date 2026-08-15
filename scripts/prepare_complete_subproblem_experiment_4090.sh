#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
ROOT_LIMIT="${ROOT_LIMIT:-256}"
MIN_SAMPLES="${MIN_SAMPLES:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-12}"
MIN_TRAINING_ROOTS="${MIN_TRAINING_ROOTS:-64}"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/complete_subproblem_n${ROOT_LIMIT}}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_ROOT/calibration"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_complete_subproblem_data.txt"

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/calibrate_helpful_subproblems.py" \
  --source-data "$SOURCE_DATA" \
  --candidates "$CANDIDATES" \
  --model "$MODEL_PATH" \
  --output-dir "$RUN_ROOT/calibration" \
  --root-limit "$ROOT_LIMIT" \
  --q-low 0.25 --q-high 0.60 \
  --min-samples "$MIN_SAMPLES" --max-samples "$MAX_SAMPLES" \
  --sample-batch 2 --max-plan-words 12 \
  --device cuda:0 --dtype bfloat16 \
  --batch-size 8 --max-input-tokens 2048 --max-new-tokens 1024 \
  --temperature 1.0 --top-p 1.0 --stop-after-boxed \
  2>&1 | tee "$RUN_ROOT/calibration.log"

conda run --no-capture-output -n "$ENV_NAME" python - \
  "$RUN_ROOT/calibration/summary.json" "$MIN_TRAINING_ROOTS" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
minimum = int(sys.argv[2])
selected = int(summary["training_roots"])
if selected < minimum:
    raise SystemExit(
        f"Causal calibration retained only {selected} roots; minimum is {minimum}. "
        "Do not spend H100 time until candidate quality or the probe size is improved."
    )
print(f"Causal calibration gate passed: {selected} >= {minimum}")
PY

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/build_complete_subproblem_experiment.py" \
  --source-data "$SOURCE_DATA" \
  --candidates "$RUN_ROOT/calibration/training_candidates.jsonl" \
  --output-dir "$RUN_ROOT/data"

echo "Complete method data prepared: $RUN_ROOT"
echo "Calibration: $RUN_ROOT/calibration/summary.json"
echo "Root data: $RUN_ROOT/data/root_train.parquet"
echo "Subproblem mix: $RUN_ROOT/data/mixed_train.parquet"
echo "Fading-plan roots: $RUN_ROOT/data/proposed_root_train.parquet"
echo "Curriculum: $RUN_ROOT/data/curriculum.jsonl"
