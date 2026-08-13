#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B-Instruct}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
LIMIT="${LIMIT:-64}"
SAMPLES="${SAMPLES:-4}"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/subproblem_feasibility_n${LIMIT}_$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_subproblem_feasibility.txt"

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/subproblem_relevance_probe.py" \
  --model "$MODEL_PATH" \
  --teacher-model "$TEACHER_MODEL" \
  --data "$DATA_FILE" \
  --output-dir "$RUN_ROOT/probe" \
  --device cuda:0 --teacher-device cuda:0 --dtype bfloat16 \
  --limit "$LIMIT" --samples-per-variant "$SAMPLES" \
  --batch-size 8 --max-input-tokens 3072 --max-new-tokens 1024 \
  --teacher-retries 2 --teacher-temperature 0.2 \
  --q-low 0.25 --q-high 0.60 --stop-after-boxed

conda run --no-capture-output -n "$ENV_NAME" python - "$RUN_ROOT/probe/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
q = summary["q_calibration"]
checks = summary["causal_checks"]
errors = []
if summary.get("valid_candidate_rate", 0.0) < 0.50:
    errors.append("valid candidate rate is below 50%")
if q.get("trainable_candidate_count", 0) < 8:
    errors.append("fewer than 8 candidates lie in the q learning band")
if checks.get("relevant_gain_over_random", 0.0) <= 0.0:
    errors.append("relevant subproblems did not beat the matched random control")
if errors:
    raise SystemExit("Subproblem feasibility gate failed: " + "; ".join(errors))
print("Subproblem feasibility gate passed.")
PY

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/build_subproblem_train_data.py" \
  --source-data "$DATA_FILE" \
  --candidates "$RUN_ROOT/probe/training_candidates.jsonl" \
  --output "$RUN_ROOT/mixed_train.parquet" \
  --root-output "$RUN_ROOT/root_train.parquet"

echo "Feasibility probe: $RUN_ROOT/probe/summary.json"
echo "Calibrated 1:1 train data: $RUN_ROOT/mixed_train.parquet"
echo "Matched root-only baseline data: $RUN_ROOT/root_train.parquet"
