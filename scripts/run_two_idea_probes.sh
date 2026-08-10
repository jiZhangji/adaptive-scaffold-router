#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$(dirname "$PROJECT_ROOT")}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-8}"
CAPABILITY_SAMPLES="${CAPABILITY_SAMPLES:-2}"
METAASK_SAMPLES="${METAASK_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/two_idea_probe}"

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python)

ENV_NAME="$ENV_NAME" MODEL_PATH="$MODEL_PATH" DATA_FILE="$DATA_FILE" \
  SCAF_REPO="${SCAF_REPO:-$ROOT/Scaf-GRPO}" \
  bash "$PROJECT_ROOT/scripts/wait_and_validate_downloads.sh"

mkdir -p "$RUN_ROOT"

echo "[1/3] Running capability-matched scaffold frontier probe..."
"${python_cmd[@]}" "$PROJECT_ROOT/frontier_probe.py" \
  --model "$MODEL_PATH" \
  --data "$DATA_FILE" \
  --output-dir "$RUN_ROOT/capability_frontier" \
  --device "$DEVICE" \
  --limit "$LIMIT" \
  --samples-per-arm "$CAPABILITY_SAMPLES" \
  --strengths 0.25,0.5,1.0 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size 4 \
  --band-low 0.25 \
  --band-high 0.60 \
  --stop-after-boxed

echo "[2/3] Compiling the capability curriculum manifest..."
"${python_cmd[@]}" "$PROJECT_ROOT/frontier_to_curriculum.py" \
  --input "$RUN_ROOT/capability_frontier/raw_results.jsonl" \
  --output-dir "$RUN_ROOT/capability_curriculum" \
  --band-low 0.25 \
  --band-high 0.60 \
  --group-size "$CAPABILITY_SAMPLES"

echo "[3/3] Running MetaAsk minimal-information mechanism probe..."
"${python_cmd[@]}" "$PROJECT_ROOT/metaask_probe.py" \
  --model "$MODEL_PATH" \
  --data "$DATA_FILE" \
  --output-dir "$RUN_ROOT/metaask" \
  --device "$DEVICE" \
  --limit "$LIMIT" \
  --samples-per-variant "$METAASK_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size 4

echo "Both first-stage probes finished."
echo "Capability summary: $RUN_ROOT/capability_frontier/summary.json"
echo "Curriculum manifest: $RUN_ROOT/capability_curriculum/curriculum.jsonl"
echo "MetaAsk summary:     $RUN_ROOT/metaask/summary.json"
echo "These are mechanism probes, not final RL training results."
