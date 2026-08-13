#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
LIMIT="${LIMIT:-64}"
MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
DIMENSIONS="${DIMENSIONS:-calculation}"
MAX_SOURCE_ACCURACY="${MAX_SOURCE_ACCURACY:-0.9}"
RUN_NAME="${RUN_NAME:-deepseek_subproblems_n${LIMIT}_${DIMENSIONS//,/-}_acc${MAX_SOURCE_ACCURACY}}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/$RUN_NAME}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set. Use: read -rsp 'DeepSeek API Key: ' DEEPSEEK_API_KEY; export DEEPSEEK_API_KEY" >&2
  exit 2
fi
if [[ ! -s "$DATA_FILE" ]]; then
  echo "Training parquet is missing: $DATA_FILE" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_deepseek_subproblems.txt"

python "$PROJECT_ROOT/generate_subproblems_deepseek.py" \
  --data "$DATA_FILE" \
  --output "$RUN_ROOT/candidates.jsonl" \
  --errors "$RUN_ROOT/errors.jsonl" \
  --summary "$RUN_ROOT/summary.json" \
  --limit "$LIMIT" \
  --model "$MODEL" \
  --dimensions "$DIMENSIONS" \
  --max-source-accuracy "$MAX_SOURCE_ACCURACY" \
  2>&1 | tee -a "$RUN_ROOT/generation.log"

echo "Generation complete: $RUN_ROOT"
echo "Candidates: $RUN_ROOT/candidates.jsonl"
echo "Summary: $RUN_ROOT/summary.json"
