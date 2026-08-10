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
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-10000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_cmd=("$PYTHON_BIN")
else
  python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python)
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  ENV_NAME="$ENV_NAME" MODEL_PATH="$MODEL_PATH" DATA_FILE="$DATA_FILE" \
    SCAF_REPO="${SCAF_REPO:-$ROOT/Scaf-GRPO}" \
    bash "$PROJECT_ROOT/scripts/wait_and_validate_downloads.sh"
fi

gpu_index="${DEVICE#cuda:}"
if [[ "$DEVICE" == cuda:* ]] && command -v nvidia-smi >/dev/null 2>&1; then
  while true; do
    gpu_status="$(nvidia-smi --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits -i "$gpu_index" | head -n 1)"
    free_mb="$(echo "$gpu_status" | cut -d, -f1 | tr -d ' ')"
    utilization="$(echo "$gpu_status" | cut -d, -f2 | tr -d ' ')"
    if [[ "$free_mb" =~ ^[0-9]+$ && "$free_mb" -ge "$MIN_FREE_GPU_MB" ]]; then
      echo "GPU $gpu_index is ready: free=${free_mb}MB, utilization=${utilization}%"
      break
    fi
    echo "GPU $gpu_index is busy: free=${free_mb:-unknown}MB, utilization=${utilization:-unknown}%; waiting..."
    sleep "$GPU_POLL_SECONDS"
  done
fi

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
  --batch-size 4 \
  --stop-after-boxed

"${python_cmd[@]}" "$PROJECT_ROOT/analyze_metaask_results.py" \
  --input "$RUN_ROOT/metaask/raw_results.jsonl" \
  --output "$RUN_ROOT/metaask/diagnostics.json"

echo "Running controlled one-bit answer-verification retry..."
"${python_cmd[@]}" "$PROJECT_ROOT/metaask_answer_retry.py" \
  --input "$RUN_ROOT/metaask/raw_results.jsonl" \
  --model "$MODEL_PATH" \
  --output-dir "$RUN_ROOT/metaask_controlled" \
  --device "$DEVICE" \
  --batch-size 4 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --stop-after-boxed

echo "Both first-stage probes finished."
echo "Capability summary: $RUN_ROOT/capability_frontier/summary.json"
echo "Curriculum manifest: $RUN_ROOT/capability_curriculum/curriculum.jsonl"
echo "MetaAsk summary:     $RUN_ROOT/metaask/summary.json"
echo "MetaAsk diagnostics: $RUN_ROOT/metaask/diagnostics.json"
echo "Controlled 1-bit:    $RUN_ROOT/metaask_controlled/summary.json"
echo "These are mechanism probes, not final RL training results."
