#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$(dirname "$PROJECT_ROOT")}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LIMIT="${LIMIT:-32}"
CAPABILITY_SAMPLES="${CAPABILITY_SAMPLES:-2}"
SUBPROBLEM_SAMPLES="${SUBPROBLEM_SAMPLES:-2}"
METAASK_SAMPLES="${METAASK_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/feasibility_screen_n${LIMIT}}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"

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

wait_for_gpu() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  while true; do
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | head -n 1 | tr -d ' ')"
    if [[ "$free_mb" =~ ^[0-9]+$ && "$free_mb" -ge "$MIN_FREE_GPU_MB" ]]; then
      echo "GPU $gpu ready: ${free_mb} MB free"
      return
    fi
    echo "GPU $gpu has ${free_mb:-unknown} MB free; waiting..."
    sleep 30
  done
}

mkdir -p "$RUN_ROOT/logs"
wait_for_gpu "$GPU0"
wait_for_gpu "$GPU1"

run_scheme1() {
  export CUDA_VISIBLE_DEVICES="$GPU0"
  "${python_cmd[@]}" "$PROJECT_ROOT/frontier_probe.py" \
    --model "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --output-dir "$RUN_ROOT/capability_frontier" \
    --device cuda:0 \
    --dtype bfloat16 \
    --limit "$LIMIT" \
    --samples-per-arm "$CAPABILITY_SAMPLES" \
    --strengths 0.25,0.5,1.0 \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size 8 \
    --band-low 0.25 \
    --band-high 0.60 \
    --stop-after-boxed

  "${python_cmd[@]}" "$PROJECT_ROOT/frontier_to_curriculum.py" \
    --input "$RUN_ROOT/capability_frontier/raw_results.jsonl" \
    --output-dir "$RUN_ROOT/capability_curriculum" \
    --band-low 0.25 \
    --band-high 0.60 \
    --group-size "$CAPABILITY_SAMPLES"

  "${python_cmd[@]}" "$PROJECT_ROOT/subproblem_relevance_probe.py" \
    --model "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --output-dir "$RUN_ROOT/subproblem_relevance" \
    --device cuda:0 \
    --dtype bfloat16 \
    --limit "$LIMIT" \
    --samples-per-variant "$SUBPROBLEM_SAMPLES" \
    --batch-size 8 \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-after-boxed
}

run_scheme2() {
  export CUDA_VISIBLE_DEVICES="$GPU1"
  "${python_cmd[@]}" "$PROJECT_ROOT/metaask_probe.py" \
    --model "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --output-dir "$RUN_ROOT/metaask" \
    --device cuda:0 \
    --dtype bfloat16 \
    --limit "$LIMIT" \
    --samples-per-variant "$METAASK_SAMPLES" \
    --batch-size 8 \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-after-boxed

  "${python_cmd[@]}" "$PROJECT_ROOT/analyze_metaask_results.py" \
    --input "$RUN_ROOT/metaask/raw_results.jsonl" \
    --output "$RUN_ROOT/metaask/diagnostics.json"

  "${python_cmd[@]}" "$PROJECT_ROOT/metaask_answer_retry.py" \
    --input "$RUN_ROOT/metaask/raw_results.jsonl" \
    --model "$MODEL_PATH" \
    --output-dir "$RUN_ROOT/metaask_controlled" \
    --device cuda:0 \
    --dtype bfloat16 \
    --batch-size 8 \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --stop-after-boxed
}

cleanup() {
  [[ -n "${pid1:-}" ]] && kill "$pid1" 2>/dev/null || true
  [[ -n "${pid2:-}" ]] && kill "$pid2" 2>/dev/null || true
}
trap cleanup INT TERM

echo "Starting Scheme 1 on physical GPU $GPU0; log: $RUN_ROOT/logs/scheme1.log"
run_scheme1 >"$RUN_ROOT/logs/scheme1.log" 2>&1 &
pid1=$!
echo "Starting Scheme 2 on physical GPU $GPU1; log: $RUN_ROOT/logs/scheme2.log"
run_scheme2 >"$RUN_ROOT/logs/scheme2.log" 2>&1 &
pid2=$!

status=0
wait "$pid1" || status=1
wait "$pid2" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "At least one branch failed. Inspect $RUN_ROOT/logs/*.log" >&2
  exit "$status"
fi

"${python_cmd[@]}" "$PROJECT_ROOT/assess_two_idea_feasibility.py" \
  --run-root "$RUN_ROOT" | tee "$RUN_ROOT/logs/assessment.log"

echo "Feasibility screen complete."
echo "Human-readable report: $RUN_ROOT/feasibility_report.md"
echo "Machine-readable report: $RUN_ROOT/feasibility_report.json"
