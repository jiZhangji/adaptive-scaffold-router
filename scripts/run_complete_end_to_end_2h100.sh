#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
ROOT_LIMIT="${ROOT_LIMIT:-256}"
MIN_SAMPLES="${MIN_SAMPLES:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-12}"
MIN_TRAINING_ROOTS="${MIN_TRAINING_ROOTS:-64}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-32}"
CALIBRATION_ROOT_WINDOW="${CALIBRATION_ROOT_WINDOW:-8}"
CALIBRATION_STOP_CHECK_INTERVAL="${CALIBRATION_STOP_CHECK_INTERVAL:-16}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_MEMORY_PERCENT="${MIN_FREE_MEMORY_PERCENT:-90}"
PREP_ROOT="${PREP_ROOT:-$PROJECT_ROOT/outputs/complete_subproblem_n${ROOT_LIMIT}_2h100}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/complete_four_way_2h100_$(date +%Y%m%d_%H%M%S)}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
mkdir -p "$PREP_ROOT/calibration"/{shard_0,shard_1} "$PREP_ROOT/data" "$RUN_ROOT/logs"
printf '%s\n' "$PREP_ROOT" > "$PROJECT_ROOT/outputs/latest_complete_subproblem_data.txt"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_complete_four_way_pilot.txt"

wait_for_gpus() {
  while true; do
    local ready=1 report="" gpu free total
    for gpu in 0 1; do
      IFS=',' read -r free total < <(
        nvidia-smi --id="$gpu" --query-gpu=memory.free,memory.total \
          --format=csv,noheader,nounits 2>/dev/null | head -n 1
      )
      free="${free//[[:space:]]/}"; total="${total//[[:space:]]/}"
      report+=" GPU${gpu}=${free:-?}/${total:-?}MiB"
      [[ "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || ready=0
      if [[ "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] && \
         (( free * 100 < total * MIN_FREE_MEMORY_PERCENT )); then
        ready=0
      fi
    done
    echo "[$(date '+%F %T')]$report"
    (( ready == 1 )) && return 0
    echo "Waiting for both GPUs to have at least ${MIN_FREE_MEMORY_PERCENT}% free memory."
    sleep "$POLL_SECONDS"
  done
}

calibrate_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n "$ENV_NAME" python \
    "$PROJECT_ROOT/calibrate_helpful_subproblems.py" \
    --source-data "$SOURCE_DATA" --candidates "$CANDIDATES" --model "$MODEL_PATH" \
    --output-dir "$PREP_ROOT/calibration/shard_$shard" \
    --root-manifest "$PREP_ROOT/calibration/shard_${shard}_roots.txt" \
    --root-limit "$ROOT_LIMIT" --num-shards 2 --shard-index "$shard" \
    --q-low 0.25 --q-high 0.60 --min-samples "$MIN_SAMPLES" \
    --max-samples "$MAX_SAMPLES" --sample-batch 2 --max-plan-words 12 \
    --root-window "$CALIBRATION_ROOT_WINDOW" \
    --device cuda:0 --dtype bfloat16 --batch-size "$CALIBRATION_BATCH_SIZE" \
    --max-input-tokens 2048 --max-new-tokens 1024 \
    --stop-check-interval "$CALIBRATION_STOP_CHECK_INTERVAL" \
    --temperature 1.0 --top-p 1.0 --stop-after-boxed
}

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/prepare_calibration_root_manifests.py" \
  --source-data "$SOURCE_DATA" --candidates "$CANDIDATES" \
  --calibration-dir "$PREP_ROOT/calibration" \
  --root-limit "$ROOT_LIMIT" --num-shards 2 --seed 42 --max-plan-words 12 \
  2>&1 | tee "$PREP_ROOT/calibration/root_manifest.log"

echo "Waiting for two free GPUs before calibration."
wait_for_gpus
echo "Starting two calibration shards."
calibrate_shard 0 0 > "$PREP_ROOT/calibration/shard_0.log" 2>&1 & C0=$!
calibrate_shard 1 1 > "$PREP_ROOT/calibration/shard_1.log" 2>&1 & C1=$!
status=0
wait "$C0" || status=1
wait "$C1" || status=1
[[ "$status" -eq 0 ]] || { echo "Calibration failed; inspect shard logs." >&2; exit 4; }

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/merge_helpful_subproblem_shards.py" \
  --shard-dir "$PREP_ROOT/calibration/shard_0" \
  --shard-dir "$PREP_ROOT/calibration/shard_1" \
  --output-dir "$PREP_ROOT/calibration" \
  2>&1 | tee "$PREP_ROOT/calibration/merge.log"

conda run --no-capture-output -n "$ENV_NAME" python - \
  "$PREP_ROOT/calibration/summary.json" "$MIN_TRAINING_ROOTS" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected, minimum = int(summary["training_roots"]), int(sys.argv[2])
if selected < minimum:
    raise SystemExit(f"Only {selected} causally useful roots; require at least {minimum}.")
print(f"Calibration gate passed: {selected} >= {minimum}")
PY

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/build_complete_subproblem_experiment.py" \
  --source-data "$SOURCE_DATA" \
  --candidates "$PREP_ROOT/calibration/training_candidates.jsonl" \
  --output-dir "$PREP_ROOT/data"

echo "Calibration and data construction complete; starting four-way training."
wait_for_gpus
SCAF_REPO="$SCAF_REPO" ENV_NAME="$ENV_NAME" MODEL_PATH="$MODEL_PATH" \
DATA_ROOT="$PREP_ROOT/data" RUN_ROOT="$RUN_ROOT" TRAIN_STEPS="$TRAIN_STEPS" \
REQUIRE_FREE_GPUS=1 AUTO_EVAL=1 \
bash "$PROJECT_ROOT/scripts/run_complete_four_way_pilot_2h100.sh"

echo "End-to-end experiment complete: $RUN_ROOT/four_way_results.md"
