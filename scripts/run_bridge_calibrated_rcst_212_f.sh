#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/powerleader/project/adaptive-scaffold-router}"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
BASE="$PROJECT_ROOT/outputs/bridge_calibrated_rcst_212"
ASSETS="$BASE/assets"
DELTA_DIR="$BASE/delta_system_preserving_v2"
CALIBRATION_DIR="$BASE/calibration_crossfit_v1"
CANDIDATES="$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl"
PROBE_42="$ASSETS/probe_seed_42.jsonl"
PROBE_314159="$ASSETS/probe_seed_314159.jsonl"
AGGREGATES="$ASSETS/rcst_aggregated.jsonl"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_REQUIRED_SECONDS="${IDLE_REQUIRED_SECONDS:-1800}"
SKIP_IDLE_WAIT="${SKIP_IDLE_WAIT:-0}"

mkdir -p "$BASE/logs" "$DELTA_DIR" "$CALIBRATION_DIR"
if ! mkdir "$BASE/run.lock" 2>/dev/null; then
  echo "Another bridge-calibration launcher owns $BASE/run.lock" >&2
  exit 3
fi
trap 'rmdir "$BASE/run.lock" 2>/dev/null || true' EXIT

for path in "$PYTHON_BIN" "$MODEL_PATH/config.json" "$CANDIDATES" \
  "$PROBE_42" "$PROBE_314159" "$AGGREGATES"; do
  test -s "$path" || { echo "Missing required input: $path" >&2; exit 2; }
done

if [[ "$SKIP_IDLE_WAIT" != "1" ]]; then
  echo "[$(date -Is)] waiting until both F GPUs remain idle for $IDLE_REQUIRED_SECONDS seconds"
  idle_since=""
  while true; do
    readarray -t gpu_rows < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
    )
    free=1
    for row in "${gpu_rows[@]}"; do
      memory="${row%%,*}"
      utilization="${row##*,}"
      memory="${memory// /}"
      utilization="${utilization// /}"
      if (( memory > 1024 || utilization > 10 )); then
        free=0
      fi
    done
    now="$(date +%s)"
    if (( free )); then
      [[ -n "$idle_since" ]] || idle_since="$now"
      idle_elapsed="$((now - idle_since))"
    else
      idle_since=""
      idle_elapsed=0
    fi
    echo "[$(date -Is)] GPU rows: ${gpu_rows[*]}; continuous_idle=${idle_elapsed}/${IDLE_REQUIRED_SECONDS}s"
    (( idle_elapsed >= IDLE_REQUIRED_SECONDS )) && break
    sleep "$POLL_SECONDS"
  done
else
  echo "[$(date -Is)] external idle monitor satisfied; starting workflow"
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
common=(
  --candidates "$CANDIDATES"
  --probes "$PROBE_42" "$PROBE_314159"
  --model "$MODEL_PATH"
  --device cuda:0
  --dtype bfloat16
  --batch-size 4
)

echo "[$(date -Is)] one-root smoke test"
rm -f "$DELTA_DIR/smoke.jsonl" "$DELTA_DIR/smoke.summary.json"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$PROJECT_ROOT/score_bridge_delta_features.py" \
  "${common[@]}" --output "$DELTA_DIR/smoke.jsonl" \
  --num-shards 212 --shard-index 0 \
  > "$BASE/logs/delta_smoke.log" 2>&1
test "$(wc -l < "$DELTA_DIR/smoke.jsonl")" -eq 3
rm -f "$DELTA_DIR/smoke.jsonl" "$DELTA_DIR/smoke.summary.json"

echo "[$(date -Is)] launching two system-preserving delta shards"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$PROJECT_ROOT/score_bridge_delta_features.py" \
  "${common[@]}" --output "$DELTA_DIR/shard_0.jsonl" \
  --num-shards 2 --shard-index 0 \
  > "$BASE/logs/delta_shard_0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$PROJECT_ROOT/score_bridge_delta_features.py" \
  "${common[@]}" --output "$DELTA_DIR/shard_1.jsonl" \
  --num-shards 2 --shard-index 1 \
  > "$BASE/logs/delta_shard_1.log" 2>&1 &
pid1=$!
printf '%s\n' "$pid0" > "$DELTA_DIR/shard_0.pid"
printf '%s\n' "$pid1" > "$DELTA_DIR/shard_1.pid"

set +e
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
set -e
printf '%s\n' "$code0" > "$DELTA_DIR/shard_0.exit"
printf '%s\n' "$code1" > "$DELTA_DIR/shard_1.exit"
if (( code0 != 0 || code1 != 0 )); then
  echo "Delta scoring failed: shard0=$code0 shard1=$code1" >&2
  exit 4
fi

total="$(( $(wc -l < "$DELTA_DIR/shard_0.jsonl") + $(wc -l < "$DELTA_DIR/shard_1.jsonl") ))"
test "$total" -eq 636 || { echo "Expected 636 delta rows, found $total" >&2; exit 5; }

echo "[$(date -Is)] fitting root-grouped cross-fitted calibrator"
"$PYTHON_BIN" "$PROJECT_ROOT/fit_bridge_calibrated_rcst.py" \
  --candidates "$CANDIDATES" \
  --aggregates "$AGGREGATES" \
  --delta-features "$DELTA_DIR/shard_0.jsonl" "$DELTA_DIR/shard_1.jsonl" \
  --output-dir "$CALIBRATION_DIR" \
  --folds 5 --bootstrap-models 32 --ridge-alpha 5.0 \
  --confidence-z 1.0 --uncertainty-floor 0.01 --min-score 0.0 \
  > "$BASE/logs/calibration.log" 2>&1

echo "[$(date -Is)] bridge calibration complete"
cat "$CALIBRATION_DIR/diagnostics.json"
