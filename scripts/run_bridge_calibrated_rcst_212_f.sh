#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/powerleader/project/adaptive-scaffold-router}"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
SCAF_REPO="${SCAF_REPO:-/home/powerleader/project/Scaf-GRPO}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
BASE="$PROJECT_ROOT/outputs/bridge_calibrated_rcst_212"
ASSETS="$BASE/assets"
DELTA_DIR="$BASE/delta_system_preserving_v2"
CALIBRATION_DIR="$BASE/calibration_crossfit_v1"
CANDIDATES="$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl"
PROBE_42="$ASSETS/probe_seed_42.jsonl"
PROBE_314159="$ASSETS/probe_seed_314159.jsonl"
AGGREGATES="$ASSETS/rcst_aggregated.jsonl"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
ALL_SELECTED="${ALL_SELECTED:-$PROJECT_ROOT/outputs/bridge_212_assets/transfer_selected_candidates.jsonl}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_REQUIRED_SECONDS="${IDLE_REQUIRED_SECONDS:-600}"
SKIP_IDLE_WAIT="${SKIP_IDLE_WAIT:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
RUN_FORMAL_TRAINING="${RUN_FORMAL_TRAINING:-1}"

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

delta_total=0
if [[ -s "$DELTA_DIR/shard_0.jsonl" && -s "$DELTA_DIR/shard_1.jsonl" ]]; then
  delta_total="$(( $(wc -l < "$DELTA_DIR/shard_0.jsonl") + $(wc -l < "$DELTA_DIR/shard_1.jsonl") ))"
fi
if (( delta_total == 636 )); then
  echo "[$(date -Is)] all 636 delta rows already exist; skipping delta scoring"
else
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
  delta_total="$(( $(wc -l < "$DELTA_DIR/shard_0.jsonl") + $(wc -l < "$DELTA_DIR/shard_1.jsonl") ))"
  test "$delta_total" -eq 636 || { echo "Expected 636 delta rows, found $delta_total" >&2; exit 5; }
fi

if [[ -s "$CALIBRATION_DIR/bridge_calibrated_selected.jsonl" \
  && -s "$CALIBRATION_DIR/diagnostics.json" \
  && "$(wc -l < "$CALIBRATION_DIR/bridge_calibrated_selected.jsonl")" -eq 212 ]]; then
  echo "[$(date -Is)] complete 212-root calibration already exists; skipping calibration"
else
  echo "[$(date -Is)] fitting root-grouped cross-fitted calibrator"
  "$PYTHON_BIN" "$PROJECT_ROOT/fit_bridge_calibrated_rcst.py" \
    --candidates "$CANDIDATES" \
    --aggregates "$AGGREGATES" \
    --delta-features "$DELTA_DIR/shard_0.jsonl" "$DELTA_DIR/shard_1.jsonl" \
    --output-dir "$CALIBRATION_DIR" \
    --folds 5 --bootstrap-models 32 --ridge-alpha 5.0 \
    --confidence-z 1.0 --uncertainty-floor 0.01 --min-score 0.0 \
    > "$BASE/logs/calibration.log" 2>&1
fi

echo "[$(date -Is)] bridge calibration complete"
cat "$CALIBRATION_DIR/diagnostics.json"

if [[ "$RUN_FORMAL_TRAINING" != "1" ]]; then
  echo "[$(date -Is)] calibration-only mode; formal training/evaluation skipped"
  exit 0
fi

echo "[$(date -Is)] building calibrated 212-root training data"
TRAIN_ROOT="$BASE/formal_train_rcst"
DATA_DIR="$TRAIN_ROOT/data"
mkdir -p "$TRAIN_ROOT/logs" "$DATA_DIR"
"$PYTHON_BIN" "$PROJECT_ROOT/build_rcst_bridge_data.py" \
  --source-data "$SOURCE_DATA" \
  --all-selected "$ALL_SELECTED" \
  --rcst-selected "$CALIBRATION_DIR/bridge_calibrated_selected.jsonl" \
  --output-dir "$DATA_DIR" \
  2>&1 | tee "$TRAIN_ROOT/logs/build_data.log"

echo "[$(date -Is)] starting formal Bridge-Calibrated RCST training on F GPU0+GPU1"
rm -f "$TRAIN_ROOT/train.exit"
test -s "$PROJECT_ROOT/scaf_integration/ray_trainer_bridge.py"
cp "$SCAF_REPO/verl/trainer/ppo/ray_trainer.py" "$TRAIN_ROOT/logs/ray_trainer.before_bridge.py"
cp "$PROJECT_ROOT/scaf_integration/ray_trainer_bridge.py" "$SCAF_REPO/verl/trainer/ppo/ray_trainer.py"
if CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 NNODES=1 TP_SIZE=1 \
  SCAF_REPO="$SCAF_REPO" \
  MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$DATA_DIR/rcst_lcb_delta_212.parquet" \
  OUTPUT_DIR="$TRAIN_ROOT/train_rcst" TRAIN_STEPS="$TRAIN_STEPS" \
  TRAIN_BATCH_SIZE=32 ROLLOUTS=8 PPO_MINI_BATCH_SIZE=32 \
  MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048 SAVE_FREQ=25 \
  ACTOR_MICRO_BATCH_SIZE=2 LOG_PROB_MICRO_BATCH_SIZE=2 REF_LOG_PROB_MICRO_BATCH_SIZE=2 \
  GPU_MEMORY_UTILIZATION=0.35 FREE_CACHE_ENGINE=true RESUME_MODE=auto \
  BRIDGE_ENABLED=true BRIDGE_ALPHA=0.1 BRIDGE_CLIP=1.0 \
  EXPERIMENT_NAME=bridge-calibrated-rcst-212 PYTHON_BIN="$PYTHON_BIN" \
  bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" \
  > "$TRAIN_ROOT/logs/train.log" 2>&1; then
  train_code=0
else
  train_code=$?
fi
printf '%s\n' "$train_code" > "$TRAIN_ROOT/train.exit"
if (( train_code != 0 )); then
  echo "Formal Bridge-Calibrated RCST training failed: exit=$train_code" >&2
  exit 6
fi

ACTOR="$TRAIN_ROOT/train_rcst/checkpoints/global_step_${TRAIN_STEPS}/actor"
MERGED="$TRAIN_ROOT/bridge_calibrated_rcst_merged"
if [[ ! -s "$MERGED/config.json" ]] || ! compgen -G "$MERGED/*.safetensors" >/dev/null; then
  test -d "$ACTOR"
  rm -rf "$MERGED"
  echo "[$(date -Is)] merging final actor checkpoint"
  "$PYTHON_BIN" -m verl.model_merger merge \
    --backend fsdp --local_dir "$ACTOR" --target_dir "$MERGED" \
    2>&1 | tee "$TRAIN_ROOT/logs/merge.log"
fi
test -s "$MERGED/config.json"
compgen -G "$MERGED/*.safetensors" >/dev/null

echo "[$(date -Is)] evaluating Bridge-Calibrated RCST on seven benchmarks"
CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 EVAL_GPUS=0,1 EVAL_N_GPUS=2 \
EVAL_BATCH_SIZE=256 EVAL_GPU_MEMORY_UTILIZATION=0.45 \
EVAL_MAX_NUM_BATCHED_TOKENS=16384 SKIP_PREPARE=1 \
PYTHON_BIN="$PYTHON_BIN" SCAF_REPO="$SCAF_REPO" \
MODEL_PATH="$MERGED" METHOD_LABEL=bridge_calibrated_rcst \
PAPER_REFERENCE=vanilla CHECKPOINT_RULE="fixed global_step_${TRAIN_STEPS}" \
OUT="$TRAIN_ROOT/eval_bridge_calibrated_rcst" \
  bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
  2>&1 | tee "$TRAIN_ROOT/logs/eval.log"

echo "[$(date -Is)] Bridge-Calibrated RCST training and evaluation complete"
echo "Results: $TRAIN_ROOT/eval_bridge_calibrated_rcst/paper_comparison.md"
