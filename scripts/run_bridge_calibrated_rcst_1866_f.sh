#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/powerleader/project/adaptive-scaffold-router}"
SCAF_REPO="${SCAF_REPO:-/home/powerleader/project/Scaf-GRPO}"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
BASE="${BASE:-$PROJECT_ROOT/outputs/bridge_calibrated_rcst_1866}"
ASSETS="$BASE/assets"
DELTA_DIR="$BASE/delta_system_preserving_v2"
SELECTION_DIR="$BASE/selection_frozen_212_v1"
TRAIN_ROOT="$BASE/formal_train"
CANDIDATES="$ASSETS/candidate_sets.jsonl"
PROBE_42="$ASSETS/probe_seed_42.jsonl"
PROBE_314159="$ASSETS/probe_seed_314159.jsonl"
CALIBRATOR="$PROJECT_ROOT/outputs/bridge_calibrated_rcst_212/calibration_crossfit_v1/calibrator_model.json"
CROSSFIT_SELECTED="$PROJECT_ROOT/outputs/bridge_calibrated_rcst_212/calibration_crossfit_v1/bridge_calibrated_selected.jsonl"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
TRAIN_STEPS="${TRAIN_STEPS:-440}"
SAVE_FREQ="${SAVE_FREQ:-44}"

mkdir -p "$BASE/logs" "$ASSETS" "$DELTA_DIR" "$SELECTION_DIR" "$TRAIN_ROOT/logs"
if ! mkdir "$BASE/run.lock" 2>/dev/null; then
  echo "Another full Bridge-Calibrated RCST launcher owns $BASE/run.lock" >&2
  exit 3
fi
pipeline_start="$(date +%s)"
printf 'pipeline_start\t%s\t%s\n' "$pipeline_start" "$(date -Is)" > "$BASE/stage_times.tsv"
trap 'code=$?; now=$(date +%s); printf "pipeline_exit_%s\t%s\t%s\n" "$code" "$now" "$(date -Is)" >> "$BASE/stage_times.tsv"; printf "%s\n" "$code" > "$BASE/pipeline.exit"; rmdir "$BASE/run.lock" 2>/dev/null || true' EXIT

mark_stage() {
  local name="$1" started="$2" now
  now="$(date +%s)"
  printf '%s\t%s\t%s\t%s\n' "$name" "$now" "$((now - started))" "$(date -Is)" \
    >> "$BASE/stage_times.tsv"
}

for path in "$PYTHON_BIN" "$MODEL_PATH/config.json" "$SOURCE_DATA" \
  "$CANDIDATES" "$PROBE_42" "$PROBE_314159" "$CALIBRATOR" "$CROSSFIT_SELECTED" \
  "$PROJECT_ROOT/apply_bridge_calibrator.py" "$PROJECT_ROOT/build_rcst_bridge_data.py" \
  "$PROJECT_ROOT/rcst_bridge_dataset.py" "$PROJECT_ROOT/scaf_integration/ray_trainer_bridge.py"; do
  test -s "$path" || { echo "Missing required input: $path" >&2; exit 2; }
done
test "$(wc -l < "$CANDIDATES")" -eq 5598
test "$(wc -l < "$PROBE_42")" -eq 5598
test "$(wc -l < "$PROBE_314159")" -eq 5598

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
if (( delta_total == 5598 )); then
  echo "[$(date -Is)] all 5,598 Bridge feature rows exist; skipping scoring"
else
  stage_start="$(date +%s)"
  echo "[$(date -Is)] scoring 5,598 candidates on two A6000 GPUs"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$PROJECT_ROOT/score_bridge_delta_features.py" \
    "${common[@]}" --output "$DELTA_DIR/shard_0.jsonl" \
    --num-shards 2 --shard-index 0 > "$BASE/logs/delta_shard_0.log" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$PROJECT_ROOT/score_bridge_delta_features.py" \
    "${common[@]}" --output "$DELTA_DIR/shard_1.jsonl" \
    --num-shards 2 --shard-index 1 > "$BASE/logs/delta_shard_1.log" 2>&1 &
  pid1=$!
  printf '%s\n' "$pid0" > "$DELTA_DIR/shard_0.pid"
  printf '%s\n' "$pid1" > "$DELTA_DIR/shard_1.pid"
  set +e
  wait "$pid0"; code0=$?
  wait "$pid1"; code1=$?
  set -e
  printf '%s\n' "$code0" > "$DELTA_DIR/shard_0.exit"
  printf '%s\n' "$code1" > "$DELTA_DIR/shard_1.exit"
  (( code0 == 0 && code1 == 0 )) || { echo "Bridge scoring failed: $code0/$code1" >&2; exit 4; }
  delta_total="$(( $(wc -l < "$DELTA_DIR/shard_0.jsonl") + $(wc -l < "$DELTA_DIR/shard_1.jsonl") ))"
  test "$delta_total" -eq 5598 || { echo "Expected 5598 rows, found $delta_total" >&2; exit 5; }
  mark_stage bridge_scoring_complete "$stage_start"
fi

if [[ -s "$SELECTION_DIR/bridge_calibrated_selected.jsonl" \
  && "$(wc -l < "$SELECTION_DIR/bridge_calibrated_selected.jsonl")" -eq 1866 ]]; then
  echo "[$(date -Is)] complete 1,866-root selection exists; skipping selection"
else
  stage_start="$(date +%s)"
  echo "[$(date -Is)] applying frozen 212-root calibrator to full candidate pool"
  "$PYTHON_BIN" "$PROJECT_ROOT/apply_bridge_calibrator.py" \
    --candidates "$CANDIDATES" \
    --delta-features "$DELTA_DIR/shard_0.jsonl" "$DELTA_DIR/shard_1.jsonl" \
    --calibrator-model "$CALIBRATOR" \
    --crossfit-selected "$CROSSFIT_SELECTED" \
    --output-dir "$SELECTION_DIR" \
    --confidence-z 1.0 --uncertainty-floor 0.01 --min-score 0.0 \
    2>&1 | tee "$BASE/logs/selection.log"
  test "$(wc -l < "$SELECTION_DIR/bridge_calibrated_selected.jsonl")" -eq 1866
  mark_stage selection_complete "$stage_start"
fi

stage_start="$(date +%s)"
echo "[$(date -Is)] building full 1,866-root Bridge training parquet"
mkdir -p "$TRAIN_ROOT/data"
"$PYTHON_BIN" "$PROJECT_ROOT/build_rcst_bridge_data.py" \
  --source-data "$SOURCE_DATA" --all-selected "$CANDIDATES" \
  --rcst-selected "$SELECTION_DIR/bridge_calibrated_selected.jsonl" \
  --output-dir "$TRAIN_ROOT/data" --expected-roots 1866 \
  2>&1 | tee "$TRAIN_ROOT/logs/build_data.log"
mark_stage data_build_complete "$stage_start"

cp "$SCAF_REPO/verl/trainer/ppo/ray_trainer.py" "$TRAIN_ROOT/logs/ray_trainer.before_bridge.py"
cp "$PROJECT_ROOT/scaf_integration/ray_trainer_bridge.py" "$SCAF_REPO/verl/trainer/ppo/ray_trainer.py"

ACTOR="$TRAIN_ROOT/train_rcst/checkpoints/global_step_${TRAIN_STEPS}/actor"
if [[ -d "$ACTOR" ]]; then
  echo "[$(date -Is)] final step-$TRAIN_STEPS checkpoint exists; skipping training"
else
  stage_start="$(date +%s)"
  echo "[$(date -Is)] starting 440-step two-GPU Bridge-Calibrated RCST training"
  rm -f "$TRAIN_ROOT/train.exit"
  if CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 NNODES=1 TP_SIZE=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" \
    TRAIN_DATA="$TRAIN_ROOT/data/rcst_lcb_delta_1866.parquet" \
    OUTPUT_DIR="$TRAIN_ROOT/train_rcst" TRAIN_STEPS="$TRAIN_STEPS" \
    TRAIN_BATCH_SIZE=32 ROLLOUTS=8 PPO_MINI_BATCH_SIZE=32 \
    MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048 SAVE_FREQ="$SAVE_FREQ" \
    ACTOR_MICRO_BATCH_SIZE=2 LOG_PROB_MICRO_BATCH_SIZE=2 REF_LOG_PROB_MICRO_BATCH_SIZE=2 \
    GPU_MEMORY_UTILIZATION=0.35 FREE_CACHE_ENGINE=true RESUME_MODE=auto \
    BRIDGE_ENABLED=true BRIDGE_ALPHA=0.1 BRIDGE_CLIP=1.0 \
    EXPERIMENT_NAME=bridge-calibrated-rcst-1866 PYTHON_BIN="$PYTHON_BIN" \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" \
    > "$TRAIN_ROOT/logs/train.log" 2>&1; then
    train_code=0
  else
    train_code=$?
  fi
  printf '%s\n' "$train_code" > "$TRAIN_ROOT/train.exit"
  (( train_code == 0 )) || { echo "Full training failed: exit=$train_code" >&2; exit 6; }
  mark_stage training_complete "$stage_start"
fi

MERGED="$TRAIN_ROOT/bridge_calibrated_rcst_1866_merged"
if [[ ! -s "$MERGED/config.json" ]] || ! compgen -G "$MERGED/*.safetensors" >/dev/null; then
  stage_start="$(date +%s)"
  rm -rf "$MERGED"
  echo "[$(date -Is)] merging final step-$TRAIN_STEPS actor"
  "$PYTHON_BIN" -m verl.model_merger merge --backend fsdp \
    --local_dir "$ACTOR" --target_dir "$MERGED" \
    2>&1 | tee "$TRAIN_ROOT/logs/merge.log"
  mark_stage merge_complete "$stage_start"
fi

stage_start="$(date +%s)"
echo "[$(date -Is)] evaluating full Bridge-Calibrated RCST on seven benchmarks"
CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 EVAL_BATCH_SIZE=256 \
EVAL_GPU_MEMORY_UTILIZATION=0.45 EVAL_MAX_NUM_BATCHED_TOKENS=16384 SKIP_PREPARE=1 \
PYTHON_BIN="$PYTHON_BIN" SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MERGED" \
METHOD_LABEL=bridge_calibrated_rcst_full_1866 PAPER_REFERENCE=vanilla \
CHECKPOINT_RULE="fixed global_step_${TRAIN_STEPS}" \
OUT="$TRAIN_ROOT/eval_bridge_calibrated_rcst_1866" \
  bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
  2>&1 | tee "$TRAIN_ROOT/logs/eval.log"
mark_stage evaluation_complete "$stage_start"

pipeline_end="$(date +%s)"
printf 'pipeline_complete\t%s\t%s\t%s\n' "$pipeline_end" \
  "$((pipeline_end - pipeline_start))" "$(date -Is)" >> "$BASE/stage_times.tsv"
echo "[$(date -Is)] full 1,866-root Bridge-Calibrated RCST complete"
echo "Timing: $BASE/stage_times.tsv"
echo "Results: $TRAIN_ROOT/eval_bridge_calibrated_rcst_1866/paper_comparison.md"
