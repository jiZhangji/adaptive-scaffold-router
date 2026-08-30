#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/powerleader/project/adaptive-scaffold-router}"
SCAF_REPO="${SCAF_REPO:-/home/powerleader/project/Scaf-GRPO}"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
ROOTS="${ROOTS:-$PROJECT_ROOT/outputs/bridge_212_assets/transfer_selected_candidates.jsonl}"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
AGGREGATES="${AGGREGATES:-$PROJECT_ROOT/outputs/bridge_calibrated_rcst_212/assets/rcst_aggregated.jsonl}"
RCST_SELECTED="${RCST_SELECTED:-$PROJECT_ROOT/outputs/bridge_212_assets/rcst_lcb_positive.jsonl}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/contrastive_bridge_ablation_212_$(date +%Y%m%d_%H%M%S)}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"

mkdir -p "$RUN_ROOT"/{data,logs,contrastive_fixed,contrastive_decay}
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_contrastive_bridge_ablation_212.txt"
date +%s > "$RUN_ROOT/start.epoch"

"$PYTHON_BIN" "$PROJECT_ROOT/build_contrastive_bridge_data.py" \
  --source-data "$SOURCE_DATA" --roots "$ROOTS" --candidates "$CANDIDATES" \
  --aggregates "$AGGREGATES" --rcst-selected "$RCST_SELECTED" \
  --output "$RUN_ROOT/data/rcst_lcb_contrastive_212.parquet" --expected-roots 212 \
  2>&1 | tee "$RUN_ROOT/logs/build_data.log"

cp "$SCAF_REPO/verl/trainer/ppo/ray_trainer.py" "$RUN_ROOT/ray_trainer.before_contrastive.py"
cp "$PROJECT_ROOT/scaf_integration/ray_trainer_bridge.py" "$SCAF_REPO/verl/trainer/ppo/ray_trainer.py"

run_arm() {
  local gpu="$1" name="$2" output="$3" decay_start="$4" decay_end="$5"
  CUDA_VISIBLE_DEVICES="$gpu" N_GPUS=1 NNODES=1 TP_SIZE=1 \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" \
  TRAIN_DATA="$RUN_ROOT/data/rcst_lcb_contrastive_212.parquet" OUTPUT_DIR="$output" \
  TRAIN_STEPS="$TRAIN_STEPS" TRAIN_BATCH_SIZE=32 ROLLOUTS=8 PPO_MINI_BATCH_SIZE=32 \
  MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048 SAVE_FREQ=25 \
  ACTOR_MICRO_BATCH_SIZE=2 LOG_PROB_MICRO_BATCH_SIZE=2 REF_LOG_PROB_MICRO_BATCH_SIZE=2 \
  GPU_MEMORY_UTILIZATION=0.35 FREE_CACHE_ENGINE=true RESUME_MODE=auto \
  BRIDGE_ENABLED=true BRIDGE_ALPHA=0.05 BRIDGE_CLIP=1.0 BRIDGE_CONTRASTIVE=true \
  BRIDGE_DECAY_START_FRACTION="$decay_start" BRIDGE_DECAY_END_FRACTION="$decay_end" \
  EXPERIMENT_NAME="$name" PYTHON_BIN="$PYTHON_BIN" \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" > "$output/launcher.log" 2>&1
}

run_arm 0 contrastive-bridge-fixed-212 "$RUN_ROOT/contrastive_fixed" 1.0 1.0 &
pid0=$!
run_arm 1 contrastive-bridge-decay-212 "$RUN_ROOT/contrastive_decay" 0.30 0.70 &
pid1=$!
printf '%s\n' "$pid0" > "$RUN_ROOT/contrastive_fixed.pid"
printf '%s\n' "$pid1" > "$RUN_ROOT/contrastive_decay.pid"
set +e
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
set -e
printf '%s\n' "$code0" > "$RUN_ROOT/contrastive_fixed.exit"
printf '%s\n' "$code1" > "$RUN_ROOT/contrastive_decay.exit"
if (( code0 != 0 || code1 != 0 )); then
  echo "Training failed: fixed=$code0 decay=$code1" >&2
  exit 1
fi

merge_arm() {
  local arm="$1"
  "$PYTHON_BIN" -m verl.model_merger merge --backend fsdp \
    --local_dir "$RUN_ROOT/$arm/checkpoints/global_step_${TRAIN_STEPS}/actor" \
    --target_dir "$RUN_ROOT/${arm}_merged" \
    > "$RUN_ROOT/logs/${arm}_merge.log" 2>&1
}
merge_arm contrastive_fixed
merge_arm contrastive_decay

eval_arm() {
  local gpu="$1" arm="$2" label="$3"
  CUDA_VISIBLE_DEVICES="$gpu" N_GPUS=1 EVAL_BATCH_SIZE=64 \
  EVAL_GPU_MEMORY_UTILIZATION=0.45 EVAL_MAX_NUM_BATCHED_TOKENS=8192 SKIP_PREPARE=1 \
  PYTHON_BIN="$PYTHON_BIN" SCAF_REPO="$SCAF_REPO" \
  MODEL_PATH="$RUN_ROOT/${arm}_merged" METHOD_LABEL="$label" PAPER_REFERENCE=vanilla \
  CHECKPOINT_RULE="$label fixed global_step_${TRAIN_STEPS}" \
  OUT="$RUN_ROOT/eval_${arm}" \
    bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
    > "$RUN_ROOT/logs/${arm}_eval.log" 2>&1
}
eval_arm 0 contrastive_fixed contrastive_bridge_fixed_212 &
epid0=$!
eval_arm 1 contrastive_decay contrastive_bridge_decay_212 &
epid1=$!
set +e
wait "$epid0"; ecode0=$?
wait "$epid1"; ecode1=$?
set -e
printf '%s\n' "$ecode0" > "$RUN_ROOT/contrastive_fixed.eval.exit"
printf '%s\n' "$ecode1" > "$RUN_ROOT/contrastive_decay.eval.exit"
if (( ecode0 != 0 || ecode1 != 0 )); then
  echo "Evaluation failed: fixed=$ecode0 decay=$ecode1" >&2
  exit 2
fi
date +%s > "$RUN_ROOT/end.epoch"
echo "Contrastive Bridge ablation complete: $RUN_ROOT"
