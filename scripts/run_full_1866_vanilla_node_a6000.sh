#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/powerleader/project/adaptive-scaffold-router}"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
ENV_PATH="${ENV_PATH:-/home/powerleader/project/envs/scaf-grpo}"
SCAF_REPO="${SCAF_REPO:-/home/powerleader/project/Scaf-GRPO}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to the 1,866-root parquet}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
EXIT_FILE="${EXIT_FILE:?Set EXIT_FILE}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
SAVE_FREQ="${SAVE_FREQ:-44}"
STAGE_1_END="${STAGE_1_END:-88}"
STAGE_2_END="${STAGE_2_END:-308}"
TOTAL_STEPS="${TOTAL_STEPS:-440}"

[[ -s "$CONDA_SH" ]] || { echo "Missing conda init: $CONDA_SH" >&2; exit 2; }
[[ -s "$TRAIN_DATA" ]] || { echo "Missing training parquet: $TRAIN_DATA" >&2; exit 2; }
[[ -s "$MODEL_PATH/config.json" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }
(( 0 < STAGE_1_END && STAGE_1_END < STAGE_2_END && STAGE_2_END < TOTAL_STEPS )) || {
  echo "Require 0 < STAGE_1_END < STAGE_2_END < TOTAL_STEPS" >&2
  exit 2
}

source "$CONDA_SH"
conda activate "$ENV_PATH"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
mkdir -p "$RUN_ROOT"/{logs,vanilla}
trap 'code=$?; printf "%s\n" "$code" > "$EXIT_FILE"' EXIT
printf '%s\n' "full_1866_vanilla_grpo_fixed_440_updates_v1" > "$RUN_ROOT/protocol.txt"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

for end_step in "$STAGE_1_END" "$STAGE_2_END" "$TOTAL_STEPS"; do
  checkpoint="$RUN_ROOT/vanilla/checkpoints/global_step_$end_step/actor"
  if [[ -d "$checkpoint" ]]; then
    echo "Vanilla global_step_$end_step already exists; continuing."
    continue
  fi
  echo "Vanilla GRPO: training/resuming to global_step_$end_step."
  CUDA_VISIBLE_DEVICES=0 RAY_TMPDIR="/tmp/vanilla1866_${UID}_${end_step}" \
  TRAIN_STEPS="$end_step" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  ROLLOUTS="$ROLLOUTS" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
  GPU_MEMORY_UTILIZATION=0.30 ACTOR_MICRO_BATCH_SIZE=2 \
  LOG_PROB_MICRO_BATCH_SIZE=2 REF_LOG_PROB_MICRO_BATCH_SIZE=2 \
  FREE_CACHE_ENGINE=true N_GPUS=1 TP_SIZE=1 RESUME_MODE=auto \
  PYTHON_BIN="$PYTHON_BIN" SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" \
  TRAIN_DATA="$TRAIN_DATA" OUTPUT_DIR="$RUN_ROOT/vanilla" \
  EXPERIMENT_NAME=full-1866-vanilla-grpo \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" \
    > "$RUN_ROOT/logs/vanilla_to_${end_step}.log" 2>&1
done

final_actor="$RUN_ROOT/vanilla/checkpoints/global_step_$TOTAL_STEPS/actor"
[[ -d "$final_actor" ]] || { echo "Missing final Vanilla checkpoint" >&2; exit 4; }

merged="$RUN_ROOT/vanilla_merged"
if [[ ! -s "$merged/config.json" ]]; then
  "$PYTHON_BIN" -m verl.model_merger merge --backend fsdp \
    --local_dir "$final_actor" --target_dir "$merged" \
    2>&1 | tee "$RUN_ROOT/logs/merge.log"
fi

CUDA_VISIBLE_DEVICES=0 N_GPUS=1 SKIP_PREPARE=1 \
PROJECT_ROOT_OVERRIDE="$PROJECT_ROOT" SCAF_REPO="$SCAF_REPO" \
MODEL_PATH="$merged" METHOD_LABEL=vanilla PAPER_REFERENCE=vanilla \
CHECKPOINT_RULE="full 1,866-root pool, fixed global_step_$TOTAL_STEPS" \
OUT="$RUN_ROOT/eval_vanilla" PYTHON_BIN="$PYTHON_BIN" \
  bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
  2>&1 | tee "$RUN_ROOT/logs/eval.log"

echo "Vanilla full-pool training and evaluation complete: $RUN_ROOT"
