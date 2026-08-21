#!/usr/bin/env bash
set -euo pipefail

# One arm of the compute-matched Vanilla vs. subproblem pilot.  This is kept
# separate from the paper reproduction launcher so server-side reproduction
# edits are never overwritten by the pilot.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SCAF_REPO:?Set SCAF_REPO to the offline Scaf-GRPO checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-Math-1.5B}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the arm parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for this arm}"

TRAIN_STEPS="${TRAIN_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
SAVE_FREQ="${SAVE_FREQ:-25}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-4}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-4}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-4}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-false}"
RESUME_MODE="${RESUME_MODE:-auto}"
N_GPUS="${N_GPUS:-1}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fair-pilot-arm}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1 VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/rollout"

VAL_FILES="['$SCAF_REPO/data/AIME24/math-verify/system-p1/test.parquet','$SCAF_REPO/data/AIME25/math-verify/system-p1/test.parquet','$SCAF_REPO/data/AMC23/math-verify/system-p1/test.parquet','$SCAF_REPO/data/MinervaMath/math-verify/system-p1/test.parquet','$SCAF_REPO/data/MATH-500/math-verify/system-p1/test.parquet','$SCAF_REPO/data/OlympiadBench/math-verify/system-p1/test.parquet','$SCAF_REPO/data/GaoKao2023en/math-verify/system-p1/test.parquet']"

cd "$SCAF_REPO"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  "data.train_files=$TRAIN_DATA" "data.val_files=$VAL_FILES" \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" data.val_batch_size=512 \
  data.dataloader_num_workers=0 \
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
  "data.max_response_length=$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=true data.truncation=error \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE" \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.temperature=1.0 \
  "actor_rollout_ref.rollout.n=$ROLLOUTS" \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.name=vllm \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.load_format=safetensors \
  "actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE" \
  actor_rollout_ref.rollout.enforce_eager=true \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  "trainer.nnodes=$NNODES" "trainer.n_gpus_per_node=$N_GPUS" \
  trainer.total_epochs=100 "trainer.total_training_steps=$TRAIN_STEPS" \
  "trainer.save_freq=$SAVE_FREQ" trainer.test_freq=-1 \
  trainer.val_before_train=false trainer.val_only=false "trainer.resume_mode=$RESUME_MODE" \
  "trainer.default_local_dir=$OUTPUT_DIR/checkpoints" \
  "trainer.rollout_data_dir=$OUTPUT_DIR/rollout" \
  trainer.logger='[console]' trainer.project_name=fair-subproblem-pilot \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  2>&1 | tee "$OUTPUT_DIR/logs/train.log"
