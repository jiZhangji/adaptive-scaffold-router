#!/usr/bin/env bash
set -euo pipefail

: "${SCAF_REPO:?Set SCAF_REPO to the official Scaf-GRPO checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
: "${DATA_FILE:?Set DATA_FILE to the official Scaf-GRPO parquet file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and logs}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${N_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
ROLLOUTS="${ROLLOUTS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
WITH_HINT="${WITH_HINT:-true}"
CURRICULUM_MANIFEST="${CURRICULUM_MANIFEST:-}"
CURRICULUM_FADE_START="${CURRICULUM_FADE_START:-0}"
CURRICULUM_FADE_END="${CURRICULUM_FADE_END:-1000000000}"
CURRICULUM_ROLLOUTS="${CURRICULUM_ROLLOUTS:-$ROLLOUTS}"
CURRICULUM_OFF_CONTEXT="${CURRICULUM_OFF_CONTEXT:-true}"
CURRICULUM_IS_CLIP_LOW="${CURRICULUM_IS_CLIP_LOW:-0.2}"
CURRICULUM_IS_CLIP_HIGH="${CURRICULUM_IS_CLIP_HIGH:-5.0}"
REPLACE_HINT_PROMPT="${REPLACE_HINT_PROMPT:-false}"
if [[ -n "$CURRICULUM_MANIFEST" && "$CURRICULUM_OFF_CONTEXT" == "true" ]]; then
  REPLACE_HINT_PROMPT=true
fi

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

cd "$SCAF_REPO"

args=(
  algorithm.adv_estimator=grpo
  "data.train_files=$DATA_FILE"
  "data.val_files=$DATA_FILE"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  data.val_batch_size=16
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  data.filter_overlong_prompts=true
  data.truncation=error
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.model.use_remove_padding=false
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.kl_loss_coef=0.0
  algorithm.use_kl_in_reward=false
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.rollout.temperature=1.0
  "actor_rollout_ref.rollout.n=$ROLLOUTS"
  actor_rollout_ref.rollout.enable_chunked_prefill=false
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.0
  trainer.nnodes=1
  "trainer.n_gpus_per_node=$N_GPUS"
  trainer.total_epochs=1
  "trainer.total_training_steps=$TOTAL_STEPS"
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.val_before_train=false
  trainer.resume_mode=disable
  "+trainer.with_hint=$WITH_HINT"
  "+trainer.replace_hint_prompt=$REPLACE_HINT_PROMPT"
  +trainer.replace_num=1
  +trainer.warmup_steps=0
  "trainer.default_local_dir=$OUTPUT_DIR/checkpoints"
  "trainer.rollout_data_dir=$OUTPUT_DIR/rollout"
  "+trainer.hint_data_dir=$OUTPUT_DIR/hints"
  trainer.logger='[console]'
  trainer.project_name=dsfl-smoke
  trainer.experiment_name=scaf-two-gpu-smoke
)

if [[ -n "$CURRICULUM_MANIFEST" ]]; then
  args+=(
    "+trainer.curriculum_manifest=$CURRICULUM_MANIFEST"
    "+trainer.curriculum_fade_start=$CURRICULUM_FADE_START"
    "+trainer.curriculum_fade_end=$CURRICULUM_FADE_END"
    "+trainer.curriculum_rollouts=$CURRICULUM_ROLLOUTS"
    "+trainer.curriculum_off_context=$CURRICULUM_OFF_CONTEXT"
    "+trainer.curriculum_is_clip_low=$CURRICULUM_IS_CLIP_LOW"
    "+trainer.curriculum_is_clip_high=$CURRICULUM_IS_CLIP_HIGH"
    +trainer.curriculum_is_length_normalize=true
  )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exec "$PYTHON_BIN" -m hint_mix_grpo.main_ppo --cfg job "${args[@]}"
fi

exec "$PYTHON_BIN" -m hint_mix_grpo.main_ppo "${args[@]}"
