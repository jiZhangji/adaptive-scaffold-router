#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SCAF_REPO:?Set SCAF_REPO}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${TRAIN_DATA:?Set TRAIN_DATA}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

MODE="${MODE:-scaf}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
SAVE_FREQ="${SAVE_FREQ:-25}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
CURRICULUM_MANIFEST="${CURRICULUM_MANIFEST:-}"
FADE_START="${FADE_START:-0}"
FADE_END="${FADE_END:-$TRAIN_STEPS}"
SCAF_WARMUP_STEPS="${SCAF_WARMUP_STEPS:-15}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$MODE" != "scaf" && "$MODE" != "fade_is" ]]; then
  echo "MODE must be scaf or fade_is" >&2
  exit 2
fi
if [[ "$MODE" == "fade_is" && ! -s "$CURRICULUM_MANIFEST" ]]; then
  echo "fade_is requires CURRICULUM_MANIFEST" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1 VLLM_USE_V1=0
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/rollout" "$OUTPUT_DIR/hints"

VAL_FILES="['$SCAF_REPO/data/AIME24/math-verify/system-p1/test.parquet','$SCAF_REPO/data/AIME25/math-verify/system-p1/test.parquet','$SCAF_REPO/data/AMC23/math-verify/system-p1/test.parquet','$SCAF_REPO/data/MinervaMath/math-verify/system-p1/test.parquet','$SCAF_REPO/data/MATH-500/math-verify/system-p1/test.parquet','$SCAF_REPO/data/OlympiadBench/math-verify/system-p1/test.parquet','$SCAF_REPO/data/GaoKao2023en/math-verify/system-p1/test.parquet']"

args=(
  algorithm.adv_estimator=grpo
  "data.train_files=$TRAIN_DATA" "data.val_files=$VAL_FILES"
  "data.train_batch_size=$TRAIN_BATCH_SIZE" data.val_batch_size=512
  data.dataloader_num_workers=0 "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=true data.truncation=error
  "actor_rollout_ref.model.path=$MODEL_PATH"
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.model.use_remove_padding=false
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.kl_loss_coef=0.0
  algorithm.use_kl_in_reward=false actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.rollout.temperature=1.0
  "actor_rollout_ref.rollout.n=$ROLLOUTS"
  actor_rollout_ref.rollout.enable_chunked_prefill=false
  actor_rollout_ref.rollout.name=vllm
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.free_cache_engine=false
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  actor_rollout_ref.actor.optim.warmup_style=constant
  actor_rollout_ref.actor.optim.weight_decay=0.0
  trainer.nnodes=1 trainer.n_gpus_per_node=1 trainer.total_epochs=100
  "trainer.total_training_steps=$TRAIN_STEPS"
  "trainer.save_freq=$SAVE_FREQ" trainer.test_freq=-1
  trainer.val_before_train=false trainer.val_only=false trainer.resume_mode=disable
  "trainer.default_local_dir=$OUTPUT_DIR/checkpoints"
  "trainer.rollout_data_dir=$OUTPUT_DIR/rollout"
  trainer.logger='[console]' trainer.project_name=complete-subproblem-pilot
  "trainer.experiment_name=$MODE"
  +trainer.with_hint=true +trainer.replace_num=1
  "+trainer.hint_data_dir=$OUTPUT_DIR/hints"
)

if [[ "$MODE" == "scaf" ]]; then
  args+=(+trainer.replace_hint_prompt=false "+trainer.warmup_steps=$SCAF_WARMUP_STEPS")
else
  args+=(
    +trainer.replace_hint_prompt=true +trainer.warmup_steps=0
    "+trainer.curriculum_manifest=$CURRICULUM_MANIFEST"
    "+trainer.curriculum_fade_start=$FADE_START"
    "+trainer.curriculum_fade_end=$FADE_END"
    "+trainer.curriculum_rollouts=$ROLLOUTS"
    +trainer.curriculum_off_context=true
    +trainer.curriculum_is_clip_low=0.2
    +trainer.curriculum_is_clip_high=5.0
    +trainer.curriculum_is_length_normalize=true
  )
fi

cd "$SCAF_REPO"
"$PYTHON_BIN" -m hint_mix_grpo.main_ppo "${args[@]}" \
  2>&1 | tee "$OUTPUT_DIR/logs/train.log"
