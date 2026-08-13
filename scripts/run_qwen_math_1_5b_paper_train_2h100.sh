#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-$PROJECT_ROOT}"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
METHOD="${METHOD:-vanilla}"
MODE="${MODE:-smoke}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SKIP_ASSET_VALIDATION="${SKIP_ASSET_VALIDATION:-0}"

if [[ "$METHOD" != "vanilla" && "$METHOD" != "scaf" ]]; then
  echo "METHOD must be vanilla or scaf" >&2; exit 1
fi
if [[ "$MODE" != "paper" && "$MODE" != "smoke" ]]; then
  echo "MODE must be paper or smoke" >&2; exit 1
fi
if [[ "$MODE" == "paper" && "${CONFIRM_FULL_REPRO:-}" != "YES" ]]; then
  echo "Run Base evaluation first, then set CONFIRM_FULL_REPRO=YES." >&2; exit 2
fi

export CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
if [[ "$SKIP_ASSET_VALIDATION" != "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_qwen_math_1_5b_repro.py" \
    --project-root "$PROJECT_ROOT" --scaf-repo "$SCAF_REPO" --validate-only >/dev/null
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/qwen_math_1_5b_${METHOD}_${MODE}_$timestamp}"
mkdir -p "$OUTPUT_DIR/logs"
printf '%s\n' "$OUTPUT_DIR" > "$PROJECT_ROOT/outputs/latest_qwen_math_1_5b_${METHOD}_train.txt"
printf '%s\n' "$$" > "$OUTPUT_DIR/pid.txt"

if [[ "$MODE" == "paper" ]]; then
  TRAIN_BATCH_SIZE=256; ROLLOUTS=8; PPO_MINI_BATCH_SIZE=64
  MAX_RESPONSE_LENGTH=2048; TOTAL_EPOCHS=10
  SAVE_FREQ=10; TEST_FREQ=10; WARMUP_STEPS=75
  VAL_BEFORE_TRAIN=true
  TOTAL_STEPS_ARG=()
else
  TRAIN_BATCH_SIZE=8; ROLLOUTS=2; PPO_MINI_BATCH_SIZE=8
  MAX_RESPONSE_LENGTH=512; TOTAL_EPOCHS=1
  SAVE_FREQ=-1; TEST_FREQ=-1; WARMUP_STEPS=0
  VAL_BEFORE_TRAIN=false
  TOTAL_STEPS_ARG=(trainer.total_training_steps=1)
fi

if [[ "$METHOD" == "scaf" ]]; then
  ENTRYPOINT=hint_mix_grpo.main_ppo; MAX_PROMPT_LENGTH=4096
else
  ENTRYPOINT=verl.trainer.main_ppo; MAX_PROMPT_LENGTH=2048
fi

VAL_FILES="['$SCAF_REPO/data/AIME24/math-verify/system-p1/test.parquet','$SCAF_REPO/data/AIME25/math-verify/system-p1/test.parquet','$SCAF_REPO/data/AMC23/math-verify/system-p1/test.parquet','$SCAF_REPO/data/MinervaMath/math-verify/system-p1/test.parquet','$SCAF_REPO/data/MATH-500/math-verify/system-p1/test.parquet','$SCAF_REPO/data/OlympiadBench/math-verify/system-p1/test.parquet','$SCAF_REPO/data/GaoKao2023en/math-verify/system-p1/test.parquet']"

args=(
  algorithm.adv_estimator=grpo "data.train_files=$TRAIN_DATA" "data.val_files=$VAL_FILES"
  "data.train_batch_size=$TRAIN_BATCH_SIZE" data.val_batch_size=512
  "data.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" data.filter_overlong_prompts=true
  data.truncation=error "data.max_response_length=$MAX_RESPONSE_LENGTH"
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
  actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.kl_loss_type=low_var_kl algorithm.use_kl_in_reward=false
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.rollout.temperature=1.0
  "actor_rollout_ref.rollout.n=$ROLLOUTS" actor_rollout_ref.rollout.enable_chunked_prefill=false
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.60
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  actor_rollout_ref.actor.optim.warmup_style=constant actor_rollout_ref.actor.optim.weight_decay=0.0
  trainer.nnodes=1 trainer.n_gpus_per_node=2 "trainer.total_epochs=$TOTAL_EPOCHS"
  "trainer.save_freq=$SAVE_FREQ" "trainer.test_freq=$TEST_FREQ"
  "trainer.val_before_train=$VAL_BEFORE_TRAIN" trainer.val_only=false trainer.resume_mode=disable
  "trainer.default_local_dir=$OUTPUT_DIR/checkpoints" "trainer.rollout_data_dir=$OUTPUT_DIR/rollout"
  trainer.logger='[console]' trainer.project_name=scaf-grpo-reproduction
  "trainer.experiment_name=qwen-math-1.5b-${METHOD}-${MODE}" "${TOTAL_STEPS_ARG[@]}"
)
if [[ "$METHOD" == "scaf" ]]; then
  args+=(+trainer.with_hint=true +trainer.replace_hint_prompt=false +trainer.replace_num=1
    "+trainer.warmup_steps=$WARMUP_STEPS" "+trainer.hint_data_dir=$OUTPUT_DIR/hints")
fi

cd "$SCAF_REPO"
echo "Starting $METHOD ($MODE) on two H100 GPUs. Output: $OUTPUT_DIR"
"$PYTHON_BIN" -m "$ENTRYPOINT" "${args[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/train.log"
echo "Training complete. Select the best validation checkpoint under $OUTPUT_DIR/checkpoints."
