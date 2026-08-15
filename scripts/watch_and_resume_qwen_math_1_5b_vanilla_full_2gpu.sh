#!/usr/bin/env bash
set -euo pipefail

# Watch two GPUs and resume the interrupted full Qwen2.5-Math-1.5B Vanilla
# GRPO reproduction from the latest FSDP checkpoint.  It never starts from
# scratch unless ALLOW_FRESH_START=1 is explicitly supplied.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
RUN_ROOT="${RUN_ROOT:-$(cat "$PROJECT_ROOT/outputs/latest_qwen_math_1_5b_vanilla_train.txt")}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
ALLOW_FRESH_START="${ALLOW_FRESH_START:-0}"
MIN_FREE_MEMORY_PERCENT="${MIN_FREE_MEMORY_PERCENT:-90}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/fr${UID}_$$}"

export CUDA_VISIBLE_DEVICES RAY_TMPDIR RAY_ADDRESS=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled

mkdir -p "$RUN_ROOT/logs" "$RAY_TMPDIR"
WATCH_LOG="$RUN_ROOT/logs/full_train_auto_resume.log"

VAL_FILES="['$SCAF_REPO/data/AIME24/math-verify/system-p1/test.parquet','$SCAF_REPO/data/AIME25/math-verify/system-p1/test.parquet','$SCAF_REPO/data/AMC23/math-verify/system-p1/test.parquet','$SCAF_REPO/data/MinervaMath/math-verify/system-p1/test.parquet','$SCAF_REPO/data/MATH-500/math-verify/system-p1/test.parquet','$SCAF_REPO/data/OlympiadBench/math-verify/system-p1/test.parquet','$SCAF_REPO/data/GaoKao2023en/math-verify/system-p1/test.parquet']"

for path in "$MODEL_PATH/config.json" "$TRAIN_DATA"; do
  [[ -s "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

latest_checkpoint() {
  find "$RUN_ROOT/checkpoints" -maxdepth 1 -type d -name 'global_step_*' \
    -printf '%f\n' 2>/dev/null | sort -V | tail -n 1
}

training_process_alive() {
  ps -eo args= | grep -F "$RUN_ROOT/checkpoints" | grep -E 'main_ppo|TaskRunner' \
    | grep -v grep >/dev/null
}

gpu_processes() {
  local gpu output=""
  IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
  for gpu in "${gpu_ids[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    output+="$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  done
  printf '%s' "$output"
}

gpu_memory_report() {
  local gpu free total report=""
  IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
  for gpu in "${gpu_ids[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    IFS=',' read -r free total < <(
      nvidia-smi --id="$gpu" --query-gpu=memory.free,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1
    )
    free="${free//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    report+="GPU${gpu}:${free:-?}/${total:-?}MiB "
  done
  printf '%s' "$report"
}

gpus_have_enough_free_memory() {
  local gpu free total
  IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
  for gpu in "${gpu_ids[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    IFS=',' read -r free total < <(
      nvidia-smi --id="$gpu" --query-gpu=memory.free,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1
    )
    free="${free//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    [[ "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || return 1
    (( free * 100 >= total * MIN_FREE_MEMORY_PERCENT )) || return 1
  done
}

run_resume() {
  cd "$SCAF_REPO"
  conda run --no-capture-output -n "$ENV_NAME" python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=$TRAIN_DATA" "data.val_files=$VAL_FILES" \
    data.train_batch_size=256 data.val_batch_size=512 \
    data.dataloader_num_workers=0 data.max_prompt_length=2048 \
    data.max_response_length=2048 data.filter_overlong_prompts=true \
    data.truncation=error \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    trainer.nnodes=1 trainer.n_gpus_per_node=2 trainer.total_epochs=10 \
    trainer.save_freq=10 trainer.test_freq=10 \
    trainer.val_before_train=false trainer.val_only=false \
    trainer.resume_mode=auto \
    "trainer.default_local_dir=$RUN_ROOT/checkpoints" \
    "trainer.rollout_data_dir=$RUN_ROOT/rollout" \
    trainer.logger='[console]' trainer.project_name=scaf-grpo-reproduction \
    trainer.experiment_name=qwen-math-1.5b-vanilla-paper-resume
}

echo "[$(date '+%F %T')] monitoring full Vanilla run: $RUN_ROOT" | tee -a "$WATCH_LOG"

restarts=0
while true; do
  checkpoint="$(latest_checkpoint)"

  if training_process_alive; then
    echo "[$(date '+%F %T')] original full training is still running; latest=${checkpoint:-none}" \
      | tee -a "$WATCH_LOG"
    sleep "$POLL_SECONDS"
    continue
  fi

  active_gpu_pids="$(gpu_processes)"
  if [[ -n "$active_gpu_pids" ]]; then
    echo "[$(date '+%F %T')] GPUs are occupied by PID(s): $active_gpu_pids; waiting" \
      | tee -a "$WATCH_LOG"
    sleep "$POLL_SECONDS"
    continue
  fi

  memory_report="$(gpu_memory_report)"
  if ! gpus_have_enough_free_memory; then
    echo "[$(date '+%F %T')] insufficient free GPU memory ($memory_report); "\
"waiting for at least ${MIN_FREE_MEMORY_PERCENT}% free on every GPU" \
      | tee -a "$WATCH_LOG"
    sleep "$POLL_SECONDS"
    continue
  fi

  if [[ -z "$checkpoint" && "$ALLOW_FRESH_START" != "1" ]]; then
    echo "No checkpoint found under $RUN_ROOT/checkpoints; refusing a fresh start." \
      | tee -a "$WATCH_LOG" >&2
    exit 3
  fi

  restarts=$((restarts + 1))
  echo "[$(date '+%F %T')] GPUs ready ($memory_report); resume attempt $restarts/$MAX_RESTARTS from ${checkpoint:-fresh}" \
    | tee -a "$WATCH_LOG"

  set +e
  run_resume 2>&1 | tee -a "$WATCH_LOG"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "[$(date '+%F %T')] full Vanilla training completed successfully" \
      | tee -a "$WATCH_LOG"
    exit 0
  fi

  echo "[$(date '+%F %T')] resume exited with status $status" | tee -a "$WATCH_LOG"
  if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
    echo "Reached MAX_RESTARTS=$MAX_RESTARTS; manual inspection required." \
      | tee -a "$WATCH_LOG" >&2
    exit "$status"
  fi
  sleep "$POLL_SECONDS"
done
