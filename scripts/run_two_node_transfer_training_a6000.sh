#!/usr/bin/env bash
set -Eeuo pipefail

# Run one fixed-budget three-stage method on two one-GPU A6000 nodes.
# Both nodes keep identical absolute paths. The global batch, rollout count,
# learning rate, and optimizer-step budget remain unchanged from the single-GPU
# comparison; only the data-parallel world size changes.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_PYTHON="${CONDA_PYTHON:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
SCAF_REPO="${SCAF_REPO:-/home/powerleader/project/Scaf-GRPO}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
SELECTED_CANDIDATES="${SELECTED_CANDIDATES:?Set SELECTED_CANDIDATES to one selected row per root}"
METHOD_LABEL="${METHOD_LABEL:-Transfer-Aware Distributed}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/two_node_transfer_$(date +%Y%m%d_%H%M%S)}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-$PROJECT_ROOT/outputs/complete_four_way_ordered_20260815_155120}"
PEER_HOST="${PEER_HOST:-powerleader@10.6.2.3}"
PEER_KEY="${PEER_KEY:-$HOME/.ssh/a6000_peer_ed25519}"
ENV_PREFIX="${ENV_PREFIX:-/home/powerleader/project/envs/scaf-grpo}"
PRECONDITION_STEPS="${PRECONDITION_STEPS:-10}"
ROOT_ALIGNED_END_STEP="${ROOT_ALIGNED_END_STEP:-35}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
SAVE_FREQ="${SAVE_FREQ:-5}"
AUTO_EVAL="${AUTO_EVAL:-1}"
RUN_LEARNABILITY="${RUN_LEARNABILITY:-1}"
LEARNABILITY_FILE="${LEARNABILITY_FILE:-}"
GATE_MODE="${GATE_MODE:-positive}"

SSH=(ssh -i "$PEER_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15)
RSYNC_SHELL="ssh -i $PEER_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"

remote() { "${SSH[@]}" "$PEER_HOST" "$@"; }

reset_loader_state() {
  local checkpoint="$1" tag="$2" file
  for file in data.pt train_dataset_state.pt current_epoch.txt; do
    if [[ -f "$checkpoint/$file" && ! -e "$checkpoint/$file.$tag.bak" ]]; then
      mv "$checkpoint/$file" "$checkpoint/$file.$tag.bak"
    fi
  done
  remote "checkpoint='$checkpoint'; tag='$tag'; \
    for file in data.pt train_dataset_state.pt current_epoch.txt; do \
      source=\"\$checkpoint/\$file\"; backup=\"\$source.\$tag.bak\"; \
      if [[ -f \"\$source\" && ! -e \"\$backup\" ]]; then mv \"\$source\" \"\$backup\"; fi; \
    done"
}

if (( PRECONDITION_STEPS < 0 || ROOT_ALIGNED_END_STEP <= PRECONDITION_STEPS || TOTAL_STEPS <= ROOT_ALIGNED_END_STEP )); then
  echo "Require 0 <= PRECONDITION_STEPS < ROOT_ALIGNED_END_STEP < TOTAL_STEPS" >&2
  exit 2
fi
for asset in "$CONDA_PYTHON" "$MODEL_PATH/config.json" "$SOURCE_DATA" "$SELECTED_CANDIDATES"; do
  [[ -s "$asset" ]] || { echo "Missing asset: $asset" >&2; exit 3; }
done

mkdir -p "$RUN_ROOT"/{data,learnability,logs,proposed}
printf '%s\n' "$METHOD_LABEL" > "$RUN_ROOT/method_label.txt"
printf '%s\n' "two_node_data_parallel_fixed_global_budget_v1" > "$RUN_ROOT/protocol.txt"

if [[ -n "$LEARNABILITY_FILE" ]]; then
  cp "$LEARNABILITY_FILE" "$RUN_ROOT/learnability/subproblem_learnability.jsonl"
elif [[ "$RUN_LEARNABILITY" == "1" && ! -s "$RUN_ROOT/learnability/subproblem_learnability.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$CONDA_PYTHON" "$PROJECT_ROOT/probe_selected_subproblem_learnability.py" \
    --candidates "$SELECTED_CANDIDATES" --model "$MODEL_PATH" \
    --output-dir "$RUN_ROOT/learnability" --samples "$ROLLOUTS" \
    --group-size "$ROLLOUTS" --batch-size 16 --job-chunk-size 128 \
    --device cuda:0 --dtype bfloat16 --max-input-tokens 2048 \
    --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 --stop-after-boxed \
    2>&1 | tee "$RUN_ROOT/logs/learnability.log"
fi

builder_args=()
case "$GATE_MODE" in
  all) ;;
  positive) builder_args+=(--positive-transfer-only) ;;
  conservative)
    builder_args+=(--positive-transfer-only --min-transfer-gain 0.25 --min-post-update-probability 0.5)
    ;;
  *) echo "GATE_MODE must be all, positive, or conservative" >&2; exit 2 ;;
esac
learnability_args=()
if [[ -s "$RUN_ROOT/learnability/subproblem_learnability.jsonl" ]]; then
  learnability_args=(--learnability "$RUN_ROOT/learnability/subproblem_learnability.jsonl")
fi
"$CONDA_PYTHON" "$PROJECT_ROOT/build_student_aware_preconditioning_experiment.py" \
  --source-data "$SOURCE_DATA" --candidates "$SELECTED_CANDIDATES" \
  "${learnability_args[@]}" --output-dir "$RUN_ROOT/data" \
  --scaffold-ready-threshold 0.5 --contrast-min 0.0 --group-size "$ROLLOUTS" \
  "${builder_args[@]}" 2>&1 | tee "$RUN_ROOT/logs/build_data.log"

precondition_rows="$($CONDA_PYTHON -c "import json; print(json.load(open('$RUN_ROOT/data/summary.json'))['precondition_rows'])")"
if (( precondition_rows > 0 && PRECONDITION_STEPS > 0 )); then
  "$CONDA_PYTHON" "$PROJECT_ROOT/normalize_subproblem_reward_data.py" \
    --input "$RUN_ROOT/data/precondition_train.parquet" \
    --output "$RUN_ROOT/data/precondition_train_compatible.parquet" \
    2>&1 | tee "$RUN_ROOT/logs/normalize.log"
fi

remote "mkdir -p '$RUN_ROOT'/{data,learnability,logs,proposed}"
rsync -aH -e "$RSYNC_SHELL" "$RUN_ROOT/data/" "$PEER_HOST:$RUN_ROOT/data/"
rsync -aH -e "$RSYNC_SHELL" "$RUN_ROOT/learnability/" "$PEER_HOST:$RUN_ROOT/learnability/"

"$PROJECT_ROOT/scripts/manage_two_node_ray_a6000.sh" start \
  2>&1 | tee "$RUN_ROOT/logs/ray_start.log"
trap '"$PROJECT_ROOT/scripts/manage_two_node_ray_a6000.sh" stop >/dev/null 2>&1 || true' EXIT

export RAY_ADDRESS=auto NCCL_SOCKET_IFNAME=eno1 GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled VLLM_USE_V1=0
common_env=(
  NNODES=2 N_GPUS=1 TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" ROLLOUTS="$ROLLOUTS"
  PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" SAVE_FREQ="$SAVE_FREQ"
  MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH"
  ACTOR_MICRO_BATCH_SIZE=2 LOG_PROB_MICRO_BATCH_SIZE=2
  REF_LOG_PROB_MICRO_BATCH_SIZE=2 FREE_CACHE_ENGINE=true
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" PYTHON_BIN="$CONDA_PYTHON"
  OUTPUT_DIR="$RUN_ROOT/proposed" RESUME_MODE=auto
)

if (( precondition_rows > 0 && PRECONDITION_STEPS > 0 )) && \
   [[ ! -d "$RUN_ROOT/proposed/checkpoints/global_step_$PRECONDITION_STEPS/actor" ]]; then
  env "${common_env[@]}" TRAIN_STEPS="$PRECONDITION_STEPS" \
    TRAIN_DATA="$RUN_ROOT/data/precondition_train_compatible.parquet" \
    GPU_MEMORY_UTILIZATION=0.35 EXPERIMENT_NAME=td-precondition \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" \
    > "$RUN_ROOT/logs/precondition_train.log" 2>&1
fi

if (( precondition_rows > 0 && PRECONDITION_STEPS > 0 )); then
  reset_loader_state "$RUN_ROOT/proposed/checkpoints/global_step_$PRECONDITION_STEPS" stage1
  fade_start="$PRECONDITION_STEPS"
else
  fade_start=0
fi

if [[ ! -d "$RUN_ROOT/proposed/checkpoints/global_step_$ROOT_ALIGNED_END_STEP/actor" ]]; then
  env "${common_env[@]}" MODE=fade_is TRAIN_STEPS="$ROOT_ALIGNED_END_STEP" \
    TRAIN_DATA="$RUN_ROOT/data/root_scaffold_train.parquet" \
    CURRICULUM_MANIFEST="$RUN_ROOT/data/curriculum.jsonl" \
    FADE_START="$fade_start" FADE_END="$ROOT_ALIGNED_END_STEP" \
    GPU_MEMORY_UTILIZATION=0.30 \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_scaffold_arm_1gpu.sh" \
    > "$RUN_ROOT/logs/root_aligned_train.log" 2>&1
fi

reset_loader_state "$RUN_ROOT/proposed/checkpoints/global_step_$ROOT_ALIGNED_END_STEP" stage2
if [[ ! -d "$RUN_ROOT/proposed/checkpoints/global_step_$TOTAL_STEPS/actor" ]]; then
  env "${common_env[@]}" TRAIN_STEPS="$TOTAL_STEPS" \
    TRAIN_DATA="$RUN_ROOT/data/root_only_train.parquet" \
    GPU_MEMORY_UTILIZATION=0.28 EXPERIMENT_NAME=td-root-only \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" \
    > "$RUN_ROOT/logs/root_only_train.log" 2>&1
fi

"$PROJECT_ROOT/scripts/manage_two_node_ray_a6000.sh" stop \
  2>&1 | tee "$RUN_ROOT/logs/ray_stop.log"
trap - EXIT

actor_dir="$RUN_ROOT/proposed/checkpoints/global_step_$TOTAL_STEPS/actor"
rsync -aH -e "$RSYNC_SHELL" \
  --include='model_world_size_2_rank_1.pt' --exclude='*' \
  "$PEER_HOST:$actor_dir/" "$actor_dir/"
test -s "$actor_dir/model_world_size_2_rank_0.pt"
test -s "$actor_dir/model_world_size_2_rank_1.pt"

if [[ "$AUTO_EVAL" != "1" ]]; then
  echo "Training complete; AUTO_EVAL=0"
  exit 0
fi

merged="$RUN_ROOT/merged_model"
"$CONDA_PYTHON" -m verl.model_merger merge --backend fsdp \
  --local_dir "$actor_dir" --target_dir "$merged" \
  2>&1 | tee "$RUN_ROOT/logs/merge.log"
CUDA_VISIBLE_DEVICES=0 N_GPUS=1 SKIP_PREPARE=1 \
  PROJECT_ROOT_OVERRIDE="$PROJECT_ROOT" SCAF_REPO="$SCAF_REPO" \
  MODEL_PATH="$merged" METHOD_LABEL="$METHOD_LABEL" \
  CHECKPOINT_RULE="two-node data parallel, fixed global_step_$TOTAL_STEPS" \
  PAPER_REFERENCE=vanilla OUT="$RUN_ROOT/eval_method" PYTHON_BIN="$CONDA_PYTHON" \
  bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
  2>&1 | tee "$RUN_ROOT/logs/eval.log"
"$CONDA_PYTHON" "$PROJECT_ROOT/scripts/summarize_new_method_vs_existing.py" \
  --baseline-run-root "$BASELINE_RUN_ROOT" --eval-root "$RUN_ROOT/eval_method" \
  --output-dir "$RUN_ROOT" --method-label "$METHOD_LABEL" \
  --training-steps "$TOTAL_STEPS" 2>&1 | tee "$RUN_ROOT/logs/comparison.log"
echo "Comparison: $RUN_ROOT/comparison.md"
