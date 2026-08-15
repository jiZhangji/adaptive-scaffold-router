#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/outputs/complete_subproblem_n256/data}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/complete_four_way_$(date +%Y%m%d_%H%M%S)}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
SAVE_FREQ="${SAVE_FREQ:-25}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
SCAFFOLD_GPU_MEMORY_UTILIZATION="${SCAFFOLD_GPU_MEMORY_UTILIZATION:-0.35}"
SCAFFOLD_MICRO_BATCH_SIZE="${SCAFFOLD_MICRO_BATCH_SIZE:-2}"
REQUIRE_FREE_GPUS="${REQUIRE_FREE_GPUS:-1}"
AUTO_EVAL="${AUTO_EVAL:-1}"
RAY_TMP_BASE="${RAY_TMP_BASE:-/tmp/c4${UID}_$$}"

ROOT_DATA="$DATA_ROOT/root_train.parquet"
MIXED_SOURCE_DATA="$DATA_ROOT/mixed_train.parquet"
MIXED_DATA="$RUN_ROOT/data/mixed_train_reward_compatible.parquet"
PROPOSED_DATA="$DATA_ROOT/proposed_root_train.parquet"
MANIFEST="$DATA_ROOT/curriculum.jsonl"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/data" "$RAY_TMP_BASE"/{v,s,p,f}
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_complete_four_way_pilot.txt"
printf '%s\n' "complete_four_way_v1" > "$RUN_ROOT/protocol.txt"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 HYDRA_FULL_ERROR=1 VLLM_USE_V1=0

for path in "$MODEL_PATH/config.json" "$ROOT_DATA" "$MIXED_SOURCE_DATA" "$PROPOSED_DATA" "$MANIFEST"; do
  [[ -s "$path" ]] || { echo "Missing asset: $path" >&2; exit 2; }
done

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/normalize_subproblem_reward_data.py" \
  --input "$MIXED_SOURCE_DATA" --output "$MIXED_DATA" \
  2>&1 | tee "$RUN_ROOT/logs/normalize_subproblem_data.log"

if [[ "$REQUIRE_FREE_GPUS" == "1" ]]; then
  for gpu in 0 1; do
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
    total="$(nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
    if [[ "$used" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] && (( used * 100 > total * 10 )); then
      echo "GPU $gpu has less than 90% free memory; refuse to start." >&2
      exit 3
    fi
  done
fi

SCAF_REPO="$SCAF_REPO" conda run --no-capture-output -n "$ENV_NAME" bash \
  "$PROJECT_ROOT/scripts/install_complete_scaf_integration.sh" \
  2>&1 | tee "$RUN_ROOT/logs/install.log"

run_vanilla() {
  CUDA_VISIBLE_DEVICES=0 RAY_TMPDIR="$RAY_TMP_BASE/v" \
  TRAIN_STEPS="$TRAIN_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  ROLLOUTS="$ROLLOUTS" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" N_GPUS=1 TP_SIZE=1 \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$ROOT_DATA" \
  OUTPUT_DIR="$RUN_ROOT/vanilla" EXPERIMENT_NAME=complete-vanilla \
  conda run --no-capture-output -n "$ENV_NAME" \
  bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh"
}

run_scaf() {
  CUDA_VISIBLE_DEVICES=1 RAY_TMPDIR="$RAY_TMP_BASE/s" \
  MODE=scaf TRAIN_STEPS="$TRAIN_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  ROLLOUTS="$ROLLOUTS" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
  GPU_MEMORY_UTILIZATION="$SCAFFOLD_GPU_MEMORY_UTILIZATION" \
  ACTOR_MICRO_BATCH_SIZE="$SCAFFOLD_MICRO_BATCH_SIZE" \
  LOG_PROB_MICRO_BATCH_SIZE="$SCAFFOLD_MICRO_BATCH_SIZE" \
  REF_LOG_PROB_MICRO_BATCH_SIZE="$SCAFFOLD_MICRO_BATCH_SIZE" \
  FREE_CACHE_ENGINE=true RESUME_MODE=auto \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$ROOT_DATA" \
  OUTPUT_DIR="$RUN_ROOT/scaf" \
  conda run --no-capture-output -n "$ENV_NAME" \
  bash "$PROJECT_ROOT/scripts/run_fixed_budget_scaffold_arm_1gpu.sh"
}

run_subproblem() {
  CUDA_VISIBLE_DEVICES=0 RAY_TMPDIR="$RAY_TMP_BASE/p" \
  TRAIN_STEPS="$TRAIN_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  ROLLOUTS="$ROLLOUTS" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" N_GPUS=1 TP_SIZE=1 \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$MIXED_DATA" \
  OUTPUT_DIR="$RUN_ROOT/subproblem" EXPERIMENT_NAME=complete-subproblem \
  conda run --no-capture-output -n "$ENV_NAME" \
  bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh"
}

run_fade_is() {
  CUDA_VISIBLE_DEVICES=1 RAY_TMPDIR="$RAY_TMP_BASE/f" \
  MODE=fade_is TRAIN_STEPS="$TRAIN_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  ROLLOUTS="$ROLLOUTS" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
  GPU_MEMORY_UTILIZATION="$SCAFFOLD_GPU_MEMORY_UTILIZATION" \
  ACTOR_MICRO_BATCH_SIZE="$SCAFFOLD_MICRO_BATCH_SIZE" \
  LOG_PROB_MICRO_BATCH_SIZE="$SCAFFOLD_MICRO_BATCH_SIZE" \
  REF_LOG_PROB_MICRO_BATCH_SIZE="$SCAFFOLD_MICRO_BATCH_SIZE" \
  FREE_CACHE_ENGINE=true RESUME_MODE=auto CURRICULUM_MANIFEST="$MANIFEST" \
  FADE_START=0 FADE_END="$TRAIN_STEPS" \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$PROPOSED_DATA" \
  OUTPUT_DIR="$RUN_ROOT/fade_is" \
  conda run --no-capture-output -n "$ENV_NAME" \
  bash "$PROJECT_ROOT/scripts/run_fixed_budget_scaffold_arm_1gpu.sh"
}

wait_for_wave() {
  local label="$1"
  shift
  local status=0 pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  if [[ "$status" -ne 0 ]]; then
    echo "$label failed; inspect the corresponding logs in $RUN_ROOT/logs." >&2
    return 1
  fi
  echo "$label completed."
}

echo "Wave 1/2: starting official Scaf and Subproblem-mix in parallel."
WAVE_PIDS=()
if [[ -d "$RUN_ROOT/subproblem/checkpoints/global_step_$TRAIN_STEPS/actor" ]]; then
  echo "Subproblem already completed at global_step_$TRAIN_STEPS; skipping."
else
  run_subproblem > "$RUN_ROOT/logs/subproblem_train.log" 2>&1 &
  WAVE_PIDS+=("$!")
fi
if [[ -d "$RUN_ROOT/scaf/checkpoints/global_step_$TRAIN_STEPS/actor" ]]; then
  echo "Scaf already completed at global_step_$TRAIN_STEPS; skipping."
else
  run_scaf > "$RUN_ROOT/logs/scaf_train.log" 2>&1 &
  WAVE_PIDS+=("$!")
fi
wait_for_wave "Wave 1 (Scaf + Subproblem)" "${WAVE_PIDS[@]}"

echo "Wave 2/2: starting Vanilla and fading-IS in parallel."
WAVE_PIDS=()
if [[ -d "$RUN_ROOT/vanilla/checkpoints/global_step_$TRAIN_STEPS/actor" ]]; then
  echo "Vanilla already completed at global_step_$TRAIN_STEPS; skipping."
else
  run_vanilla > "$RUN_ROOT/logs/vanilla_train.log" 2>&1 &
  WAVE_PIDS+=("$!")
fi
if [[ -d "$RUN_ROOT/fade_is/checkpoints/global_step_$TRAIN_STEPS/actor" ]]; then
  echo "Fade-IS already completed at global_step_$TRAIN_STEPS; skipping."
else
  run_fade_is > "$RUN_ROOT/logs/fade_is_train.log" 2>&1 &
  WAVE_PIDS+=("$!")
fi
wait_for_wave "Wave 2 (Vanilla + Fade-IS)" "${WAVE_PIDS[@]}"

echo "All four training arms completed: $RUN_ROOT"
if [[ "$AUTO_EVAL" == "1" ]]; then
  echo "Starting merged-checkpoint downstream evaluation."
  RUN_ROOT="$RUN_ROOT" TRAIN_STEPS="$TRAIN_STEPS" \
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" ROLLOUTS="$ROLLOUTS" \
  SCAF_REPO="$SCAF_REPO" ENV_NAME="$ENV_NAME" \
  bash "$PROJECT_ROOT/scripts/run_complete_four_way_downstream_2h100.sh" \
    2>&1 | tee "$RUN_ROOT/logs/downstream_all.log"
else
  echo "AUTO_EVAL=0; run scripts/run_complete_four_way_downstream_2h100.sh manually."
fi
