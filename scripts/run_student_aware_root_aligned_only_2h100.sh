#!/usr/bin/env bash
set -euo pipefail

# Train only the proposed student-aware arm. Existing Vanilla/Scaf/Subproblem
# checkpoints and evaluations are reused for the final comparison.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
PREP_ROOT="${PREP_ROOT:-$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100}"
SELECTED_CANDIDATES="${SELECTED_CANDIDATES:-$PREP_ROOT/calibration/training_candidates.jsonl}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-$(cat "$PROJECT_ROOT/outputs/latest_complete_four_way_pilot.txt")}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/student_aware_root_aligned_$(date +%Y%m%d_%H%M%S)}"
TRAIN_GPU="${TRAIN_GPU:-0}"
PRECONDITION_STEPS="${PRECONDITION_STEPS:-10}"
ROOT_ALIGNED_END_STEP="${ROOT_ALIGNED_END_STEP:-35}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
SAVE_FREQ="${SAVE_FREQ:-5}"
SCAFFOLD_READY_THRESHOLD="${SCAFFOLD_READY_THRESHOLD:-0.50}"
CONTRAST_MIN="${CONTRAST_MIN:-0.0}"
RUN_SUBPROBLEM_PROBE="${RUN_SUBPROBLEM_PROBE:-1}"
POSITIVE_TRANSFER_ONLY="${POSITIVE_TRANSFER_ONLY:-0}"
MIN_TRANSFER_GAIN="${MIN_TRANSFER_GAIN:-}"
MIN_POST_UPDATE_PROBABILITY="${MIN_POST_UPDATE_PROBABILITY:-}"
AUTO_EVAL="${AUTO_EVAL:-1}"
EVAL_GPUS="${EVAL_GPUS:-0,1}"
EVAL_N_GPUS="${EVAL_N_GPUS:-2}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
ROOT_ONLY_GPU_MEMORY_UTILIZATION="${ROOT_ONLY_GPU_MEMORY_UTILIZATION:-0.30}"
ROOT_ONLY_ACTOR_MICRO_BATCH_SIZE="${ROOT_ONLY_ACTOR_MICRO_BATCH_SIZE:-2}"
ROOT_ONLY_LOG_PROB_MICRO_BATCH_SIZE="${ROOT_ONLY_LOG_PROB_MICRO_BATCH_SIZE:-2}"
ROOT_ONLY_REF_LOG_PROB_MICRO_BATCH_SIZE="${ROOT_ONLY_REF_LOG_PROB_MICRO_BATCH_SIZE:-2}"
RAY_TMP_BASE="${RAY_TMP_BASE:-/tmp/sar${UID}_$$}"

[[ -s "$CONDA_SH" ]] || { echo "Conda initialization script is missing: $CONDA_SH" >&2; exit 2; }
source "$CONDA_SH"

reset_cross_stage_dataloader_state() {
  local checkpoint_dir="$1"
  local stage_tag="$2"
  local state_file backup_file
  for state_file in data.pt train_dataset_state.pt current_epoch.txt; do
    if [[ -f "$checkpoint_dir/$state_file" ]]; then
      backup_file="$checkpoint_dir/$state_file.$stage_tag.bak"
      if [[ ! -e "$backup_file" ]]; then
        mv "$checkpoint_dir/$state_file" "$backup_file"
      fi
    fi
  done
}

if (( PRECONDITION_STEPS < 0 || ROOT_ALIGNED_END_STEP <= PRECONDITION_STEPS || TOTAL_STEPS <= ROOT_ALIGNED_END_STEP )); then
  echo "Require 0 <= PRECONDITION_STEPS < ROOT_ALIGNED_END_STEP < TOTAL_STEPS" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"/{data,logs,learnability,proposed} "$RAY_TMP_BASE"/{p,r,c}
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_student_aware_root_aligned.txt"
printf '%s\n' "$BASELINE_RUN_ROOT" > "$RUN_ROOT/baseline_run_root.txt"
printf '%s\n' "student_aware_precondition_root_aligned_v1" > "$RUN_ROOT/protocol.txt"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF

for asset in "$MODEL_PATH/config.json" "$SOURCE_DATA" "$SELECTED_CANDIDATES"; do
  [[ -s "$asset" ]] || { echo "Missing asset: $asset" >&2; exit 3; }
done

LEARNABILITY_ARGS=()
if [[ "$RUN_SUBPROBLEM_PROBE" == "1" ]]; then
  echo "Probing selected subproblem learnability; baseline training is not repeated."
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" conda run --no-capture-output -n "$ENV_NAME" python \
    "$PROJECT_ROOT/probe_selected_subproblem_learnability.py" \
    --candidates "$SELECTED_CANDIDATES" --model "$MODEL_PATH" \
    --output-dir "$RUN_ROOT/learnability" --samples "$ROLLOUTS" \
    --group-size "$ROLLOUTS" --batch-size 16 --job-chunk-size 128 \
    --device cuda:0 --dtype bfloat16 --max-input-tokens 2048 \
    --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 --stop-after-boxed \
    2>&1 | tee "$RUN_ROOT/logs/subproblem_probe.log"
  LEARNABILITY_ARGS=(--learnability "$RUN_ROOT/learnability/subproblem_learnability.jsonl")
elif [[ -s "$RUN_ROOT/learnability/subproblem_learnability.jsonl" ]]; then
  LEARNABILITY_ARGS=(--learnability "$RUN_ROOT/learnability/subproblem_learnability.jsonl")
fi

BUILDER_ARGS=()
if [[ "$POSITIVE_TRANSFER_ONLY" == "1" ]]; then
  BUILDER_ARGS+=(--positive-transfer-only)
fi
if [[ -n "$MIN_TRANSFER_GAIN" ]]; then
  BUILDER_ARGS+=(--min-transfer-gain "$MIN_TRANSFER_GAIN")
fi
if [[ -n "$MIN_POST_UPDATE_PROBABILITY" ]]; then
  BUILDER_ARGS+=(--min-post-update-probability "$MIN_POST_UPDATE_PROBABILITY")
fi

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/build_student_aware_preconditioning_experiment.py" \
  --source-data "$SOURCE_DATA" --candidates "$SELECTED_CANDIDATES" \
  "${LEARNABILITY_ARGS[@]}" --output-dir "$RUN_ROOT/data" \
  --scaffold-ready-threshold "$SCAFFOLD_READY_THRESHOLD" \
  --contrast-min "$CONTRAST_MIN" --group-size "$ROLLOUTS" \
  "${BUILDER_ARGS[@]}" \
  2>&1 | tee "$RUN_ROOT/logs/build_data.log"

PRECONDITION_ROWS="$(conda run --no-capture-output -n "$ENV_NAME" python - "$RUN_ROOT/data/summary.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["precondition_rows"]))
PY
)"

if (( PRECONDITION_ROWS > 0 && PRECONDITION_STEPS > 0 )); then
  conda run --no-capture-output -n "$ENV_NAME" python \
    "$PROJECT_ROOT/normalize_subproblem_reward_data.py" \
    --input "$RUN_ROOT/data/precondition_train.parquet" \
    --output "$RUN_ROOT/data/precondition_train_compatible.parquet" \
    2>&1 | tee "$RUN_ROOT/logs/normalize_precondition.log"

  if [[ ! -d "$RUN_ROOT/proposed/checkpoints/global_step_$PRECONDITION_STEPS/actor" ]]; then
    echo "Stage 1/3: student-aware subproblem preconditioning to step $PRECONDITION_STEPS."
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RAY_TMPDIR="$RAY_TMP_BASE/p" \
    TRAIN_STEPS="$PRECONDITION_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    ROLLOUTS="$ROLLOUTS" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
    GPU_MEMORY_UTILIZATION=0.45 ACTOR_MICRO_BATCH_SIZE=2 \
    LOG_PROB_MICRO_BATCH_SIZE=2 REF_LOG_PROB_MICRO_BATCH_SIZE=2 \
    FREE_CACHE_ENGINE=true N_GPUS=1 TP_SIZE=1 RESUME_MODE=auto \
    SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" \
    TRAIN_DATA="$RUN_ROOT/data/precondition_train_compatible.parquet" \
    OUTPUT_DIR="$RUN_ROOT/proposed" EXPERIMENT_NAME=student-aware-precondition \
    conda run --no-capture-output -n "$ENV_NAME" \
      bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" \
      > "$RUN_ROOT/logs/precondition_train.log" 2>&1
  fi
  FADE_START="$PRECONDITION_STEPS"
else
  echo "No candidate is below the scaffold-usability gate; skipping independent preconditioning."
  FADE_START=0
fi

SCAF_REPO="$SCAF_REPO" conda run --no-capture-output -n "$ENV_NAME" \
  bash "$PROJECT_ROOT/scripts/install_complete_scaf_integration.sh" \
  2>&1 | tee "$RUN_ROOT/logs/install_integration.log"

if [[ ! -d "$RUN_ROOT/proposed/checkpoints/global_step_$ROOT_ALIGNED_END_STEP/actor" ]]; then
  # Stage 2 uses a different parquet dataset from Stage 1. Preserve the model
  # and optimizer checkpoint, but do not restore the old StatefulDataLoader
  # cursor into the new dataset.
  reset_cross_stage_dataloader_state \
    "$RUN_ROOT/proposed/checkpoints/global_step_$PRECONDITION_STEPS" "stage1"
  echo "Stage 2/3: root-aligned minimal-scaffold GRPO with fading and IS."
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RAY_TMPDIR="$RAY_TMP_BASE/r" \
  MODE=fade_is TRAIN_STEPS="$ROOT_ALIGNED_END_STEP" \
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" ROLLOUTS="$ROLLOUTS" \
  PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
  GPU_MEMORY_UTILIZATION=0.35 ACTOR_MICRO_BATCH_SIZE=2 \
  LOG_PROB_MICRO_BATCH_SIZE=2 REF_LOG_PROB_MICRO_BATCH_SIZE=2 \
  FREE_CACHE_ENGINE=true RESUME_MODE=auto \
  CURRICULUM_MANIFEST="$RUN_ROOT/data/curriculum.jsonl" \
  FADE_START="$FADE_START" FADE_END="$ROOT_ALIGNED_END_STEP" \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" \
  TRAIN_DATA="$RUN_ROOT/data/root_scaffold_train.parquet" \
  OUTPUT_DIR="$RUN_ROOT/proposed" \
  conda run --no-capture-output -n "$ENV_NAME" \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_scaffold_arm_1gpu.sh" \
    > "$RUN_ROOT/logs/root_aligned_train.log" 2>&1
fi

if [[ ! -d "$RUN_ROOT/proposed/checkpoints/global_step_$TOTAL_STEPS/actor" ]]; then
  # Stage 3 switches from scaffolded roots to the naked-root parquet.
  reset_cross_stage_dataloader_state \
    "$RUN_ROOT/proposed/checkpoints/global_step_$ROOT_ALIGNED_END_STEP" "stage2"
  echo "Stage 3/3: root-only consolidation to step $TOTAL_STEPS."
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RAY_TMPDIR="$RAY_TMP_BASE/c" \
  TRAIN_STEPS="$TOTAL_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  ROLLOUTS="$ROLLOUTS" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
  GPU_MEMORY_UTILIZATION="$ROOT_ONLY_GPU_MEMORY_UTILIZATION" \
  ACTOR_MICRO_BATCH_SIZE="$ROOT_ONLY_ACTOR_MICRO_BATCH_SIZE" \
  LOG_PROB_MICRO_BATCH_SIZE="$ROOT_ONLY_LOG_PROB_MICRO_BATCH_SIZE" \
  REF_LOG_PROB_MICRO_BATCH_SIZE="$ROOT_ONLY_REF_LOG_PROB_MICRO_BATCH_SIZE" \
  FREE_CACHE_ENGINE=true N_GPUS=1 TP_SIZE=1 RESUME_MODE=auto \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" \
  TRAIN_DATA="$RUN_ROOT/data/root_only_train.parquet" \
  OUTPUT_DIR="$RUN_ROOT/proposed" EXPERIMENT_NAME=student-aware-root-only \
  conda run --no-capture-output -n "$ENV_NAME" \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" \
    > "$RUN_ROOT/logs/root_only_train.log" 2>&1
fi

echo "Proposed training completed: $RUN_ROOT/proposed/checkpoints/global_step_$TOTAL_STEPS"
if [[ "$AUTO_EVAL" != "1" ]]; then
  echo "AUTO_EVAL=0; evaluation was skipped."
  exit 0
fi

MERGED="$RUN_ROOT/student_aware_merged"
if [[ ! -s "$MERGED/config.json" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$RUN_ROOT/proposed/checkpoints/global_step_$TOTAL_STEPS/actor" \
    --target_dir "$MERGED" \
    2>&1 | tee "$RUN_ROOT/logs/merge.log"
fi

CUDA_VISIBLE_DEVICES="$EVAL_GPUS" N_GPUS="$EVAL_N_GPUS" SKIP_PREPARE=1 \
PROJECT_ROOT_OVERRIDE="$PROJECT_ROOT" SCAF_REPO="$SCAF_REPO" \
MODEL_PATH="$MERGED" METHOD_LABEL=student_aware \
CHECKPOINT_RULE="student-aware preconditioning + root-aligned fading + root-only, fixed global_step_$TOTAL_STEPS" \
PAPER_REFERENCE=vanilla OUT="$RUN_ROOT/eval_student_aware" \
conda run --no-capture-output -n "$ENV_NAME" \
  bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
  2>&1 | tee "$RUN_ROOT/logs/eval.log"

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/scripts/summarize_student_aware_vs_existing.py" \
  --baseline-run-root "$BASELINE_RUN_ROOT" --proposed-run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT" --training-steps "$TOTAL_STEPS" \
  2>&1 | tee "$RUN_ROOT/logs/comparison.log"

echo "Comparison: $RUN_ROOT/student_aware_comparison.md"
