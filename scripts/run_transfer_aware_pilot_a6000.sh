#!/usr/bin/env bash
set -Eeuo pipefail

# Transfer-Aware pilot with the same 212-root training pool, 50 training steps,
# and seven-dataset evaluation protocol as the existing Student-Aware run.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
SCAF_REPO="${SCAF_REPO:-$INSTALL_ROOT/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
PREP_ROOT="${PREP_ROOT:-$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100}"
ANCHORS="${ANCHORS:-$PREP_ROOT/calibration/training_candidates.jsonl}"
ORIGINAL_CANDIDATES="${ORIGINAL_CANDIDATES:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-$PROJECT_ROOT/outputs/complete_four_way_ordered_20260815_155120}"
STUDENT_AWARE_RUN="${STUDENT_AWARE_RUN:-$PROJECT_ROOT/outputs/student_aware_root_aligned_20260816_183357}"
LEARNABILITY_SOURCE="${LEARNABILITY_SOURCE:-$STUDENT_AWARE_RUN/learnability/subproblem_learnability.jsonl}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/transfer_aware_pilot_$(date +%Y%m%d_%H%M%S)}"
PROBE_ROOTS="${PROBE_ROOTS:-16}"
ROOT_SAMPLES="${ROOT_SAMPLES:-4}"
PROBE_STEPS="${PROBE_STEPS:-2}"
TRAIN_GPU="${TRAIN_GPU:-0}"
AUTO_EVAL="${AUTO_EVAL:-1}"

[[ -s "$CONDA_SH" ]] || { echo "Conda initialization script is missing: $CONDA_SH" >&2; exit 2; }
source "$CONDA_SH"

mkdir -p "$RUN_ROOT"/{selection,logs}
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_transfer_aware_pilot.txt"
printf '%s\n' "same_root_transfer_probe_hybrid_v1" > "$RUN_ROOT/protocol.txt"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

for asset in "$MODEL_PATH/config.json" "$SOURCE_DATA" "$ANCHORS" "$ORIGINAL_CANDIDATES"; do
  [[ -s "$asset" ]] || { echo "Missing asset: $asset" >&2; exit 3; }
done

CANDIDATES="$RUN_ROOT/selection/candidate_sets.jsonl"
TRANSFER_RESULTS="$RUN_ROOT/selection/transfer_results.jsonl"
SELECTED="$RUN_ROOT/selection/transfer_selected_candidates.jsonl"

echo "Stage 0a: recover matched K/P/C sets from the original DeepSeek candidates."
conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/filter_original_transfer_candidates.py" \
  --anchors "$ANCHORS" --original-candidates "$ORIGINAL_CANDIDATES" \
  --output "$CANDIDATES" --min-complete-roots "$PROBE_ROOTS" \
  2>&1 | tee "$RUN_ROOT/logs/candidate_filter.log"

echo "Stage 0b: temporary LoRA updates and same-root no-hint probes."
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/probe_subproblem_transfer.py" \
  --candidates "$CANDIDATES" --model "$MODEL_PATH" --output "$TRANSFER_RESULTS" \
  --root-limit "$PROBE_ROOTS" --root-samples "$ROOT_SAMPLES" \
  --probe-steps "$PROBE_STEPS" --device cuda:0 \
  2>&1 | tee "$RUN_ROOT/logs/transfer_probe.log"

echo "Stage 0c: select max-transfer candidates and retain anchors elsewhere."
conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/select_transfer_aware_candidates.py" \
  --anchors "$ANCHORS" --candidates "$CANDIDATES" \
  --transfer-results "$TRANSFER_RESULTS" --output "$SELECTED" \
  2>&1 | tee "$RUN_ROOT/logs/selection.log"

echo "Stages 1-3: run the matched Student-Aware training protocol with new selection."
if [[ -s "$LEARNABILITY_SOURCE" ]]; then
  mkdir -p "$RUN_ROOT/learnability"
  cp "$LEARNABILITY_SOURCE" "$RUN_ROOT/learnability/subproblem_learnability.jsonl"
fi
SELECTED_CANDIDATES="$SELECTED" RUN_ROOT="$RUN_ROOT" \
BASELINE_RUN_ROOT="$BASELINE_RUN_ROOT" TRAIN_GPU="$TRAIN_GPU" \
RUN_SUBPROBLEM_PROBE=0 AUTO_EVAL="$AUTO_EVAL" EVAL_GPUS="$TRAIN_GPU" EVAL_N_GPUS=1 \
SCAF_REPO="$SCAF_REPO" ENV_NAME="$ENV_NAME" MODEL_PATH="$MODEL_PATH" \
SOURCE_DATA="$SOURCE_DATA" PREP_ROOT="$PREP_ROOT" \
  bash "$PROJECT_ROOT/scripts/run_student_aware_root_aligned_only_2h100.sh"

if [[ -f "$RUN_ROOT/student_aware_comparison.md" ]]; then
  cp "$RUN_ROOT/student_aware_comparison.md" "$RUN_ROOT/transfer_aware_comparison.md"
fi

echo "Transfer-Aware pilot complete: $RUN_ROOT"
