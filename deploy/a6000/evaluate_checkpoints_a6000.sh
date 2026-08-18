#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
ENV_PREFIX="${ENV_PREFIX:-$INSTALL_ROOT/envs/scaf-grpo}"
SCAF_REPO="${SCAF_REPO:-$INSTALL_ROOT/Scaf-GRPO}"
RUN_ID="${RUN_ID:-student_aware_root_aligned_20260816_183357}"
RUN_ROOT="$PROJECT_ROOT/outputs/$RUN_ID"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/a6000_student_aware_checkpoint_eval}"
GPU="${GPU:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.45}"
EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-8192}"
PYTHON_BIN="$ENV_PREFIX/bin/python"

mkdir -p "$OUT_ROOT" "$RUN_ROOT/merged_checkpoints"

for step in 10 35 50; do
  actor="$RUN_ROOT/proposed/checkpoints/global_step_$step/actor"
  merged="$RUN_ROOT/merged_checkpoints/global_step_$step"
  out="$OUT_ROOT/step_$step"

  if [[ ! -d "$actor" ]]; then
    echo "global_step_$step is unavailable; skipping." >&2
    continue
  fi

  if [[ "$step" == "50" && -s "$RUN_ROOT/student_aware_merged/config.json" ]]; then
    merged="$RUN_ROOT/student_aware_merged"
  elif [[ ! -s "$merged/config.json" ]]; then
    echo "===== MERGE global_step_$step ====="
    "$PYTHON_BIN" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$actor" \
      --target_dir "$merged"
  fi

  echo
  echo "===== EVALUATE global_step_$step ====="
  CUDA_VISIBLE_DEVICES="$GPU" \
  N_GPUS=1 \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  EVAL_GPU_MEMORY_UTILIZATION="$EVAL_GPU_MEMORY_UTILIZATION" \
  EVAL_MAX_NUM_BATCHED_TOKENS="$EVAL_MAX_NUM_BATCHED_TOKENS" \
  PYTHON_BIN="$PYTHON_BIN" \
  SKIP_PREPARE=1 \
  SCAF_REPO="$SCAF_REPO" \
  MODEL_PATH="$merged" \
  METHOD_LABEL="student_aware_step_$step" \
  PAPER_REFERENCE=vanilla \
  CHECKPOINT_RULE="Student-Aware fixed global_step_$step" \
  OUT="$out" \
    bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh"
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_checkpoint_eval.py" \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/checkpoint_comparison.md"

echo
echo "Checkpoint comparison: $OUT_ROOT/checkpoint_comparison.md"
