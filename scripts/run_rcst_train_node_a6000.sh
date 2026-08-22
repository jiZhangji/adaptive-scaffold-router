#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/powerleader/project/adaptive-scaffold-router}"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
SELECTED_CANDIDATES="${SELECTED_CANDIDATES:?Set SELECTED_CANDIDATES}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
METHOD_LABEL="${METHOD_LABEL:?Set METHOD_LABEL}"
EXIT_FILE="${EXIT_FILE:?Set EXIT_FILE}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-$PROJECT_ROOT/outputs/complete_four_way_ordered_20260815_155120}"

mkdir -p "$RUN_ROOT/logs"
printf '%s\n' "$METHOD_LABEL" > "$RUN_ROOT/method_label.txt"
trap 'code=$?; printf "%s\n" "$code" > "$EXIT_FILE"' EXIT

SELECTED_CANDIDATES="$SELECTED_CANDIDATES" RUN_ROOT="$RUN_ROOT" \
BASELINE_RUN_ROOT="$BASELINE_RUN_ROOT" RUN_SUBPROBLEM_PROBE=1 \
LEARNABILITY_TRANSFER_ONLY=1 POSITIVE_TRANSFER_ONLY=1 \
AUTO_EVAL=1 EVAL_GPUS=0 EVAL_N_GPUS=1 \
  bash "$PROJECT_ROOT/scripts/run_student_aware_root_aligned_only_2h100.sh"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/summarize_new_method_vs_existing.py" \
  --baseline-run-root "$BASELINE_RUN_ROOT" \
  --eval-root "$RUN_ROOT/eval_student_aware" --output-dir "$RUN_ROOT" \
  --method-label "$METHOD_LABEL" --training-steps 50
