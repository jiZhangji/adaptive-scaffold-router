#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELECTED_CANDIDATES="${SELECTED_CANDIDATES:-$PROJECT_ROOT/outputs/transfer_selected_full_212.jsonl}"
LEARNABILITY_FILE="${LEARNABILITY_FILE:-$PROJECT_ROOT/outputs/transfer_aware_positive_gated_full_20260821_1920/learnability/subproblem_learnability.jsonl}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/two_node_distributed_smoke_$(date +%Y%m%d_%H%M%S)}"

args=(
  SELECTED_CANDIDATES="$SELECTED_CANDIDATES"
  METHOD_LABEL="Two-Node Distributed Smoke"
  RUN_ROOT="$RUN_ROOT"
  PRECONDITION_STEPS=1 ROOT_ALIGNED_END_STEP=2 TOTAL_STEPS=3
  TRAIN_BATCH_SIZE=32 ROLLOUTS=8 PPO_MINI_BATCH_SIZE=32
  SAVE_FREQ=1 AUTO_EVAL=0 GATE_MODE=positive
)
if [[ -s "$LEARNABILITY_FILE" ]]; then
  args+=(LEARNABILITY_FILE="$LEARNABILITY_FILE" RUN_LEARNABILITY=0)
else
  args+=(RUN_LEARNABILITY=1)
fi

env "${args[@]}" bash "$PROJECT_ROOT/scripts/run_two_node_transfer_training_a6000.sh"
echo "Two-node distributed smoke completed: $RUN_ROOT"
