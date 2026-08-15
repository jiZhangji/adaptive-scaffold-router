#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
RUN_ROOT="${RUN_ROOT:-$(cat "$PROJECT_ROOT/outputs/latest_complete_four_way_pilot.txt")}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"

merge_arm() {
  local method="$1" actor target checkpoint
  checkpoint="$RUN_ROOT/$method/checkpoints/global_step_$TRAIN_STEPS"
  actor="$checkpoint/actor"
  target="$RUN_ROOT/${method}_merged"
  [[ -d "$actor" ]] || { echo "Missing final checkpoint: $actor" >&2; return 1; }
  if [[ ! -s "$target/config.json" ]]; then
    mkdir -p "$target"
    conda run --no-capture-output -n "$ENV_NAME" python -m verl.model_merger merge \
      --backend fsdp --local_dir "$actor" --target_dir "$target" \
      2>&1 | tee "$RUN_ROOT/logs/${method}_merge.log"
  fi
}

evaluate_arm() {
  local method="$1" reference="$2"
  CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 SKIP_PREPARE=1 \
  PROJECT_ROOT_OVERRIDE="$PROJECT_ROOT" SCAF_REPO="$SCAF_REPO" \
  MODEL_PATH="$RUN_ROOT/${method}_merged" METHOD_LABEL="$method" \
  CHECKPOINT_RULE="fixed global_step_$TRAIN_STEPS checkpoint, merged to Hugging Face format" \
  PAPER_REFERENCE="$reference" OUT="$RUN_ROOT/eval_$method" \
  conda run --no-capture-output -n "$ENV_NAME" \
    bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
    2>&1 | tee "$RUN_ROOT/logs/${method}_eval.log"
}

for method in vanilla scaf subproblem fade_is; do
  merge_arm "$method"
done

evaluate_arm vanilla vanilla
evaluate_arm scaf scaf
evaluate_arm subproblem vanilla
evaluate_arm fade_is vanilla

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/scripts/summarize_complete_four_way.py" \
  --run-root "$RUN_ROOT" --training-steps "$TRAIN_STEPS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" --rollouts "$ROLLOUTS" \
  2>&1 | tee "$RUN_ROOT/logs/four_way_summary.log"

echo "Complete comparison: $RUN_ROOT/four_way_results.md"
