#!/usr/bin/env bash
set -euo pipefail

# Watch a running fair pilot.  If the main launcher is still alive, it owns
# downstream merge/evaluation and this watcher only waits.  If training has
# finished but the launcher disappeared before downstream work started, this
# script takes over exactly once using an atomic lock directory.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
RUN_ROOT="${RUN_ROOT:-$(cat "$PROJECT_ROOT/outputs/latest_fair_subproblem_pilot.txt")}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

echo "Watching fair pilot: $RUN_ROOT"
echo "Downstream work will use the final fixed step: $TRAIN_STEPS"

checkpoint_ready() {
  [[ -d "$RUN_ROOT/vanilla/checkpoints/global_step_$TRAIN_STEPS/actor" &&
     -d "$RUN_ROOT/subproblem/checkpoints/global_step_$TRAIN_STEPS/actor" ]]
}

launcher_alive() {
  local pid
  pid="$(cat "$RUN_ROOT/pid.txt" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

while true; do
  if [[ -f "$RUN_ROOT/fair_pilot_results.md" ]]; then
    echo "Downstream results already exist: $RUN_ROOT/fair_pilot_results.md"
    exit 0
  fi

  if [[ -f "$RUN_ROOT/logs/vanilla_merge.log" ||
        -f "$RUN_ROOT/logs/subproblem_merge.log" ||
        -f "$RUN_ROOT/logs/vanilla_eval_all.log" ]]; then
    echo "The main launcher has already started downstream work."
    echo "Monitor $RUN_ROOT/logs/*_merge.log and $RUN_ROOT/logs/*_eval_all.log"
    exit 0
  fi

  if ! checkpoint_ready; then
    vanilla_step="$(find "$RUN_ROOT/vanilla/rollout" -maxdepth 1 -name '*.jsonl' \
      -printf '%f\n' 2>/dev/null | sed 's/\.jsonl$//' | sort -n | tail -n 1)"
    subproblem_step="$(find "$RUN_ROOT/subproblem/rollout" -maxdepth 1 -name '*.jsonl' \
      -printf '%f\n' 2>/dev/null | sed 's/\.jsonl$//' | sort -n | tail -n 1)"
    echo "[$(date '+%F %T')] waiting: vanilla=${vanilla_step:-0}/$TRAIN_STEPS subproblem=${subproblem_step:-0}/$TRAIN_STEPS"
    sleep "$POLL_SECONDS"
    continue
  fi

  if launcher_alive; then
    echo "[$(date '+%F %T')] both checkpoints ready; main launcher still owns downstream work"
    sleep "$POLL_SECONDS"
    continue
  fi

  if ! mkdir "$RUN_ROOT/downstream.lock" 2>/dev/null; then
    echo "Another downstream process owns $RUN_ROOT/downstream.lock"
    exit 0
  fi
  trap 'rmdir "$RUN_ROOT/downstream.lock" 2>/dev/null || true' EXIT
  break
done

merge_checkpoint() {
  local label="$1"
  local actor="$RUN_ROOT/$label/checkpoints/global_step_$TRAIN_STEPS/actor"
  local target="$RUN_ROOT/${label}_merged"
  rm -rf "$target"
  conda run --no-capture-output -n "$ENV_NAME" \
    python -m verl.model_merger merge \
    --backend fsdp --local_dir "$actor" --target_dir "$target" \
    2>&1 | tee "$RUN_ROOT/logs/${label}_merge.log"
  [[ -s "$target/config.json" ]] || { echo "Missing merged config: $target" >&2; exit 1; }
  compgen -G "$target/*.safetensors" >/dev/null || {
    echo "Missing merged weights: $target" >&2; exit 1;
  }
}

merge_checkpoint vanilla
merge_checkpoint subproblem

run_eval() {
  local label="$1" reference="$2" model="$3" out="$4"
  CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 SKIP_PREPARE=1 \
  SCAF_REPO="$SCAF_REPO" MODEL_PATH="$model" \
  METHOD_LABEL="$label" PAPER_REFERENCE="$reference" OUT="$out" \
  conda run --no-capture-output -n "$ENV_NAME" \
    bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
    2>&1 | tee "$RUN_ROOT/logs/${label}_eval_all.log"
}

run_eval vanilla vanilla "$RUN_ROOT/vanilla_merged" "$RUN_ROOT/eval_vanilla"
run_eval subproblem scaf "$RUN_ROOT/subproblem_merged" "$RUN_ROOT/eval_subproblem"

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/scripts/summarize_fair_subproblem_pilot.py" \
  --vanilla-eval "$RUN_ROOT/eval_vanilla" \
  --subproblem-eval "$RUN_ROOT/eval_subproblem" \
  --output-dir "$RUN_ROOT" \
  --training-steps "$TRAIN_STEPS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --rollouts "$ROLLOUTS" \
  2>&1 | tee "$RUN_ROOT/logs/final_summary.log"

echo "Downstream work complete: $RUN_ROOT/fair_pilot_results.md"
