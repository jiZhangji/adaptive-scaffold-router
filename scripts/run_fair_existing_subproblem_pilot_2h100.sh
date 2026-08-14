#!/usr/bin/env bash
set -euo pipefail

# Compute-matched feasibility experiment for the already calibrated n=256 data.
# GPU 0 trains Vanilla; GPU 1 trains the 1:1 root/subproblem curriculum.  Both
# arms start from the same weights and receive exactly the same number of
# prompts, rollouts, optimizer steps, and maximum response tokens.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
PILOT_ROOT="${PILOT_ROOT:-$PROJECT_ROOT/outputs/existing_subproblem_pilot_n256}"
ROOT_DATA="${ROOT_DATA:-$PILOT_ROOT/root_train.parquet}"
MIXED_DATA="${MIXED_DATA:-$PILOT_ROOT/mixed_train.parquet}"

TRAIN_STEPS="${TRAIN_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUTS="${ROLLOUTS:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
SAVE_FREQ="${SAVE_FREQ:-25}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
REQUIRE_FREE_GPUS="${REQUIRE_FREE_GPUS:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/fair_subproblem_pilot_$TIMESTAMP}"
RAY_TMP_BASE="${RAY_TMP_BASE:-/tmp/fp${UID}_$$}"
mkdir -p "$RUN_ROOT/logs"
mkdir -p "$RAY_TMP_BASE"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_fair_subproblem_pilot.txt"
printf '%s\n' "$$" > "$RUN_ROOT/pid.txt"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1 PYTHONHASHSEED=42

for path in \
  "$MODEL_PATH/config.json" \
  "$ROOT_DATA" \
  "$MIXED_DATA" \
  "$SCAF_REPO/data/AIME24/math-verify/system-p1/test.parquet"; do
  [[ -s "$path" ]] || { echo "Missing required offline asset: $path" >&2; exit 2; }
done
compgen -G "$MODEL_PATH/*.safetensors" >/dev/null || {
  echo "No model weights found under $MODEL_PATH" >&2; exit 2;
}

if [[ "$REQUIRE_FREE_GPUS" == "1" ]]; then
  for gpu in 0 1; do
    active="$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    if [[ -n "$active" ]]; then
      echo "GPU $gpu already has compute process(es): $active" >&2
      echo "Wait for the existing job, or set REQUIRE_FREE_GPUS=0 only if sharing is intentional." >&2
      exit 3
    fi
  done
fi

conda run --no-capture-output -n "$ENV_NAME" python - \
  "$ROOT_DATA" "$MIXED_DATA" "$PILOT_ROOT/calibration/summary.json" \
  "$RUN_ROOT/data_validation.json" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root_path, mixed_path, calibration_path, output_path = map(Path, sys.argv[1:])
root = pd.read_parquet(root_path)
mixed = pd.read_parquet(mixed_path)
if len(root) == 0 or len(mixed) != 2 * len(root):
    raise SystemExit(
        f"Expected non-empty 1:1 matched data; root={len(root)}, mixed={len(mixed)}"
    )

is_subproblem = []
for value in mixed["extra_info"]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    is_subproblem.append(bool((value or {}).get("is_subproblem", False)))
subproblem_count = sum(is_subproblem)
if subproblem_count != len(root):
    raise SystemExit(
        f"Expected {len(root)} subproblems in mixed data, found {subproblem_count}"
    )

calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
payload = {
    "root_rows": len(root),
    "mixed_rows": len(mixed),
    "mixed_root_rows": len(mixed) - subproblem_count,
    "mixed_subproblem_rows": subproblem_count,
    "calibration_training_roots": calibration.get("training_roots"),
    "calibration_training_candidates": calibration.get("training_candidates"),
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

cat > "$RUN_ROOT/protocol.txt" <<EOF
protocol=compute_matched_subproblem_pilot_v1
initial_model=$MODEL_PATH
training_steps_per_arm=$TRAIN_STEPS
prompts_per_step=$TRAIN_BATCH_SIZE
rollouts_per_prompt=$ROLLOUTS
max_prompt_tokens=$MAX_PROMPT_LENGTH
max_response_tokens=$MAX_RESPONSE_LENGTH
generated_trajectories_per_arm=$((TRAIN_STEPS * TRAIN_BATCH_SIZE * ROLLOUTS))
checkpoint_rule=final fixed-step checkpoint
vanilla_data=$ROOT_DATA
subproblem_data=$MIXED_DATA
important_limitation=q calibrated only; no matched-random causal relevance filter
EOF

run_arm() {
  local gpu="$1" label="$2" train_data="$3" output_dir="$4"
  local ray_tag="${label:0:1}"
  local ray_tmp="$RAY_TMP_BASE/$ray_tag"
  mkdir -p "$ray_tmp"
  CUDA_VISIBLE_DEVICES="$gpu" \
  RAY_TMPDIR="$ray_tmp" RAY_ADDRESS="" \
  TRAIN_STEPS="$TRAIN_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  ROLLOUTS="$ROLLOUTS" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" SAVE_FREQ="$SAVE_FREQ" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  EXPERIMENT_NAME="fair-pilot-$label" N_GPUS=1 TP_SIZE=1 \
  SCAF_REPO="$SCAF_REPO" \
  MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$train_data" OUTPUT_DIR="$output_dir" \
  conda run --no-capture-output -n "$ENV_NAME" \
    bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh" \
    >"$RUN_ROOT/logs/${label}_train.log" 2>&1
}

echo "Starting compute-matched training in parallel."
echo "GPU 0: Vanilla; GPU 1: Subproblem. Run root: $RUN_ROOT"
run_arm 0 vanilla "$ROOT_DATA" "$RUN_ROOT/vanilla" &
vanilla_pid=$!
run_arm 1 subproblem "$MIXED_DATA" "$RUN_ROOT/subproblem" &
subproblem_pid=$!
printf '%s\n' "$vanilla_pid" > "$RUN_ROOT/vanilla.pid"
printf '%s\n' "$subproblem_pid" > "$RUN_ROOT/subproblem.pid"

cleanup() {
  kill "$vanilla_pid" "$subproblem_pid" 2>/dev/null || true
}
trap cleanup INT TERM

status=0
wait "$vanilla_pid" || { echo "Vanilla training failed." >&2; status=1; }
wait "$subproblem_pid" || { echo "Subproblem training failed." >&2; status=1; }
trap - INT TERM
[[ "$status" -eq 0 ]] || {
  echo "Inspect $RUN_ROOT/logs/*_train.log" >&2
  exit "$status"
}

merge_final_checkpoint() {
  local label="$1" train_dir="$2" target_dir="$3"
  local checkpoint_name checkpoint_dir actor_dir step
  checkpoint_name="$(find "$train_dir/checkpoints" -maxdepth 1 -type d \
    -name 'global_step_*' -printf '%f\n' | sort -V | tail -n 1)"
  [[ -n "$checkpoint_name" ]] || {
    echo "No checkpoint found for $label under $train_dir/checkpoints" >&2; return 1;
  }
  step="${checkpoint_name##*_}"
  [[ "$step" == "$TRAIN_STEPS" ]] || {
    echo "$label stopped at step $step; expected fixed final step $TRAIN_STEPS" >&2; return 1;
  }
  checkpoint_dir="$train_dir/checkpoints/$checkpoint_name"
  actor_dir="$checkpoint_dir/actor"
  [[ -d "$actor_dir" ]] || {
    echo "Missing FSDP actor directory: $actor_dir" >&2; return 1;
  }
  rm -rf "$target_dir"
  conda run --no-capture-output -n "$ENV_NAME" python -m verl.model_merger merge \
    --backend fsdp --local_dir "$actor_dir" --target_dir "$target_dir" \
    2>&1 | tee "$RUN_ROOT/logs/${label}_merge.log"
  [[ -s "$target_dir/config.json" ]] || {
    echo "Merged model is missing config.json: $target_dir" >&2; return 1;
  }
  compgen -G "$target_dir/*.safetensors" >/dev/null || {
    echo "Merged model has no safetensors weights: $target_dir" >&2; return 1;
  }
  printf '%s\n' "$checkpoint_dir" > "$RUN_ROOT/${label}_final_checkpoint.txt"
}

echo "Merging both fixed step-$TRAIN_STEPS checkpoints."
merge_final_checkpoint vanilla "$RUN_ROOT/vanilla" "$RUN_ROOT/vanilla_merged"
merge_final_checkpoint subproblem "$RUN_ROOT/subproblem" "$RUN_ROOT/subproblem_merged"

run_eval() {
  local label="$1" model="$2" reference="$3" out="$4"
  CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 SKIP_PREPARE=1 \
  PROJECT_ROOT_OVERRIDE="$PROJECT_ROOT" SCAF_REPO="$SCAF_REPO" \
  MODEL_PATH="$model" METHOD_LABEL="$label" PAPER_REFERENCE="$reference" OUT="$out" \
  conda run --no-capture-output -n "$ENV_NAME" \
    bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh" \
    2>&1 | tee "$RUN_ROOT/logs/${label}_eval_all.log"
}

echo "Evaluating Vanilla with the unified seven-benchmark protocol."
run_eval vanilla "$RUN_ROOT/vanilla_merged" vanilla "$RUN_ROOT/eval_vanilla"
echo "Evaluating Subproblem with the identical protocol."
run_eval subproblem "$RUN_ROOT/subproblem_merged" scaf "$RUN_ROOT/eval_subproblem"

conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/scripts/summarize_fair_subproblem_pilot.py" \
  --vanilla-eval "$RUN_ROOT/eval_vanilla" \
  --subproblem-eval "$RUN_ROOT/eval_subproblem" \
  --output-dir "$RUN_ROOT" \
  --training-steps "$TRAIN_STEPS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --rollouts "$ROLLOUTS" \
  2>&1 | tee "$RUN_ROOT/logs/final_summary.log"

echo "Fair pilot complete: $RUN_ROOT/fair_pilot_results.md"
