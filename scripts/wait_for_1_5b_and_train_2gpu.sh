#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_MODEL="${POLICY_MODEL:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B-Instruct}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EXPECTED_WEIGHT_BYTES="${EXPECTED_WEIGHT_BYTES:-3087467144}"
LIMIT="${LIMIT:-64}"
SAMPLES="${SAMPLES:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable on this instance." >&2
  exit 2
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -lt 2 ]]; then
  echo "Two GPUs are required; detected $gpu_count." >&2
  exit 2
fi

if [[ ! -s "$DATA_FILE" ]]; then
  echo "Training data is missing: $DATA_FILE" >&2
  exit 2
fi

quick_model_ready() {
  local model="$1"
  [[ -s "$model/config.json" ]] || return 1
  [[ -s "$model/tokenizer_config.json" ]] || return 1
  [[ -s "$model/model.safetensors" ]] || return 1
  [[ "$(stat -c %s "$model/model.safetensors" 2>/dev/null || echo 0)" \
      -eq "$EXPECTED_WEIGHT_BYTES" ]] || return 1
  if find "$model" -name '*.incomplete' -print -quit | grep -q .; then
    return 1
  fi
}

describe_model() {
  local label="$1" model="$2" size=0 incomplete=0
  size="$(stat -c %s "$model/model.safetensors" 2>/dev/null || echo 0)"
  incomplete="$(find "$model" -name '*.incomplete' 2>/dev/null | wc -l)"
  if quick_model_ready "$model"; then
    echo "$label=ready(${size}B)"
  else
    echo "$label=waiting(weight=${size}B,incomplete=${incomplete})"
  fi
}

echo "Monitoring shared model directories before two-GPU training."
echo "Policy:  $POLICY_MODEL"
echo "Teacher: $TEACHER_MODEL"
while ! quick_model_ready "$POLICY_MODEL" || ! quick_model_ready "$TEACHER_MODEL"; do
  echo "[$(date '+%F %T')] $(describe_model policy "$POLICY_MODEL"); $(describe_model teacher "$TEACHER_MODEL")"
  sleep "$POLL_SECONDS"
done

echo "Both weight files have appeared. Running one full offline integrity check."
validation_log="$(mktemp)"
until conda run --no-capture-output -n "$ENV_NAME" python - \
  "$POLICY_MODEL" "$TEACHER_MODEL" "$EXPECTED_WEIGHT_BYTES" >"$validation_log" 2>&1 <<'PY'
import sys
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

expected = int(sys.argv[3])
for raw in sys.argv[1:3]:
    model = Path(raw)
    weight = model / "model.safetensors"
    if weight.stat().st_size != expected:
        raise RuntimeError(f"Unexpected weight size for {model}: {weight.stat().st_size}")
    if list(model.rglob("*.incomplete")):
        raise RuntimeError(f"Incomplete download files remain under {model}")
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    with safe_open(str(weight), framework="pt", device="cpu") as handle:
        tensors = len(handle.keys())
    print(
        f"[READY] {model.name}: type={config.model_type}, "
        f"tokenizer={len(tokenizer)}, tensors={tensors}"
    )
PY
do
  echo "[$(date '+%F %T')] Files are visible but final integrity validation is not ready; retrying in ${POLL_SECONDS}s."
  tail -n 8 "$validation_log" || true
  sleep "$POLL_SECONDS"
done
cat "$validation_log"
rm -f "$validation_log"

echo "Assets validated. Starting Vanilla GRPO and subproblem GRPO on two GPUs."
ENV_NAME="$ENV_NAME" \
MODEL_PATH="$POLICY_MODEL" \
TEACHER_MODEL="$TEACHER_MODEL" \
DATA_FILE="$DATA_FILE" \
LIMIT="$LIMIT" SAMPLES="$SAMPLES" TRAIN_STEPS="$TRAIN_STEPS" \
bash "$PROJECT_ROOT/scripts/run_vanilla_vs_subproblem_parallel_2h100.sh"

run_root="$(cat "$PROJECT_ROOT/outputs/latest_vanilla_vs_subproblem_parallel.txt")"
echo "All requested work completed: $run_root"
