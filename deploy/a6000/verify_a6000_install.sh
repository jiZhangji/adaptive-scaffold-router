#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
ENV_PREFIX="${ENV_PREFIX:-$INSTALL_ROOT/envs/scaf-grpo}"
SCAF_REPO="${SCAF_REPO:-$INSTALL_ROOT/Scaf-GRPO}"
RUN_ID="${RUN_ID:-student_aware_root_aligned_20260816_183357}"
RUN_ROOT="$PROJECT_ROOT/outputs/$RUN_ID"

fail=0
check_file() {
  local path="$1"
  if [[ -s "$path" ]]; then
    echo "OK   $path"
  else
    echo "MISS $path"
    fail=1
  fi
}

echo "===== GPU ====="
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader

echo
echo "===== PYTHON PACKAGES ====="
"$ENV_PREFIX/bin/python" - <<'PY'
import torch
import transformers
import vllm
import verl
import modelscope

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("modelscope", modelscope.__version__)
print("verl", verl.__file__)
PY

echo
echo "===== REQUIRED FILES ====="
check_file "$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet"
check_file "$PROJECT_ROOT/models/Qwen2.5-Math-1.5B/config.json"
check_file "$RUN_ROOT/student_aware_merged/config.json"
check_file "$SCAF_REPO/data/AIME24/math-verify/system-p1/test.parquet"
check_file "$SCAF_REPO/data/AMC23/math-verify/system-p1/test.parquet"

echo
echo "===== CHECKPOINTS ====="
for step in 10 35 50; do
  actor="$RUN_ROOT/proposed/checkpoints/global_step_$step/actor"
  if [[ -d "$actor" ]]; then
    echo "OK   global_step_$step"
  else
    echo "MISS global_step_$step"
    fail=1
  fi
done

if (( fail != 0 )); then
  echo
  echo "Verification failed; do not start evaluation yet." >&2
  exit 4
fi

echo
echo "A6000 installation verified."
