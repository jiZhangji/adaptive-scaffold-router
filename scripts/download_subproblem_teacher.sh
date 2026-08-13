#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B-Instruct}"
MAX_WORKERS="${MAX_WORKERS:-8}"

mkdir -p "$LOCAL_DIR"
conda run --no-capture-output -n "$ENV_NAME" python - "$MODEL_ID" "$LOCAL_DIR" "$MAX_WORKERS" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, local_dir, max_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    max_workers=max_workers,
)
print(f"Teacher model ready: {path}")
PY

echo "The teacher is used only to decompose reference solutions; the RL policy remains the base model."
