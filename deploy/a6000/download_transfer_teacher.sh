#!/usr/bin/env bash
set -Eeuo pipefail

INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
ENV_PREFIX="${ENV_PREFIX:-$INSTALL_ROOT/envs/scaf-grpo}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Math-7B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-$INSTALL_ROOT/adaptive-scaffold-router/models/Qwen2.5-Math-7B-Instruct}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MS_CLI="$ENV_PREFIX/bin/modelscope"

[[ -x "$MS_CLI" ]] || { echo "ModelScope CLI is missing: $MS_CLI" >&2; exit 2; }
mkdir -p "$LOCAL_DIR"

if [[ -s "$LOCAL_DIR/config.json" ]] && find "$LOCAL_DIR" -type f \
  \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +1G -print -quit | grep -q .; then
  echo "Transfer teacher already exists: $LOCAL_DIR"
  exit 0
fi

echo "Downloading $MODEL_ID to $LOCAL_DIR"
"$MS_CLI" download "$MODEL_ID" --local-dir "$LOCAL_DIR" --max-workers "$MAX_WORKERS"
echo "Transfer teacher ready: $LOCAL_DIR"
