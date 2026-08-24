#!/usr/bin/env bash
set -Eeuo pipefail

# Synchronize the reproducible A6000 workspace from 10.6.2.4 to 10.6.2.3.
# Run this script on the source server. It never starts a GPU workload and it
# intentionally excludes historical outputs and the unused 7B model.

SOURCE_ROOT="${SOURCE_ROOT:-/home/powerleader/project}"
TARGET_ROOT="${TARGET_ROOT:-/home/powerleader/project}"
TARGET_HOST="${TARGET_HOST:-powerleader@10.6.2.3}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/a6000_peer_ed25519}"
SOURCE_REPO="$SOURCE_ROOT/adaptive-scaffold-router"
TARGET_REPO="$TARGET_ROOT/adaptive-scaffold-router"
SCAF_REPO="$SOURCE_ROOT/Scaf-GRPO"
ENV_PREFIX="$SOURCE_ROOT/envs/scaf-grpo"
MODEL_DIR="$SOURCE_REPO/models/Qwen2.5-Math-1.5B"
DATA_DIR="$SOURCE_REPO/data"

SSH_OPTIONS=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
)
RSYNC_SHELL="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"

remote() {
  ssh "${SSH_OPTIONS[@]}" "$TARGET_HOST" "$@"
}

sync_tree() {
  local source="$1"
  local target="$2"
  shift 2
  [[ -d "$source" ]] || { echo "Missing source directory: $source" >&2; exit 3; }
  remote "mkdir -p '$target'"
  rsync -aH --partial --info=progress2 -e "$RSYNC_SHELL" "$@" \
    "$source/" "$TARGET_HOST:$target/"
}

sync_file() {
  local source="$1"
  local target_dir="$2"
  [[ -s "$source" ]] || { echo "Missing source file: $source" >&2; exit 3; }
  remote "mkdir -p '$target_dir'"
  rsync -aH --partial --info=progress2 -e "$RSYNC_SHELL" \
    "$source" "$TARGET_HOST:$target_dir/"
}

[[ -s "$SSH_KEY" ]] || { echo "Missing peer SSH key: $SSH_KEY" >&2; exit 2; }
command -v rsync >/dev/null || { echo "rsync is missing on the source server" >&2; exit 2; }

echo "===== PEER CONNECTIVITY ====="
remote "set -e; echo host=\$(hostname); echo ips=\$(hostname -I); command -v rsync; \
  if test -x /opt/miniconda3/bin/conda; then echo target_conda=/opt/miniconda3/bin/conda; \
  else echo target_conda=missing_using_synced_prefix_python; fi"

echo "===== PROJECT CODE ====="
sync_tree "$SOURCE_REPO" "$TARGET_REPO" \
  --exclude='outputs/' \
  --exclude='models/' \
  --exclude='data/' \
  --exclude='.pytest_cache/' \
  --exclude='**/__pycache__/'

echo "===== SCAF-GRPO CODE AND BENCHMARK DATA ====="
sync_tree "$SCAF_REPO" "$TARGET_ROOT/Scaf-GRPO" \
  --exclude='outputs/' \
  --exclude='checkpoints/' \
  --exclude='wandb/' \
  --exclude='**/__pycache__/'

echo "===== CONDA ENVIRONMENT ====="
sync_tree "$ENV_PREFIX" "$TARGET_ROOT/envs/scaf-grpo" \
  --exclude='**/__pycache__/'

echo "===== 1.5B MODEL ====="
sync_tree "$MODEL_DIR" "$TARGET_REPO/models/Qwen2.5-Math-1.5B"

echo "===== TRAINING DATA ====="
sync_tree "$DATA_DIR" "$TARGET_REPO/data"

echo "===== REQUIRED DEEPSEEK AND CALIBRATION ASSETS ====="
sync_file \
  "$SOURCE_REPO/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl" \
  "$TARGET_REPO/outputs/deepseek_zero_reward_subproblems_all"
if [[ -s "$SOURCE_REPO/outputs/deepseek_zero_reward_subproblems_all/candidates.summary.json" ]]; then
  sync_file \
    "$SOURCE_REPO/outputs/deepseek_zero_reward_subproblems_all/candidates.summary.json" \
    "$TARGET_REPO/outputs/deepseek_zero_reward_subproblems_all"
fi
sync_file \
  "$SOURCE_REPO/outputs/complete_subproblem_n768_2h100/calibration/training_candidates.jsonl" \
  "$TARGET_REPO/outputs/complete_subproblem_n768_2h100/calibration"

LEARNABILITY="$SOURCE_REPO/outputs/student_aware_root_aligned_20260816_183357/learnability/subproblem_learnability.jsonl"
if [[ -s "$LEARNABILITY" ]]; then
  sync_file "$LEARNABILITY" \
    "$TARGET_REPO/outputs/student_aware_root_aligned_20260816_183357/learnability"
fi

echo "===== REGISTER AND VERIFY ENVIRONMENT ====="
remote "set -e; \
  if test -x /opt/miniconda3/bin/conda; then \
    /opt/miniconda3/bin/conda config --add envs_dirs '$TARGET_ROOT/envs' 2>/dev/null || true; \
    /opt/miniconda3/bin/conda run -p '$TARGET_ROOT/envs/scaf-grpo' python -c \"import torch, flash_attn; print('python_ok'); print('torch=' + torch.__version__); print('cuda=' + str(torch.version.cuda)); print('gpu=' + torch.cuda.get_device_name(0)); print('flash_attn=' + flash_attn.__version__)\"; \
  else \
    '$TARGET_ROOT/envs/scaf-grpo/bin/python' -c \"import torch, flash_attn; print('python_ok'); print('torch=' + torch.__version__); print('cuda=' + str(torch.version.cuda)); print('gpu=' + torch.cuda.get_device_name(0)); print('flash_attn=' + flash_attn.__version__)\"; \
  fi; \
  test -s '$TARGET_REPO/models/Qwen2.5-Math-1.5B/config.json'; \
  test -s '$TARGET_REPO/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet'; \
  test -s '$TARGET_REPO/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl'; \
  echo verification=passed"

echo "===== SYNC COMPLETE ====="
echo "Target: $TARGET_HOST:$TARGET_ROOT"
echo "Historical outputs and Qwen2.5-Math-7B-Instruct were intentionally not copied."
