#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===== MODEL SCOPE LOGIN FIRST ====="
if [[ -z "${MS_TOKEN:-}" ]]; then
  read -rsp "ModelScope access token: " MS_TOKEN
  echo
fi
export MS_TOKEN

# Do not block on Hugging Face before the private ModelScope artifacts have
# been authenticated and downloaded.
SKIP_TRAINING_DATA=1 bash "$SCRIPT_DIR/bootstrap_a6000.sh"
bash "$SCRIPT_DIR/download_modelscope_artifacts.sh"
bash "$SCRIPT_DIR/restore_modelscope_artifacts.sh"

echo
echo "===== TRAINING DATA (MIRROR FIRST) ====="
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$(pwd)/data/DeepScaleR" \
  BASE_URL="${HF_DATA_BASE_URL:-https://hf-mirror.com/datasets/hkuzxc/scaf-grpo-dataset/resolve/main}" \
  MIRROR_BASE_URL="${HF_DATA_MIRROR_URL:-https://hf-mirror.com/datasets/hkuzxc/scaf-grpo-dataset/resolve/main}" \
  bash scripts/download_scaf_data.sh

bash "$SCRIPT_DIR/verify_a6000_install.sh"

if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  bash "$SCRIPT_DIR/evaluate_checkpoints_a6000.sh"
else
  echo
  echo "Setup is complete. Evaluation was not started."
  echo "Run later with:"
  echo "  bash $SCRIPT_DIR/evaluate_checkpoints_a6000.sh"
fi
