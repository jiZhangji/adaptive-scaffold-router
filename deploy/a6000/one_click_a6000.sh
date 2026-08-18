#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/bootstrap_a6000.sh"
bash "$SCRIPT_DIR/download_modelscope_artifacts.sh"
bash "$SCRIPT_DIR/restore_modelscope_artifacts.sh"
bash "$SCRIPT_DIR/verify_a6000_install.sh"

if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  bash "$SCRIPT_DIR/evaluate_checkpoints_a6000.sh"
else
  echo
  echo "Setup is complete. Evaluation was not started."
  echo "Run later with:"
  echo "  bash $SCRIPT_DIR/evaluate_checkpoints_a6000.sh"
fi
