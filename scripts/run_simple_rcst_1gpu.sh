#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_DIR:?Set DATA_DIR to prepare_simple_rcst.py output}"
: "${SCAF_REPO:?Set SCAF_REPO to the existing Scaf-GRPO checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the base model}"
: "${RUN_DIR:?Set RUN_DIR to a new experiment directory}"
ARM="${ARM:-method}"
PYTHON_BIN="${PYTHON_BIN:-python}"
[[ "$ARM" == method || "$ARM" == vanilla ]] || { echo 'ARM must be method or vanilla' >&2; exit 2; }

# One ordered stream keeps optimizer state across the phase boundary. A separate
# trainer invocation or checkpoint restart is unnecessary.
PLAN="$("$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
s = json.loads((root / 'summary.json').read_text())
if (s['method'] != 'simple_rcst_no_hint_v1' or s['shuffle'] is not False
        or s['filter_overlong_prompts'] is not False or s['bridge_enabled'] is not False
        or s['guided_root'] is not False):
    raise SystemExit('Incompatible simple RCST schedule')
if not (isinstance(s['steps'], int) and s['steps'] > 1
        and isinstance(s['batch_size'], int) and s['batch_size'] > 1
        and s['rows_per_arm'] == s['steps'] * s['batch_size']):
    raise SystemExit('Invalid step/batch schedule')
for name in ('method_train.parquet', 'vanilla_train.parquet'):
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != s['file_sha256'][name]:
        raise SystemExit('Prepared data changed: ' + name)
print(s['steps'], s['batch_size'])
PY
)"
read -r STEPS BATCH <<< "$PLAN"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'arm=%s steps=%s batch=%s shuffle=false hints=false bridge=false\n' "$ARM" "$STEPS" "$BATCH"
  exit 0
fi
[[ ! -e "$RUN_DIR" ]] || { echo 'RUN_DIR must be new; existing experiments are preserved' >&2; exit 2; }
mkdir -p "$RUN_DIR"
cp "$DATA_DIR/summary.json" "$RUN_DIR/data_protocol.json"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export SCAF_REPO MODEL_PATH PYTHON_BIN
export TRAIN_DATA="$DATA_DIR/${ARM}_train.parquet" OUTPUT_DIR="$RUN_DIR"
export TRAIN_STEPS="$STEPS" TRAIN_BATCH_SIZE="$BATCH" PPO_MINI_BATCH_SIZE="$BATCH"
export SAVE_FREQ="$STEPS" RESUME_MODE=disable N_GPUS=1 NNODES=1 TP_SIZE=1
export PLAIN_GRPO=1 BRIDGE_ENABLED=0 TRAIN_SHUFFLE=false FILTER_OVERLONG_PROMPTS=false
export EXPERIMENT_NAME="simple-rcst-${ARM}"
bash "$PROJECT_ROOT/scripts/run_fixed_budget_grpo_arm_1gpu.sh"
