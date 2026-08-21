#!/usr/bin/env bash
set -Eeuo pipefail

# Probe all 212 training roots against their original DeepSeek K/P/C candidates.
# This script deliberately stops after selection; it never generates candidates
# and never launches GRPO training before selected scaffolds are audited.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
ANCHORS="${ANCHORS:-$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100/calibration/training_candidates.jsonl}"
ORIGINAL_CANDIDATES="${ORIGINAL_CANDIDATES:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/transfer_aware_full_212_$(date +%Y%m%d_%H%M%S)}"
SEED_RESULTS="${SEED_RESULTS:-$PROJECT_ROOT/outputs/transfer_aware_deepseek_20260820_014611/selection/transfer_results.jsonl}"
TRAIN_GPU="${TRAIN_GPU:-0}"
ROOT_SAMPLES="${ROOT_SAMPLES:-4}"
PROBE_STEPS="${PROBE_STEPS:-2}"

source "$CONDA_SH"
mkdir -p "$RUN_ROOT"/{selection,logs}
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_transfer_aware_full_212.txt"
printf '%s\n' "full_212_same_root_transfer_probe_v1" > "$RUN_ROOT/protocol.txt"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CANDIDATES="$RUN_ROOT/selection/candidate_sets.jsonl"
TRANSFER_RESULTS="$RUN_ROOT/selection/transfer_results.jsonl"
SELECTED="$RUN_ROOT/selection/transfer_selected_candidates.jsonl"

echo "Stage 0a: retain all 212 original DeepSeek K/P/C candidate sets."
conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/filter_original_transfer_candidates.py" \
  --anchors "$ANCHORS" --original-candidates "$ORIGINAL_CANDIDATES" \
  --output "$CANDIDATES" --min-complete-roots 212 \
  --allow-empty-minimal-plan \
  2>&1 | tee "$RUN_ROOT/logs/candidate_filter.log"

if [[ ! -e "$TRANSFER_RESULTS" && -s "$SEED_RESULTS" ]]; then
  cp "$SEED_RESULTS" "$TRANSFER_RESULTS"
  echo "Seeded the 48 already completed candidate probes from: $SEED_RESULTS"
fi

echo "Stage 0b: probe all 212 roots (636 candidate updates); existing rows resume safely."
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/probe_subproblem_transfer.py" \
  --candidates "$CANDIDATES" --model "$MODEL_PATH" --output "$TRANSFER_RESULTS" \
  --root-limit 212 --root-samples "$ROOT_SAMPLES" --probe-steps "$PROBE_STEPS" \
  --device cuda:0 \
  2>&1 | tee "$RUN_ROOT/logs/transfer_probe.log"

echo "Stage 0c: select the maximum-transfer candidate for every root."
conda run --no-capture-output -n "$ENV_NAME" python \
  "$PROJECT_ROOT/select_transfer_aware_candidates.py" \
  --anchors "$ANCHORS" --candidates "$CANDIDATES" \
  --transfer-results "$TRANSFER_RESULTS" --output "$SELECTED" \
  2>&1 | tee "$RUN_ROOT/logs/selection.log"

conda run --no-capture-output -n "$ENV_NAME" python - "$SELECTED" \
  > "$RUN_ROOT/selection/selected_scaffold_audit.json" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
empty = [
    {"root_id": row["root_id"], "candidate_id": row["id"], "dimension": row.get("dimension")}
    for row in rows if not str(row.get("minimal_plan", "")).strip()
]
print(json.dumps({
    "selected_roots": len(rows),
    "empty_minimal_plan_selected": len(empty),
    "requires_scaffold_resolution_before_training": bool(empty),
    "details": empty,
}, ensure_ascii=False, indent=2))
PY

cat "$RUN_ROOT/selection/selected_scaffold_audit.json"
echo "Full transfer selection completed. GRPO was intentionally not started."
