#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end overnight queue. It starts formal work only after the actual
# released parquet files pass checksum, row-count, and same-212-root checks.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
ANCHORS="${ANCHORS:-$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100/calibration/training_candidates.jsonl}"
TD_DATA_DIR="${TD_DATA_DIR:-$PROJECT_ROOT/outputs/td_grokking_artifact/code/data/DeepScaleR-hard}"
QUEUE_ROOT="${QUEUE_ROOT:-$PROJECT_ROOT/outputs/td_grokking_overnight_$(date +%Y%m%d_%H%M%S)}"
RUN_ALL_SELECTED="${RUN_ALL_SELECTED:-1}"
RUN_POSITIVE="${RUN_POSITIVE:-1}"
RUN_CONSERVATIVE="${RUN_CONSERVATIVE:-1}"

mkdir -p "$QUEUE_ROOT"/{data,logs,runs,learnability}
printf '%s\n' "$QUEUE_ROOT" > "$PROJECT_ROOT/outputs/latest_td_grokking_overnight.txt"

if TD_DATA_DIR="$TD_DATA_DIR" \
  bash "$PROJECT_ROOT/scripts/fetch_td_grokking_artifact.sh" \
  > "$QUEUE_ROOT/logs/fetch.log" 2>&1; then
  :
else
  status=$?
  echo "TD-Grokking released parquet is unavailable; no formal experiment started." \
    | tee "$QUEUE_ROOT/BLOCKED.txt"
  echo "fetch_exit=$status" >> "$QUEUE_ROOT/BLOCKED.txt"
  exit "$status"
fi

"$PYTHON_BIN" "$PROJECT_ROOT/prepare_td_grokking_candidates.py" \
  --anchors "$ANCHORS" --root-only "$TD_DATA_DIR/root_only.parquet" \
  --sub-only "$TD_DATA_DIR/sub_only.parquet" \
  --output "$QUEUE_ROOT/data/candidate_sets.jsonl" \
  --require-matched-roots 212 --min-candidates-per-root 2 \
  2>&1 | tee "$QUEUE_ROOT/logs/align.log"

probe_root="$QUEUE_ROOT/probe"
CANDIDATES="$QUEUE_ROOT/data/candidate_sets.jsonl" RUN_ROOT="$probe_root" \
  bash "$PROJECT_ROOT/scripts/run_two_node_transfer_probe_a6000.sh" \
  2>&1 | tee "$QUEUE_ROOT/logs/transfer_probe.log"
selected="$probe_root/selection/transfer_selected_candidates.jsonl"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$PROJECT_ROOT/probe_selected_subproblem_learnability.py" \
  --candidates "$selected" --model "$PROJECT_ROOT/models/Qwen2.5-Math-1.5B" \
  --output-dir "$QUEUE_ROOT/learnability" --samples 8 --group-size 8 \
  --batch-size 16 --job-chunk-size 128 --device cuda:0 --dtype bfloat16 \
  --max-input-tokens 2048 --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 \
  --stop-after-boxed 2>&1 | tee "$QUEUE_ROOT/logs/learnability.log"
learnability="$QUEUE_ROOT/learnability/subproblem_learnability.jsonl"

run_one() {
  local slug="$1" label="$2" gate="$3"
  local run_root="$QUEUE_ROOT/runs/$slug"
  SELECTED_CANDIDATES="$selected" LEARNABILITY_FILE="$learnability" \
    RUN_LEARNABILITY=0 GATE_MODE="$gate" METHOD_LABEL="$label" \
    RUN_ROOT="$run_root" AUTO_EVAL=1 \
    bash "$PROJECT_ROOT/scripts/run_two_node_transfer_training_a6000.sh" \
    > "$QUEUE_ROOT/logs/$slug.log" 2>&1
}

[[ "$RUN_ALL_SELECTED" == "1" ]] && \
  run_one all_selected "TD Candidates: All-Selected" all
[[ "$RUN_POSITIVE" == "1" ]] && \
  run_one positive "TD Candidates: Positive-Gated" positive
[[ "$RUN_CONSERVATIVE" == "1" ]] && \
  run_one conservative "TD Candidates: Conservative-Gated" conservative

"$PYTHON_BIN" - "$QUEUE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "runs").glob("*/comparison.json")):
    payload = json.loads(path.read_text())
    row = {"method": payload["method_label"], "macro": payload["macro"]["proposed"]}
    row.update(payload["results"]["proposed"])
    rows.append(row)
(root / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = ["# TD-Grokking Candidate Overnight Results", "", "| Method | Macro |", "|---|---:|"]
lines += [f"| {row['method']} | {row['macro']:.1%} |" for row in rows]
(root / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo "Overnight queue complete: $QUEUE_ROOT/results.md"
