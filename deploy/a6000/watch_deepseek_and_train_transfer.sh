#!/usr/bin/env bash
set -Eeuo pipefail

INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
PROJECT_ROOT="${PROJECT_ROOT:-$INSTALL_ROOT/adaptive-scaffold-router}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
ORIGINAL_CANDIDATES="${ORIGINAL_CANDIDATES:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
ANCHORS="${ANCHORS:-$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100/calibration/training_candidates.jsonl}"
PILOT_LOG="${PILOT_LOG:-$INSTALL_ROOT/transfer_aware_auto_train.log}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PROBE_ROOTS="${PROBE_ROOTS:-16}"

source "$CONDA_SH"
mkdir -p "$INSTALL_ROOT"

valid_candidates() {
  [[ -s "$ORIGINAL_CANDIDATES" ]] || return 1
  "$INSTALL_ROOT/envs/$ENV_NAME/bin/python" - "$ORIGINAL_CANDIDATES" "$ANCHORS" <<'PY'
import json, sys
from collections import Counter, defaultdict

candidates, anchors = sys.argv[1:]
anchor_roots = {str(json.loads(line)["root_id"]) for line in open(anchors, encoding="utf-8") if line.strip()}
by_root = defaultdict(set)
for line in open(candidates, encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    root = str(row.get("root_id", ""))
    dim = str(row.get("dimension", ""))
    if root in anchor_roots and dim in {"knowledge", "planning", "calculation"}:
        by_root[root].add(dim)
complete = sum({"knowledge", "planning", "calculation"}.issubset(dims) for dims in by_root.values())
if complete < 1:
    raise SystemExit(1)
print(f"valid original DeepSeek candidates: complete_roots={complete}, rows={sum(map(len, by_root.values()))}")
PY
}

echo "[$(date '+%F %T')] Waiting for original DeepSeek candidates: $ORIGINAL_CANDIDATES" >> "$PILOT_LOG"
while ! valid_candidates >> "$PILOT_LOG" 2>&1; do
  sleep "$POLL_SECONDS"
done

if pgrep -af '[r]un_transfer_aware_pilot_a6000.sh' >/dev/null; then
  echo "[$(date '+%F %T')] Transfer-Aware pilot already running; exiting watcher." >> "$PILOT_LOG"
  exit 0
fi

RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/transfer_aware_deepseek_$(date +%Y%m%d_%H%M%S)}"
echo "[$(date '+%F %T')] Candidates ready; starting Transfer-Aware training at $RUN_ROOT" >> "$PILOT_LOG"

cd "$PROJECT_ROOT"
ORIGINAL_CANDIDATES="$ORIGINAL_CANDIDATES" \
ANCHORS="$ANCHORS" PROBE_ROOTS="$PROBE_ROOTS" RUN_ROOT="$RUN_ROOT" \
RUN_SUBPROBLEM_PROBE=0 AUTO_EVAL=1 \
  bash scripts/run_transfer_aware_pilot_a6000.sh >> "$PILOT_LOG" 2>&1

echo "[$(date '+%F %T')] Transfer-Aware training and evaluation finished." >> "$PILOT_LOG"
