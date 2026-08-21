#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATES="${CANDIDATES:?Set CANDIDATES to the aligned candidate JSONL}"
ANCHORS="${ANCHORS:-$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100/calibration/training_candidates.jsonl}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/td_transfer_probe_$(date +%Y%m%d_%H%M%S)}"
PEER_HOST="${PEER_HOST:-powerleader@10.6.2.3}"
PEER_KEY="${PEER_KEY:-$HOME/.ssh/a6000_peer_ed25519}"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
ROOT_SAMPLES="${ROOT_SAMPLES:-4}"
PROBE_STEPS="${PROBE_STEPS:-2}"
ROOT_LIMIT="${ROOT_LIMIT:-0}"
RSYNC_SHELL="ssh -i $PEER_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
SSH=(ssh -i "$PEER_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15)

remote() { "${SSH[@]}" "$PEER_HOST" "$@"; }

mkdir -p "$RUN_ROOT"/{selection,logs}
cp "$CANDIDATES" "$RUN_ROOT/selection/candidate_sets.jsonl"
remote "mkdir -p '$RUN_ROOT'/{selection,logs}"
rsync -aH -e "$RSYNC_SHELL" "$RUN_ROOT/selection/candidate_sets.jsonl" \
  "$PEER_HOST:$RUN_ROOT/selection/candidate_sets.jsonl"

local_output="$RUN_ROOT/selection/shard0.jsonl"
remote_output="$RUN_ROOT/selection/shard1.jsonl"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$PROJECT_ROOT/probe_subproblem_transfer.py" \
  --candidates "$RUN_ROOT/selection/candidate_sets.jsonl" --model "$MODEL_PATH" \
  --output "$local_output" --root-limit "$ROOT_LIMIT" \
  --num-shards 2 --shard-index 0 --root-samples "$ROOT_SAMPLES" \
  --probe-steps "$PROBE_STEPS" --device cuda:0 \
  > "$RUN_ROOT/logs/shard0.log" 2>&1 &
local_pid=$!
printf '%s\n' "$local_pid" > "$RUN_ROOT/shard0.pid"

remote "cd '$PROJECT_ROOT'; nohup env CUDA_VISIBLE_DEVICES=0 \
  '$PYTHON_BIN' '$PROJECT_ROOT/probe_subproblem_transfer.py' \
  --candidates '$RUN_ROOT/selection/candidate_sets.jsonl' --model '$MODEL_PATH' \
  --output '$remote_output' --root-limit '$ROOT_LIMIT' \
  --num-shards 2 --shard-index 1 --root-samples '$ROOT_SAMPLES' \
  --probe-steps '$PROBE_STEPS' --device cuda:0 \
  > '$RUN_ROOT/logs/shard1.log' 2>&1 < /dev/null & echo \$! > '$RUN_ROOT/shard1.pid'"
remote_pid="$(remote "cat '$RUN_ROOT/shard1.pid'")"

while kill -0 "$local_pid" 2>/dev/null || remote "kill -0 '$remote_pid' 2>/dev/null"; do
  sleep 60
done
wait "$local_pid"
remote "wait '$remote_pid' 2>/dev/null || true"
rsync -aH -e "$RSYNC_SHELL" "$PEER_HOST:$remote_output" "$remote_output"

"$PYTHON_BIN" "$PROJECT_ROOT/merge_transfer_probe_shards.py" \
  --inputs "$local_output" "$remote_output" \
  --output "$RUN_ROOT/selection/transfer_results.jsonl" \
  --candidates "$RUN_ROOT/selection/candidate_sets.jsonl" --require-complete \
  2>&1 | tee "$RUN_ROOT/logs/merge.log"
"$PYTHON_BIN" "$PROJECT_ROOT/select_transfer_aware_candidates.py" \
  --anchors "$ANCHORS" --candidates "$RUN_ROOT/selection/candidate_sets.jsonl" \
  --transfer-results "$RUN_ROOT/selection/transfer_results.jsonl" \
  --output "$RUN_ROOT/selection/transfer_selected_candidates.jsonl" \
  2>&1 | tee "$RUN_ROOT/logs/select.log"
echo "$RUN_ROOT/selection/transfer_selected_candidates.jsonl"
