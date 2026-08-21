#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:?usage: finalize_two_node_transfer_probe_a6000.sh RUN_ROOT}"
PEER_HOST="${PEER_HOST:-powerleader@10.6.2.3}"
PEER_KEY="${PEER_KEY:-$HOME/.ssh/a6000_peer_ed25519}"
ENV_PREFIX="${ENV_PREFIX:-/home/powerleader/project/envs/scaf-grpo}"
ANCHORS="${ANCHORS:-$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100/calibration/training_candidates.jsonl}"

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/selection"

local_pid="$(cat "$RUN_ROOT/shard0.pid")"
remote_pid="$(ssh -i "$PEER_KEY" -o IdentitiesOnly=yes "$PEER_HOST" \
  "cat '$PROJECT_ROOT/$RUN_ROOT/shard1.pid'")"

while :; do
  local_alive=0
  remote_alive=0
  kill -0 "$local_pid" 2>/dev/null && local_alive=1
  ssh -i "$PEER_KEY" -o IdentitiesOnly=yes "$PEER_HOST" \
    "kill -0 '$remote_pid' 2>/dev/null" && remote_alive=1 || true
  local_count="$(wc -l < "$RUN_ROOT/selection/shard0.jsonl")"
  remote_count="$(ssh -i "$PEER_KEY" -o IdentitiesOnly=yes "$PEER_HOST" \
    "wc -l < '$PROJECT_ROOT/$RUN_ROOT/selection/shard1.jsonl'")"
  printf '%s local=%s remote=%s alive=%s/%s\n' \
    "$(date '+%F %T')" "$local_count" "$remote_count" "$local_alive" "$remote_alive"
  [[ "$local_alive" -eq 0 && "$remote_alive" -eq 0 ]] && break
  sleep 30
done

scp -q -i "$PEER_KEY" -o IdentitiesOnly=yes \
  "$PEER_HOST:$PROJECT_ROOT/$RUN_ROOT/selection/shard1.jsonl" \
  "$RUN_ROOT/selection/shard1.jsonl"

/opt/miniconda3/bin/conda run --no-capture-output -p "$ENV_PREFIX" python \
  merge_transfer_probe_shards.py \
  --inputs "$RUN_ROOT/selection/shard0.jsonl" "$RUN_ROOT/selection/shard1.jsonl" \
  --output "$RUN_ROOT/selection/transfer_results.jsonl" \
  --candidates "$RUN_ROOT/selection/candidate_sets.jsonl" \
  --require-complete

/opt/miniconda3/bin/conda run --no-capture-output -p "$ENV_PREFIX" python \
  select_transfer_aware_candidates.py \
  --anchors "$ANCHORS" \
  --candidates "$RUN_ROOT/selection/candidate_sets.jsonl" \
  --transfer-results "$RUN_ROOT/selection/transfer_results.jsonl" \
  --output "$RUN_ROOT/selection/transfer_selected_candidates.jsonl"

/opt/miniconda3/bin/conda run --no-capture-output -p "$ENV_PREFIX" python - \
  "$RUN_ROOT/selection/transfer_selected_candidates.jsonl" \
  > "$RUN_ROOT/selection/selected_scaffold_audit.json" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
empty = [
    {"root_id": row["root_id"], "candidate_id": row["id"], "dimension": row.get("dimension")}
    for row in rows
    if not str(row.get("minimal_plan", "")).strip()
]
print(json.dumps({
    "selected_roots": len(rows),
    "empty_minimal_plan_selected": len(empty),
    "requires_scaffold_resolution_before_training": bool(empty),
    "details": empty,
}, ensure_ascii=False, indent=2))
PY

cat "$RUN_ROOT/selection/selected_scaffold_audit.json"
echo "Two-node transfer probe merged and selected. Formal GRPO was not started."
