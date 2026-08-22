#!/usr/bin/env bash
set -Eeuo pipefail

# Fastest layout for the measured 1-GbE cluster: replicated probes and
# independent fixed-budget training runs. No cross-node gradient synchronization.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
PEER_HOST="${PEER_HOST:-powerleader@10.6.2.3}"
PEER_KEY="${PEER_KEY:-$HOME/.ssh/a6000_peer_ed25519}"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/outputs/transfer_aware_full_212_distributed_20260821_1636/selection/candidate_sets.jsonl}"
ANCHORS="${ANCHORS:-$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100/calibration/training_candidates.jsonl}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/rcst_full_$(date +%Y%m%d_%H%M%S)}"
SEED_A="${SEED_A:-42}"
SEED_B="${SEED_B:-314159}"
ROOT_SAMPLES="${ROOT_SAMPLES:-8}"
PROBE_STEPS="${PROBE_STEPS:-2}"
CONFIDENCE_Z="${CONFIDENCE_Z:-1.0}"
POLL_SECONDS="${POLL_SECONDS:-30}"

SSH=(ssh -i "$PEER_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15)
RSYNC_SHELL="ssh -i $PEER_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
remote() { "${SSH[@]}" "$PEER_HOST" "$@"; }

for path in "$PYTHON_BIN" "$CANDIDATES" "$ANCHORS"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
candidate_count="$(wc -l < "$CANDIDATES")"
[[ "$candidate_count" -eq 636 ]] || {
  echo "Expected 636 candidate rows, found $candidate_count" >&2
  exit 2
}
if pgrep -af '[v]erl.trainer.main_ppo|[h]int_mix_grpo.main_ppo|[p]robe_subproblem_transfer' >/dev/null; then
  echo "Server A GPU is busy" >&2
  exit 3
fi
if remote "pgrep -af '[v]erl.trainer.main_ppo|[h]int_mix_grpo.main_ppo|[p]robe_subproblem_transfer' >/dev/null"; then
  echo "Server B GPU is busy" >&2
  exit 3
fi

mkdir -p "$RUN_ROOT"/{selection,logs}
remote "mkdir -p '$RUN_ROOT'/selection '$RUN_ROOT'/logs"
printf '%s\n' "rcst_replicated_probe_2x8_then_independent_training_v1" > "$RUN_ROOT/protocol.txt"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_rcst_full.txt"

probe_a="$RUN_ROOT/selection/probe_seed_${SEED_A}.jsonl"
probe_b="$RUN_ROOT/selection/probe_seed_${SEED_B}.jsonl"
exit_a="$RUN_ROOT/probe_a.exit"
exit_b="$RUN_ROOT/probe_b.exit"
rm -f "$exit_a"
remote "rm -f '$exit_b'"

nohup env PROJECT_ROOT="$PROJECT_ROOT" CANDIDATES="$CANDIDATES" \
  OUTPUT="$probe_a" EXIT_FILE="$exit_a" SEED="$SEED_A" \
  ROOT_SAMPLES="$ROOT_SAMPLES" PROBE_STEPS="$PROBE_STEPS" \
  bash "$PROJECT_ROOT/scripts/run_rcst_probe_node_a6000.sh" \
  > "$RUN_ROOT/logs/probe_seed_${SEED_A}.log" 2>&1 < /dev/null &
pid_a=$!
printf '%s\n' "$pid_a" > "$RUN_ROOT/probe_a.pid"

remote "cd '$PROJECT_ROOT'; nohup env PROJECT_ROOT='$PROJECT_ROOT' \
  CANDIDATES='$CANDIDATES' OUTPUT='$probe_b' EXIT_FILE='$exit_b' \
  SEED='$SEED_B' ROOT_SAMPLES='$ROOT_SAMPLES' PROBE_STEPS='$PROBE_STEPS' \
  bash '$PROJECT_ROOT/scripts/run_rcst_probe_node_a6000.sh' \
  > '$RUN_ROOT/logs/probe_seed_${SEED_B}.log' 2>&1 < /dev/null & echo \$! > '$RUN_ROOT/probe_b.pid'"

while [[ ! -s "$exit_a" ]] || ! remote "test -s '$exit_b'"; do
  count_a="$(if [[ -f "$probe_a" ]]; then wc -l < "$probe_a"; else echo 0; fi)"
  count_b="$(remote "if [[ -f '$probe_b' ]]; then wc -l < '$probe_b'; else echo 0; fi")"
  printf '%s probe A=%s/636 B=%s/636\n' "$(date '+%F %T')" "$count_a" "$count_b"
  sleep "$POLL_SECONDS"
done
[[ "$(cat "$exit_a")" == "0" ]] || { echo "Server A probe failed" >&2; exit 4; }
[[ "$(remote "cat '$exit_b'")" == "0" ]] || { echo "Server B probe failed" >&2; exit 4; }

rsync -aH -e "$RSYNC_SHELL" "$PEER_HOST:$probe_b" "$probe_b"
[[ "$(wc -l < "$probe_a")" -eq 636 ]] || { echo "Incomplete A probe" >&2; exit 4; }
[[ "$(wc -l < "$probe_b")" -eq 636 ]] || { echo "Incomplete B probe" >&2; exit 4; }

aggregate="$RUN_ROOT/selection/rcst_aggregated.jsonl"
mean_selected="$RUN_ROOT/selection/rcst_mean_positive.jsonl"
lcb_selected="$RUN_ROOT/selection/rcst_lcb_positive.jsonl"
common=(
  --anchors "$ANCHORS" --candidates "$CANDIDATES"
  --inputs "$probe_a" "$probe_b" --aggregated-output "$aggregate"
  --confidence-z "$CONFIDENCE_Z" --min-replicates 2 --min-score 0.0
)
"$PYTHON_BIN" "$PROJECT_ROOT/aggregate_rcst_probes.py" "${common[@]}" \
  --policy mean_positive --output "$mean_selected" \
  2>&1 | tee "$RUN_ROOT/logs/select_mean.log"
"$PYTHON_BIN" "$PROJECT_ROOT/aggregate_rcst_probes.py" "${common[@]}" \
  --policy lcb_positive --output "$lcb_selected" \
  2>&1 | tee "$RUN_ROOT/logs/select_lcb.log"

rsync -aH -e "$RSYNC_SHELL" "$RUN_ROOT/selection/" "$PEER_HOST:$RUN_ROOT/selection/"

train_a="$RUN_ROOT/train_mean_positive"
train_b="$RUN_ROOT/train_lcb_positive"
exit_train_a="$RUN_ROOT/train_a.exit"
exit_train_b="$RUN_ROOT/train_b.exit"
rm -f "$exit_train_a"
remote "rm -f '$exit_train_b'; mkdir -p '$train_b/logs'"
mkdir -p "$train_a/logs"

nohup env PROJECT_ROOT="$PROJECT_ROOT" SELECTED_CANDIDATES="$mean_selected" \
  RUN_ROOT="$train_a" METHOD_LABEL="RCST Mean-Positive" \
  EXIT_FILE="$exit_train_a" \
  bash "$PROJECT_ROOT/scripts/run_rcst_train_node_a6000.sh" \
  > "$train_a/logs/rcst_launcher.log" 2>&1 < /dev/null &
train_pid_a=$!
printf '%s\n' "$train_pid_a" > "$RUN_ROOT/train_a.pid"

remote "cd '$PROJECT_ROOT'; nohup env PROJECT_ROOT='$PROJECT_ROOT' \
  SELECTED_CANDIDATES='$lcb_selected' RUN_ROOT='$train_b' \
  METHOD_LABEL='RCST LCB-Positive' EXIT_FILE='$exit_train_b' \
  bash '$PROJECT_ROOT/scripts/run_rcst_train_node_a6000.sh' \
  > '$train_b/logs/rcst_launcher.log' 2>&1 < /dev/null & echo \$! > '$RUN_ROOT/train_b.pid'"

while [[ ! -s "$exit_train_a" ]] || ! remote "test -s '$exit_train_b'"; do
  step_a="$(find "$train_a/proposed/checkpoints" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)"
  step_b="$(remote "find '$train_b/proposed/checkpoints' -maxdepth 1 -type d -name 'global_step_*' -printf '%f\\n' 2>/dev/null | sort -V | tail -n 1 || true")"
  printf '%s training A=%s B=%s\n' "$(date '+%F %T')" "${step_a:-starting}" "${step_b:-starting}"
  sleep "$POLL_SECONDS"
done
[[ "$(cat "$exit_train_a")" == "0" ]] || { echo "Server A training failed" >&2; exit 5; }
[[ "$(remote "cat '$exit_train_b'")" == "0" ]] || { echo "Server B training failed" >&2; exit 5; }

mkdir -p "$RUN_ROOT/remote_lcb_results"
rsync -aH -e "$RSYNC_SHELL" \
  --include='*/' --include='*.json' --include='*.md' --exclude='*' \
  "$PEER_HOST:$train_b/" "$RUN_ROOT/remote_lcb_results/"
echo "RCST suite complete: $RUN_ROOT"
