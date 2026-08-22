#!/usr/bin/env bash
set -Eeuo pipefail

# Largest currently candidate-complete experiment:
#   1,866 zero-reward roots x 3 DeepSeek candidates x 2 probe seeds.
# Server A coordinates and trains Vanilla; Server B trains RCST LCB-Positive.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/powerleader/project/envs/scaf-grpo/bin/python}"
PEER_HOST="${PEER_HOST:-powerleader@10.6.2.3}"
PEER_KEY="${PEER_KEY:-$HOME/.ssh/a6000_peer_ed25519}"
SOURCE_DATA="${SOURCE_DATA:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
RAW_CANDIDATES="${RAW_CANDIDATES:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/full_1866_rcst_vs_vanilla_$(date +%Y%m%d_%H%M%S)}"
EXPECTED_ROOTS="${EXPECTED_ROOTS:-1866}"
EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-5598}"
SEED_A="${SEED_A:-42}"
SEED_B="${SEED_B:-314159}"
ROOT_SAMPLES="${ROOT_SAMPLES:-8}"
PROBE_STEPS="${PROBE_STEPS:-2}"
CONFIDENCE_Z="${CONFIDENCE_Z:-1.0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STAGE_1_END="${STAGE_1_END:-88}"
STAGE_2_END="${STAGE_2_END:-308}"
TOTAL_STEPS="${TOTAL_STEPS:-440}"
SAVE_FREQ="${SAVE_FREQ:-44}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

SSH=(ssh -i "$PEER_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15)
RSYNC_SHELL="ssh -i $PEER_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
remote() { "${SSH[@]}" "$PEER_HOST" "$@"; }

for path in "$PYTHON_BIN" "$SOURCE_DATA" "$RAW_CANDIDATES"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ "$(wc -l < "$RAW_CANDIDATES")" -eq "$EXPECTED_CANDIDATES" ]] || {
  echo "Raw candidate count is not $EXPECTED_CANDIDATES" >&2
  exit 2
}
remote "test -x '$PYTHON_BIN' && test -s '$PROJECT_ROOT/models/Qwen2.5-Math-1.5B/config.json'" || {
  echo "Peer environment/model is incomplete" >&2
  exit 2
}

SUITE_FILES=(
  prepare_full_rcst_inputs.py
  probe_subproblem_transfer.py
  aggregate_rcst_probes.py
  build_student_aware_preconditioning_experiment.py
  scripts/run_rcst_probe_node_a6000.sh
  scripts/run_full_1866_vanilla_node_a6000.sh
  scripts/run_full_1866_rcst_node_a6000.sh
  scripts/summarize_full_rcst_vs_vanilla.py
)
suite_file_args="${SUITE_FILES[*]}"
local_suite_hash="$(
  cd "$PROJECT_ROOT"
  sha256sum "${SUITE_FILES[@]}" | sha256sum | awk '{print $1}'
)"
peer_suite_hash="$(remote "cd '$PROJECT_ROOT'; sha256sum $suite_file_args | sha256sum | awk '{print \$1}'")"
if [[ "$local_suite_hash" != "$peer_suite_hash" ]]; then
  echo "Experiment code checksum mismatch: A=$local_suite_hash B=$peer_suite_hash" >&2
  exit 2
fi

if pgrep -af '[v]erl.trainer.main_ppo|[h]int_mix_grpo.main_ppo|[p]robe_subproblem_transfer' >/dev/null; then
  echo "Server A GPU is busy" >&2
  exit 3
fi
if remote "pgrep -af '[v]erl.trainer.main_ppo|[h]int_mix_grpo.main_ppo|[p]robe_subproblem_transfer' >/dev/null"; then
  echo "Server B GPU is busy" >&2
  exit 3
fi

mkdir -p "$RUN_ROOT"/{data,selection,logs}
remote "mkdir -p '$RUN_ROOT'/{data,selection,logs}"
printf '%s\n' "full_1866_rcst_lcb_vs_vanilla_two_node_v1" > "$RUN_ROOT/protocol.txt"
printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_full_1866_rcst_vs_vanilla.txt"
cat > "$RUN_ROOT/config.env" <<EOF
expected_roots=$EXPECTED_ROOTS
expected_candidates=$EXPECTED_CANDIDATES
probe_seeds=$SEED_A,$SEED_B
root_samples_per_seed=$ROOT_SAMPLES
probe_steps=$PROBE_STEPS
confidence_z=$CONFIDENCE_Z
stage_1_end=$STAGE_1_END
stage_2_end=$STAGE_2_END
total_steps=$TOTAL_STEPS
batch_size=32
rollouts=8
EOF

echo "Preparing all $EXPECTED_ROOTS candidate-complete roots."
"$PYTHON_BIN" "$PROJECT_ROOT/prepare_full_rcst_inputs.py" \
  --source-data "$SOURCE_DATA" --raw-candidates "$RAW_CANDIDATES" \
  --output-dir "$RUN_ROOT/data" --expected-roots "$EXPECTED_ROOTS" \
  2>&1 | tee "$RUN_ROOT/logs/prepare.log"

candidates="$RUN_ROOT/data/candidate_sets.jsonl"
anchors="$RUN_ROOT/data/fallback_anchors.jsonl"
vanilla_data="$RUN_ROOT/data/vanilla_root_train.parquet"
[[ "$(wc -l < "$candidates")" -eq "$EXPECTED_CANDIDATES" ]] || exit 4
[[ "$(wc -l < "$anchors")" -eq "$EXPECTED_ROOTS" ]] || exit 4
rsync -aH -e "$RSYNC_SHELL" "$RUN_ROOT/data/" "$PEER_HOST:$RUN_ROOT/data/"
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Preflight complete; probes and training were not started."
  exit 0
fi

probe_a="$RUN_ROOT/selection/probe_seed_${SEED_A}.jsonl"
probe_b="$RUN_ROOT/selection/probe_seed_${SEED_B}.jsonl"
exit_a="$RUN_ROOT/probe_a.exit"
exit_b="$RUN_ROOT/probe_b.exit"
rm -f "$exit_a"
remote "rm -f '$exit_b'"

nohup env PROJECT_ROOT="$PROJECT_ROOT" CANDIDATES="$candidates" \
  OUTPUT="$probe_a" EXIT_FILE="$exit_a" SEED="$SEED_A" ROOT_LIMIT=0 \
  ROOT_SAMPLES="$ROOT_SAMPLES" PROBE_STEPS="$PROBE_STEPS" \
  bash "$PROJECT_ROOT/scripts/run_rcst_probe_node_a6000.sh" \
  > "$RUN_ROOT/logs/probe_seed_${SEED_A}.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/probe_a.pid"

remote "cd '$PROJECT_ROOT'; nohup env PROJECT_ROOT='$PROJECT_ROOT' \
  CANDIDATES='$candidates' OUTPUT='$probe_b' EXIT_FILE='$exit_b' \
  SEED='$SEED_B' ROOT_LIMIT=0 ROOT_SAMPLES='$ROOT_SAMPLES' \
  PROBE_STEPS='$PROBE_STEPS' \
  bash '$PROJECT_ROOT/scripts/run_rcst_probe_node_a6000.sh' \
  > '$RUN_ROOT/logs/probe_seed_${SEED_B}.log' 2>&1 < /dev/null & \
  echo \$! > '$RUN_ROOT/probe_b.pid'"

while [[ ! -s "$exit_a" ]] || ! remote "test -s '$exit_b'"; do
  count_a="$(if [[ -f "$probe_a" ]]; then wc -l < "$probe_a"; else echo 0; fi)"
  count_b="$(remote "if [[ -f '$probe_b' ]]; then wc -l < '$probe_b'; else echo 0; fi")"
  printf '%s probe A=%s/%s B=%s/%s\n' \
    "$(date '+%F %T')" "$count_a" "$EXPECTED_CANDIDATES" \
    "$count_b" "$EXPECTED_CANDIDATES"
  sleep "$POLL_SECONDS"
done
[[ "$(cat "$exit_a")" == "0" ]] || { echo "Server A probe failed" >&2; exit 5; }
[[ "$(remote "cat '$exit_b'")" == "0" ]] || { echo "Server B probe failed" >&2; exit 5; }
rsync -aH -e "$RSYNC_SHELL" "$PEER_HOST:$probe_b" "$probe_b"
[[ "$(wc -l < "$probe_a")" -eq "$EXPECTED_CANDIDATES" ]] || exit 5
[[ "$(wc -l < "$probe_b")" -eq "$EXPECTED_CANDIDATES" ]] || exit 5

aggregate="$RUN_ROOT/selection/rcst_aggregated.jsonl"
selected="$RUN_ROOT/selection/rcst_lcb_positive.jsonl"
"$PYTHON_BIN" "$PROJECT_ROOT/aggregate_rcst_probes.py" \
  --anchors "$anchors" --candidates "$candidates" \
  --inputs "$probe_a" "$probe_b" --aggregated-output "$aggregate" \
  --confidence-z "$CONFIDENCE_Z" --min-replicates 2 --min-score 0.0 \
  --policy lcb_positive --output "$selected" \
  2>&1 | tee "$RUN_ROOT/logs/select_lcb.log"
[[ "$(wc -l < "$selected")" -eq "$EXPECTED_ROOTS" ]] || exit 6
rsync -aH -e "$RSYNC_SHELL" "$RUN_ROOT/selection/" "$PEER_HOST:$RUN_ROOT/selection/"

vanilla_root="$RUN_ROOT/train_vanilla"
rcst_root="$RUN_ROOT/train_rcst_lcb"
vanilla_exit="$RUN_ROOT/train_vanilla.exit"
rcst_exit="$RUN_ROOT/train_rcst.exit"
rm -f "$vanilla_exit"
remote "rm -f '$rcst_exit'; mkdir -p '$rcst_root/logs'"
mkdir -p "$vanilla_root/logs"

nohup env PROJECT_ROOT="$PROJECT_ROOT" TRAIN_DATA="$vanilla_data" \
  RUN_ROOT="$vanilla_root" EXIT_FILE="$vanilla_exit" \
  STAGE_1_END="$STAGE_1_END" STAGE_2_END="$STAGE_2_END" \
  TOTAL_STEPS="$TOTAL_STEPS" SAVE_FREQ="$SAVE_FREQ" \
  bash "$PROJECT_ROOT/scripts/run_full_1866_vanilla_node_a6000.sh" \
  > "$vanilla_root/logs/launcher.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/train_vanilla.pid"

remote "cd '$PROJECT_ROOT'; nohup env PROJECT_ROOT='$PROJECT_ROOT' \
  SELECTED_CANDIDATES='$selected' RUN_ROOT='$rcst_root' EXIT_FILE='$rcst_exit' \
  PRECONDITION_STEPS='$STAGE_1_END' ROOT_ALIGNED_END_STEP='$STAGE_2_END' \
  TOTAL_STEPS='$TOTAL_STEPS' SAVE_FREQ='$SAVE_FREQ' \
  bash '$PROJECT_ROOT/scripts/run_full_1866_rcst_node_a6000.sh' \
  > '$rcst_root/logs/launcher.log' 2>&1 < /dev/null & \
  echo \$! > '$RUN_ROOT/train_rcst.pid'"

while [[ ! -s "$vanilla_exit" ]] || ! remote "test -s '$rcst_exit'"; do
  vanilla_step="$(find "$vanilla_root/vanilla/checkpoints" -maxdepth 1 -type d \
    -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)"
  rcst_step="$(remote "find '$rcst_root/proposed/checkpoints' -maxdepth 1 -type d \
    -name 'global_step_*' -printf '%f\\n' 2>/dev/null | sort -V | tail -n 1 || true")"
  printf '%s training Vanilla=%s RCST-LCB=%s\n' "$(date '+%F %T')" \
    "${vanilla_step:-starting}" "${rcst_step:-starting}"
  sleep "$POLL_SECONDS"
done
[[ "$(cat "$vanilla_exit")" == "0" ]] || { echo "Vanilla failed" >&2; exit 7; }
[[ "$(remote "cat '$rcst_exit'")" == "0" ]] || { echo "RCST failed" >&2; exit 7; }

remote_results="$RUN_ROOT/remote_rcst_results"
mkdir -p "$remote_results"
rsync -aH -e "$RSYNC_SHELL" \
  --include='*/' --include='*.json' --include='*.md' --include='*.txt' --exclude='*' \
  "$PEER_HOST:$rcst_root/" "$remote_results/"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/summarize_full_rcst_vs_vanilla.py" \
  --vanilla-eval "$vanilla_root/eval_vanilla" \
  --rcst-eval "$remote_results/eval_student_aware" \
  --output-dir "$RUN_ROOT/final_comparison" \
  --training-steps "$TOTAL_STEPS" --training-roots "$EXPECTED_ROOTS" \
  2>&1 | tee "$RUN_ROOT/logs/final_comparison.log"

echo "Full RCST-vs-Vanilla suite complete: $RUN_ROOT"
echo "Comparison: $RUN_ROOT/final_comparison/comparison.md"
