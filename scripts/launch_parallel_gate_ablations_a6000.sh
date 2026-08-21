#!/usr/bin/env bash
set -Eeuo pipefail

# Launch two independent fixed-budget runs, one per A6000 server. This is
# faster than cross-node FSDP on the measured 1-GbE link and keeps every
# training/evaluation hyperparameter equal to the existing fair comparison.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEER_HOST="${PEER_HOST:-powerleader@10.6.2.3}"
PEER_KEY="${PEER_KEY:-$HOME/.ssh/a6000_peer_ed25519}"
SELECTED="${SELECTED:-$PROJECT_ROOT/outputs/transfer_selected_full_212.jsonl}"
LEARNABILITY_A="${LEARNABILITY_A:-$PROJECT_ROOT/outputs/transfer_aware_all_selected_full_20260821_1920/learnability/subproblem_learnability.jsonl}"
LEARNABILITY_B="${LEARNABILITY_B:-$PROJECT_ROOT/outputs/transfer_aware_positive_gated_full_20260821_1920/learnability/subproblem_learnability.jsonl}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_A="${RUN_A:-$PROJECT_ROOT/outputs/transfer_gate_moderate_$STAMP}"
RUN_B="${RUN_B:-$PROJECT_ROOT/outputs/transfer_gate_conservative_$STAMP}"

SSH=(ssh -i "$PEER_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15)

for path in "$SELECTED" "$LEARNABILITY_A"; do
  [[ -s "$path" ]] || { echo "Missing local input: $path" >&2; exit 2; }
done
"${SSH[@]}" "$PEER_HOST" "test -s '$SELECTED' && test -s '$LEARNABILITY_B'" || {
  echo "Peer candidate or learnability input is missing." >&2
  exit 2
}
if pgrep -af '[v]erl.trainer.main_ppo|[h]int_mix_grpo.main_ppo' >/dev/null; then
  echo "Local GPU already has a training process." >&2
  exit 3
fi
if "${SSH[@]}" "$PEER_HOST" "pgrep -af '[v]erl.trainer.main_ppo|[h]int_mix_grpo.main_ppo' >/dev/null"; then
  echo "Peer GPU already has a training process." >&2
  exit 3
fi

mkdir -p "$RUN_A"/{learnability,logs}
cp "$LEARNABILITY_A" "$RUN_A/learnability/subproblem_learnability.jsonl"
printf '%s\n' "Transfer-Aware Moderate Gate: gain>=0.25, post>=0.5" > "$RUN_A/method_label.txt"
printf '%s\n' "independent_node_fixed_budget_v1" > "$RUN_A/parallel_protocol.txt"

"${SSH[@]}" "$PEER_HOST" "mkdir -p '$RUN_B'/learnability '$RUN_B'/logs; \
  cp '$LEARNABILITY_B' '$RUN_B/learnability/subproblem_learnability.jsonl'; \
  printf '%s\\n' 'Transfer-Aware Strict Gate: gain>=0.5, post>=0.75' > '$RUN_B/method_label.txt'; \
  printf '%s\\n' 'independent_node_fixed_budget_v1' > '$RUN_B/parallel_protocol.txt'"

nohup env SELECTED_CANDIDATES="$SELECTED" RUN_ROOT="$RUN_A" \
  RUN_SUBPROBLEM_PROBE=0 POSITIVE_TRANSFER_ONLY=1 \
  MIN_TRANSFER_GAIN=0.25 MIN_POST_UPDATE_PROBABILITY=0.5 \
  AUTO_EVAL=1 EVAL_GPUS=0 EVAL_N_GPUS=1 \
  bash "$PROJECT_ROOT/scripts/run_student_aware_root_aligned_only_2h100.sh" \
  > "$RUN_A/logs/launcher.log" 2>&1 < /dev/null &
pid_a=$!

"${SSH[@]}" "$PEER_HOST" "cd '$PROJECT_ROOT'; nohup env \
  SELECTED_CANDIDATES='$SELECTED' RUN_ROOT='$RUN_B' \
  RUN_SUBPROBLEM_PROBE=0 POSITIVE_TRANSFER_ONLY=1 \
  MIN_TRANSFER_GAIN=0.5 MIN_POST_UPDATE_PROBABILITY=0.75 \
  AUTO_EVAL=1 EVAL_GPUS=0 EVAL_N_GPUS=1 \
  bash '$PROJECT_ROOT/scripts/run_student_aware_root_aligned_only_2h100.sh' \
  > '$RUN_B/logs/launcher.log' 2>&1 < /dev/null & echo \$! > '$RUN_B/launcher.pid'"
pid_b="$("${SSH[@]}" "$PEER_HOST" "cat '$RUN_B/launcher.pid'")"

printf '%s\n' "$pid_a" > "$RUN_A/launcher.pid"
cat <<EOF
Server A PID=$pid_a
Server A run=$RUN_A
Server B PID=$pid_b
Server B run=$RUN_B
EOF
