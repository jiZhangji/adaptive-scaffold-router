#!/usr/bin/env bash
set -Eeuo pipefail

# Run this watchdog on A6000-C. It waits for F to become reachable over the
# private LAN, then starts the existing idle-GPU launcher on F exactly once.

F_SSH_HOST="${F_SSH_HOST:-a6000-f}"
CHECK_SECONDS="${CHECK_SECONDS:-60}"
F_IDLE_SECONDS="${F_IDLE_SECONDS:-1800}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/powerleader/project/adaptive-scaffold-router}"
LOCAL_PROJECT="${LOCAL_PROJECT:-/home/powerleader/project/adaptive-scaffold-router}"
WATCH_ROOT="${WATCH_ROOT:-$LOCAL_PROJECT/outputs/f_boot_watchdog}"
LOCK_DIR="$WATCH_ROOT/watch.lock"
STATE_FILE="$WATCH_ROOT/state.txt"

mkdir -p "$WATCH_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another F watchdog owns $LOCK_DIR" >&2
  exit 3
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

write_state() {
  local value="$1"
  printf 'timestamp=%s\nstatus=%s\nf_host=%s\n' \
    "$(date -Is)" "$value" "$F_SSH_HOST" > "$STATE_FILE.tmp"
  mv "$STATE_FILE.tmp" "$STATE_FILE"
}

log "watching $F_SSH_HOST; check every ${CHECK_SECONDS}s"
write_state waiting_for_f

while true; do
  if ! ssh "$F_SSH_HOST" true >/dev/null 2>&1; then
    log "F is offline or SSH is unavailable"
    write_state waiting_for_f
    sleep "$CHECK_SECONDS"
    continue
  fi

  log "F is reachable; checking and launching the Bridge-Calibrated RCST workflow"
  write_state f_reachable
  set +e
  ssh "$F_SSH_HOST" \
    "PROJECT_ROOT='$REMOTE_PROJECT' IDLE_SECONDS='$F_IDLE_SECONDS' bash -s" <<'REMOTE'
set -Eeuo pipefail

BASE="$PROJECT_ROOT/outputs/bridge_calibrated_rcst_212"
PYTHON_BIN="/home/powerleader/project/envs/scaf-grpo/bin/python"
FINAL_RESULT="$BASE/formal_train_rcst/eval_bridge_calibrated_rcst/paper_comparison.md"
mkdir -p "$BASE/logs"

if [[ -s "$FINAL_RESULT" ]]; then
  echo "Bridge-Calibrated RCST result already exists: $FINAL_RESULT"
  exit 0
fi

if pgrep -f '[m]onitor_gpu_idle_then_run.py.*bridge_calibrated_rcst_212' >/dev/null \
  || pgrep -f '[r]un_bridge_calibrated_rcst_212_f.sh' >/dev/null \
  || pgrep -f '[s]core_bridge_delta_features.py' >/dev/null \
  || pgrep -f '[f]it_bridge_calibrated_rcst.py' >/dev/null; then
  echo "Bridge-Calibrated RCST workflow is already running on F"
  exit 0
fi

for path in \
  "$PYTHON_BIN" \
  "$PROJECT_ROOT/monitor_gpu_idle_then_run.py" \
  "$PROJECT_ROOT/scripts/run_bridge_calibrated_rcst_212_f.sh" \
  "$PROJECT_ROOT/build_rcst_bridge_data.py" \
  "$PROJECT_ROOT/rcst_bridge_dataset.py" \
  "$PROJECT_ROOT/scaf_integration/ray_trainer_bridge.py" \
  "$PROJECT_ROOT/models/Qwen2.5-Math-1.5B/config.json"; do
  test -s "$path" || { echo "Missing required file on F: $path" >&2; exit 21; }
done

# These locks are stale when none of the workflow processes above exists.
rm -f "$BASE/idle_monitor.lock"
rmdir "$BASE/run.lock" 2>/dev/null || true

nohup "$PYTHON_BIN" "$PROJECT_ROOT/monitor_gpu_idle_then_run.py" \
  --gpu-indices 0 1 \
  --idle-seconds "$IDLE_SECONDS" \
  --poll-seconds 60 \
  --max-utilization 10 \
  --max-memory-mib 1024 \
  --state-file "$BASE/idle_monitor_state.json" \
  --lock-file "$BASE/idle_monitor.lock" \
  --workdir "$PROJECT_ROOT" \
  -- env SKIP_IDLE_WAIT=1 RUN_FORMAL_TRAINING=1 \
    bash scripts/run_bridge_calibrated_rcst_212_f.sh \
  > "$BASE/logs/idle_monitor.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$BASE/idle_monitor.pid"
sleep 2
kill -0 "$pid"
echo "Started F idle monitor: pid=$pid, idle_required=${IDLE_SECONDS}s"
REMOTE
  remote_code=$?
  set -e

  if (( remote_code == 0 )); then
    log "F workflow is running or already complete; watchdog finished"
    write_state launched_or_complete
    exit 0
  fi

  log "F was reachable but launch validation failed (exit=$remote_code); retrying"
  write_state launch_failed_retrying
  sleep "$CHECK_SECONDS"
done
