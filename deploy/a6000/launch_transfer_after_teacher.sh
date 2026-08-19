#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/powerleader/project/adaptive-scaffold-router}"
TEACHER_DIR="${TEACHER_DIR:-$PROJECT_ROOT/models/Qwen2.5-Math-7B-Instruct}"
LOG="${LOG:-/home/powerleader/project/transfer_aware_pilot.log}"
PROBE_ROOTS="${PROBE_ROOTS:-16}"
ROOT_SAMPLES="${ROOT_SAMPLES:-4}"
PROBE_STEPS="${PROBE_STEPS:-2}"

weight_ready() {
  [[ -s "$TEACHER_DIR/config.json" ]] &&
    find "$TEACHER_DIR" -type f -name '*.safetensors' -size +1G | grep -q . &&
    ! find "$TEACHER_DIR" -type f -name '*.incomplete' -print -quit | grep -q .
}

echo "[$(date '+%F %T')] Waiting for transfer teacher: $TEACHER_DIR"
while ! weight_ready; do
  sleep 30
done

echo "[$(date '+%F %T')] Teacher ready; starting Transfer-Aware pilot."
cd "$PROJECT_ROOT"
PROBE_ROOTS="$PROBE_ROOTS" ROOT_SAMPLES="$ROOT_SAMPLES" PROBE_STEPS="$PROBE_STEPS" \
  bash scripts/run_transfer_aware_pilot_a6000.sh >> "$LOG" 2>&1
echo "[$(date '+%F %T')] Transfer-Aware pilot finished."
