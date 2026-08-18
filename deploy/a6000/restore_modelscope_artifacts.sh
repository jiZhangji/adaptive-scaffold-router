#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
MS_DIR="${MS_DIR:-$INSTALL_ROOT/modelscope/student-aware-grpo-migration}"
RUN_ID="${RUN_ID:-student_aware_root_aligned_20260816_183357}"
RUN_ROOT="$PROJECT_ROOT/outputs/$RUN_ID"
BASELINE_ROOT="$PROJECT_ROOT/outputs/complete_four_way_ordered_20260815_155120"
SUBPROBLEM_ROOT="$PROJECT_ROOT/outputs/complete_subproblem_n768_2h100"

mkdir -p \
  "$PROJECT_ROOT/models" \
  "$PROJECT_ROOT/data" \
  "$PROJECT_ROOT/outputs" \
  "$RUN_ROOT" \
  "$BASELINE_ROOT" \
  "$SUBPROBLEM_ROOT"

restore_archive() {
  local stem="$1"
  local destination="$2"
  local -a parts=()

  mapfile -t parts < <(
    find "$MS_DIR" -type f -name "$stem.tar.zst.part-*" -print | sort
  )

  if (( ${#parts[@]} == 0 )); then
    echo "Archive not found, skipping: $stem"
    return 0
  fi

  echo
  echo "===== RESTORE $stem ====="
  mkdir -p "$destination"
  for part in "${parts[@]}"; do
    cat "$part"
  done | zstd -dc | tar -xf - -C "$destination"
}

# The GitHub checkout is the authoritative source code. The archived source
# snapshot remains in MS_DIR for audit/recovery and is not overlaid here.
restore_archive "Qwen2.5-Math-1.5B-base-model" "$PROJECT_ROOT/models"
restore_archive "subproblem-generation-data" "$SUBPROBLEM_ROOT"
restore_archive "baseline-evaluation-results" "$BASELINE_ROOT"
restore_archive "student-aware-final" "$RUN_ROOT"

echo
echo "===== COPY DIRECT MERGED MODEL ====="
MERGED_CONFIG="$(find "$MS_DIR" -type f \
  -path '*/student_aware_merged/config.json' -print -quit || true)"
if [[ -n "$MERGED_CONFIG" ]]; then
  MERGED_SOURCE="$(dirname "$MERGED_CONFIG")"
  rm -rf "$RUN_ROOT/student_aware_merged"
  cp -a "$MERGED_SOURCE" "$RUN_ROOT/student_aware_merged"
  echo "Merged model: $RUN_ROOT/student_aware_merged"
else
  echo "Direct merged model was not found." >&2
fi

echo
echo "===== RESTORE METHOD ARTIFACTS ====="
METHOD_ROOT="$(find "$MS_DIR" -type d \
  -path "*/method_artifacts/$RUN_ID" -print -quit || true)"
if [[ -n "$METHOD_ROOT" ]]; then
  cp -a "$METHOD_ROOT/." "$RUN_ROOT/"
  echo "Method artifacts copied from: $METHOD_ROOT"
else
  echo "Method artifact directory was not found; archive contents are retained." >&2
fi

mkdir -p "$RUN_ROOT/proposed/checkpoints"
for step in 10 35 50; do
  ACTOR="$(find "$MS_DIR" -type d \
    -path "*/global_step_${step}/actor" -print -quit || true)"
  if [[ -n "$ACTOR" ]]; then
    SOURCE_STEP="$(dirname "$ACTOR")"
    TARGET_STEP="$RUN_ROOT/proposed/checkpoints/global_step_$step"
    rm -rf "$TARGET_STEP"
    cp -a "$SOURCE_STEP" "$TARGET_STEP"
    echo "Restored global_step_$step"
  fi
done

printf '%s\n' "$RUN_ROOT" > "$PROJECT_ROOT/outputs/latest_student_aware_root_aligned.txt"

echo
echo "Restore complete: $RUN_ROOT"
