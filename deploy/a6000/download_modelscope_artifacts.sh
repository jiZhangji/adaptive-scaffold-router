#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
ENV_PREFIX="${ENV_PREFIX:-$INSTALL_ROOT/envs/scaf-grpo}"
MS_REPO_ID="${MS_REPO_ID:-shimian123/student-aware-grpo-migration}"
MS_DIR="${MS_DIR:-$INSTALL_ROOT/modelscope/student-aware-grpo-migration}"
RUN_ID="${RUN_ID:-student_aware_root_aligned_20260816_183357}"
MIGRATION_ID="${MIGRATION_ID:-student_aware_migration_20260817_144959}"

MS_CLI="$ENV_PREFIX/bin/modelscope"
[[ -x "$MS_CLI" ]] || {
  echo "ModelScope CLI is missing: $MS_CLI" >&2
  echo "Run bootstrap_a6000.sh first." >&2
  exit 2
}

mkdir -p "$MS_DIR"

if [[ -n "${MS_TOKEN:-}" ]]; then
  "$MS_CLI" login --token "$MS_TOKEN"
else
  read -rsp "ModelScope access token: " TOKEN
  echo
  "$MS_CLI" login --token "$TOKEN"
  unset TOKEN
fi

if ! "$MS_CLI" download --help 2>&1 | grep -q -- "--include"; then
  echo "This ModelScope CLI does not expose selective --include downloads." >&2
  echo "Upgrade modelscope instead of downloading the entire repository." >&2
  exit 3
fi

download_pattern() {
  local pattern="$1"
  echo
  echo "===== DOWNLOAD $pattern ====="
  "$MS_CLI" download \
    --model "$MS_REPO_ID" \
    --include "$pattern" \
    --local_dir "$MS_DIR"
}

# Directly loadable final model.
download_pattern "student_aware_merged/*"

# Essential archive: source snapshot, base model, selected data, final run,
# baseline evaluations, environment manifests, and checksums.
download_pattern "migration/$MIGRATION_ID/*"

# Step 10/35/50 checkpoints, representative rollouts, logs, and evaluations.
download_pattern "method_artifacts/$RUN_ID/*"

echo
echo "ModelScope download complete: $MS_DIR"
