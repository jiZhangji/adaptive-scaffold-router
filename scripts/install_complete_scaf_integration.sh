#!/usr/bin/env bash
set -euo pipefail

: "${SCAF_REPO:?Set SCAF_REPO to the offline Scaf-GRPO checkout}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$PROJECT_ROOT/install_complete_scaf_integration.py" \
  --project-root "$PROJECT_ROOT" \
  --scaf-repo "$SCAF_REPO"
