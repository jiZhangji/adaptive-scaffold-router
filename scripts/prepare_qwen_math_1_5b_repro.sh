#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MAX_WORKERS="${MAX_WORKERS:-8}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_cmd=("$PYTHON_BIN")
else
  python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python)
fi

echo "Preparing Qwen/Qwen2.5-Math-1.5B and its official Scaf-GRPO parquet."
echo "Model target: $PROJECT_ROOT/models/Qwen2.5-Math-1.5B"
echo "Dataset target: $PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet"
echo "Scaf-GRPO checkout: $SCAF_REPO"

"${python_cmd[@]}" "$PROJECT_ROOT/scripts/prepare_qwen_math_1_5b_repro.py" \
  --project-root "$PROJECT_ROOT" --scaf-repo "$SCAF_REPO" \
  --max-workers "$MAX_WORKERS" "$@"

echo "Assets downloaded, paper context configuration applied, and files validated."
