#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/deepseek_zero_reward_subproblems_all}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"

if [[ "${CONFIRM_FULL_GENERATION:-}" != "YES" ]]; then
  conda run --no-capture-output -n "${ENV_NAME:-scaf-grpo}" python - "$DATA_FILE" <<'PY'
import sys
import pyarrow.parquet as pq

rows = pq.read_table(sys.argv[1], columns=["accuracy"]).to_pylist()
zero = sum(float(row.get("accuracy", 0.0) or 0.0) == 0.0 for row in rows)
print(f"Zero-reward roots: {zero}")
print(f"Requested candidates: {zero * 3} (knowledge/planning/calculation)")
PY
  echo "This can issue one paid API request per zero-reward root." >&2
  echo "Re-run with CONFIRM_FULL_GENERATION=YES after the 64-root quality screen passes." >&2
  exit 2
fi

# One API call per zero-reward root returns three distinct candidates. LIMIT=0
# means all eligible roots; MAX_SOURCE_ACCURACY=0 restricts generation to the
# roots that currently have no successful rollout signal in the Scaf dataset.
RUN_ROOT="$RUN_ROOT" \
DATA_FILE="$DATA_FILE" \
LIMIT=0 \
MAX_SOURCE_ACCURACY=0 \
DIMENSIONS=knowledge,planning,calculation \
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}" \
bash "$PROJECT_ROOT/scripts/generate_subproblems_deepseek.sh"
