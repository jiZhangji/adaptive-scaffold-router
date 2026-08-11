#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Math-1.5B}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-Math-1.5B}"
DATASET_ID="${DATASET_ID:-hkuzxc/scaf-grpo-dataset}"
DATASET_FILE="${DATASET_FILE:-Qwen2.5-Math-1.5B.parquet}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MODEL_PATH="$PROJECT_ROOT/models/$MODEL_NAME"
DATA_FILE="$PROJECT_ROOT/data/DeepScaleR/$DATASET_FILE"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_cmd=("$PYTHON_BIN")
else
  python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python)
fi

echo "Project root: $PROJECT_ROOT"
echo "Model target: $MODEL_PATH"
echo "Dataset target: $DATA_FILE"
echo "Existing complete files are reused; interrupted Hugging Face downloads resume automatically."

"${python_cmd[@]}" "$PROJECT_ROOT/scripts/download_probe_assets.py" \
  --project-root "$PROJECT_ROOT" \
  --model-id "$MODEL_ID" \
  --model-name "$MODEL_NAME" \
  --dataset-id "$DATASET_ID" \
  --dataset-file "$DATASET_FILE" \
  --max-workers "$MAX_WORKERS"

"${python_cmd[@]}" "$PROJECT_ROOT/scripts/validate_assets.py" \
  --model "$MODEL_PATH" \
  --dataset "$DATA_FILE"

echo "Assets downloaded and validated."
echo "MODEL_PATH=$MODEL_PATH"
echo "DATA_FILE=$DATA_FILE"
