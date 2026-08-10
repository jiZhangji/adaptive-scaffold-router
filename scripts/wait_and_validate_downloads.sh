#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$(dirname "$PROJECT_ROOT")}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet}"
SCAF_REPO="${SCAF_REPO:-$ROOT/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-0}"
REPORT_PATH="${REPORT_PATH:-$PROJECT_ROOT/outputs/preflight_validation.json}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH" >&2
  exit 1
fi

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python)
start_time="$(date +%s)"
log_file="$(mktemp -t scaffold_asset_check.XXXXXX)"
trap 'rm -f "$log_file"' EXIT

echo "Waiting for downloads to become complete"
echo "  model:   $MODEL_PATH"
echo "  dataset: $DATA_FILE"
echo "  env:     $ENV_NAME"

while ! "${python_cmd[@]}" "$PROJECT_ROOT/scripts/validate_assets.py" \
  --model "$MODEL_PATH" --dataset "$DATA_FILE" --quiet >"$log_file" 2>&1; do
  now="$(date +%s)"
  elapsed="$((now - start_time))"
  if [[ "$WAIT_TIMEOUT" -gt 0 && "$elapsed" -ge "$WAIT_TIMEOUT" ]]; then
    echo "Timed out after ${elapsed}s while waiting for downloads" >&2
    cat "$log_file" >&2
    exit 1
  fi
  model_size="$(du -sh "$MODEL_PATH" 2>/dev/null | awk '{print $1}' || true)"
  data_size="$(du -h "$DATA_FILE" 2>/dev/null | awk '{print $1}' || true)"
  echo "[$(date '+%F %T')] not complete yet; model=${model_size:-missing}, dataset=${data_size:-missing}"
  tail -n 8 "$log_file" || true
  sleep "$POLL_SECONDS"
done

echo "Downloads are complete. Running full validation..."
"${python_cmd[@]}" "$PROJECT_ROOT/scripts/validate_assets.py" \
  --model "$MODEL_PATH" --dataset "$DATA_FILE" --report "$REPORT_PATH"

echo "Running repository unit tests..."
cd "$PROJECT_ROOT"
"${python_cmd[@]}" -m unittest discover -s tests -v

if [[ -d "$SCAF_REPO/hint_mix_grpo" ]]; then
  echo "Checking Scaf-GRPO integration imports..."
  PYTHONPATH="$PROJECT_ROOT:$SCAF_REPO${PYTHONPATH:+:$PYTHONPATH}" \
    "${python_cmd[@]}" -m py_compile \
      "$PROJECT_ROOT/capability_scaffold.py" \
      "$PROJECT_ROOT/scaf_curriculum_adapter.py" \
      "$SCAF_REPO/hint_mix_grpo/trainer/ray_trainer.py"
  PYTHONPATH="$PROJECT_ROOT:$SCAF_REPO${PYTHONPATH:+:$PYTHONPATH}" \
    "${python_cmd[@]}" -c \
      "from hint_mix_grpo.trainer.ray_trainer import RayPPOTrainer; print('Scaf-GRPO trainer import passed')"
else
  echo "Scaf-GRPO checkout was not found at $SCAF_REPO; skipping its import check."
fi

echo "Preflight validation passed. Report: $REPORT_PATH"
