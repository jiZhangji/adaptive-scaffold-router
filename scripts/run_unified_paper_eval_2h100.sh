#!/usr/bin/env bash
set -euo pipefail

# Evaluate any of the four comparison arms under one immutable protocol.
# MODEL_PATH must be a directly loadable Hugging Face directory.  For a
# trained arm, first merge the selected best validation checkpoint.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to Base or a merged trained checkpoint}"
METHOD_LABEL="${METHOD_LABEL:-}"
case "$METHOD_LABEL" in
  base|vanilla|scaf|subproblem) ;;
  *) echo "METHOD_LABEL must be base, vanilla, scaf, or subproblem" >&2; exit 2 ;;
esac

if [[ ! -s "$MODEL_PATH/config.json" ]]; then
  echo "MODEL_PATH is not a loadable Hugging Face model: $MODEL_PATH" >&2
  exit 2
fi
if ! compgen -G "$MODEL_PATH/*.safetensors" >/dev/null; then
  echo "No safetensors weights found in MODEL_PATH: $MODEL_PATH" >&2
  echo "Merge the selected FSDP checkpoint before evaluation." >&2
  exit 2
fi

if [[ "$METHOD_LABEL" == "subproblem" ]]; then
  PAPER_REFERENCE="${PAPER_REFERENCE:-scaf}"
else
  PAPER_REFERENCE="${PAPER_REFERENCE:-$METHOD_LABEL}"
fi

SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}" \
MODEL_PATH="$MODEL_PATH" METHOD_LABEL="$METHOD_LABEL" \
PAPER_REFERENCE="$PAPER_REFERENCE" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
N_GPUS="${N_GPUS:-2}" \
bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh"
