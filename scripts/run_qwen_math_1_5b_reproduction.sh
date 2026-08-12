#!/usr/bin/env bash
set -euo pipefail

# One entry point for the networked download machine and the offline GPU machine.
# Stages are intentionally explicit because one shell cannot launch work on a
# different server instance by itself.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
ENV_NAME="${ENV_NAME:-scaf-grpo}"
STAGE="${STAGE:-base}"
MODEL_PATH="$PROJECT_ROOT/models/Qwen2.5-Math-1.5B"
TRAIN_DATA="$PROJECT_ROOT/data/DeepScaleR/Qwen2.5-Math-1.5B.parquet"

case "$STAGE" in
  prepare)
    SCAF_REPO="$SCAF_REPO" ENV_NAME="$ENV_NAME" \
      bash "$PROJECT_ROOT/scripts/prepare_qwen_math_1_5b_repro.sh"
    ;;
  prepare-base)
    SCAF_REPO="$SCAF_REPO" ENV_NAME="$ENV_NAME" \
      bash "$PROJECT_ROOT/scripts/prepare_qwen_math_1_5b_repro.sh"
    SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" PAPER_REFERENCE=base \
      bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh"
    ;;
  base)
    if [[ ! -f "$MODEL_PATH/model.safetensors" || ! -f "$TRAIN_DATA" ]]; then
      echo "Assets are missing. Run STAGE=prepare on the networked instance first." >&2
      exit 2
    fi
    SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" PAPER_REFERENCE=base \
      bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_eval_2h100.sh"
    ;;
  vanilla-smoke|scaf-smoke)
    method="${STAGE%-smoke}"
    SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$TRAIN_DATA" \
      METHOD="$method" MODE=smoke \
      bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_train_2h100.sh"
    ;;
  baseline-smokes)
    for method in vanilla scaf; do
      SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$TRAIN_DATA" \
        METHOD="$method" MODE=smoke \
        bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_train_2h100.sh"
    done
    ;;
  vanilla-paper|scaf-paper)
    method="${STAGE%-paper}"
    SCAF_REPO="$SCAF_REPO" MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$TRAIN_DATA" \
      METHOD="$method" MODE=paper CONFIRM_FULL_REPRO="${CONFIRM_FULL_REPRO:-}" \
      bash "$PROJECT_ROOT/scripts/run_qwen_math_1_5b_paper_train_2h100.sh"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    echo "Use: prepare, prepare-base, base, baseline-smokes, vanilla-smoke," >&2
    echo "     scaf-smoke, vanilla-paper, or scaf-paper" >&2
    exit 2
    ;;
esac
