#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAF_REPO="${SCAF_REPO:-$(dirname "$PROJECT_ROOT")/Scaf-GRPO}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B}"
PAPER_REFERENCE="${PAPER_REFERENCE:-base}"
METHOD_LABEL="${METHOD_LABEL:-$PAPER_REFERENCE}"
PROTOCOL_ID="scaf-grpo-greedy-pass1-v1"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${N_GPUS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
RUN_NAME="${RUN_NAME:-qwen_math_1_5b_${PAPER_REFERENCE}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$PROJECT_ROOT/outputs/$RUN_NAME}"
CHECKPOINT_RULE="${CHECKPOINT_RULE:-}"

export CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled

if [[ "$SKIP_PREPARE" != "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_qwen_math_1_5b_repro.py" \
    --project-root "$PROJECT_ROOT" --scaf-repo "$SCAF_REPO" --validate-only >/dev/null
fi

mkdir -p "$OUT/logs"
printf '%s\n' "$OUT" > "$PROJECT_ROOT/outputs/latest_qwen_math_1_5b_eval.txt"
printf '%s\n' "$$" > "$OUT/pid.txt"

# Persist the exact evaluation contract next to every result.  A score without
# this file must not be mixed into the unified Base/Vanilla/Scaf/Subproblem
# comparison table.
"$PYTHON_BIN" - "$OUT/evaluation_protocol.json" "$MODEL_PATH" \
  "$METHOD_LABEL" "$PAPER_REFERENCE" "$PROTOCOL_ID" "$CHECKPOINT_RULE" <<'PY'
import json, os, sys
from pathlib import Path

output, model, method, reference, protocol_id, checkpoint_rule = sys.argv[1:]
payload = {
    "protocol_id": protocol_id,
    "method": method,
    "paper_reference": reference,
    "model_path": str(Path(model).resolve()),
    "decoding": {
        "metric": "pass@1",
        "n_samples": 1,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "prompt_length": 2048,
        "response_length": 2048,
    },
    "verification": "Scaf-GRPO verl.trainer.main_eval math-verify/symeval pipeline",
    "checkpoint_rule": checkpoint_rule or (
        "base pretrained weights" if method == "base"
        else "best validation checkpoint, merged to Hugging Face format"
    ),
    "benchmarks": [
        "AIME24", "AIME25", "AMC23", "MinervaMath", "MATH-500",
        "OlympiadBench", "GaoKao2023en",
    ],
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

declare -A DATASETS=(
  [AIME24]="data/AIME24/math-verify/system-p1/test.parquet"
  [AIME25]="data/AIME25/math-verify/system-p1/test.parquet"
  [AMC23]="data/AMC23/math-verify/system-p1/test.parquet"
  [MinervaMath]="data/MinervaMath/math-verify/system-p1/test.parquet"
  [MATH-500]="data/MATH-500/math-verify/system-p1/test.parquet"
  [OlympiadBench]="data/OlympiadBench/math-verify/system-p1/test.parquet"
  [GaoKao2023en]="data/GaoKao2023en/math-verify/system-p1/test.parquet"
)
ORDER=(AIME24 AIME25 AMC23 MinervaMath MATH-500 OlympiadBench GaoKao2023en)

cd "$SCAF_REPO"
for dataset in "${ORDER[@]}"; do
  data_path="$SCAF_REPO/${DATASETS[$dataset]}"
  dataset_out="$OUT/$dataset"
  save_path="$dataset_out/generation_output.parquet"
  mkdir -p "$dataset_out"
  if [[ -f "$dataset_out/metric.json" ]]; then
    echo "[$dataset] metric exists; skipping"
    continue
  fi

  echo "[$dataset] generating unified greedy pass@1 on $N_GPUS GPUs"
  "$PYTHON_BIN" -m verl.trainer.main_generation \
    trainer.nnodes=1 trainer.n_gpus_per_node="$N_GPUS" \
    data.path="$data_path" data.batch_size=1024 data.prompt_key=prompt \
    data.n_samples=1 data.output_path="$save_path" \
    model.path="$MODEL_PATH" +model.trust_remote_code=True \
    rollout.do_sample=False rollout.temperature=0.0 rollout.top_p=1.0 rollout.top_k=-1 \
    rollout.prompt_length=2048 rollout.response_length=2048 \
    rollout.tensor_model_parallel_size=1 rollout.gpu_memory_utilization=0.72 \
    rollout.max_num_batched_tokens=32768 rollout.log_prob_micro_batch_size_per_gpu=1 \
    2>&1 | tee "$OUT/logs/${dataset}_generation.log"

  "$PYTHON_BIN" -m verl.trainer.main_eval \
    data.path="$save_path" data.prompt_key=prompt data.response_key=responses \
    data.data_source_key=data_source data.reward_model_key=reward_model \
    2>&1 | tee "$OUT/logs/${dataset}_eval.log"
done

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/summarize_paper_eval.py --run-root "$OUT" \
  --model-size 1.5b --paper-reference "$PAPER_REFERENCE" --method-label "$METHOD_LABEL" \
  | tee "$OUT/logs/summary.log"
echo "Evaluation complete: $OUT/paper_comparison.md"
