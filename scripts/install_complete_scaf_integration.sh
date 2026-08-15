#!/usr/bin/env bash
set -euo pipefail

: "${SCAF_REPO:?Set SCAF_REPO to the offline Scaf-GRPO checkout}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINER="$SCAF_REPO/hint_mix_grpo/trainer/ray_trainer.py"

cp "$PROJECT_ROOT/capability_scaffold.py" "$SCAF_REPO/capability_scaffold.py"
cp "$PROJECT_ROOT/scaf_curriculum_adapter.py" "$SCAF_REPO/scaf_curriculum_adapter.py"
cp "$PROJECT_ROOT/scaf_integration/curriculum_runtime.py" \
  "$SCAF_REPO/hint_mix_grpo/curriculum_runtime.py"

cd "$SCAF_REPO"

apply_once() {
  local marker="$1" patch_file="$2"
  if rg -q "$marker" "$TRAINER"; then
    echo "Already installed: $marker"
  else
    patch --forward -p1 < "$PROJECT_ROOT/$patch_file"
  fi
}

apply_once "curriculum_manifest = load_optional_manifest" \
  scaf_integration/ray_trainer_curriculum.patch
apply_once "curriculum_off_context = bool" \
  scaf_integration/ray_trainer_off_context.patch
# Older installations used len(batch.batch), which returns the number of tensor
# fields instead of the response batch dimension.  Apply the one-line fix only
# when that old expression is still present.
if rg -q 'len\(batch\.batch\), dtype=torch\.bool' "$TRAINER"; then
  patch --forward -p1 < "$PROJECT_ROOT/scaf_integration/ray_trainer_mask_fix.patch"
fi

python -m py_compile \
  capability_scaffold.py \
  scaf_curriculum_adapter.py \
  hint_mix_grpo/curriculum_runtime.py \
  hint_mix_grpo/trainer/ray_trainer.py

echo "Complete Scaf/gradient curriculum integration is ready: $SCAF_REPO"
