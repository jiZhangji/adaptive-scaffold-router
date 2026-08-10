#!/usr/bin/env bash
set -euo pipefail

: "${SCAF_REPO:?Set SCAF_REPO to a clean official Scaf-GRPO checkout}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cp "$PROJECT_ROOT/capability_scaffold.py" "$SCAF_REPO/capability_scaffold.py"
cp "$PROJECT_ROOT/scaf_curriculum_adapter.py" "$SCAF_REPO/scaf_curriculum_adapter.py"
cp "$PROJECT_ROOT/scaf_integration/curriculum_runtime.py" \
  "$SCAF_REPO/hint_mix_grpo/curriculum_runtime.py"

cd "$SCAF_REPO"
patch --forward -p1 < "$PROJECT_ROOT/scaf_integration/ray_trainer_curriculum.patch"
patch --forward -p1 < "$PROJECT_ROOT/scaf_integration/ray_trainer_off_context.patch"
patch --forward -p1 < "$PROJECT_ROOT/scaf_integration/ray_trainer_mask_fix.patch"

python -m py_compile \
  capability_scaffold.py \
  scaf_curriculum_adapter.py \
  hint_mix_grpo/curriculum_runtime.py \
  hint_mix_grpo/trainer/ray_trainer.py

echo "Capability-matched curriculum integration installed in $SCAF_REPO"
