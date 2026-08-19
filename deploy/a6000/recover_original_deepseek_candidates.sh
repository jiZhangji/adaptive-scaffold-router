#!/usr/bin/env bash
set -Eeuo pipefail

INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
ENV_PREFIX="${ENV_PREFIX:-$INSTALL_ROOT/envs/scaf-grpo}"
PROJECT_ROOT="${PROJECT_ROOT:-$INSTALL_ROOT/adaptive-scaffold-router}"
MS_REPO_ID="${MS_REPO_ID:-shimian123/student-aware-grpo-migration}"
REMOTE_PREFIX="${REMOTE_PREFIX:-migration/full_outputs_supplement_stream_20260818/chunks}"
WORK_ROOT="${WORK_ROOT:-$INSTALL_ROOT/recover_deepseek}"
TARGET_REL="${TARGET_REL:-outputs/deepseek_zero_reward_subproblems_all/candidates.jsonl}"
TARGET="$PROJECT_ROOT/$TARGET_REL"
MAX_PART="${MAX_PART:-27}"
MS_CLI="$ENV_PREFIX/bin/modelscope"

mkdir -p "$WORK_ROOT" "$(dirname "$TARGET")"

for index in $(seq 0 "$MAX_PART"); do
  part="$(printf 'adaptive-scaffold-router-outputs-full.tar.zst.part-%04d' "$index")"
  local_part="$WORK_ROOT/$REMOTE_PREFIX/$part"
  if [[ ! -s "$local_part" ]]; then
    "$MS_CLI" download "$MS_REPO_ID" \
      --include "$REMOTE_PREFIX/$part" --local-dir "$WORK_ROOT"
  fi

  echo "Trying recovery with parts 0000..$(printf '%04d' "$index")"
  set +e
  cat "$WORK_ROOT/$REMOTE_PREFIX"/adaptive-scaffold-router-outputs-full.tar.zst.part-* |
    zstd -dc 2>/dev/null |
    tar -xOf - "$TARGET_REL" > "$TARGET.tmp" 2>/dev/null
  status=$?
  set -e
  if [[ -s "$TARGET.tmp" ]] && "$ENV_PREFIX/bin/python" - "$TARGET.tmp" <<'PY'
import json, sys
from collections import Counter
p=sys.argv[1]
rows=[json.loads(line) for line in open(p, encoding='utf-8') if line.strip()]
dims=Counter(str(row.get('dimension')) for row in rows)
roots=len({str(row.get('root_id')) for row in rows})
if not rows or not {'knowledge','planning','calculation'}.issubset(dims):
    raise SystemExit('Recovered file is incomplete')
print({'rows': len(rows), 'roots': roots, 'dimensions': dict(dims)})
PY
  then
    mv "$TARGET.tmp" "$TARGET"
    echo "Recovered: $TARGET"
    exit 0
  fi
  rm -f "$TARGET.tmp"
done

echo "Original candidates were not present in the available archive prefix." >&2
exit 1
