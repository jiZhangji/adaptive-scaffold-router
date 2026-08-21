#!/usr/bin/env bash
set -Eeuo pipefail

# Fetch and strictly validate the released TD-Grokking parquet files.
# The public GitHub repository currently contains the code and manifest but
# omits *.parquet.  This script never treats a README/manifest-only checkout as
# usable data.  A user-supplied archive URL or pre-populated data directory is
# accepted and verified with the released row counts and SHA256 hashes.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/outputs/td_grokking_artifact}"
CODE_DIR="${CODE_DIR:-$ARTIFACT_ROOT/code}"
DATA_DIR="${TD_DATA_DIR:-$CODE_DIR/data/DeepScaleR-hard}"
ARCHIVE_URL="${TD_GROKKING_ARCHIVE_URL:-}"
GITHUB_REPO="${TD_GROKKING_GITHUB_REPO:-https://github.com/BachOzean/TD-Grokking.git}"
OFFICIAL_REPO="${TD_GROKKING_OFFICIAL_REPO:-https://anonymous.4open.science/r/TD-Grokking-6567/}"
CONDA_PYTHON="${CONDA_PYTHON:-/home/powerleader/project/envs/scaf-grpo/bin/python}"

mkdir -p "$ARTIFACT_ROOT"

validate() {
  "$CONDA_PYTHON" - "$DATA_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"missing manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
missing = []
for view, expected in manifest["views"].items():
    path = root / Path(expected["path"]).name
    if not path.is_file():
        missing.append(str(path))
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise SystemExit(f"sha256 mismatch for {path}: {digest}")
    rows = len(pd.read_parquet(path))
    if rows != int(expected["rows"]):
        raise SystemExit(f"row-count mismatch for {path}: {rows}")
if missing:
    raise SystemExit("missing released parquet files:\n" + "\n".join(missing))
print(json.dumps({"status": "validated", "data_dir": str(root)}, indent=2))
PY
}

if validate 2>/dev/null; then
  exit 0
fi

if [[ -n "$ARCHIVE_URL" ]]; then
  archive="$ARTIFACT_ROOT/td_grokking_artifact.zip"
  echo "Downloading user-supplied TD-Grokking archive."
  curl -fL --retry 5 --connect-timeout 20 -o "$archive" "$ARCHIVE_URL"
  unpacked="$ARTIFACT_ROOT/unpacked_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$unpacked"
  unzip -q "$archive" -d "$unpacked"
  found="$(find "$unpacked" -type f -name sub_only.parquet -print -quit)"
  [[ -n "$found" ]] || { echo "Archive contains no sub_only.parquet" >&2; exit 42; }
  DATA_DIR="$(dirname "$found")"
  validate
  printf '%s\n' "$DATA_DIR" > "$ARTIFACT_ROOT/data_dir.txt"
  exit 0
fi

if [[ ! -d "$CODE_DIR/.git" ]]; then
  echo "Trying the paper's anonymous artifact checkout."
  if ! git clone --depth 1 "$OFFICIAL_REPO" "$CODE_DIR"; then
    if [[ -e "$CODE_DIR" ]]; then
      mv "$CODE_DIR" "$CODE_DIR.failed.$(date +%Y%m%d_%H%M%S)"
    fi
    echo "Anonymous artifact is unavailable; cloning the public code mirror."
    git clone --depth 1 "$GITHUB_REPO" "$CODE_DIR"
  fi
fi

if validate; then
  printf '%s\n' "$DATA_DIR" > "$ARTIFACT_ROOT/data_dir.txt"
  exit 0
fi

cat >&2 <<EOF
TD-Grokking code was downloaded, but the released parquet data is absent.
The public checkout contains only README.md and manifest.json under:
  $DATA_DIR

Formal training was not started. Supply a verified archive through
TD_GROKKING_ARCHIVE_URL or copy root_only.parquet, sub_only.parquet, and
mixed.parquet into that directory, then rerun this script.
EOF
exit 42
