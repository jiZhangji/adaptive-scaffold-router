#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/DeepScaleR}"
FILE_NAME="${FILE_NAME:-Qwen2.5-Math-1.5B.parquet}"
EXPECTED_SHA256="${EXPECTED_SHA256:-e20a6bd8e1e01ddc02a708dacac94dca406c0053de092a076e4f3910127d213e}"
BASE_URL="https://huggingface.co/datasets/hkuzxc/scaf-grpo-dataset/resolve/main"
TARGET="$DATA_DIR/$FILE_NAME"

mkdir -p "$DATA_DIR"

if [[ ! -f "$TARGET" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 3 \
      "$BASE_URL/$FILE_NAME?download=true" -o "$TARGET"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 -O "$TARGET" "$BASE_URL/$FILE_NAME?download=true"
  else
    echo "Install curl or wget before downloading the dataset." >&2
    exit 1
  fi
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA256="$(sha256sum "$TARGET" | awk '{print $1}')"
  if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "SHA256 mismatch for $TARGET" >&2
    echo "expected: $EXPECTED_SHA256" >&2
    echo "actual:   $ACTUAL_SHA256" >&2
    exit 1
  fi
fi

echo "Dataset ready: $TARGET"
