#!/usr/bin/env python3
"""Normalize legacy subproblem data sources to registered math reward sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LEGACY_SUFFIX = "::verifiable_subproblem"


def normalize_data_source(value: Any) -> Any:
    if isinstance(value, str) and value.endswith(LEGACY_SUFFIX):
        return value[: -len(LEGACY_SUFFIX)]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(args.input)
    if "data_source" not in frame:
        raise ValueError("Training parquet has no data_source column")
    before = frame["data_source"].astype(str)
    frame["data_source"] = frame["data_source"].map(normalize_data_source)
    after = frame["data_source"].astype(str)
    remaining = int(after.str.endswith(LEGACY_SUFFIX).sum())
    if remaining:
        raise ValueError(f"Legacy reward sources remain after normalization: {remaining}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(frame),
        "normalized_rows": int((before != after).sum()),
        "data_sources": sorted(set(after.tolist())),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
