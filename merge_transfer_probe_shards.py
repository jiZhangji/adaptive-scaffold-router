#!/usr/bin/env python3
"""Merge independently written transfer-probe shards by candidate ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    merged: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    for path in args.inputs:
        for row in read_jsonl(path):
            candidate_id = str(row["candidate_id"])
            if candidate_id in merged:
                duplicate_rows += 1
                continue
            merged[candidate_id] = row

    expected_ids: set[str] = set()
    if args.candidates:
        expected_ids = {
            str(row.get("id", row.get("candidate_id", "")))
            for row in read_jsonl(args.candidates)
        }
        expected_ids.discard("")
    missing = sorted(expected_ids - set(merged))

    if args.require_complete and missing:
        raise SystemExit(f"Missing {len(missing)} candidate results")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for candidate_id in sorted(merged):
            handle.write(json.dumps(merged[candidate_id], ensure_ascii=False) + "\n")

    summary = {
        "input_files": [str(path) for path in args.inputs],
        "unique_candidates": len(merged),
        "duplicate_rows_ignored": duplicate_rows,
        "expected_candidates": len(expected_ids) if expected_ids else None,
        "missing_candidates": len(missing) if expected_ids else None,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
