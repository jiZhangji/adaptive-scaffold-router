#!/usr/bin/env python3
"""Restrict original DeepSeek K/P/C candidates to the matched training roots."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from calibrate_helpful_subproblems import minimal_plan, read_jsonl


DIMENSIONS = {"knowledge", "planning", "calculation"}


def run(args: argparse.Namespace) -> None:
    anchors = read_jsonl(args.anchors)
    roots = {str(row["root_id"]): row for row in anchors}
    original = read_jsonl(args.original_candidates)
    by_root: dict[str, list[dict]] = defaultdict(list)
    for row in original:
        root_id = str(row.get("root_id", ""))
        dimension = str(row.get("dimension", ""))
        if root_id not in roots or dimension not in DIMENSIONS:
            continue
        row = dict(row)
        row["candidate_origin"] = "original_deepseek_three_candidate_set"
        row["minimal_plan"] = str(row.get("minimal_plan") or minimal_plan(row, 12))
        if row["minimal_plan"]:
            by_root[root_id].append(row)

    complete = {
        root_id: rows
        for root_id, rows in by_root.items()
        if DIMENSIONS.issubset({str(row["dimension"]) for row in rows})
    }
    if len(complete) < args.min_complete_roots:
        raise ValueError(
            f"Only {len(complete)} matched roots have all K/P/C candidates; "
            f"required at least {args.min_complete_roots}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for root_id in sorted(complete):
        selected_by_dimension = {}
        for row in complete[root_id]:
            selected_by_dimension.setdefault(str(row["dimension"]), row)
        output_rows.extend(selected_by_dimension[dimension] for dimension in sorted(DIMENSIONS))
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "anchor_roots": len(roots),
        "original_candidate_rows": len(original),
        "matched_complete_roots": len(complete),
        "output_rows": len(output_rows),
        "dimension_counts": dict(Counter(str(row["dimension"]) for row in output_rows)),
        "source": str(args.original_candidates),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--original-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-complete-roots", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
