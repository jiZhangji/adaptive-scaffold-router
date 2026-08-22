#!/usr/bin/env python3
"""Prepare the largest candidate-complete RCST training pool.

The DeepSeek artifact contains three standalone subproblems for each covered
zero-reward root, but it does not contain the legacy q-help calibration used by
the 212-root pilot.  This script therefore creates a synthetic fallback anchor
for every root.  The anchor is deliberately disabled for transfer: downstream
training uses it as a naked-root control unless replicated RCST probing accepts
one of the three real candidates.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from calibrate_helpful_subproblems import minimal_plan, read_jsonl
from capability_scaffold import stable_question_key


DIMENSIONS = ("calculation", "knowledge", "planning")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fallback_anchor(candidate: dict[str, Any]) -> dict[str, Any]:
    anchor = copy.deepcopy(candidate)
    anchor.update(
        {
            "minimal_plan": "",
            "success_probability": 0.0,
            "no_help_probability": 0.0,
            "random_plan_probability": 0.0,
            "gain_over_no_help": 0.0,
            "gain_over_random": 0.0,
            "trainable": True,
            "selection_policy": "synthetic_root_only_fallback",
            "anchor_origin": "uncalibrated_full_pool_root_only_fallback",
        }
    )
    return anchor


def run(args: argparse.Namespace) -> None:
    import pandas as pd

    raw = read_jsonl(args.raw_candidates)
    by_root: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_dimensions = 0
    for source_row in raw:
        row = copy.deepcopy(source_row)
        root_id = str(row.get("root_id", ""))
        dimension = str(row.get("dimension", ""))
        if not root_id or dimension not in DIMENSIONS:
            continue
        if dimension in by_root[root_id]:
            duplicate_dimensions += 1
            continue
        row["candidate_origin"] = "original_deepseek_three_candidate_set"
        row["minimal_plan"] = str(row.get("minimal_plan") or minimal_plan(row, 12))
        by_root[root_id][dimension] = row

    complete = {
        root_id: rows
        for root_id, rows in by_root.items()
        if all(dimension in rows for dimension in DIMENSIONS)
    }
    if len(complete) != args.expected_roots:
        raise ValueError(
            f"Expected {args.expected_roots} complete roots, found {len(complete)}"
        )

    source = pd.read_parquet(args.source_data)
    source_by_question: dict[str, dict[str, Any]] = {}
    duplicate_source_questions = 0
    for row in source.to_dict(orient="records"):
        key = stable_question_key(str(row["question"]))
        if key in source_by_question:
            duplicate_source_questions += 1
            continue
        source_by_question[key] = row

    candidates: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    missing_source: list[str] = []
    for root_id in sorted(complete):
        rows = complete[root_id]
        question = str(rows[DIMENSIONS[0]]["question"])
        source_row = source_by_question.get(stable_question_key(question))
        if source_row is None:
            missing_source.append(root_id)
            continue
        ordered = [rows[dimension] for dimension in DIMENSIONS]
        candidates.extend(ordered)
        # Planning is deterministic and human-readable, but it is never used as
        # a subproblem unless an RCST probe explicitly replaces this fallback.
        anchors.append(fallback_anchor(rows["planning"]))
        root_rows.append(copy.deepcopy(source_row))

    if missing_source:
        raise ValueError(
            f"{len(missing_source)} candidate roots do not match source data; "
            f"examples: {missing_source[:5]}"
        )
    expected_candidates = args.expected_roots * len(DIMENSIONS)
    if len(candidates) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} candidate rows, found {len(candidates)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "candidate_sets.jsonl", candidates)
    write_jsonl(args.output_dir / "fallback_anchors.jsonl", anchors)
    pd.DataFrame(root_rows).to_parquet(
        args.output_dir / "vanilla_root_train.parquet", index=False
    )

    summary = {
        "source_rows": len(source),
        "raw_candidate_rows": len(raw),
        "complete_roots": len(root_rows),
        "candidate_rows": len(candidates),
        "anchor_rows": len(anchors),
        "dimensions": dict(Counter(str(row["dimension"]) for row in candidates)),
        "candidates_per_root": len(DIMENSIONS),
        "empty_minimal_plan_rows": sum(
            not str(row.get("minimal_plan", "")).strip() for row in candidates
        ),
        "duplicate_candidate_dimensions_ignored": duplicate_dimensions,
        "duplicate_source_questions_ignored": duplicate_source_questions,
        "fallback_policy": "root-only unless replicated RCST LCB is positive",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-roots", type=int, default=1866)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
