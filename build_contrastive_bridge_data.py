#!/usr/bin/env python3
"""Build RCST-LCB-gated data with a same-root low-transfer control subproblem."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from capability_scaffold import stable_question_key


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_same_root_controls(
    candidates: list[dict], aggregates: list[dict], positives: dict[str, dict]
) -> dict[str, dict]:
    candidates_by_root: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_root[str(candidate["root_id"])].append(candidate)
    scores = {
        str(row["candidate_id"]): float(row["gain_lcb"])
        for row in aggregates
    }
    controls: dict[str, dict] = {}
    for root_id, positive in positives.items():
        alternatives = [
            candidate
            for candidate in candidates_by_root[root_id]
            if str(candidate["id"]) != str(positive["id"])
            and str(candidate["id"]) in scores
        ]
        if not alternatives:
            raise ValueError(f"No scored control candidate for accepted root {root_id}")
        controls[root_id] = min(
            alternatives, key=lambda candidate: (scores[str(candidate["id"])], str(candidate["id"]))
        )
    return controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--rcst-selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-roots", type=int, default=212)
    args = parser.parse_args()

    import pandas as pd

    source = pd.read_parquet(args.source_data)
    source_by_question = {
        stable_question_key(str(row["question"])): row
        for row in source.to_dict(orient="records")
    }
    root_rows_by_id = {}
    for row in read_jsonl(args.roots):
        root_rows_by_id.setdefault(str(row["root_id"]), row)
    selected = read_jsonl(args.rcst_selected)
    positives = {
        str(row["root_id"]): row
        for row in selected
        if not bool(row.get("rcst_abstained", False))
    }
    controls = select_same_root_controls(
        read_jsonl(args.candidates), read_jsonl(args.aggregates), positives
    )

    rows = []
    for root_id in sorted(root_rows_by_id):
        root = root_rows_by_id[root_id]
        source_row = source_by_question.get(stable_question_key(str(root["question"])))
        if source_row is None:
            raise ValueError(f"Root not found in source parquet: {root_id}")
        row = copy.deepcopy(source_row)
        positive = positives.get(root_id)
        control = controls.get(root_id)
        extra = copy.deepcopy(row.get("extra_info") or {})
        extra.update(
            {
                "bridge_root_id": root_id,
                "bridge_mode": "rcst_lcb_same_root_contrastive",
                "bridge_enabled": positive is not None,
                "bridge_subproblem": str(positive.get("subproblem", "")) if positive else "",
                "bridge_subproblem_answer": str(positive.get("subproblem_answer", "")) if positive else "",
                "bridge_candidate_id": str(positive.get("id", "")) if positive else "",
                "bridge_control_subproblem": str(control.get("subproblem", "")) if control else "",
                "bridge_control_subproblem_answer": str(control.get("subproblem_answer", "")) if control else "",
                "bridge_control_candidate_id": str(control.get("id", "")) if control else "",
            }
        )
        row["extra_info"] = extra
        rows.append(row)

    if len(rows) != args.expected_roots:
        raise ValueError(f"Expected {args.expected_roots} roots, found {len(rows)}")
    if sum(bool(row["extra_info"]["bridge_enabled"]) for row in rows) != len(positives):
        raise AssertionError("Enabled root count does not match accepted RCST roots")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    summary = {
        "roots": len(rows),
        "bridge_enabled": len(positives),
        "bridge_abstained": len(rows) - len(positives),
        "control_policy": "lowest_same_root_transfer_lcb_excluding_selected",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
