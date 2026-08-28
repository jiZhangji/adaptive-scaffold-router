#!/usr/bin/env python3
"""Build matched 212-root data for Delta-only and RCST-gated Delta training."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from capability_scaffold import stable_question_key


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--all-selected", type=Path, required=True)
    parser.add_argument("--rcst-selected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-roots", type=int, default=212)
    args = parser.parse_args()

    import pandas as pd

    source = pd.read_parquet(args.source_data)
    by_question = {
        stable_question_key(str(row["question"])): row
        for row in source.to_dict(orient="records")
    }
    all_selected_raw = read_jsonl(args.all_selected)
    all_selected_by_root: dict[str, dict] = {}
    for candidate in all_selected_raw:
        root_id = str(candidate["root_id"])
        previous = all_selected_by_root.get(root_id)
        if previous is not None and str(previous["question"]) != str(candidate["question"]):
            raise ValueError(f"Inconsistent questions for root: {root_id}")
        all_selected_by_root.setdefault(root_id, candidate)
    all_selected = [all_selected_by_root[root_id] for root_id in sorted(all_selected_by_root)]
    rcst_rows = read_jsonl(args.rcst_selected)
    rcst_accepted = {
        str(row["root_id"]): row
        for row in rcst_rows
        if not bool(row.get("rcst_abstained", False))
    }

    def build(mode: str) -> list[dict]:
        rows = []
        seen = set()
        for candidate in all_selected:
            root_id = str(candidate["root_id"])
            if root_id in seen:
                raise ValueError(f"Duplicate root in all-selected file: {root_id}")
            seen.add(root_id)
            source_row = by_question.get(stable_question_key(str(candidate["question"])))
            if source_row is None:
                raise ValueError(f"Root not found in source parquet: {root_id}")
            row = copy.deepcopy(source_row)
            chosen = candidate if mode == "all" else rcst_accepted.get(root_id)
            extra = copy.deepcopy(row.get("extra_info") or {})
            extra.update(
                {
                    "bridge_root_id": root_id,
                    "bridge_mode": mode,
                    "bridge_enabled": chosen is not None,
                    "bridge_subproblem": str(chosen.get("subproblem", "")) if chosen else "",
                    "bridge_subproblem_answer": str(chosen.get("subproblem_answer", "")) if chosen else "",
                    "bridge_candidate_id": str(chosen.get("id", "")) if chosen else "",
                }
            )
            row["extra_info"] = extra
            rows.append(row)
        if len(rows) != args.expected_roots:
            raise ValueError(f"Expected {args.expected_roots} roots, found {len(rows)}")
        return rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = build("all")
    rcst_gate_rows = build("rcst_lcb")
    pd.DataFrame(all_rows).to_parquet(
        args.output_dir / f"delta_all_{args.expected_roots}.parquet", index=False
    )
    pd.DataFrame(rcst_gate_rows).to_parquet(
        args.output_dir / f"rcst_lcb_delta_{args.expected_roots}.parquet", index=False
    )
    summary = {
        "roots": args.expected_roots,
        "all_enabled": sum(bool((row.get("extra_info") or {}).get("bridge_enabled")) for row in all_rows),
        "rcst_enabled": sum(
            bool((row.get("extra_info") or {}).get("bridge_enabled")) for row in rcst_gate_rows
        ),
        "rcst_abstained": args.expected_roots - len(rcst_accepted),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
