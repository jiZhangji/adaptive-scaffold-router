#!/usr/bin/env python3
"""Build matched datasets and a fading-plan curriculum manifest."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from build_subproblem_train_data import _read_candidates, build_mixed_rows
from capability_scaffold import stable_question_key


def extra_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    return copy.deepcopy(value or {})


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = _read_candidates(args.candidates)
    if not candidates:
        raise ValueError("No q_help-calibrated candidates were selected")
    frame = pd.read_parquet(args.source_data)
    rows = frame.to_dict(orient="records")
    mixed, mixed_summary = build_mixed_rows(rows, candidates, args.seed, None)
    matched_roots = mixed[::2]

    candidate_by_key = {
        stable_question_key(str(candidate["question"])): candidate
        for candidate in candidates
    }
    proposed_rows = []
    manifest = []
    for root in matched_roots:
        row = copy.deepcopy(root)
        question = str(row["question"])
        candidate = candidate_by_key[stable_question_key(question)]
        plan = str(candidate["minimal_plan"])
        original = row.get("planning_skeleton_parts")
        if hasattr(original, "tolist"):
            import numpy as np

            row["planning_skeleton_parts"] = np.asarray([plan], dtype=object)
            row["knowledge_components_parts"] = np.asarray([], dtype=object)
            row["solution_breakdown_parts"] = np.asarray([], dtype=object)
        else:
            row["planning_skeleton_parts"] = [plan]
            row["knowledge_components_parts"] = []
            row["solution_breakdown_parts"] = []
        extra = extra_dict(row.get("extra_info"))
        extra.update(
            {
                "minimal_plan": plan,
                "q_help": float(candidate["success_probability"]),
                "no_help_probability": float(candidate["no_help_probability"]),
                "random_plan_probability": float(candidate["random_plan_probability"]),
            }
        )
        row["extra_info"] = extra
        proposed_rows.append(row)
        manifest.append(
            {
                "root_id": str(candidate["root_id"]),
                "question_key": stable_question_key(question),
                "phase": "guided_root",
                "reason": "Answer-free minimal plan places root success in the learning band and beats controls.",
                "root_success": float(candidate["no_help_probability"]),
                "informative_probability": 0.0,
                "selected_scaffold": "planning@100",
                "selected_scaffold_success": float(candidate["success_probability"]),
                "active_subproblem_ids": [str(candidate["id"])],
                "unresolved_subproblem_ids": [],
                "prerequisite_mastery": 1.0,
                "diagnostics": {
                    "random_plan_probability": float(candidate["random_plan_probability"]),
                    "gain_over_no_help": float(candidate["gain_over_no_help"]),
                    "gain_over_random": float(candidate["gain_over_random"]),
                    "minimal_plan": plan,
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matched_roots).to_parquet(args.output_dir / "root_train.parquet", index=False)
    pd.DataFrame(mixed).to_parquet(args.output_dir / "mixed_train.parquet", index=False)
    pd.DataFrame(proposed_rows).to_parquet(
        args.output_dir / "proposed_root_train.parquet", index=False
    )
    with (args.output_dir / "curriculum.jsonl").open("w", encoding="utf-8") as handle:
        for decision in manifest:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    summary = {
        **mixed_summary,
        "proposed_root_rows": len(proposed_rows),
        "curriculum_rows": len(manifest),
        "q_definition": "P(root correct | answer-free minimal plan)",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
