#!/usr/bin/env python3
"""Build a short preconditioning stage followed by root-aligned training.

The builder reuses the already calibrated root/candidate pairs.  Candidates
below the configurable scaffold-usability threshold receive a short 1:1
root/subproblem preconditioning stage.  Every selected root then enters a
minimal-plan curriculum followed by root-only consolidation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from build_subproblem_train_data import _subproblem_row
from capability_scaffold import informative_group_probability, stable_question_key
from calibrate_helpful_subproblems import minimal_plan, read_jsonl


def extra_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    return copy.deepcopy(value or {})


def candidate_diagnostics(
    candidate: dict[str, Any],
    learnability: dict[str, Any] | None,
    group_size: int,
) -> dict[str, Any]:
    q_help = float(candidate["success_probability"])
    q_no = float(candidate["no_help_probability"])
    q_random = float(candidate["random_plan_probability"])
    relevance = q_help - max(q_no, q_random)
    p_sub = None if learnability is None else float(learnability["p_sub"])
    contrast = (
        informative_group_probability(p_sub, group_size)
        if p_sub is not None
        else None
    )
    final_score = max(relevance, 0.0) * (contrast if contrast is not None else 1.0)
    return {
        "q_help": q_help,
        "q_no": q_no,
        "q_random": q_random,
        "relevance_score": relevance,
        "p_sub": p_sub,
        "contrast_score": contrast,
        "final_score": final_score,
    }


def needs_preconditioning(
    diagnostics: dict[str, Any],
    scaffold_ready_threshold: float,
    contrast_min: float,
) -> bool:
    if float(diagnostics["q_help"]) >= scaffold_ready_threshold:
        return False
    p_sub = diagnostics.get("p_sub")
    if p_sub is not None and not 0.0 < float(p_sub) < 1.0:
        return False
    contrast = diagnostics.get("contrast_score")
    return contrast is None or float(contrast) >= contrast_min


def should_use_transfer_candidate(
    candidate: dict[str, Any],
    positive_transfer_only: bool,
    min_transfer_gain: float | None = None,
    min_post_update_probability: float | None = None,
) -> bool:
    if (
        not positive_transfer_only
        and min_transfer_gain is None
        and min_post_update_probability is None
    ):
        return True
    probe = candidate.get("transfer_probe", {})
    gain = probe.get("proxy_transfer_gain")
    post = probe.get("post_update_probability")
    if gain is None:
        return False
    if positive_transfer_only and float(gain) <= 0.0:
        return False
    if min_transfer_gain is not None and float(gain) < min_transfer_gain:
        return False
    if min_post_update_probability is not None:
        if post is None or float(post) < min_post_update_probability:
            return False
    return True


def make_scaffold_row(root: dict[str, Any], plan: str, metadata: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(root)
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
    extra.update(metadata)
    row["extra_info"] = extra
    return row


def run(args: argparse.Namespace) -> None:
    import pandas as pd

    if not 0.0 <= args.scaffold_ready_threshold <= 1.0:
        raise ValueError("scaffold_ready_threshold must be in [0, 1]")
    if not 0.0 <= args.contrast_min <= 1.0:
        raise ValueError("contrast_min must be in [0, 1]")

    candidates = read_jsonl(args.candidates)
    learnability_rows = read_jsonl(args.learnability) if args.learnability else []
    learnability = {
        str(row["candidate_id"]): row for row in learnability_rows
    }
    source = pd.read_parquet(args.source_data)
    by_question = {
        stable_question_key(str(row["question"])): row
        for row in source.to_dict(orient="records")
    }

    root_only_rows: list[dict[str, Any]] = []
    root_scaffold_rows: list[dict[str, Any]] = []
    precondition_rows: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    curriculum: list[dict[str, Any]] = []
    missing = 0
    missing_plan = 0
    transfer_enabled = 0
    root_only_controls = 0

    for candidate in candidates:
        root = by_question.get(stable_question_key(str(candidate["question"])))
        if root is None:
            missing += 1
            continue
        use_subproblem = should_use_transfer_candidate(
            candidate,
            args.positive_transfer_only,
            args.min_transfer_gain,
            args.min_post_update_probability,
        )
        plan = (
            str(candidate.get("minimal_plan") or minimal_plan(candidate, args.max_plan_words))
            if use_subproblem
            else ""
        )
        scaffold_available = bool(plan)
        if use_subproblem:
            transfer_enabled += 1
            missing_plan += int(not scaffold_available)
        else:
            root_only_controls += 1
        diagnostics = candidate_diagnostics(
            candidate,
            learnability.get(str(candidate["id"])),
            args.group_size,
        )
        precondition = use_subproblem and needs_preconditioning(
            diagnostics, args.scaffold_ready_threshold, args.contrast_min
        )
        stage = (
            "precondition"
            if precondition
            else ("root_scaffold" if scaffold_available else "root_only_control")
        )
        metadata = {
            "student_aware_stage": stage,
            "minimal_plan": plan,
            "selected_candidate_id": str(candidate["id"]),
            "selected_dimension": str(candidate.get("dimension", "")),
            "transfer_candidate_enabled": use_subproblem,
            "scaffold_available": scaffold_available,
            **diagnostics,
        }
        root_copy = copy.deepcopy(root)
        root_only_rows.append(root_copy)
        root_scaffold_rows.append(
            make_scaffold_row(root, plan, metadata)
            if scaffold_available
            else copy.deepcopy(root)
        )
        if precondition:
            precondition_rows.extend(
                [copy.deepcopy(root), _subproblem_row(root, {**candidate, "trainable": True})]
            )
        selected_pairs.append(
            {
                "root_id": str(candidate["root_id"]),
                "candidate_id": str(candidate["id"]),
                "dimension": str(candidate.get("dimension", "")),
                "subproblem": str(candidate["subproblem"]),
                "subproblem_answer": str(candidate["subproblem_answer"]),
                "minimal_plan": plan,
                "current_stage": stage,
                "transfer_candidate_enabled": use_subproblem,
                "scaffold_available": scaffold_available,
                **diagnostics,
            }
        )
        curriculum.append(
            {
                "root_id": str(candidate["root_id"]),
                "question_key": stable_question_key(str(root["question"])),
                "phase": "guided_root" if scaffold_available else "unguided_root",
                "reason": (
                    "Use the selected minimal scaffold when transfer is enabled and a safe "
                    "answer-free plan exists; otherwise retain an unguided root control."
                ),
                "root_success": float(diagnostics["q_no"]),
                "informative_probability": informative_group_probability(
                    float(diagnostics["q_no"]), args.group_size
                ),
                "selected_scaffold": "planning@100" if scaffold_available else None,
                "selected_scaffold_success": float(diagnostics["q_help"]),
                "active_subproblem_ids": (
                    [str(candidate["id"])] if use_subproblem else []
                ),
                "unresolved_subproblem_ids": (
                    [str(candidate["id"])] if precondition else []
                ),
                "prerequisite_mastery": 0.0 if precondition else 1.0,
                "diagnostics": metadata,
            }
        )

    if not root_only_rows:
        raise ValueError("No selected candidates matched the source data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(root_only_rows).to_parquet(
        args.output_dir / "root_only_train.parquet", index=False
    )
    pd.DataFrame(root_scaffold_rows).to_parquet(
        args.output_dir / "root_scaffold_train.parquet", index=False
    )
    if precondition_rows:
        pd.DataFrame(precondition_rows).to_parquet(
            args.output_dir / "precondition_train.parquet", index=False
        )
    with (args.output_dir / "selected_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "curriculum.jsonl").open("w", encoding="utf-8") as handle:
        for row in curriculum:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "selected_roots": len(root_only_rows),
        "precondition_roots": sum(
            row["current_stage"] == "precondition" for row in selected_pairs
        ),
        "root_scaffold_roots": sum(
            row["current_stage"] == "root_scaffold" for row in selected_pairs
        ),
        "precondition_rows": len(precondition_rows),
        "learnability_rows": len(learnability_rows),
        "missing_source_rows": missing,
        "missing_safe_plan_rows": missing_plan,
        "transfer_enabled_roots": transfer_enabled,
        "root_only_control_roots": root_only_controls,
        "positive_transfer_only": args.positive_transfer_only,
        "min_transfer_gain": args.min_transfer_gain,
        "min_post_update_probability": args.min_post_update_probability,
        "scaffold_ready_threshold": args.scaffold_ready_threshold,
        "contrast_min": args.contrast_min,
        "group_size": args.group_size,
        "calibration_note": (
            "p_sub/contrast are used when a learnability file is supplied; otherwise the "
            "existing q_help root-relevance calibration controls the preconditioning gate."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--learnability", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scaffold-ready-threshold", type=float, default=0.50)
    parser.add_argument("--contrast-min", type=float, default=0.0)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-plan-words", type=int, default=12)
    parser.add_argument("--positive-transfer-only", action="store_true")
    parser.add_argument("--min-transfer-gain", type=float, default=None)
    parser.add_argument("--min-post-update-probability", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
