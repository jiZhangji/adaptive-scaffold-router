#!/usr/bin/env python3
"""Align released TD-Grokking subproblems to this project's training roots.

The TD-Grokking release uses its own ``root_problem_id`` values.  A formal
comparison with our existing 212-root runs is only possible when those roots
can be matched by normalized question text.  This adapter deliberately fails
when the requested coverage is not met instead of silently mixing different
root pools.
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


def as_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def as_dict(value: Any) -> dict[str, Any]:
    value = as_python(value)
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def prompt_question(value: Any) -> str:
    value = as_python(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("content", "")).strip()
    if isinstance(value, (list, tuple)):
        messages = [as_python(item) for item in value]
        for message in reversed(messages):
            if isinstance(message, dict) and str(message.get("role", "")) == "user":
                return str(message.get("content", "")).strip()
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"]).strip()
    return ""


def row_prompt(row: dict[str, Any]) -> Any:
    value = row.get("prompt")
    return row.get("question") if value is None else value


def target_answer(row: dict[str, Any]) -> str:
    reward_model = as_dict(row.get("reward_model"))
    extra = as_dict(row.get("extra_info"))
    for value in (
        reward_model.get("ground_truth"),
        reward_model.get("reference"),
        extra.get("ground_truth"),
        extra.get("answer"),
        row.get("answer"),
    ):
        value = as_python(value)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def root_identifier(row: dict[str, Any]) -> str:
    extra = as_dict(row.get("extra_info"))
    return str(extra.get("root_problem_id") or row.get("problem_id") or "").strip()


def subproblem_identifier(row: dict[str, Any], fallback_index: int) -> str:
    extra = as_dict(row.get("extra_info"))
    return str(
        extra.get("subproblem_id")
        or row.get("problem_id")
        or f"subproblem_{fallback_index}"
    ).strip()


def run(args: argparse.Namespace) -> None:
    import pandas as pd

    anchors = read_jsonl(args.anchors)
    root_rows = pd.read_parquet(args.root_only).to_dict(orient="records")
    sub_rows = pd.read_parquet(args.sub_only).to_dict(orient="records")

    td_root_question: dict[str, str] = {}
    for row in root_rows:
        identifier = root_identifier(row)
        question = prompt_question(row_prompt(row))
        if identifier and question:
            td_root_question[identifier] = question

    anchor_by_question = {
        stable_question_key(str(row["question"])): row for row in anchors
    }
    td_to_anchor: dict[str, dict[str, Any]] = {}
    for identifier, question in td_root_question.items():
        anchor = anchor_by_question.get(stable_question_key(question))
        if anchor is not None:
            td_to_anchor[identifier] = anchor

    by_td_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_subproblems = 0
    for row in sub_rows:
        identifier = root_identifier(row)
        question = prompt_question(row_prompt(row))
        answer = target_answer(row)
        if not identifier or not question or not answer:
            invalid_subproblems += 1
            continue
        by_td_root[identifier].append(row)

    output_rows: list[dict[str, Any]] = []
    matched_roots: set[str] = set()
    per_root_counts: Counter[int] = Counter()
    same_answer_nodes = 0
    for td_root_id in sorted(td_to_anchor):
        anchor = td_to_anchor[td_root_id]
        children = by_td_root.get(td_root_id, [])
        valid_children: list[dict[str, Any]] = []
        for child in children:
            subproblem = prompt_question(row_prompt(child))
            answer = target_answer(child)
            if args.exclude_same_answer and answer == str(anchor.get("reference", "")).strip():
                same_answer_nodes += 1
                continue
            valid_children.append(child)
        if len(valid_children) < args.min_candidates_per_root:
            continue

        root_id = str(anchor["root_id"])
        matched_roots.add(root_id)
        per_root_counts[len(valid_children)] += 1
        fallback_plan = str(anchor.get("minimal_plan") or minimal_plan(anchor, 12))
        for index, child in enumerate(valid_children, start=1):
            subproblem_id = subproblem_identifier(child, index)
            candidate = copy.deepcopy(anchor)
            candidate.update(
                {
                    "id": f"{root_id}::td_grokking::{subproblem_id}",
                    "root_id": root_id,
                    "dimension": f"td_node_{index:02d}",
                    "subproblem": prompt_question(row_prompt(child)),
                    "subproblem_answer": target_answer(child),
                    "minimal_plan": fallback_plan,
                    "candidate_origin": "td_grokking_released_subproblem",
                    "td_root_problem_id": td_root_id,
                    "td_subproblem_id": subproblem_id,
                }
            )
            output_rows.append(candidate)

    if len(matched_roots) < args.require_matched_roots:
        raise ValueError(
            f"TD-Grokking covers only {len(matched_roots)} of {len(anchors)} anchor roots; "
            f"formal comparison requires at least {args.require_matched_roots}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "anchor_roots": len(anchors),
        "td_root_rows": len(root_rows),
        "td_subproblem_rows": len(sub_rows),
        "matched_roots": len(matched_roots),
        "output_candidates": len(output_rows),
        "invalid_subproblem_rows": invalid_subproblems,
        "excluded_same_answer_nodes": same_answer_nodes,
        "candidate_count_histogram": {
            str(key): value for key, value in sorted(per_root_counts.items())
        },
        "comparison_status": (
            "same_root_pool_ready"
            if len(matched_roots) == len(anchors)
            else "partial_overlap_not_directly_comparable"
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--root-only", type=Path, required=True)
    parser.add_argument("--sub-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-matched-roots", type=int, default=212)
    parser.add_argument("--min-candidates-per-root", type=int, default=2)
    parser.add_argument("--exclude-same-answer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
