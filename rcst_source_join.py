"""Unambiguous full-question joins; source IDs are provenance, never join keys."""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any


def question_key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty full question is required")
    return " ".join(value.split())


def source_index(rows: list[dict]) -> dict[str, list[dict]]:
    index = defaultdict(list)
    for row in rows:
        index[question_key(row["question"])].append(row)
    return dict(index)


def unique_source(index: dict, question: str) -> dict:
    matches = index.get(question_key(question), [])
    if len(matches) != 1:
        raise ValueError(f"Full question requires exactly one source match; got {len(matches)}")
    return copy.deepcopy(matches[0])


def attach_sources(roots: list[dict], source_rows: list[dict]) -> list[dict]:
    index, seen, output = source_index(source_rows), set(), []
    for root in roots:
        root_id = str(root["root_id"])
        if root["root_id"] is None or not root_id.strip() or root_id in seen:
            raise ValueError("Missing or duplicate root_id")
        seen.add(root_id)
        source = unique_source(index, root["question"])
        item = copy.deepcopy(root)
        for target, origin in (
            ("reference_solution", "solution"),
            ("knowledge_components", "knowledge_components_parts"),
            ("planning_skeleton", "planning_skeleton_parts"),
            ("solution_breakdown", "solution_breakdown_parts"),
        ):
            value = source.get(origin)
            item[target] = value.tolist() if hasattr(value, "tolist") else copy.deepcopy(value)
        item["source_join_audit"] = {
            "match_basis": "normalized_full_question",
            "source_id": str(source.get("id", "")),
            "mathematical_correctness_verified": False,
        }
        output.append(item)
    return output
