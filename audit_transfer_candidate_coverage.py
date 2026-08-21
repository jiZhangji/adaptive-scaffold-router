#!/usr/bin/env python3
"""Audit safe mappings between training anchors and original K/P/C candidates."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DIMENSIONS = {"knowledge", "planning", "calculation"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\s+", "", text)
    return text.casefold()


def normalized_answer(value: object) -> str:
    text = normalized_text(value)
    text = text.replace("\\boxed{", "").replace("}", "")
    return text


def legacy_normalize(value: object) -> str:
    text = re.sub(r"\\boxed\s*\{(.*?)\}", r"\1", str(value or ""), flags=re.S)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def safe_minimal_plan(row: dict, max_words: int = 12) -> tuple[str, str]:
    """Try every existing field instead of rejecting after a leaky source_step."""
    sub_answer = legacy_normalize(row.get("subproblem_answer"))
    root_answer = legacy_normalize(row.get("reference"))
    for key in ("source_step", "relation", "subproblem"):
        value = " ".join(str(row.get(key, "")).split())
        plan = " ".join(re.findall(r"\S+", value)[:max_words]).strip(" .")
        normalized = legacy_normalize(plan)
        if not plan:
            continue
        if sub_answer and sub_answer in normalized:
            continue
        if root_answer and root_answer in normalized:
            continue
        return plan, key
    return "", "none"


def unique_index(rows: dict[str, dict], key_fn) -> dict[object, str]:
    buckets: dict[object, list[str]] = defaultdict(list)
    for root_id, row in rows.items():
        key = key_fn(row)
        if key and all(key if isinstance(key, tuple) else [key]):
            buckets[key].append(root_id)
    return {key: ids[0] for key, ids in buckets.items() if len(ids) == 1}


def run(args: argparse.Namespace) -> None:
    anchor_rows = read_jsonl(args.anchors)
    anchors = {str(row["root_id"]): row for row in anchor_rows}

    original_rows = read_jsonl(args.original_candidates)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in original_rows:
        if str(row.get("dimension", "")) in DIMENSIONS:
            grouped[str(row.get("root_id", ""))].append(row)
    complete_groups = {
        root_id: rows
        for root_id, rows in grouped.items()
        if DIMENSIONS.issubset({str(row.get("dimension", "")) for row in rows})
    }
    original_roots = {root_id: rows[0] for root_id, rows in complete_groups.items()}

    safe_plan_counts: Counter[str] = Counter()
    safe_plan_missing_rows = []
    for root_id in anchors:
        for row in complete_groups.get(root_id, []):
            plan, source = safe_minimal_plan(row)
            safe_plan_counts[source] += 1
            if not plan:
                safe_plan_missing_rows.append(
                    {
                        "root_id": root_id,
                        "dimension": row.get("dimension"),
                        "subproblem_answer": row.get("subproblem_answer"),
                        "reference": row.get("reference"),
                        "source_step": row.get("source_step"),
                        "relation": row.get("relation"),
                        "subproblem": row.get("subproblem"),
                    }
                )

    question_raw = unique_index(original_roots, lambda row: str(row.get("question", "")))
    question_norm = unique_index(original_roots, lambda row: normalized_text(row.get("question")))
    question_answer = unique_index(
        original_roots,
        lambda row: (
            normalized_text(row.get("question")),
            normalized_answer(row.get("reference")),
        ),
    )
    source_question = unique_index(
        original_roots,
        lambda row: (str(row.get("source_id", "")), normalized_text(row.get("question"))),
    )

    matched_original: set[str] = set()
    audit = []
    counts: Counter[str] = Counter()
    for anchor_id, anchor in anchors.items():
        target = None
        method = "unmatched"
        candidates = [
            ("root_id", anchor_id if anchor_id in complete_groups else None),
            ("question_raw_unique", question_raw.get(str(anchor.get("question", "")))),
            ("question_normalized_unique", question_norm.get(normalized_text(anchor.get("question")))),
            (
                "question_answer_unique",
                question_answer.get(
                    (
                        normalized_text(anchor.get("question")),
                        normalized_answer(anchor.get("reference")),
                    )
                ),
            ),
            (
                "source_question_unique",
                source_question.get(
                    (str(anchor.get("source_id", "")), normalized_text(anchor.get("question")))
                ),
            ),
        ]
        for candidate_method, candidate_id in candidates:
            if candidate_id and candidate_id not in matched_original:
                method = candidate_method
                target = candidate_id
                break
        if target:
            matched_original.add(target)
        counts[method] += 1
        audit.append(
            {
                "anchor_root_id": anchor_id,
                "matched_original_root_id": target,
                "match_method": method,
                "source_id": anchor.get("source_id"),
                "question": anchor.get("question"),
                "reference": anchor.get("reference"),
            }
        )

    report = {
        "anchor_roots": len(anchors),
        "complete_original_roots": len(complete_groups),
        "safe_matched_roots": len(anchors) - counts["unmatched"],
        "unmatched_roots": counts["unmatched"],
        "match_methods": dict(counts),
        "safe_plan_sources": dict(safe_plan_counts),
        "safe_plan_missing_rows": len(safe_plan_missing_rows),
        "safe_plan_missing_roots": len(
            {row["root_id"] for row in safe_plan_missing_rows}
        ),
        "safe_plan_missing_details": safe_plan_missing_rows,
        "policy": "Only exact root IDs or unique exact/Unicode-whitespace-normalized question mappings are accepted.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in audit:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--original-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
