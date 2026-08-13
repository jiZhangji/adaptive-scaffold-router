#!/usr/bin/env python3
"""Build a 1:1 root/subproblem GRPO parquet from calibrated candidates."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from capability_scaffold import stable_question_key


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ground_truth(candidate: dict[str, Any]) -> str:
    value = str(candidate["subproblem_answer"]).strip()
    if value.startswith("\\boxed{") and value.endswith("}"):
        value = value[len("\\boxed{") : -1]
    return value


def _subproblem_row(root: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(root)
    question = str(candidate["subproblem"])
    answer = _ground_truth(candidate)
    messages = [{"role": "user", "content": question}]
    # Current verl datasets use `prompt`; Scaf data additionally carries
    # `question`.  Updating both keeps the verifier and tokenizer views aligned.
    row["question"] = question
    original_prompt = row.get("prompt")
    if hasattr(original_prompt, "tolist"):
        import numpy as np

        row["prompt"] = np.asarray(messages, dtype=object)
    else:
        row["prompt"] = messages
    reward_model = copy.deepcopy(row.get("reward_model") or {})
    reward_model["ground_truth"] = answer
    reward_model.setdefault("style", "rule")
    row["reward_model"] = reward_model
    # Keep the physical dtype of the official `id` column.  The unique child
    # identity lives in extra_info, avoiding a mixed int/string parquet column.
    row["id"] = root.get("id")
    row["data_source"] = f"{root.get('data_source', 'math')}::verifiable_subproblem"
    extra = copy.deepcopy(row.get("extra_info") or {})
    extra.update(
        {
            "is_subproblem": True,
            "parent_source_id": str(candidate.get("source_id", "")),
            "parent_question": str(candidate["question"]),
            "subproblem_q": float(candidate["success_probability"]),
            "subproblem_key": stable_question_key(question),
        }
    )
    row["extra_info"] = extra
    for column in (
        "knowledge_components_parts",
        "planning_skeleton_parts",
        "solution_breakdown_parts",
    ):
        if column in row:
            original = row[column]
            if hasattr(original, "tolist"):
                import numpy as np

                row[column] = np.asarray([], dtype=object)
            else:
                row[column] = []
    for column in ("solution", "model_think", "model_answer", "parsed_output"):
        if column in row:
            row[column] = None
    if "accuracy" in row:
        row["accuracy"] = 0.0
    return row


def build_mixed_rows(
    root_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    seed: int,
    max_pairs: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [item for item in candidates if bool(item.get("trainable", True))]
    if max_pairs is not None:
        candidates = candidates[:max_pairs]
    by_question: dict[str, dict[str, Any]] = {}
    for row in root_rows:
        by_question[stable_question_key(str(row["question"]))] = row

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing = 0
    for candidate in candidates:
        root = by_question.get(stable_question_key(str(candidate["question"])))
        if root is None:
            missing += 1
            continue
        pairs.append((copy.deepcopy(root), _subproblem_row(root, candidate)))
    if not pairs:
        raise ValueError("No calibrated candidates matched the source parquet")

    # Interleave the two views so every local region of the shuffled training
    # stream retains the intended 1:1 pressure on root and prerequisite.
    rng = random.Random(seed)
    rng.shuffle(pairs)
    mixed = [row for pair in pairs for row in pair]
    summary = {
        "pairs": len(pairs),
        "root_rows": len(pairs),
        "subproblem_rows": len(pairs),
        "total_rows": len(mixed),
        "candidate_rows_not_matched": missing,
        "mix_ratio": "1:1",
    }
    return mixed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root-output",
        type=Path,
        default=None,
        help="Optional matched root-only baseline parquet.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(args.source_data)
    mixed, summary = build_mixed_rows(
        frame.to_dict(orient="records"),
        _read_candidates(args.candidates),
        args.seed,
        args.max_pairs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mixed).to_parquet(args.output, index=False)
    if args.root_output is not None:
        args.root_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mixed[::2]).to_parquet(args.root_output, index=False)
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Mixed training data: {args.output}")
    if args.root_output is not None:
        print(f"Matched root-only data: {args.root_output}")


if __name__ == "__main__":
    main()
