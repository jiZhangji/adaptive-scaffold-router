#!/usr/bin/env python3
"""Measure whether selected subproblems create informative GRPO groups.

This probe is intentionally separate from root-relevance calibration.  It can
be run on the already selected DeepSeek candidates without regenerating the
teacher data or retraining any baseline.  Results are resumable at the
candidate/sample level and are consumed by the student-aware data builder.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from capability_scaffold import informative_group_probability
from calibrate_helpful_subproblems import append_jsonl, read_jsonl
from frontier_probe import SYSTEM_PROMPT, build_math_verifier
from metaask_probe import LocalGenerator, chat_prompt


def aggregate_learnability(
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    group_size: int,
) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_candidate[str(row["candidate_id"])].append(row)

    output = []
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        rows = by_candidate.get(candidate_id, [])
        probability = (
            sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0
        )
        output.append(
            {
                "root_id": str(candidate["root_id"]),
                "candidate_id": candidate_id,
                "dimension": str(candidate.get("dimension", "")),
                "p_sub": probability,
                "contrast_score": informative_group_probability(probability, group_size),
                "num_samples": len(rows),
                "correct_samples": sum(bool(row["correct"]) for row in rows),
            }
        )
    return output


def run(args: argparse.Namespace) -> None:
    if args.samples < 1 or args.group_size < 2:
        raise ValueError("samples must be positive and group_size must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "subproblem_rollouts.jsonl"
    summary_path = args.output_dir / "subproblem_learnability.jsonl"

    candidates = read_jsonl(args.candidates)
    existing = read_jsonl(result_path) if result_path.exists() else []
    completed = {
        (str(row["candidate_id"]), int(row["sample_index"])) for row in existing
    }

    jobs: list[tuple[dict[str, Any], int]] = []
    for candidate in candidates:
        for sample_index in range(args.samples):
            key = (str(candidate["id"]), sample_index)
            if key not in completed:
                jobs.append((candidate, sample_index))

    print(
        f"candidates={len(candidates)} completed_rollouts={len(existing)} "
        f"pending_rollouts={len(jobs)}",
        flush=True,
    )
    if jobs:
        generator = LocalGenerator(args)
        verifier = build_math_verifier()
        for start in range(0, len(jobs), args.job_chunk_size):
            chunk = jobs[start : start + args.job_chunk_size]
            prompts = [
                chat_prompt(generator.tokenizer, SYSTEM_PROMPT, str(candidate["subproblem"]))
                for candidate, _ in chunk
            ]
            outputs = generator.generate(
                prompts,
                args.max_new_tokens,
                args.seed + start,
                stop_after_boxed=args.stop_after_boxed,
            )
            for (candidate, sample_index), output in zip(chunk, outputs):
                answer = str(candidate["subproblem_answer"])
                row = {
                    "root_id": str(candidate["root_id"]),
                    "candidate_id": str(candidate["id"]),
                    "dimension": str(candidate.get("dimension", "")),
                    "sample_index": sample_index,
                    "question": str(candidate["subproblem"]),
                    "reference": answer,
                    "correct": verifier(output["text"], answer),
                    **output,
                }
                append_jsonl(result_path, row)
                existing.append(row)
            print(
                f"processed={min(start + len(chunk), len(jobs))}/{len(jobs)}",
                flush=True,
            )

    summaries = aggregate_learnability(candidates, existing, args.group_size)
    with summary_path.open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "candidates": len(candidates),
        "rollouts": len(existing),
        "samples_per_candidate": args.samples,
        "group_size": args.group_size,
        "informative_candidates": sum(
            0.0 < float(row["p_sub"]) < 1.0 for row in summaries
        ),
        "mean_p_sub": (
            sum(float(row["p_sub"]) for row in summaries) / len(summaries)
            if summaries
            else 0.0
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--job-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-after-boxed", action="store_true")
    parser.add_argument("--stop-check-interval", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
