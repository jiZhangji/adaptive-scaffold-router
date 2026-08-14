#!/usr/bin/env python3
"""Calibrate existing DeepSeek K/P/C candidates with the student model.

This is a resumable pilot-data builder.  It estimates each candidate's solve
probability from repeated student rollouts, keeps candidates in the informative
q band, and deterministically selects at most one candidate per root.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from frontier_probe import SYSTEM_PROMPT, build_math_verifier
from metaask_probe import LocalGenerator, chat_prompt


DIMENSIONS = ("knowledge", "planning", "calculation")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def select_complete_roots(
    rows: list[dict[str, Any]], root_limit: int, seed: int
) -> list[dict[str, Any]]:
    by_root: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        root_id = str(row.get("root_id", ""))
        dimension = str(row.get("dimension", ""))
        if root_id and dimension in DIMENSIONS:
            by_root[root_id][dimension] = row
    complete = sorted(
        root_id
        for root_id, candidates in by_root.items()
        if all(dimension in candidates for dimension in DIMENSIONS)
    )
    rng = random.Random(seed)
    rng.shuffle(complete)
    if root_limit > 0:
        complete = complete[:root_limit]
    selected = []
    for root_id in complete:
        selected.extend(by_root[root_id][dimension] for dimension in DIMENSIONS)
    return selected


def choose_training_candidates(
    candidates: list[dict[str, Any]],
    q_by_id: dict[str, float],
    q_low: float,
    q_high: float,
) -> list[dict[str, Any]]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dimension_rank = {name: index for index, name in enumerate(DIMENSIONS)}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        q = q_by_id.get(candidate_id)
        if q is None or not q_low <= q <= q_high:
            continue
        item = dict(candidate)
        item["success_probability"] = q
        item["trainable"] = True
        by_root[str(candidate["root_id"])].append(item)

    chosen = []
    for root_id in sorted(by_root):
        options = by_root[root_id]
        chosen.append(
            min(
                options,
                key=lambda item: (
                    abs(float(item["success_probability"]) - 0.5),
                    len(str(item["subproblem"])),
                    dimension_rank.get(str(item.get("dimension")), len(DIMENSIONS)),
                    str(item["id"]),
                ),
            )
        )
    return chosen


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    if not 0 <= args.q_low <= args.q_high <= 1:
        raise ValueError("q band must satisfy 0 <= q_low <= q_high <= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_complete_roots(
        read_jsonl(args.candidates), args.root_limit, args.seed
    )
    if not candidates:
        raise ValueError("No complete K/P/C roots were found")

    selected_path = args.output_dir / "selected_candidates.jsonl"
    write_jsonl(selected_path, candidates)
    result_path = args.output_dir / "subproblem_results.jsonl"
    existing = read_jsonl(result_path) if result_path.exists() else []
    completed = {
        (str(row["id"]), int(row["sample_index"])) for row in existing
    }

    generator = LocalGenerator(args)
    verifier = build_math_verifier()
    jobs = [
        {**candidate, "sample_index": sample_index}
        for candidate in candidates
        for sample_index in range(args.samples_per_candidate)
        if (str(candidate["id"]), sample_index) not in completed
    ]
    print(
        f"roots={len(candidates) // len(DIMENSIONS)} candidates={len(candidates)} "
        f"completed_rollouts={len(completed)} remaining_rollouts={len(jobs)}",
        flush=True,
    )

    for start in range(0, len(jobs), args.job_chunk_size):
        chunk = jobs[start : start + args.job_chunk_size]
        prompts = [
            chat_prompt(generator.tokenizer, SYSTEM_PROMPT, str(job["subproblem"]))
            for job in chunk
        ]
        outputs = generator.generate(
            prompts,
            args.max_new_tokens,
            args.seed + 10_000 + start,
            stop_after_boxed=args.stop_after_boxed,
        )
        for job, output in zip(chunk, outputs):
            record = {
                "id": str(job["id"]),
                "root_id": str(job["root_id"]),
                "dimension": str(job["dimension"]),
                "sample_index": int(job["sample_index"]),
                "subproblem": str(job["subproblem"]),
                "reference": str(job["subproblem_answer"]),
                "correct": verifier(output["text"], str(job["subproblem_answer"])),
                **output,
            }
            append_jsonl(result_path, record)
            existing.append(record)
        print(f"processed={min(start + len(chunk), len(jobs))}/{len(jobs)}", flush=True)

    selected_ids = {str(candidate["id"]) for candidate in candidates}
    relevant_results = [row for row in existing if str(row.get("id")) in selected_ids]
    by_id: dict[str, list[bool]] = defaultdict(list)
    for row in relevant_results:
        by_id[str(row["id"])].append(bool(row["correct"]))
    q_by_id = {
        candidate_id: sum(values) / len(values)
        for candidate_id, values in by_id.items()
        if len(values) >= args.samples_per_candidate
    }

    calibrated = []
    for candidate in candidates:
        item = dict(candidate)
        q = q_by_id.get(str(candidate["id"]))
        item["success_probability"] = q
        item["in_q_band"] = q is not None and args.q_low <= q <= args.q_high
        calibrated.append(item)
    training = choose_training_candidates(candidates, q_by_id, args.q_low, args.q_high)
    write_jsonl(args.output_dir / "calibrated_candidates.jsonl", calibrated)
    write_jsonl(args.output_dir / "training_candidates.jsonl", training)

    q_histogram = Counter(str(q) for q in q_by_id.values())
    dimension_summary = {}
    for dimension in DIMENSIONS:
        ids = [str(row["id"]) for row in candidates if row["dimension"] == dimension]
        values = [q_by_id[item_id] for item_id in ids if item_id in q_by_id]
        dimension_summary[dimension] = {
            "candidates": len(values),
            "mean_q": sum(values) / len(values) if values else 0.0,
            "in_band": sum(args.q_low <= value <= args.q_high for value in values),
        }
    summary = {
        "selected_roots": len(candidates) // len(DIMENSIONS),
        "selected_candidates": len(candidates),
        "samples_per_candidate": args.samples_per_candidate,
        "completed_rollouts": len(relevant_results),
        "q_low": args.q_low,
        "q_high": args.q_high,
        "q_histogram": dict(sorted(q_histogram.items())),
        "dimension_summary": dimension_summary,
        "training_roots": len(training),
        "training_candidates": len(training),
        "model": str(args.model),
        "important_limitation": (
            "Pilot q calibration only; matched-random causal relevance filtering is not yet applied."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-limit", type=int, default=256)
    parser.add_argument("--samples-per-candidate", type=int, default=4)
    parser.add_argument("--q-low", type=float, default=0.25)
    parser.add_argument("--q-high", type=float, default=0.60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--job-chunk-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-after-boxed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
