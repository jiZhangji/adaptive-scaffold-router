#!/usr/bin/env python3
"""Select subproblems by the student's ability to use them on the root problem.

Unlike the earlier pilot, q is measured on the original root problem after a
short answer-free plan is supplied.  A matched plan from another root is used
as a causal control.  Sampling is sequential and stops for a root once a
candidate enters the learning band and beats both controls.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capability_scaffold import stable_question_key
from frontier_probe import SYSTEM_PROMPT, build_math_verifier
from metaask_probe import LocalGenerator, chat_prompt


DIMENSIONS = ("knowledge", "planning", "calculation")


@dataclass
class RootCalibrationState:
    root_id: str
    candidates: list[dict[str, Any]]
    question: str
    reference: str
    records: list[dict[str, Any]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def words(text: str) -> list[str]:
    return re.findall(r"\S+", " ".join(str(text).split()))


def normalize(text: str) -> str:
    text = re.sub(r"\\boxed\s*\{(.*?)\}", r"\1", str(text), flags=re.S)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def minimal_plan(candidate: dict[str, Any], max_words: int) -> str:
    for key in ("source_step", "relation", "subproblem"):
        value = " ".join(str(candidate.get(key, "")).split())
        if value:
            break
    else:
        return ""
    plan = " ".join(words(value)[:max_words]).strip(" .")
    answer = normalize(str(candidate.get("subproblem_answer", "")))
    root_answer = normalize(str(candidate.get("reference", "")))
    normalized_plan = normalize(plan)
    if not plan or (answer and answer in normalized_plan) or (
        root_answer and root_answer in normalized_plan
    ):
        return ""
    return plan


def root_prompt(question: str, plan: str = "") -> str:
    if not plan:
        return question
    return (
        f"{question}\n\nMinimal plan (no answer is provided): {plan}\n"
        "Infer every missing step yourself and solve the original problem."
    )


def mean_correct(rows: list[dict[str, Any]]) -> float:
    return sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0


def choose_candidate(
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    q_low: float,
    q_high: float,
    min_gain: float,
    min_samples: int = 1,
) -> dict[str, Any] | None:
    baseline_rows = [row for row in records if row["variant"] == "no_help"]
    if len(baseline_rows) < min_samples:
        return None
    baseline = mean_correct(baseline_rows)
    choices = []
    for candidate in candidates:
        candidate_rows = [
            row for row in records if row.get("candidate_id") == candidate["id"]
        ]
        relevant_rows = [
            row for row in candidate_rows if row["variant"] == "relevant_plan"
        ]
        random_rows = [
            row for row in candidate_rows if row["variant"] == "random_plan"
        ]
        if len(relevant_rows) < min_samples or len(random_rows) < min_samples:
            continue
        relevant = mean_correct(relevant_rows)
        random_q = mean_correct(random_rows)
        if (
            q_low <= relevant <= q_high
            and relevant > baseline + min_gain
            and relevant > random_q + min_gain
        ):
            choices.append(
                {
                    **candidate,
                    "success_probability": relevant,
                    "no_help_probability": baseline,
                    "random_plan_probability": random_q,
                    "gain_over_no_help": relevant - baseline,
                    "gain_over_random": relevant - random_q,
                    "trainable": True,
                }
            )
    if not choices:
        return None
    midpoint = (q_low + q_high) / 2
    return min(
        choices,
        key=lambda row: (
            len(words(row["minimal_plan"])),
            abs(float(row["success_probability"]) - midpoint),
            DIMENSIONS.index(str(row.get("dimension", "calculation"))),
            str(row["id"]),
        ),
    )


def expected_sample_keys(
    root_id: str,
    candidates: list[dict[str, Any]],
    sample_index: int,
) -> set[tuple[str, str, str, int]]:
    keys = {(root_id, "", "no_help", sample_index)}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        keys.add((root_id, candidate_id, "relevant_plan", sample_index))
        keys.add((root_id, candidate_id, "random_plan", sample_index))
    return keys


def completed_sample_prefix(
    root_id: str,
    candidates: list[dict[str, Any]],
    completed: set[tuple[str, str, str, int]],
    max_samples: int,
) -> int:
    """Return the first incomplete sample index for one root.

    A process killed in the middle of a large cross-root batch can leave holes.
    Scanning from zero makes restart fill those holes before scheduling new
    sample indices, so every previously flushed JSONL record remains reusable.
    """
    for sample_index in range(max_samples):
        if not expected_sample_keys(root_id, candidates, sample_index).issubset(completed):
            return sample_index
    return max_samples


def build_pending_jobs(
    state: RootCalibrationState,
    completed: set[tuple[str, str, str, int]],
    random_plans: dict[tuple[str, str], str],
    min_samples: int,
    max_samples: int,
    sample_batch: int,
) -> tuple[list[tuple[str, str, str, int, str]], int]:
    """Build only the missing jobs for the root's next sequential sample wave."""
    start_sample = completed_sample_prefix(
        state.root_id, state.candidates, completed, max_samples
    )
    if start_sample >= max_samples:
        return [], start_sample
    end_sample = min(
        max_samples,
        max(min_samples, start_sample + sample_batch),
    )
    jobs: list[tuple[str, str, str, int, str]] = []
    for sample_index in range(start_sample, end_sample):
        baseline_key = (state.root_id, "", "no_help", sample_index)
        if baseline_key not in completed:
            jobs.append(
                (
                    state.root_id,
                    "",
                    "no_help",
                    sample_index,
                    root_prompt(state.question),
                )
            )
        for candidate in state.candidates:
            candidate_id = str(candidate["id"])
            for variant, plan in (
                ("relevant_plan", candidate["minimal_plan"]),
                ("random_plan", random_plans[(state.root_id, candidate_id)]),
            ):
                key = (state.root_id, candidate_id, variant, sample_index)
                if key not in completed:
                    jobs.append(
                        (
                            state.root_id,
                            candidate_id,
                            variant,
                            sample_index,
                            root_prompt(state.question, plan),
                        )
                    )
    return jobs, end_sample


def run(args: argparse.Namespace) -> None:
    import pandas as pd

    if not 0 <= args.q_low <= args.q_high <= 1:
        raise ValueError("q band must satisfy 0 <= q_low <= q_high <= 1")
    if args.min_samples < 1 or args.max_samples < args.min_samples:
        raise ValueError("sample limits must satisfy 1 <= min <= max")
    if args.root_window < 1 or args.batch_size < 1:
        raise ValueError("root_window and batch_size must both be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "help_results.jsonl"

    source = pd.read_parquet(args.source_data)
    by_question = {
        stable_question_key(str(row["question"])): row
        for row in source.to_dict(orient="records")
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in read_jsonl(args.candidates):
        plan = minimal_plan(candidate, args.max_plan_words)
        if plan and stable_question_key(str(candidate["question"])) in by_question:
            grouped[str(candidate["root_id"])].append(
                {**candidate, "minimal_plan": plan}
            )
    roots = sorted(grouped)
    random.Random(args.seed).shuffle(roots)
    roots = roots[: args.root_limit] if args.root_limit > 0 else roots
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("sharding must satisfy num_shards >= 1 and 0 <= shard_index < num_shards")
    roots = roots[args.shard_index :: args.num_shards]
    if len(roots) < 2:
        raise ValueError("At least two roots with non-leaking plans are required")

    # Deterministic dimension-matched controls from another root.  A control
    # that accidentally contains the current root answer is rejected.
    random_plans: dict[tuple[str, str], str] = {}
    for index, root_id in enumerate(roots):
        current = grouped[root_id][0]
        current_reference = normalize(str(current.get("reference", "")))
        for candidate in grouped[root_id]:
            target_dimension = str(candidate["dimension"])
            control = ""
            for offset in range(1, len(roots)):
                other = grouped[roots[(index + offset) % len(roots)]]
                ordered = sorted(
                    other,
                    key=lambda row: (
                        str(row["dimension"]) != target_dimension,
                        str(row["id"]),
                    ),
                )
                for option in ordered:
                    plan = str(option["minimal_plan"])
                    if not current_reference or current_reference not in normalize(plan):
                        control = plan
                        break
                if control:
                    break
            if not control:
                raise ValueError(f"No non-leaking random control found for {root_id}")
            random_plans[(root_id, str(candidate["id"]))] = control

    existing = read_jsonl(result_path) if result_path.exists() else []
    completed = {
        (
            str(row["root_id"]),
            str(row.get("candidate_id", "")),
            str(row["variant"]),
            int(row["sample_index"]),
        )
        for row in existing
    }
    generator = LocalGenerator(args)
    verifier = build_math_verifier()
    records_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing:
        records_by_root[str(row["root_id"])].append(row)

    states: list[RootCalibrationState] = []
    for root_id in roots:
        candidates = grouped[root_id]
        source_row = by_question[stable_question_key(str(candidates[0]["question"]))]
        states.append(
            RootCalibrationState(
                root_id=root_id,
                candidates=candidates,
                question=str(source_row["question"]),
                reference=str(source_row["reward_model"]["ground_truth"]),
                records=records_by_root[root_id],
            )
        )

    selected_by_root: dict[str, dict[str, Any]] = {}
    pending: list[RootCalibrationState] = []
    finished_count = 0
    for state in states:
        chosen = choose_candidate(
            state.candidates,
            state.records,
            args.q_low,
            args.q_high,
            args.min_gain,
            args.min_samples,
        )
        sample_prefix = completed_sample_prefix(
            state.root_id, state.candidates, completed, args.max_samples
        )
        if chosen is not None:
            selected_by_root[state.root_id] = chosen
            finished_count += 1
        elif sample_prefix >= args.max_samples:
            finished_count += 1
        else:
            pending.append(state)

    print(
        f"resume completed_roots={finished_count}/{len(roots)} "
        f"selected_roots={len(selected_by_root)} pending_roots={len(pending)} "
        f"root_window={args.root_window} inference_batch={args.batch_size}",
        flush=True,
    )

    round_index = 0
    while pending:
        active = pending[: args.root_window]
        queued_jobs: list[tuple[str, str, str, int, str]] = []
        for state in active:
            jobs, _ = build_pending_jobs(
                state,
                completed,
                random_plans,
                args.min_samples,
                args.max_samples,
                args.sample_batch,
            )
            queued_jobs.extend(jobs)
        if not queued_jobs:
            raise RuntimeError("Pending calibration roots produced no generation jobs")

        state_by_root = {state.root_id: state for state in active}
        # Flush each inference micro-batch immediately. If the process is
        # interrupted, a restart reuses every completed record and regenerates
        # only the holes in the unfinished cross-root wave.
        for batch_start in range(0, len(queued_jobs), args.batch_size):
            batch_jobs = queued_jobs[batch_start : batch_start + args.batch_size]
            prompts = [
                chat_prompt(generator.tokenizer, SYSTEM_PROMPT, job[4])
                for job in batch_jobs
            ]
            outputs = generator.generate(
                prompts,
                args.max_new_tokens,
                args.seed
                + args.shard_index * 1_000_000
                + round_index * 10_000
                + batch_start,
                stop_after_boxed=args.stop_after_boxed,
            )
            for job, output in zip(batch_jobs, outputs):
                root_id, candidate_id, variant, sample_index, _ = job
                state = state_by_root[root_id]
                record = {
                    "root_id": root_id,
                    "candidate_id": candidate_id,
                    "variant": variant,
                    "sample_index": sample_index,
                    "question": state.question,
                    "reference": state.reference,
                    "correct": verifier(output["text"], state.reference),
                    **output,
                }
                append_jsonl(result_path, record)
                existing.append(record)
                completed.add((root_id, candidate_id, variant, sample_index))
                state.records.append(record)

        still_active: list[RootCalibrationState] = []
        for state in active:
            chosen = choose_candidate(
                state.candidates,
                state.records,
                args.q_low,
                args.q_high,
                args.min_gain,
                args.min_samples,
            )
            sample_prefix = completed_sample_prefix(
                state.root_id, state.candidates, completed, args.max_samples
            )
            if chosen is not None:
                selected_by_root[state.root_id] = chosen
                finished_count += 1
                status = (
                    f"selected={chosen['dimension']} "
                    f"q_help={chosen['success_probability']:.3f}"
                )
                print(
                    f"[{finished_count}/{len(roots)}] {state.root_id} {status}",
                    flush=True,
                )
            elif sample_prefix >= args.max_samples:
                finished_count += 1
                print(
                    f"[{finished_count}/{len(roots)}] {state.root_id} "
                    "no causally useful candidate in band",
                    flush=True,
                )
            else:
                still_active.append(state)

        pending = still_active + pending[len(active) :]
        round_index += 1
        print(
            f"batch_round={round_index} active_roots={len(active)} "
            f"generated_jobs={len(queued_jobs)} completed_roots={finished_count}/{len(roots)} "
            f"pending_roots={len(pending)}",
            flush=True,
        )

    selected = [
        selected_by_root[root_id]
        for root_id in roots
        if root_id in selected_by_root
    ]

    selected_path = args.output_dir / "training_candidates.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "selected_roots": len(roots),
        "training_roots": len(selected),
        "training_candidate_rate": len(selected) / len(roots) if roots else 0.0,
        "q_definition": "P(root correct | relevant answer-free minimal plan)",
        "q_low": args.q_low,
        "q_high": args.q_high,
        "min_samples": args.min_samples,
        "max_samples": args.max_samples,
        "sample_batch": args.sample_batch,
        "root_window": args.root_window,
        "inference_batch_size": args.batch_size,
        "max_plan_words": args.max_plan_words,
        "causal_filter": "relevant plan must beat no-help and dimension-matched random plan",
        "model": str(args.model),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-limit", type=int, default=256)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--q-low", type=float, default=0.25)
    parser.add_argument("--q-high", type=float, default=0.60)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--sample-batch", type=int, default=2)
    parser.add_argument(
        "--root-window",
        type=int,
        default=8,
        help="Number of root questions scheduled together before model micro-batching.",
    )
    parser.add_argument("--max-plan-words", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-after-boxed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
