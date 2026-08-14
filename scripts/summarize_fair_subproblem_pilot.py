#!/usr/bin/env python3
"""Summarize compute-matched Vanilla and subproblem pilot evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = (
    "AIME24",
    "AIME25",
    "AMC23",
    "MinervaMath",
    "MATH-500",
    "OlympiadBench",
    "GaoKao2023en",
)


def read_pass_at_1(run_dir: Path, dataset: str) -> float:
    path = run_dir / dataset / "metric.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        raise ValueError(f"Empty metric file: {path}")
    record = next(iter(payload.values()))
    return float(record["pass@1"])


def collect(run_dir: Path) -> dict[str, float]:
    return {dataset: read_pass_at_1(run_dir, dataset) for dataset in DATASETS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-eval", type=Path, required=True)
    parser.add_argument("--subproblem-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-steps", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    args = parser.parse_args()

    vanilla = collect(args.vanilla_eval)
    subproblem = collect(args.subproblem_eval)
    vanilla_macro = sum(vanilla.values()) / len(vanilla)
    subproblem_macro = sum(subproblem.values()) / len(subproblem)

    rows = []
    for dataset in DATASETS:
        rows.append(
            {
                "dataset": dataset,
                "vanilla": vanilla[dataset],
                "subproblem": subproblem[dataset],
                "delta": subproblem[dataset] - vanilla[dataset],
            }
        )

    payload = {
        "protocol": "compute_matched_subproblem_pilot_v1",
        "training_steps_per_arm": args.training_steps,
        "prompts_per_step": args.train_batch_size,
        "rollouts_per_prompt": args.rollouts,
        "generated_trajectories_per_arm": (
            args.training_steps * args.train_batch_size * args.rollouts
        ),
        "checkpoint_rule": "final fixed-step checkpoint",
        "evaluation": "greedy pass@1, 2048 response tokens, official math verifier",
        "rows": rows,
        "macro": {
            "vanilla": vanilla_macro,
            "subproblem": subproblem_macro,
            "delta": subproblem_macro - vanilla_macro,
        },
        "interpretation_limit": (
            "Pilot q-calibrated data only; matched-random causal relevance filtering "
            "was not applied. Run multiple seeds before making a paper claim."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fair_pilot_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Compute-matched Vanilla vs. Subproblem Pilot",
        "",
        f"- Training steps per arm: {args.training_steps}",
        f"- Prompts per step: {args.train_batch_size}",
        f"- Rollouts per prompt: {args.rollouts}",
        f"- Generated trajectories per arm: {payload['generated_trajectories_per_arm']}",
        "- Checkpoint: final fixed-step checkpoint (no best-checkpoint selection)",
        "- Evaluation: greedy pass@1 with the same seven datasets and verifier",
        "",
        "| Dataset | Vanilla | Subproblem | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['vanilla']:.1%} | "
            f"{row['subproblem']:.1%} | {row['delta'] * 100:+.1f} pp |"
        )
    lines.append(
        f"| **Macro average** | **{vanilla_macro:.1%}** | "
        f"**{subproblem_macro:.1%}** | "
        f"**{(subproblem_macro - vanilla_macro) * 100:+.1f} pp** |"
    )
    lines.extend(
        [
            "",
            "This is a mechanism pilot, not a final paper result. The current candidates "
            "were filtered by student success probability q, but not yet by a matched-random "
            "causal relevance test. Multiple training seeds are also required.",
            "",
        ]
    )
    report = "\n".join(lines)
    (args.output_dir / "fair_pilot_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
