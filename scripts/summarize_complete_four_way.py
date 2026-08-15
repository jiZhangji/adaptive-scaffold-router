#!/usr/bin/env python3
"""Summarize Vanilla, Scaf, subproblem-mix, and fading-IS evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = (
    "AIME24", "AIME25", "AMC23", "MinervaMath",
    "MATH-500", "OlympiadBench", "GaoKao2023en",
)
METHODS = ("vanilla", "scaf", "subproblem", "fade_is")


def score(path: Path) -> float:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return float(next(iter(value.values()))["pass@1"])


def collect(root: Path, method: str) -> dict[str, float]:
    return {
        dataset: score(root / f"eval_{method}" / dataset / "metric.json")
        for dataset in DATASETS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-steps", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    args = parser.parse_args()

    results = {method: collect(args.run_root, method) for method in METHODS}
    macro = {
        method: sum(values.values()) / len(values)
        for method, values in results.items()
    }
    payload = {
        "protocol": "complete_four_way_v1",
        "matched_root_set": True,
        "training_steps": args.training_steps,
        "train_batch_size": args.train_batch_size,
        "root_rollouts_per_prompt": args.rollouts,
        "budget_note": (
            "Optimizer updates and base root rollout budgets are matched. Scaf and fading-IS "
            "may issue auxiliary hinted rollouts; their extra generation cost must be reported."
        ),
        "q_definition": "P(root correct | relevant answer-free minimal plan)",
        "results": results,
        "macro": macro,
        "delta_vs_vanilla": {
            method: macro[method] - macro["vanilla"] for method in METHODS
        },
    }
    (args.run_root / "four_way_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    titles = {
        "vanilla": "Vanilla",
        "scaf": "Scaf-GRPO",
        "subproblem": "Subproblem 1:1",
        "fade_is": "Fade + IS",
    }
    lines = [
        "# Complete Four-Way Comparison",
        "",
        "| Dataset | Vanilla | Scaf-GRPO | Subproblem 1:1 | Fade + IS |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        lines.append(
            "| " + dataset + " | " + " | ".join(
                f"{results[method][dataset]:.1%}" for method in METHODS
            ) + " |"
        )
    lines.append(
        "| **Macro average** | "
        + " | ".join(f"**{macro[method]:.1%}**" for method in METHODS)
        + " |"
    )
    lines.extend(["", "## Delta vs Vanilla", ""])
    for method in METHODS[1:]:
        lines.append(
            f"- {titles[method]}: {(macro[method] - macro['vanilla']) * 100:+.2f} pp"
        )
    lines.extend(
        [
            "",
            "All arms use the same selected root set, initialization, optimizer-step budget, "
            "batch size, rollout count, response limit, verifier, and downstream decoding. "
            "Scaf-GRPO and Fade+IS can consume additional hinted rollouts, so GPU-hours and "
            "generated tokens must accompany accuracy in the final paper table.",
            "",
        ]
    )
    report = "\n".join(lines)
    (args.run_root / "four_way_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
