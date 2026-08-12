#!/usr/bin/env python3
"""Aggregate seven Scaf-GRPO metrics and compare them with paper Table 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = [
    "AIME24",
    "AIME25",
    "AMC23",
    "MinervaMath",
    "MATH-500",
    "OlympiadBench",
    "GaoKao2023en",
]

PAPER = {
    "1.5b": {
        "base": [7.2, 3.3, 32.5, 14.7, 32.8, 20.6, 20.0],
        "vanilla": [13.3, 10.0, 47.5, 28.3, 72.2, 34.8, 57.4],
        "scaf": [20.0, 13.3, 60.0, 29.1, 73.4, 36.6, 57.9],
    },
    "7b": {
        "base": [13.3, 13.3, 42.5, 16.5, 53.6, 18.2, 35.1],
        "vanilla": [30.0, 13.3, 60.0, 33.4, 75.8, 41.3, 62.6],
        "scaf": [43.3, 20.0, 70.0, 36.4, 80.0, 43.3, 63.4],
    },
}


def read_pass_at_one(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for value in payload.values():
        if isinstance(value, dict) and "pass@1" in value:
            return 100.0 * float(value["pass@1"])
    raise ValueError(f"No pass@1 metric found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-size", choices=sorted(PAPER), default="1.5b")
    parser.add_argument(
        "--paper-reference", choices=["base", "vanilla", "scaf"], default="base"
    )
    args = parser.parse_args()

    scores = {
        dataset: read_pass_at_one(args.run_root / dataset / "metric.json")
        for dataset in DATASETS
    }
    reference = dict(zip(DATASETS, PAPER[args.model_size][args.paper_reference]))
    macro = sum(scores.values()) / len(scores)
    paper_macro = sum(reference.values()) / len(reference)

    report = {
        "model_size": args.model_size,
        "paper_reference": args.paper_reference,
        "scores_percent": scores,
        "macro_average_percent": macro,
        "paper_scores_percent": reference,
        "paper_macro_average_percent": paper_macro,
        "macro_gap_pp": macro - paper_macro,
    }
    (args.run_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Qwen2.5-Math-{args.model_size.upper()} paper comparison",
        "",
        f"Paper reference: **{args.paper_reference}**",
        "",
        "| Dataset | Reproduced | Paper | Gap |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        lines.append(
            f"| {dataset} | {scores[dataset]:.1f}% | {reference[dataset]:.1f}% | "
            f"{scores[dataset] - reference[dataset]:+.1f} pp |"
        )
    lines.extend(
        [
            f"| **Macro average** | **{macro:.1f}%** | **{paper_macro:.1f}%** | "
            f"**{macro - paper_macro:+.1f} pp** |",
            "",
            "The paper reports greedy pass@1 and selects the best training checkpoint.",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    (args.run_root / "paper_comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
