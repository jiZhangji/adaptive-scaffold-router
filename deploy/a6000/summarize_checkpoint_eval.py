#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = (
    "AIME24",
    "AIME25",
    "AMC23",
    "MinervaMath",
    "MATH-500",
    "OlympiadBench",
    "GaoKao2023en",
)
STEPS = (10, 35, 50)


def read_score(path: Path) -> float | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for value in payload.values():
        if isinstance(value, dict) and "pass@1" in value:
            return float(value["pass@1"]) * 100.0
    return None


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def delta(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return "—"
    return f"{right - left:+.1f} pp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scores: dict[int, dict[str, float | None]] = {}
    for step in STEPS:
        scores[step] = {
            dataset: read_score(args.root / f"step_{step}" / dataset / "metric.json")
            for dataset in DATASETS
        }

    lines = [
        "# Student-Aware checkpoint comparison",
        "",
        "| Dataset | Step 10 | Step 35 | Step 50 | 10→35 | 35→50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for dataset in DATASETS:
        s10 = scores[10][dataset]
        s35 = scores[35][dataset]
        s50 = scores[50][dataset]
        lines.append(
            f"| {dataset} | {fmt(s10)} | {fmt(s35)} | {fmt(s50)} | "
            f"{delta(s10, s35)} | {delta(s35, s50)} |"
        )

    means: dict[int, float | None] = {}
    for step in STEPS:
        present = [value for value in scores[step].values() if value is not None]
        means[step] = sum(present) / len(present) if len(present) == len(DATASETS) else None

    lines.append(
        f"| **Macro** | **{fmt(means[10])}** | **{fmt(means[35])}** | "
        f"**{fmt(means[50])}** | **{delta(means[10], means[35])}** | "
        f"**{delta(means[35], means[50])}** |"
    )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A drop already visible at Step 10 points to preconditioning/data selection.",
            "- A Step 10→35 drop points to scaffold-conditioned training.",
            "- A Step 35→50 drop points to root-only consolidation or over-training.",
            "- Check every dataset and Macro; do not select a checkpoint from AMC23 alone.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
