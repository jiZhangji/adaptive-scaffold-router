#!/usr/bin/env python3
"""Summarize a compute-matched full-pool RCST versus Vanilla experiment."""

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


def load_summary(root: Path) -> dict[str, Any]:
    path = root / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores_percent", {})
    missing = [dataset for dataset in DATASETS if dataset not in scores]
    if missing:
        raise ValueError(f"{path} is missing datasets: {missing}")
    return payload


def run(args: argparse.Namespace) -> None:
    vanilla = load_summary(args.vanilla_eval)
    rcst = load_summary(args.rcst_eval)
    vanilla_scores = vanilla["scores_percent"]
    rcst_scores = rcst["scores_percent"]
    rows = []
    for dataset in DATASETS:
        baseline = float(vanilla_scores[dataset])
        proposed = float(rcst_scores[dataset])
        rows.append(
            {
                "dataset": dataset,
                "vanilla_percent": baseline,
                "rcst_lcb_percent": proposed,
                "delta_pp": proposed - baseline,
            }
        )
    vanilla_macro = float(vanilla["macro_average_percent"])
    rcst_macro = float(rcst["macro_average_percent"])
    payload = {
        "protocol": "full-1866-root-compute-matched-v1",
        "training_steps": args.training_steps,
        "training_roots": args.training_roots,
        "rows": rows,
        "macro": {
            "vanilla_percent": vanilla_macro,
            "rcst_lcb_percent": rcst_macro,
            "delta_pp": rcst_macro - vanilla_macro,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Full 1,866-root RCST LCB-Positive vs Vanilla GRPO",
        "",
        f"Training roots: **{args.training_roots}**; fixed updates: **{args.training_steps}**.",
        "",
        "| Dataset | Vanilla GRPO | RCST LCB-Positive | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['vanilla_percent']:.1f}% | "
            f"{row['rcst_lcb_percent']:.1f}% | {row['delta_pp']:+.1f} pp |"
        )
    lines.append(
        f"| **Macro average** | **{vanilla_macro:.1f}%** | "
        f"**{rcst_macro:.1f}%** | **{rcst_macro - vanilla_macro:+.1f} pp** |"
    )
    (args.output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-eval", type=Path, required=True)
    parser.add_argument("--rcst-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-steps", type=int, default=440)
    parser.add_argument("--training-roots", type=int, default=1866)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
