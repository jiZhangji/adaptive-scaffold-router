#!/usr/bin/env python3
"""Compare one fixed-budget method against the existing three baseline arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = (
    "AIME24", "AIME25", "AMC23", "MinervaMath",
    "MATH-500", "OlympiadBench", "GaoKao2023en",
)


def read_score(path: Path) -> float:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for value in payload.values():
        if isinstance(value, dict) and "pass@1" in value:
            return float(value["pass@1"])
    raise ValueError(f"No pass@1 in {path}")


def collect(root: Path) -> dict[str, float]:
    return {name: read_score(root / name / "metric.json") for name in DATASETS}


def run(args: argparse.Namespace) -> None:
    results: dict[str, dict[str, float]] = {}
    labels = {
        "vanilla": "Vanilla GRPO",
        "scaf": "SCAF-GRPO",
        "subproblem": "Static Subproblem 1:1",
        "proposed": args.method_label,
    }
    for method in ("vanilla", "scaf", "subproblem"):
        root = args.baseline_run_root / f"eval_{method}"
        if all((root / name / "metric.json").is_file() for name in DATASETS):
            results[method] = collect(root)
    results["proposed"] = collect(args.eval_root)
    macro = {key: sum(value.values()) / len(value) for key, value in results.items()}
    payload = {
        "method_label": args.method_label,
        "training_steps": args.training_steps,
        "baseline_run_root": str(args.baseline_run_root.resolve()),
        "eval_root": str(args.eval_root.resolve()),
        "results": results,
        "macro": macro,
        "delta_pp": {
            key: (macro["proposed"] - value) * 100
            for key, value in macro.items() if key != "proposed"
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    methods = [key for key in ("vanilla", "scaf", "subproblem", "proposed") if key in results]
    lines = [
        f"# {args.method_label} vs Existing Results", "",
        "| Dataset | " + " | ".join(labels[key] for key in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for dataset in DATASETS:
        lines.append(
            f"| {dataset} | "
            + " | ".join(f"{results[key][dataset]:.1%}" for key in methods)
            + " |"
        )
    lines.append(
        "| **Macro average** | "
        + " | ".join(f"**{macro[key]:.1%}**" for key in methods)
        + " |"
    )
    report = "\n".join(lines) + "\n"
    (args.output_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--training-steps", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
