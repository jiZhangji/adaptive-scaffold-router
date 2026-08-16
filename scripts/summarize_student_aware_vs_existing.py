#!/usr/bin/env python3
"""Compare one newly trained method with already evaluated baseline arms."""

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


def read_score(path: Path) -> float:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for value in payload.values():
        if isinstance(value, dict) and "pass@1" in value:
            return float(value["pass@1"])
    raise ValueError(f"No pass@1 metric found in {path}")


def collect(eval_root: Path) -> dict[str, float]:
    return {
        dataset: read_score(eval_root / dataset / "metric.json")
        for dataset in DATASETS
    }


def maybe_collect(root: Path, method: str) -> dict[str, float] | None:
    eval_root = root / f"eval_{method}"
    if not all((eval_root / dataset / "metric.json").is_file() for dataset in DATASETS):
        return None
    return collect(eval_root)


def run(args: argparse.Namespace) -> None:
    results: dict[str, dict[str, float]] = {}
    for method in ("vanilla", "scaf", "subproblem"):
        values = maybe_collect(args.baseline_run_root, method)
        if values is not None:
            results[method] = values
    results["student_aware"] = collect(args.proposed_run_root / "eval_student_aware")

    macro = {
        method: sum(scores.values()) / len(scores)
        for method, scores in results.items()
    }
    payload = {
        "protocol": "student_aware_precondition_root_aligned_v1",
        "baseline_run_root": str(args.baseline_run_root.resolve()),
        "proposed_run_root": str(args.proposed_run_root.resolve()),
        "baseline_retrained": False,
        "training_steps": args.training_steps,
        "results": results,
        "macro": macro,
        "delta_pp": {
            method: (macro["student_aware"] - score) * 100.0
            for method, score in macro.items()
            if method != "student_aware"
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "student_aware_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    labels = {
        "vanilla": "Vanilla (existing)",
        "scaf": "Scaf-GRPO (existing)",
        "subproblem": "Static Subproblem 1:1 (existing)",
        "student_aware": "Student-Aware Proposed",
    }
    methods = [method for method in ("vanilla", "scaf", "subproblem", "student_aware") if method in results]
    lines = [
        "# Student-Aware Proposed vs Existing Results",
        "",
        "The Vanilla/Scaf/static-subproblem values are reused from the supplied baseline run; "
        "they were not retrained by this launcher.",
        "",
        "| Dataset | " + " | ".join(labels[method] for method in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for dataset in DATASETS:
        lines.append(
            f"| {dataset} | "
            + " | ".join(f"{results[method][dataset]:.1%}" for method in methods)
            + " |"
        )
    lines.append(
        "| **Macro average** | "
        + " | ".join(f"**{macro[method]:.1%}**" for method in methods)
        + " |"
    )
    lines.extend(["", "## Proposed deltas", ""])
    for method in methods:
        if method != "student_aware":
            lines.append(
                f"- vs {labels[method]}: "
                f"{(macro['student_aware'] - macro[method]) * 100:+.2f} pp"
            )
    lines.extend(
        [
            "",
            f"Proposed checkpoint budget: {args.training_steps} optimizer steps. ",
            "All accuracy numbers must use the same greedy pass@1 verifier protocol. ",
            "Baseline paths are recorded in the accompanying JSON for auditability.",
            "",
        ]
    )
    report = "\n".join(lines)
    (args.output_dir / "student_aware_comparison.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--proposed-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-steps", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
