#!/usr/bin/env python3
"""Aggregate replicated same-root probes and select RCST training targets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from calibrate_helpful_subproblems import read_jsonl


def paired_differences(row: dict[str, Any]) -> list[float]:
    before = row.get("baseline_correct", [])
    after = row.get("post_update_correct", [])
    if len(before) != len(after) or not before:
        raise ValueError(f"Unpaired probe outcomes for {row.get('candidate_id')}")
    return [float(bool(post)) - float(bool(pre)) for pre, post in zip(before, after)]


def aggregate_records(
    rows: list[dict[str, Any]], confidence_z: float
) -> dict[str, Any]:
    differences: list[float] = []
    before: list[float] = []
    after: list[float] = []
    final_losses: list[float] = []
    seeds: list[int] = []
    for row in rows:
        differences.extend(paired_differences(row))
        before.extend(float(bool(value)) for value in row["baseline_correct"])
        after.extend(float(bool(value)) for value in row["post_update_correct"])
        if row.get("losses"):
            final_losses.append(float(row["losses"][-1]))
        if row.get("probe_seed") is not None:
            seeds.append(int(row["probe_seed"]))
    mean_gain = statistics.fmean(differences)
    standard_error = (
        statistics.stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0
    )
    return {
        "baseline_probability": statistics.fmean(before),
        "post_update_probability": statistics.fmean(after),
        "mean_transfer_gain": mean_gain,
        "gain_standard_error": standard_error,
        "gain_lcb": mean_gain - confidence_z * standard_error,
        "replicates": len(rows),
        "total_root_samples": len(differences),
        "probe_seeds": sorted(set(seeds)),
        "mean_final_loss": statistics.fmean(final_losses) if final_losses else None,
    }


def aggregate_all(
    probe_rows: list[dict[str, Any]], confidence_z: float, min_replicates: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        grouped[(str(row["root_id"]), str(row["candidate_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (root_id, candidate_id), rows in sorted(grouped.items()):
        if len(rows) < min_replicates:
            raise ValueError(
                f"{candidate_id} has {len(rows)} replicates; require {min_replicates}"
            )
        aggregate = aggregate_records(rows, confidence_z)
        output.append(
            {
                "root_id": root_id,
                "candidate_id": candidate_id,
                "dimension": str(rows[0].get("dimension", "")),
                **aggregate,
            }
        )
    return output


def attach_probe(
    candidate: dict[str, Any], aggregate: dict[str, Any], policy: str, score: float
) -> dict[str, Any]:
    chosen = copy.deepcopy(candidate)
    chosen["transfer_probe"] = {
        "baseline_probability": aggregate["baseline_probability"],
        "post_update_probability": aggregate["post_update_probability"],
        "proxy_transfer_gain": aggregate["mean_transfer_gain"],
        "rcst_score": score,
        "gain_standard_error": aggregate["gain_standard_error"],
        "gain_lcb": aggregate["gain_lcb"],
        "replicates": aggregate["replicates"],
        "total_root_samples": aggregate["total_root_samples"],
        "probe_seeds": aggregate["probe_seeds"],
        "mean_final_loss": aggregate["mean_final_loss"],
    }
    chosen["selection_policy"] = f"rcst_{policy}"
    chosen["success_probability"] = 0.0
    chosen["no_help_probability"] = float(chosen.get("no_help_probability", 0.0))
    chosen["random_plan_probability"] = float(
        chosen.get("random_plan_probability", 0.0)
    )
    chosen["gain_over_no_help"] = -chosen["no_help_probability"]
    chosen["gain_over_random"] = -chosen["random_plan_probability"]
    chosen["trainable"] = True
    chosen["transfer_selected_for_preconditioning"] = True
    chosen["rcst_abstained"] = False
    return chosen


def select_candidates(
    anchors: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    aggregates: list[dict[str, Any]],
    policy: str,
    min_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        by_root[str(row["root_id"])].append(row)
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    accepted = 0
    changed = 0
    score_field = "mean_transfer_gain" if policy == "mean_positive" else "gain_lcb"
    for anchor in anchors:
        root_id = str(anchor["root_id"])
        choices = by_root.get(root_id, [])
        if not choices:
            raise ValueError(f"No RCST probes for root {root_id}")
        best = max(
            choices,
            key=lambda row: (
                float(row[score_field]),
                float(row["mean_transfer_gain"]),
                float(row["post_update_probability"]),
                str(row["candidate_id"]),
            ),
        )
        score = float(best[score_field])
        if score > min_score:
            chosen = attach_probe(
                candidates[str(best["candidate_id"])], best, policy, score
            )
            accepted += 1
            changed += str(chosen["id"]) != str(anchor["id"])
        else:
            chosen = copy.deepcopy(anchor)
            chosen["selection_policy"] = "rcst_abstain_fallback"
            chosen["rcst_abstained"] = True
        selected.append(chosen)
        audit.append(
            {
                "root_id": root_id,
                "anchor_candidate_id": str(anchor["id"]),
                "selected_candidate_id": str(chosen["id"]),
                "policy": chosen["selection_policy"],
                "best_candidate_id": str(best["candidate_id"]),
                "best_mean_gain": best["mean_transfer_gain"],
                "best_gain_lcb": best["gain_lcb"],
                "accepted": score > min_score,
            }
        )
    summary = {
        "policy": policy,
        "score_field": score_field,
        "min_score": min_score,
        "total_roots": len(selected),
        "accepted_roots": accepted,
        "abstained_roots": len(selected) - accepted,
        "changed_candidate_roots": changed,
    }
    return selected, audit, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    anchors = read_jsonl(args.anchors)
    candidate_rows = read_jsonl(args.candidates)
    candidates = {str(row["id"]): row for row in candidate_rows}
    probe_rows: list[dict[str, Any]] = []
    for path in args.inputs:
        probe_rows.extend(read_jsonl(path))
    aggregates = aggregate_all(probe_rows, args.confidence_z, args.min_replicates)
    expected = len(candidates)
    if len(aggregates) != expected:
        raise ValueError(f"Aggregated {len(aggregates)} candidates; expected {expected}")
    selected, audit, summary = select_candidates(
        anchors, candidates, aggregates, args.policy, args.min_score
    )
    if len(selected) != len(anchors):
        raise ValueError("Selected output does not cover every anchor root")
    write_jsonl(args.aggregated_output, aggregates)
    write_jsonl(args.output, selected)
    write_jsonl(args.output.with_name(args.output.stem + "_audit.jsonl"), audit)
    payload = {
        **summary,
        "candidate_records": len(aggregates),
        "probe_rows": len(probe_rows),
        "confidence_z": args.confidence_z,
        "min_replicates": args.min_replicates,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--aggregated-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("mean_positive", "lcb_positive"), required=True
    )
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-replicates", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
