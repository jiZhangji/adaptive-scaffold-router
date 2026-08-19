#!/usr/bin/env python3
"""Choose the highest proxy-transfer candidate and retain unprobed anchors."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from calibrate_helpful_subproblems import read_jsonl


def run(args: argparse.Namespace) -> None:
    anchors = read_jsonl(args.anchors)
    candidates = {str(row["id"]): row for row in read_jsonl(args.candidates)}
    probes = read_jsonl(args.transfer_results)
    probes_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in probes:
        probes_by_root[str(row["root_id"])].append(row)

    selected: list[dict[str, Any]] = []
    transfer_selected = 0
    changed_dimension = 0
    records: list[dict[str, Any]] = []
    for anchor in anchors:
        root_id = str(anchor["root_id"])
        root_probes = probes_by_root.get(root_id, [])
        if root_probes:
            best = max(
                root_probes,
                key=lambda row: (
                    float(row["proxy_transfer_gain"]),
                    float(row["post_update_probability"]),
                    -float(row["losses"][-1]) if row.get("losses") else 0.0,
                    str(row["candidate_id"]),
                ),
            )
            chosen = copy.deepcopy(candidates[str(best["candidate_id"])])
            chosen["transfer_probe"] = {
                key: best[key]
                for key in (
                    "baseline_probability",
                    "post_update_probability",
                    "proxy_transfer_gain",
                    "root_samples",
                    "probe_steps",
                    "learning_rate",
                    "losses",
                )
            }
            chosen["selection_policy"] = "max_same_root_proxy_transfer_gain"
            # A transfer-selected candidate is intentionally routed through
            # Stage 1.  Its copied anchor q_help described another candidate
            # and must not be reused as a scaffold-usability measurement.
            chosen["success_probability"] = 0.0
            chosen["transfer_selected_for_preconditioning"] = True
            transfer_selected += 1
            changed_dimension += str(chosen.get("id")) != str(anchor.get("id"))
        else:
            chosen = copy.deepcopy(anchor)
            chosen["selection_policy"] = "existing_calibrated_fallback"
        selected.append(chosen)
        records.append(
            {
                "root_id": root_id,
                "anchor_candidate_id": str(anchor["id"]),
                "selected_candidate_id": str(chosen["id"]),
                "selection_policy": chosen["selection_policy"],
                "proxy_transfer_gain": (
                    chosen.get("transfer_probe", {}).get("proxy_transfer_gain")
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = args.output.with_name(args.output.stem + "_audit.jsonl")
    with audit.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "total_training_roots": len(selected),
        "transfer_selected_roots": transfer_selected,
        "fallback_roots": len(selected) - transfer_selected,
        "selection_changed_candidate": changed_dimension,
        "comparison_control": "same 212-root pool; only candidate identity changes for probed roots",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--transfer-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
