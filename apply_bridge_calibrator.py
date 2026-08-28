#!/usr/bin/env python3
"""Apply the frozen 212-root Bridge calibrator to a larger candidate pool.

Roots used to fit the calibrator retain their root-grouped cross-fitted
selection. All other roots are scored only by the frozen bootstrap ensemble;
full-pool transfer labels are deliberately neither required nor read.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from bridge_prompt_utils import BRIDGE_PROMPT_VERSION
from fit_bridge_calibrated_rcst import FEATURE_NAMES, assemble_rows, feature_vector


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def deserialize_model(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": np.asarray(payload["mean"], dtype=np.float64),
        "scale": np.asarray(payload["scale"], dtype=np.float64),
        "intercept": float(payload["intercept"]),
        "coef": np.asarray(payload["coef"], dtype=np.float64),
    }


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model["coef"]


def select_frozen(
    rows: list[dict[str, Any]],
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    crossfit_selected: list[dict[str, Any]],
    confidence_z: float,
    uncertainty_floor: float,
    min_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_root: dict[str, list[int]] = defaultdict(list)
    candidate_ids: set[str] = set()
    for index, row in enumerate(rows):
        by_root[row["root_id"]].append(index)
        candidate_ids.add(row["candidate_id"])
    crossfit_by_root = {str(row["root_id"]): row for row in crossfit_selected}
    unknown_crossfit = sorted(set(crossfit_by_root) - set(by_root))
    if unknown_crossfit:
        raise ValueError(f"cross-fit roots absent from full pool: {unknown_crossfit[:5]}")

    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    accepted = 0
    source_counts: dict[str, int] = defaultdict(int)
    for root_id in sorted(by_root):
        indices = by_root[root_id]
        if len(indices) != 3:
            raise ValueError(f"{root_id} has {len(indices)} candidates, expected 3")
        lcbs = {
            index: float(pred_mean[index])
            - confidence_z
            * math.sqrt(float(pred_std[index]) ** 2 + uncertainty_floor**2)
            for index in indices
        }
        best_index = max(
            indices,
            key=lambda index: (lcbs[index], float(pred_mean[index]), rows[index]["candidate_id"]),
        )
        if root_id in crossfit_by_root:
            candidate = copy.deepcopy(crossfit_by_root[root_id])
            candidate_id = str(candidate["id"])
            if candidate_id not in candidate_ids:
                raise ValueError(f"cross-fit candidate absent from full pool: {candidate_id}")
            accepted_flag = not bool(candidate.get("rcst_abstained", False))
            source = "root_grouped_crossfit_212"
            calibration = copy.deepcopy(candidate.get("bridge_calibration") or {})
        else:
            candidate = copy.deepcopy(rows[best_index]["candidate"])
            accepted_flag = lcbs[best_index] > min_score
            source = "frozen_212_bootstrap_ensemble"
            calibration = {
                "protocol": "frozen_212_bridge_calibrator_v1",
                "predicted_transfer_gain": float(pred_mean[best_index]),
                "prediction_std": float(pred_std[best_index]),
                "predicted_gain_lcb": lcbs[best_index],
                "confidence_z": confidence_z,
                "uncertainty_floor": uncertainty_floor,
                "prompt_version": BRIDGE_PROMPT_VERSION,
            }
        calibration["selection_source"] = source
        candidate["bridge_calibration"] = calibration
        candidate["trainable"] = True
        candidate["selection_policy"] = (
            "bridge_calibrated_rcst_lcb" if accepted_flag else "bridge_calibrated_rcst_abstain"
        )
        candidate["transfer_selected_for_preconditioning"] = accepted_flag
        candidate["rcst_abstained"] = not accepted_flag
        selected.append(candidate)
        accepted += int(accepted_flag)
        source_counts[source] += 1
        audit.append(
            {
                "root_id": root_id,
                "candidate_id": str(candidate["id"]),
                "dimension": str(candidate.get("dimension", "")),
                "accepted": accepted_flag,
                "selection_source": source,
                "predicted_gain": calibration.get("predicted_transfer_gain"),
                "predicted_std": calibration.get("prediction_std"),
                "predicted_lcb": calibration.get("predicted_gain_lcb"),
            }
        )
    summary = {
        "roots": len(by_root),
        "candidates": len(rows),
        "accepted_roots": accepted,
        "abstained_roots": len(by_root) - accepted,
        "selection_sources": dict(source_counts),
        "uses_full_pool_transfer_labels": False,
        "protocol": "crossfit_212_plus_frozen_unseen_v1",
    }
    return selected, audit, summary


def run(args: argparse.Namespace) -> None:
    candidates = read_jsonl(args.candidates)
    deltas: list[dict[str, Any]] = []
    for path in args.delta_features:
        deltas.extend(read_jsonl(path))
    dummy_aggregates = [
        {"candidate_id": row["id"], "mean_transfer_gain": 0.0, "gain_lcb": 0.0}
        for row in candidates
    ]
    rows = assemble_rows(candidates, dummy_aggregates, deltas)
    x = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    model_payload = json.loads(args.calibrator_model.read_text(encoding="utf-8"))
    if model_payload.get("prompt_version") != BRIDGE_PROMPT_VERSION:
        raise ValueError("calibrator and delta prompt versions do not match")
    if tuple(model_payload.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("calibrator feature schema does not match current code")
    models = [deserialize_model(row) for row in model_payload["models"]]
    predictions = np.stack([predict(model, x) for model in models])
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0, ddof=1) if len(models) > 1 else np.zeros(len(rows))
    selected, audit, summary = select_frozen(
        rows,
        pred_mean,
        pred_std,
        read_jsonl(args.crossfit_selected),
        args.confidence_z,
        args.uncertainty_floor,
        args.min_score,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "bridge_calibrated_selected.jsonl", selected)
    write_jsonl(args.output_dir / "bridge_calibrated_audit.jsonl", audit)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--delta-features", type=Path, nargs="+", required=True)
    parser.add_argument("--calibrator-model", type=Path, required=True)
    parser.add_argument("--crossfit-selected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--uncertainty-floor", type=float, default=0.01)
    parser.add_argument("--min-score", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
