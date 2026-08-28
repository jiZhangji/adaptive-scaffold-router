#!/usr/bin/env python3
"""Cross-fitted calibration from bridge influence to same-root transfer gain."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from bridge_prompt_utils import BRIDGE_PROMPT_VERSION


CONTINUOUS_FEATURES = (
    "delta_mean",
    "delta_std",
    "delta_lcb",
    "delta_median",
    "delta_positive_rate",
    "root_logprob_mean",
    "root_baseline_accuracy",
    "response_length_mean",
    "response_length_std",
    "subproblem_answer_logprob",
    "subproblem_answer_tokens",
    "subproblem_tokens",
    "delta_abs_mean",
    "delta_signal_to_noise",
    "delta_x_root_difficulty",
    "delta_root_centered",
    "delta_lcb_root_centered",
    "delta_root_rank",
    "answer_root_rank",
)
DIMENSIONS = ("calculation", "knowledge", "planning")
FEATURE_NAMES = CONTINUOUS_FEATURES + tuple(f"dimension_{value}" for value in DIMENSIONS)


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def root_rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    output = [0.0] * len(values)
    denominator = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        output[index] = rank / denominator
    return output


def assemble_rows(
    candidates: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_id = {str(row["id"]): row for row in candidates}
    aggregate_by_id = {str(row["candidate_id"]): row for row in aggregates}
    delta_by_id: dict[str, dict[str, Any]] = {}
    for row in deltas:
        candidate_id = str(row["candidate_id"])
        if candidate_id in delta_by_id:
            raise ValueError(f"duplicate delta feature row: {candidate_id}")
        if row.get("prompt_version") != BRIDGE_PROMPT_VERSION:
            raise ValueError(
                f"{candidate_id} uses {row.get('prompt_version')}; "
                f"expected {BRIDGE_PROMPT_VERSION}"
            )
        delta_by_id[candidate_id] = row
    common = sorted(set(candidate_by_id) & set(aggregate_by_id) & set(delta_by_id))
    if len(common) != len(aggregate_by_id):
        raise ValueError(
            f"joined {len(common)} candidates, but aggregates contain {len(aggregate_by_id)}"
        )
    rows: list[dict[str, Any]] = []
    for candidate_id in common:
        candidate = candidate_by_id[candidate_id]
        aggregate = aggregate_by_id[candidate_id]
        delta = delta_by_id[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "root_id": str(candidate["root_id"]),
                "dimension": str(candidate.get("dimension", "")),
                "candidate": candidate,
                "target": float(aggregate["mean_transfer_gain"]),
                "target_lcb": float(aggregate["gain_lcb"]),
                "delta": delta,
            }
        )
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_root[row["root_id"]].append(row)
    for root_id, root_rows in by_root.items():
        if len(root_rows) != 3:
            raise ValueError(f"{root_id} has {len(root_rows)} joined candidates, expected 3")
        delta_means = [float(row["delta"]["delta_mean"]) for row in root_rows]
        delta_lcbs = [float(row["delta"]["delta_lcb"]) for row in root_rows]
        answer_scores = [
            float(row["delta"]["subproblem_answer_logprob"]) for row in root_rows
        ]
        delta_ranks = root_rank(delta_means)
        answer_ranks = root_rank(answer_scores)
        for index, row in enumerate(root_rows):
            delta = row["delta"]
            mean = float(delta["delta_mean"])
            std = float(delta["delta_std"])
            baseline = float(delta["root_baseline_accuracy"])
            row["derived"] = {
                "delta_abs_mean": abs(mean),
                "delta_signal_to_noise": mean / (std + 1e-4),
                "delta_x_root_difficulty": mean * (1.0 - baseline),
                "delta_root_centered": mean - float(np.mean(delta_means)),
                "delta_lcb_root_centered": float(delta["delta_lcb"])
                - float(np.mean(delta_lcbs)),
                "delta_root_rank": delta_ranks[index],
                "answer_root_rank": answer_ranks[index],
            }
    return sorted(rows, key=lambda row: (row["root_id"], row["candidate_id"]))


def feature_vector(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for name in CONTINUOUS_FEATURES:
        source = row["derived"] if name in row["derived"] else row["delta"]
        value = float(source[name])
        values.append(value if math.isfinite(value) else 0.0)
    dimension = row["dimension"]
    values.extend(float(dimension == name) for name in DIMENSIONS)
    return values


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    rhs = design.T @ y
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ rhs
    return {
        "mean": mean,
        "scale": scale,
        "intercept": float(coefficients[0]),
        "coef": coefficients[1:],
    }


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model["coef"]


def bootstrap_models(
    x: np.ndarray,
    y: np.ndarray,
    roots: np.ndarray,
    count: int,
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    unique_roots = np.unique(roots)
    models: list[dict[str, Any]] = []
    for _ in range(count):
        sampled = rng.choice(unique_roots, size=len(unique_roots), replace=True)
        indices = np.concatenate([np.flatnonzero(roots == root) for root in sampled])
        models.append(fit_ridge(x[indices], y[indices], alpha))
    return models


def safe_correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    result = spearmanr(left, right) if rank else pearsonr(left, right)
    return float(result.statistic)


def select_crossfit(
    rows: list[dict[str, Any]],
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    fold_ids: np.ndarray,
    confidence_z: float,
    uncertainty_floor: float,
    min_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_root: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_root[row["root_id"]].append(index)
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    accepted_actual: list[float] = []
    accepted = 0
    top1_hits = 0
    for root_id in sorted(by_root):
        indices = by_root[root_id]
        lcbs = {
            index: float(pred_mean[index])
            - confidence_z
            * math.sqrt(float(pred_std[index]) ** 2 + uncertainty_floor**2)
            for index in indices
        }
        best_index = max(
            indices,
            key=lambda index: (
                lcbs[index],
                float(pred_mean[index]),
                rows[index]["candidate_id"],
            ),
        )
        oracle_index = max(indices, key=lambda index: rows[index]["target"])
        top1_hits += int(best_index == oracle_index)
        accepted_flag = lcbs[best_index] > min_score
        candidate = copy.deepcopy(rows[best_index]["candidate"])
        calibration = {
            "protocol": "root_grouped_crossfit_ridge_bootstrap_v1",
            "fold": int(fold_ids[best_index]),
            "predicted_transfer_gain": float(pred_mean[best_index]),
            "prediction_std": float(pred_std[best_index]),
            "predicted_gain_lcb": lcbs[best_index],
            "confidence_z": confidence_z,
            "uncertainty_floor": uncertainty_floor,
            "prompt_version": BRIDGE_PROMPT_VERSION,
        }
        candidate["bridge_calibration"] = calibration
        candidate["trainable"] = True
        if accepted_flag:
            accepted += 1
            accepted_actual.append(float(rows[best_index]["target"]))
            candidate["selection_policy"] = "bridge_calibrated_rcst_lcb"
            candidate["transfer_selected_for_preconditioning"] = True
            candidate["rcst_abstained"] = False
        else:
            candidate["selection_policy"] = "bridge_calibrated_rcst_abstain"
            candidate["transfer_selected_for_preconditioning"] = False
            candidate["rcst_abstained"] = True
        selected.append(candidate)
        audit.append(
            {
                "root_id": root_id,
                "selected_candidate_id": rows[best_index]["candidate_id"],
                "selected_dimension": rows[best_index]["dimension"],
                "crossfit_fold": int(fold_ids[best_index]),
                "predicted_gain": float(pred_mean[best_index]),
                "predicted_std": float(pred_std[best_index]),
                "predicted_lcb": lcbs[best_index],
                "accepted": accepted_flag,
                "diagnostic_actual_gain": float(rows[best_index]["target"]),
                "diagnostic_actual_lcb": float(rows[best_index]["target_lcb"]),
                "oracle_candidate_id": rows[oracle_index]["candidate_id"],
                "oracle_gain": float(rows[oracle_index]["target"]),
            }
        )
    summary = {
        "total_roots": len(by_root),
        "accepted_roots": accepted,
        "abstained_roots": len(by_root) - accepted,
        "selected_actual_gain_mean": float(np.mean(accepted_actual)) if accepted_actual else 0.0,
        "selected_actual_positive_rate": (
            sum(value > 0 for value in accepted_actual) / len(accepted_actual)
            if accepted_actual
            else 0.0
        ),
        "candidate_top1_accuracy": top1_hits / len(by_root),
    }
    return selected, audit, summary


def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(),
        "intercept": model["intercept"],
        "coef": model["coef"].tolist(),
    }


def run(args: argparse.Namespace) -> None:
    candidates = read_jsonl([args.candidates])
    aggregates = read_jsonl([args.aggregates])
    deltas = read_jsonl(args.delta_features)
    rows = assemble_rows(candidates, aggregates, deltas)
    x = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    y = np.asarray([row["target"] for row in rows], dtype=np.float64)
    roots = np.asarray([row["root_id"] for row in rows], dtype=object)
    unique_roots = np.asarray(sorted(set(roots)), dtype=object)
    if len(unique_roots) < args.folds:
        raise ValueError("number of roots must be at least the number of folds")

    rng = np.random.default_rng(args.seed)
    shuffled = unique_roots.copy()
    rng.shuffle(shuffled)
    root_fold = {root: index % args.folds for index, root in enumerate(shuffled)}
    fold_ids = np.asarray([root_fold[root] for root in roots], dtype=np.int64)
    pred_mean = np.zeros(len(rows), dtype=np.float64)
    pred_std = np.zeros(len(rows), dtype=np.float64)
    for fold in range(args.folds):
        train = fold_ids != fold
        test = ~train
        models = bootstrap_models(
            x[train],
            y[train],
            roots[train],
            args.bootstrap_models,
            args.ridge_alpha,
            args.seed + fold * 1009,
        )
        predictions = np.stack([predict(model, x[test]) for model in models])
        pred_mean[test] = predictions.mean(axis=0)
        pred_std[test] = predictions.std(axis=0, ddof=1) if len(models) > 1 else 0.0

    selected, audit, selection_summary = select_crossfit(
        rows,
        pred_mean,
        pred_std,
        fold_ids,
        args.confidence_z,
        args.uncertainty_floor,
        args.min_score,
    )
    delta_mean = np.asarray([float(row["delta"]["delta_mean"]) for row in rows])
    diagnostics = {
        "prompt_version": BRIDGE_PROMPT_VERSION,
        "roots": len(unique_roots),
        "candidates": len(rows),
        "folds": args.folds,
        "bootstrap_models_per_fold": args.bootstrap_models,
        "ridge_alpha": args.ridge_alpha,
        "delta_to_transfer_pearson": safe_correlation(delta_mean, y),
        "delta_to_transfer_spearman": safe_correlation(delta_mean, y, rank=True),
        "crossfit_prediction_pearson": safe_correlation(pred_mean, y),
        "crossfit_prediction_spearman": safe_correlation(pred_mean, y, rank=True),
        "crossfit_rmse": float(np.sqrt(np.mean((pred_mean - y) ** 2))),
        "crossfit_mae": float(np.mean(np.abs(pred_mean - y))),
        **selection_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "bridge_calibrated_selected.jsonl", selected)
    write_jsonl(args.output_dir / "bridge_calibrated_audit.jsonl", audit)
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "root_id": row["root_id"],
                "candidate_id": row["candidate_id"],
                "dimension": row["dimension"],
                "fold": int(fold_ids[index]),
                "delta_mean": float(delta_mean[index]),
                "actual_transfer_gain": float(y[index]),
                "predicted_transfer_gain": float(pred_mean[index]),
                "prediction_std": float(pred_std[index]),
            }
        )
    write_jsonl(args.output_dir / "crossfit_predictions.jsonl", prediction_rows)

    final_models = bootstrap_models(
        x,
        y,
        roots,
        args.bootstrap_models,
        args.ridge_alpha,
        args.seed + 99991,
    )
    model_payload = {
        "protocol": "bridge_calibrated_rcst_ridge_bootstrap_v1",
        "prompt_version": BRIDGE_PROMPT_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "ridge_alpha": args.ridge_alpha,
        "training_roots": len(unique_roots),
        "training_candidates": len(rows),
        "models": [serializable_model(model) for model in final_models],
        "warning": "Use this all-root model only on new roots; 212-root training selections use cross-fitted predictions.",
    }
    (args.output_dir / "calibrator_model.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Bridge-Calibrated RCST diagnostics",
        "",
        f"- Roots: {diagnostics['roots']}",
        f"- Candidates: {diagnostics['candidates']}",
        f"- Delta→transfer Spearman: {diagnostics['delta_to_transfer_spearman']:.4f}",
        f"- Cross-fit prediction Spearman: {diagnostics['crossfit_prediction_spearman']:.4f}",
        f"- Cross-fit RMSE: {diagnostics['crossfit_rmse']:.4f}",
        f"- Accepted roots: {diagnostics['accepted_roots']}/{diagnostics['total_roots']}",
        f"- Selected actual mean gain (diagnostic only): {diagnostics['selected_actual_gain_mean']:.4f}",
        f"- Selected actual positive rate (diagnostic only): {diagnostics['selected_actual_positive_rate']:.4f}",
        f"- Candidate top-1 accuracy: {diagnostics['candidate_top1_accuracy']:.4f}",
        "",
        "Every 212-root selection is cross-fitted by root. The all-root model is reserved for unseen roots.",
    ]
    (args.output_dir / "diagnostics.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--delta-features", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-models", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--uncertainty-floor", type=float, default=0.01)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
