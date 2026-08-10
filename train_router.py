from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                question_hash = hashlib.sha1(
                    record["question"].encode("utf-8")
                ).hexdigest()[:12]
                record["item_key"] = f"{record['id']}::{question_hash}"
                records.append(record)
    if not records:
        raise ValueError("No router records were loaded")
    return records


def split_item_ids(
    item_ids: list[str], seed: int, train_fraction: float, validation_fraction: float
) -> tuple[set[str], set[str], set[str]]:
    ids = sorted(set(item_ids))
    random.Random(seed).shuffle(ids)
    count = len(ids)
    train_end = max(1, int(count * train_fraction))
    validation_end = max(train_end + 1, int(count * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, count - 1)
    if validation_end <= train_end or validation_end >= count:
        raise ValueError("At least three item groups are required for train/validation/test")
    return set(ids[:train_end]), set(ids[train_end:validation_end]), set(ids[validation_end:])


def expected_calibration_error(
    labels: list[int], probabilities: list[float], bins: int = 10
) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    error = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        indices = [
            index
            for index, probability in enumerate(probabilities)
            if low <= probability < high or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not indices:
            continue
        confidence = sum(probabilities[index] for index in indices) / len(indices)
        accuracy = sum(labels[index] for index in indices) / len(indices)
        error += len(indices) / total * abs(confidence - accuracy)
    return error


def _records_to_frame(records: list[dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(
        {
            "question": [record["question"] for record in records],
            "arm_name": [record["arm_name"] for record in records],
            "arm_kind": [record["arm_kind"] for record in records],
            "arm_strength": [float(record["arm_strength"]) for record in records],
            "hint_tokens": [float(record["hint_tokens"]) for record in records],
        }
    )


def build_router_pipeline(max_features: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    features = ColumnTransformer(
        [
            (
                "word_question",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features),
                "question",
            ),
            (
                "char_question",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=max_features,
                ),
                "question",
            ),
            (
                "arm",
                OneHotEncoder(handle_unknown="ignore"),
                ["arm_name", "arm_kind"],
            ),
            (
                "numeric",
                StandardScaler(),
                ["arm_strength", "hint_tokens"],
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=0,
                ),
            ),
        ]
    )


def calibrate_router(router: Any, validation_frame: Any, validation_labels: list[int]):
    from sklearn.calibration import CalibratedClassifierCV

    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(router), method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(router, method="sigmoid", cv="prefit")
    calibrated.fit(validation_frame, validation_labels)
    return calibrated


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_routing(
    records: list[dict[str, Any]],
    probabilities: list[float],
    target_success: float,
    max_fallback_arms: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_trajectory: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record, probability in zip(records, probabilities):
        enriched = dict(record)
        enriched["predicted_success"] = float(probability)
        enriched["action_cost"] = float(record["input_tokens"] + record["output_tokens"])
        trajectory_key = (record["item_key"], int(record["sample_index"]))
        by_trajectory[trajectory_key].append(enriched)

    direct_success = []
    direct_cost = []
    fallback_success = []
    fallback_cost = []
    fallback_calls = []
    public_scaf_success = []
    public_scaf_cost = []
    public_scaf_calls = []
    oracle_success = []
    oracle_cost = []
    strongest_success = []
    strongest_cost = []
    decisions = []

    for (item_key, sample_index), arms in sorted(by_trajectory.items()):
        item_id = arms[0]["id"]
        arms.sort(key=lambda row: row["arm_order"])
        eligible = [row for row in arms if row["predicted_success"] >= target_success]
        if eligible:
            selected = min(eligible, key=lambda row: (row["action_cost"], row["arm_order"]))
        else:
            selected = max(
                arms,
                key=lambda row: (
                    row["predicted_success"] - 0.02 * row["hint_tokens"] / 100.0,
                    -row["action_cost"],
                ),
            )
        direct_success.append(float(selected["correct"]))
        direct_cost.append(selected["action_cost"])

        remaining = [row for row in arms if row["arm_name"] != selected["arm_name"]]
        remaining.sort(
            key=lambda row: (
                -(row["predicted_success"] / max(row["action_cost"], 1.0)),
                row["arm_order"],
            )
        )
        attempted = [selected]
        if not selected["correct"]:
            for row in remaining[:max_fallback_arms]:
                attempted.append(row)
                if row["correct"]:
                    break
        fallback_success.append(float(any(row["correct"] for row in attempted)))
        fallback_cost.append(sum(row["action_cost"] for row in attempted))
        fallback_calls.append(float(len(attempted)))

        public_arms = [
            row for row in arms if row["arm_kind"] in {"none", "knowledge", "solution"}
        ]
        scaf_cost = 0.0
        scaf_calls = 0
        scaf_correct = False
        for row in public_arms:
            scaf_cost += row["action_cost"]
            scaf_calls += 1
            if row["correct"]:
                scaf_correct = True
                break
        public_scaf_success.append(float(scaf_correct))
        public_scaf_cost.append(scaf_cost)
        public_scaf_calls.append(float(scaf_calls))

        correct_arms = [row for row in arms if row["correct"]]
        oracle_success.append(float(bool(correct_arms)))
        oracle_choice = (
            min(correct_arms, key=lambda row: row["action_cost"])
            if correct_arms
            else min(arms, key=lambda row: row["action_cost"])
        )
        oracle_cost.append(oracle_choice["action_cost"])

        solution_arms = [row for row in arms if row["arm_kind"] == "solution"]
        strongest = max(
            solution_arms or arms,
            key=lambda row: (row["arm_strength"], row["arm_order"]),
        )
        strongest_success.append(float(strongest["correct"]))
        strongest_cost.append(strongest["action_cost"])
        decisions.append(
            {
                "id": item_id,
                "item_key": item_key,
                "sample_index": sample_index,
                "selected_arm": selected["arm_name"],
                "predicted_success": selected["predicted_success"],
                "correct": bool(selected["correct"]),
                "action_cost": selected["action_cost"],
                "fallback_arms": [row["arm_name"] for row in attempted],
                "fallback_correct": any(row["correct"] for row in attempted),
            }
        )

    metrics = {
        "num_test_examples": len({item_key for item_key, _ in by_trajectory}),
        "num_test_trajectories": len(by_trajectory),
        "direct_router": {
            "accuracy": _average(direct_success),
            "average_total_tokens": _average(direct_cost),
            "average_calls": 1.0,
        },
        "router_with_bounded_fallback": {
            "accuracy": _average(fallback_success),
            "average_total_tokens": _average(fallback_cost),
            "average_calls": _average(fallback_calls),
            "max_fallback_arms": max_fallback_arms,
        },
        "public_scaf_progressive": {
            "accuracy": _average(public_scaf_success),
            "average_total_tokens": _average(public_scaf_cost),
            "average_calls": _average(public_scaf_calls),
        },
        "full_information_oracle": {
            "accuracy": _average(oracle_success),
            "average_total_tokens": _average(oracle_cost),
        },
        "strongest_solution": {
            "accuracy": _average(strongest_success),
            "average_total_tokens": _average(strongest_cost),
            "average_calls": 1.0,
        },
    }
    return metrics, decisions


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import joblib
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    records = load_records(args.input)
    train_ids, validation_ids, test_ids = split_item_ids(
        [record["item_key"] for record in records],
        args.seed,
        args.train_fraction,
        args.validation_fraction,
    )
    train_records = [record for record in records if record["item_key"] in train_ids]
    validation_records = [
        record for record in records if record["item_key"] in validation_ids
    ]
    test_records = [record for record in records if record["item_key"] in test_ids]
    train_labels = [int(record["correct"]) for record in train_records]
    validation_labels = [int(record["correct"]) for record in validation_records]
    test_labels = [int(record["correct"]) for record in test_records]
    if len(set(train_labels)) < 2 or len(set(validation_labels)) < 2:
        raise ValueError("Train and validation splits must each contain both reward classes")

    router = build_router_pipeline(args.max_features)
    router.fit(_records_to_frame(train_records), train_labels)
    calibrated = calibrate_router(
        router, _records_to_frame(validation_records), validation_labels
    )
    probabilities = calibrated.predict_proba(_records_to_frame(test_records))[:, 1].tolist()
    routing_metrics, decisions = evaluate_routing(
        test_records,
        probabilities,
        args.target_success,
        args.max_fallback_arms,
    )
    probability_metrics = {
        "brier_score": brier_score_loss(test_labels, probabilities),
        "log_loss": log_loss(test_labels, probabilities, labels=[0, 1]),
        "roc_auc": (
            roc_auc_score(test_labels, probabilities) if len(set(test_labels)) == 2 else None
        ),
        "expected_calibration_error": expected_calibration_error(
            test_labels, probabilities, args.calibration_bins
        ),
        "positive_rate": _average([float(label) for label in test_labels]),
    }
    result = {
        "split": {
            "train_examples": len(train_ids),
            "validation_examples": len(validation_ids),
            "test_examples": len(test_ids),
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "test_records": len(test_records),
        },
        "probability_metrics": probability_metrics,
        "routing_metrics": routing_metrics,
        "config": {
            "target_success": args.target_success,
            "max_fallback_arms": args.max_fallback_arms,
            "seed": args.seed,
            "max_features": args.max_features,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, args.output_dir / "router.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a calibrated cost-aware scaffold router")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--target-success", type=float, default=0.5)
    parser.add_argument("--max-fallback-arms", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_fraction <= 0 or args.validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave a test split")
    if args.max_fallback_arms < 0:
        raise ValueError("max-fallback-arms must be non-negative")
    result = train_and_evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
