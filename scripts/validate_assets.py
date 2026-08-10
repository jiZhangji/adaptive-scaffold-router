#!/usr/bin/env python3
"""Offline structural validation for the model and Scaf-GRPO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_DATASET_SHA256 = "e20a6bd8e1e01ddc02a708dacac94dca406c0053de092a076e4f3910127d213e"
EXPECTED_DATASET_ROWS = 12_880
REQUIRED_DATASET_COLUMNS = {
    "question",
    "reward_model",
    "knowledge_components_parts",
    "planning_skeleton_parts",
    "solution_breakdown_parts",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_files(model_path: Path) -> list[Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index.get("weight_map", {}).values()))
        if not names:
            raise ValueError("model.safetensors.index.json has an empty weight_map")
        return [model_path / name for name in names]
    return sorted(model_path.glob("*.safetensors"))


def validate_model(model_path: Path, min_weight_bytes: int) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    incomplete = list(model_path.rglob("*.incomplete"))
    if incomplete:
        raise ValueError(f"Model download still has {len(incomplete)} incomplete file(s)")

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    weight_files = _weight_files(model_path)
    missing = [str(path) for path in weight_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model shard(s): {missing}")
    if not weight_files:
        raise FileNotFoundError("No .safetensors model weights were found")

    weight_bytes = sum(path.stat().st_size for path in weight_files)
    if weight_bytes < min_weight_bytes:
        raise ValueError(
            f"Model weights are only {weight_bytes / 1e9:.2f} GB; "
            f"expected at least {min_weight_bytes / 1e9:.2f} GB"
        )

    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    tensor_count = 0
    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            tensor_count += len(handle.keys())
    if tensor_count == 0:
        raise ValueError("Safetensors files contain no tensors")

    loaded_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if len(tokenizer) <= 0:
        raise ValueError("Tokenizer vocabulary is empty")

    return {
        "path": str(model_path.resolve()),
        "model_type": getattr(loaded_config, "model_type", config.get("model_type")),
        "weight_files": len(weight_files),
        "weight_bytes": weight_bytes,
        "tensor_count": tensor_count,
        "tokenizer_size": len(tokenizer),
    }


def validate_dataset(dataset_path: Path) -> dict[str, Any]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    actual_sha256 = sha256_file(dataset_path)
    if actual_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Dataset SHA256 mismatch: expected {EXPECTED_DATASET_SHA256}, got {actual_sha256}"
        )

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(dataset_path)
    columns = set(parquet.schema_arrow.names)
    missing_columns = sorted(REQUIRED_DATASET_COLUMNS - columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if parquet.metadata.num_rows != EXPECTED_DATASET_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_DATASET_ROWS} dataset rows, got {parquet.metadata.num_rows}"
        )

    return {
        "path": str(dataset_path.resolve()),
        "sha256": actual_sha256,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": sorted(columns),
        "bytes": dataset_path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-model-bytes", type=int, default=2_500_000_000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {"ok": False, "model": None, "dataset": None, "errors": []}

    try:
        report["model"] = validate_model(args.model, args.min_model_bytes)
    except Exception as exc:  # Waiting mode needs one concise, actionable error.
        report["errors"].append(f"model: {exc}")

    try:
        report["dataset"] = validate_dataset(args.dataset)
    except Exception as exc:
        report["errors"].append(f"dataset: {exc}")

    report["ok"] = not report["errors"]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quiet or not report["ok"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
