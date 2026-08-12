#!/usr/bin/env python3
"""Download and validate Qwen2.5-Math-1.5B Scaf-GRPO reproduction assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REQUIRED_DATA_COLUMNS = {
    "question",
    "reward_model",
    "knowledge_components_parts",
    "planning_skeleton_parts",
    "solution_breakdown_parts",
}

# This is the context extension documented in the official Scaf-GRPO README.
PAPER_CONTEXT_CONFIG = {
    "sliding_window": None,
    "use_sliding_window": False,
    "rope_theta": 15000,
    "max_position_embeddings": 6144,
}


def weight_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return [model_dir / name for name in sorted(set(index["weight_map"].values()))]
    return sorted(model_dir.glob("*.safetensors"))


def patch_context_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    backup_path = model_dir / "config.huggingface-original.json"
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    before = {key: config.get(key) for key in PAPER_CONTEXT_CONFIG}
    config.update(PAPER_CONTEXT_CONFIG)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"before": before, "after": PAPER_CONTEXT_CONFIG, "backup": str(backup_path)}


def validate(model_dir: Path, dataset_path: Path, scaf_repo: Path) -> dict[str, Any]:
    incomplete = list(model_dir.rglob("*.incomplete"))
    if incomplete:
        raise RuntimeError(f"Model still has {len(incomplete)} incomplete download file(s)")

    weights = weight_files(model_dir)
    missing_weights = [str(path) for path in weights if not path.is_file()]
    if not weights or missing_weights:
        raise RuntimeError(f"Missing model weights: {missing_weights or 'no safetensors found'}")
    weight_bytes = sum(path.stat().st_size for path in weights)
    if weight_bytes < 2_500_000_000:
        raise RuntimeError(f"1.5B weights are unexpectedly small: {weight_bytes / 1e9:.2f} GB")

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(dataset_path)
    columns = set(parquet.schema_arrow.names)
    missing_columns = sorted(REQUIRED_DATA_COLUMNS - columns)
    if missing_columns:
        raise RuntimeError(f"Training parquet is missing columns: {missing_columns}")
    if parquet.metadata.num_rows != 12_880:
        raise RuntimeError(f"Expected 12,880 training rows, found {parquet.metadata.num_rows}")

    eval_relpaths = {
        "AIME24": "data/AIME24/math-verify/system-p1/test.parquet",
        "AIME25": "data/AIME25/math-verify/system-p1/test.parquet",
        "AMC23": "data/AMC23/math-verify/system-p1/test.parquet",
        "MinervaMath": "data/MinervaMath/math-verify/system-p1/test.parquet",
        "MATH-500": "data/MATH-500/math-verify/system-p1/test.parquet",
        "OlympiadBench": "data/OlympiadBench/math-verify/system-p1/test.parquet",
        "GaoKao2023en": "data/GaoKao2023en/math-verify/system-p1/test.parquet",
    }
    missing_eval = [name for name, rel in eval_relpaths.items() if not (scaf_repo / rel).is_file()]
    if missing_eval:
        raise RuntimeError(f"Official evaluation data missing for: {missing_eval}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    actual_context = {key: config.get(key) for key in PAPER_CONTEXT_CONFIG}
    if actual_context != PAPER_CONTEXT_CONFIG:
        raise RuntimeError(f"Paper context config mismatch: {actual_context}")

    return {
        "ok": True,
        "model_path": str(model_dir.resolve()),
        "weight_files": len(weights),
        "weight_bytes": weight_bytes,
        "context_config": actual_context,
        "training_dataset": str(dataset_path.resolve()),
        "training_rows": parquet.metadata.num_rows,
        "evaluation_datasets": list(eval_relpaths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scaf-repo", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset-id", default="hkuzxc/scaf-grpo-dataset")
    parser.add_argument("--dataset-file", default="Qwen2.5-Math-1.5B.parquet")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    scaf_repo = args.scaf_repo.expanduser().resolve()
    model_dir = project_root / "models" / "Qwen2.5-Math-1.5B"
    dataset_path = project_root / "data" / "DeepScaleR" / args.dataset_file
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.validate_only:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub in the selected Conda environment") from exc
        snapshot_download(
            repo_id=args.model_id,
            repo_type="model",
            local_dir=model_dir,
            max_workers=args.max_workers,
        )
        hf_hub_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            filename=args.dataset_file,
            local_dir=dataset_path.parent,
        )

    patch_report = patch_context_config(model_dir)
    report = validate(model_dir, dataset_path, scaf_repo)
    report["config_patch"] = patch_report
    report_path = project_root / "outputs" / "qwen_math_1_5b_asset_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
