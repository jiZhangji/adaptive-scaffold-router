from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the two-idea probe model and dataset.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--model-name", default="Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset-id", default="hkuzxc/scaf-grpo-dataset")
    parser.add_argument("--dataset-file", default="Qwen2.5-Math-1.5B.parquet")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it in the selected Conda environment."
        ) from exc

    project_root = args.project_root.expanduser().resolve()
    model_dir = project_root / "models" / args.model_name
    data_dir = project_root / "data" / "DeepScaleR"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model_id,
        repo_type="model",
        local_dir=model_dir,
        max_workers=args.max_workers,
    )
    dataset_path = hf_hub_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        filename=args.dataset_file,
        local_dir=data_dir,
    )

    result = {
        "model_id": args.model_id,
        "model_path": str(model_dir),
        "dataset_id": args.dataset_id,
        "dataset_path": str(Path(dataset_path).resolve()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
