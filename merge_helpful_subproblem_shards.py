#!/usr/bin/env python3
"""Merge disjoint q_help calibration shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    summaries = []
    seen_roots = set()
    for shard_dir in args.shard_dir:
        summary = json.loads((shard_dir / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        for candidate in read_jsonl(shard_dir / "training_candidates.jsonl"):
            root_id = str(candidate["root_id"])
            if root_id in seen_roots:
                raise ValueError(f"Calibration shards overlap at root {root_id}")
            seen_roots.add(root_id)
            candidates.append(candidate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.sort(key=lambda row: str(row["root_id"]))
    with (args.output_dir / "training_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    selected_roots = sum(int(summary["selected_roots"]) for summary in summaries)
    payload = {
        "selected_roots": selected_roots,
        "training_roots": len(candidates),
        "training_candidate_rate": len(candidates) / selected_roots if selected_roots else 0.0,
        "q_definition": "P(root correct | relevant answer-free minimal plan)",
        "q_low": summaries[0]["q_low"],
        "q_high": summaries[0]["q_high"],
        "num_shards": len(summaries),
        "shards": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
