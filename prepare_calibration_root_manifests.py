#!/usr/bin/env python3
"""Freeze calibration root assignments and recover them from partial JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from calibrate_helpful_subproblems import minimal_plan, read_jsonl
from capability_scaffold import stable_question_key


def read_root_history(path: Path) -> tuple[list[str], Counter[str]]:
    order: list[str] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    if not path.exists():
        return order, counts
    for row in read_jsonl(path):
        root_id = str(row["root_id"])
        counts[root_id] += 1
        if root_id not in seen:
            seen.add(root_id)
            order.append(root_id)
    return order, counts


def build_manifests(
    eligible_roots: list[str],
    histories: list[tuple[list[str], Counter[str]]],
    target_sizes: list[int],
    seed: int,
) -> list[list[str]]:
    """Prefer every previously sampled root, then deterministically fill holes.

    A root can occur in two shard files after a bad resume with a changed
    candidate universe. In that case, retain it in the shard holding more
    records; that is normally its original, more-complete assignment.
    """
    eligible = set(eligible_roots)
    owners: dict[str, int] = {}
    all_historical = set().union(*(set(order) for order, _ in histories))
    for root_id in all_historical:
        choices = [
            shard_index
            for shard_index, (order, _) in enumerate(histories)
            if root_id in set(order)
        ]
        owners[root_id] = max(
            choices,
            key=lambda shard_index: (
                histories[shard_index][1][root_id],
                -histories[shard_index][0].index(root_id),
                -shard_index,
            ),
        )

    manifests: list[list[str]] = []
    assigned: set[str] = set()
    for shard_index, ((order, _), target) in enumerate(zip(histories, target_sizes)):
        recovered = [
            root_id
            for root_id in order
            if root_id in eligible and owners.get(root_id) == shard_index
        ]
        recovered = recovered[:target]
        manifests.append(recovered)
        assigned.update(recovered)

    pool = sorted(eligible - assigned)
    random.Random(seed).shuffle(pool)
    cursor = 0
    for shard_index, target in enumerate(target_sizes):
        needed = target - len(manifests[shard_index])
        manifests[shard_index].extend(pool[cursor : cursor + needed])
        cursor += needed
        if len(manifests[shard_index]) != target:
            raise ValueError(
                f"Not enough eligible roots for shard {shard_index}: "
                f"{len(manifests[shard_index])} < {target}"
            )

    flattened = [root_id for manifest in manifests for root_id in manifest]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Frozen shard manifests must be disjoint")
    return manifests


def read_manifest(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--root-limit", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-plan-words", type=int, default=12)
    args = parser.parse_args()

    if args.root_limit < args.num_shards or args.num_shards < 1:
        raise ValueError("root_limit must be positive and at least num_shards")

    import pandas as pd

    source = pd.read_parquet(args.source_data)
    question_keys = {
        stable_question_key(str(row["question"]))
        for row in source.to_dict(orient="records")
    }
    eligible_roots: list[str] = []
    seen: set[str] = set()
    for candidate in read_jsonl(args.candidates):
        root_id = str(candidate["root_id"])
        if root_id in seen:
            continue
        if not minimal_plan(candidate, args.max_plan_words):
            continue
        if stable_question_key(str(candidate["question"])) not in question_keys:
            continue
        seen.add(root_id)
        eligible_roots.append(root_id)

    args.calibration_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = [
        args.calibration_dir / f"shard_{index}_roots.txt"
        for index in range(args.num_shards)
    ]
    target_sizes = [
        len(range(index, args.root_limit, args.num_shards))
        for index in range(args.num_shards)
    ]

    if all(path.exists() for path in manifest_paths):
        manifests = [read_manifest(path) for path in manifest_paths]
        for index, (manifest, target) in enumerate(zip(manifests, target_sizes)):
            if len(manifest) != target:
                raise ValueError(
                    f"Frozen shard {index} has {len(manifest)} roots; expected {target}"
                )
            missing = set(manifest) - set(eligible_roots)
            if missing:
                raise ValueError(f"Frozen shard {index} has missing candidate roots: {missing}")
        mode = "reused"
        histories = [read_root_history(args.calibration_dir / f"shard_{i}" / "help_results.jsonl") for i in range(args.num_shards)]
    else:
        if any(path.exists() for path in manifest_paths):
            raise ValueError("Either all frozen shard manifests must exist or none may exist")
        histories = [
            read_root_history(
                args.calibration_dir / f"shard_{index}" / "help_results.jsonl"
            )
            for index in range(args.num_shards)
        ]
        manifests = build_manifests(
            eligible_roots,
            histories,
            target_sizes,
            args.seed,
        )
        for path, manifest in zip(manifest_paths, manifests):
            path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
        mode = "created"

    summary: dict[str, Any] = {
        "mode": mode,
        "root_limit": args.root_limit,
        "eligible_roots": len(eligible_roots),
        "shards": [],
    }
    for index, (manifest, (history_order, _)) in enumerate(zip(manifests, histories)):
        summary["shards"].append(
            {
                "shard_index": index,
                "manifest_roots": len(manifest),
                "recovered_roots": len(set(manifest) & set(history_order)),
                "manifest": str(manifest_paths[index]),
            }
        )
    summary_path = args.calibration_dir / "root_manifest_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
