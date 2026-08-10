from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from capability_scaffold import (
    ControllerConfig,
    RootState,
    RolloutArm,
    compile_manifest,
)


def load_frontier_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("Frontier file is empty")
    return records


def build_root_states(records: list[dict[str, Any]]) -> list[RootState]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["question"])].append(record)

    roots = []
    for question, question_records in grouped.items():
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in question_records:
            by_arm[str(record["arm_name"])].append(record)
        none_records = by_arm.get("none")
        if not none_records:
            raise ValueError(f"Question has no no-hint arm: {question[:80]}")
        first = question_records[0]
        scaffolds = []
        for arm_name, arm_records in by_arm.items():
            if arm_name == "none":
                continue
            arm_first = arm_records[0]
            scaffolds.append(
                RolloutArm(
                    name=arm_name,
                    rewards=tuple(float(item["correct"]) for item in arm_records),
                    hint_tokens=int(round(sum(item["hint_tokens"] for item in arm_records) / len(arm_records))),
                    strength=float(arm_first["arm_strength"]),
                    kind=str(arm_first["arm_kind"]),
                )
            )
        roots.append(
            RootState(
                id=str(first["id"]),
                question=question,
                answer=str(first["reference"]),
                root_rewards=tuple(float(item["correct"]) for item in none_records),
                scaffolds=tuple(scaffolds),
            )
        )
    return roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert frontier rollout records into a capability curriculum manifest."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band-low", type=float, default=0.25)
    parser.add_argument("--band-high", type=float, default=0.60)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-hint-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ControllerConfig(
        band_low=args.band_low,
        band_high=args.band_high,
        group_size=args.group_size,
        max_hint_tokens=args.max_hint_tokens,
    )
    roots = build_root_states(load_frontier_records(args.input))
    decisions, summary = compile_manifest(roots, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "curriculum.jsonl").open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
