from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_frontiers(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["id"]: row for row in rows}


def chosen_arm(row: dict[str, Any], choice_key: str) -> dict[str, Any]:
    choice = row[choice_key]
    return next(arm for arm in row["arms"] if arm["arm_name"] == choice)


def compare_frontiers(
    weaker: dict[str, dict[str, Any]],
    stronger: dict[str, dict[str, Any]],
    choice_key: str,
) -> dict[str, Any]:
    common_ids = sorted(set(weaker) & set(stronger))
    if not common_ids:
        raise ValueError("The two runs have no common example ids")

    transitions: Counter[str] = Counter()
    weaker_hint_tokens = []
    stronger_hint_tokens = []
    weaker_no_hint = []
    stronger_no_hint = []
    changed = 0
    fewer_hint_tokens = 0
    more_hint_tokens = 0
    equal_hint_tokens = 0

    for item_id in common_ids:
        weak_row = weaker[item_id]
        strong_row = stronger[item_id]
        weak_arm = chosen_arm(weak_row, choice_key)
        strong_arm = chosen_arm(strong_row, choice_key)
        transitions[f"{weak_arm['arm_name']} -> {strong_arm['arm_name']}"] += 1
        changed += weak_arm["arm_name"] != strong_arm["arm_name"]
        weak_tokens = float(weak_arm["hint_tokens"])
        strong_tokens = float(strong_arm["hint_tokens"])
        weaker_hint_tokens.append(weak_tokens)
        stronger_hint_tokens.append(strong_tokens)
        weaker_no_hint.append(float(weak_row["no_hint_p"]))
        stronger_no_hint.append(float(strong_row["no_hint_p"]))
        if strong_tokens < weak_tokens:
            fewer_hint_tokens += 1
        elif strong_tokens > weak_tokens:
            more_hint_tokens += 1
        else:
            equal_hint_tokens += 1

    count = len(common_ids)
    return {
        "common_examples": count,
        "choice_key": choice_key,
        "choice_changed_fraction": changed / count,
        "stronger_uses_fewer_hint_tokens_fraction": fewer_hint_tokens / count,
        "stronger_uses_more_hint_tokens_fraction": more_hint_tokens / count,
        "equal_hint_tokens_fraction": equal_hint_tokens / count,
        "weaker_average_selected_hint_tokens": sum(weaker_hint_tokens) / count,
        "stronger_average_selected_hint_tokens": sum(stronger_hint_tokens) / count,
        "weaker_average_no_hint_accuracy": sum(weaker_no_hint) / count,
        "stronger_average_no_hint_accuracy": sum(stronger_no_hint) / count,
        "transitions": dict(transitions.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the scaffold frontier for two model capability levels."
    )
    parser.add_argument("--weaker", type=Path, required=True)
    parser.add_argument("--stronger", type=Path, required=True)
    parser.add_argument(
        "--choice-key",
        choices=("utility_choice", "threshold_choice", "band_choice"),
        default="threshold_choice",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_frontiers(
        load_frontiers(args.weaker),
        load_frontiers(args.stronger),
        args.choice_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
