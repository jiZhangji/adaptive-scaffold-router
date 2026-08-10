from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FALLBACK_QUESTION = "Is the most uncertain intermediate assumption valid?"


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    self_asked = [row for row in records if row["variant"] == "self_asked_verification"]
    oracle_histogram = Counter(row.get("oracle_answer", "MISSING") for row in self_asked)
    accuracy_by_oracle: dict[str, list[bool]] = defaultdict(list)
    for row in self_asked:
        accuracy_by_oracle[str(row.get("oracle_answer", "MISSING"))].append(bool(row["correct"]))

    by_key: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for row in records:
        by_key[(row["id"], int(row["sample_index"]))][row["variant"]] = bool(row["correct"])
    paired = [value for value in by_key.values() if "no_help" in value and "self_asked_verification" in value]

    return {
        "self_asked_trajectories": len(self_asked),
        "oracle_answer_histogram": dict(oracle_histogram),
        "fallback_question_count": sum(
            row.get("verification_question") == FALLBACK_QUESTION for row in self_asked
        ),
        "empty_state_count": sum(not str(row.get("tentative_state", "")).strip() for row in self_asked),
        "average_verification_question_words": (
            sum(len(str(row.get("verification_question", "")).split()) for row in self_asked)
            / len(self_asked)
            if self_asked
            else 0.0
        ),
        "accuracy_by_oracle_answer": {
            key: sum(values) / len(values) for key, values in accuracy_by_oracle.items()
        },
        "paired_comparison": {
            "pairs": len(paired),
            "both_correct": sum(row["no_help"] and row["self_asked_verification"] for row in paired),
            "both_wrong": sum(not row["no_help"] and not row["self_asked_verification"] for row in paired),
            "metaask_rescues": sum(not row["no_help"] and row["self_asked_verification"] for row in paired),
            "metaask_harms": sum(row["no_help"] and not row["self_asked_verification"] for row in paired),
        },
        "question_examples": [
            {
                "oracle_answer": row.get("oracle_answer"),
                "correct": row["correct"],
                "verification_question": row.get("verification_question"),
            }
            for row in self_asked[:8]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    result = analyze(records)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
