from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from frontier_probe import _as_text_list, build_math_verifier, load_probe_rows, stable_item_id
from metaask_probe import LocalGenerator, chat_prompt


ACTIONS = ("NONE", "KNOWLEDGE", "PLANNING", "STEP")
ACTION_COLUMNS = {
    "KNOWLEDGE": "knowledge_components_parts",
    "PLANNING": "planning_skeleton_parts",
    "STEP": "solution_breakdown_parts",
}
ACTION_VARIANTS = {
    "policy_action": "policy",
    "random_action": "random",
    "confidence_action": "confidence",
    "oracle_action": "oracle",
}


def parse_action(text: str) -> str | None:
    match = re.search(r"\b(NONE|KNOWLEDGE|PLANNING|STEP)\b", text.upper())
    return match.group(1) if match else None


def prompt_for_action(tokenizer: Any, question: str) -> str:
    return chat_prompt(
        tokenizer,
        "Choose a help action. Output exactly one label and nothing else.",
        f"Problem: {question}\n"
        "Choose one: NONE, KNOWLEDGE, PLANNING, STEP.\n"
        "NONE means solve independently. KNOWLEDGE asks for one relevant fact. "
        "PLANNING asks for one short plan. STEP asks for one concrete next step.\n"
        "Label:",
    )


def prompt_for_confidence(tokenizer: Any, question: str) -> str:
    return chat_prompt(
        tokenizer,
        "Classify the smallest type of help needed. Output exactly one label and nothing else.",
        f"Problem: {question}\n"
        "Use NONE if confident. Otherwise choose KNOWLEDGE for a missing fact, "
        "PLANNING for an approach, or STEP for the next derivation.\nLabel:",
    )


def assistance(row: dict[str, Any], action: str) -> str:
    column = ACTION_COLUMNS.get(action)
    if not column:
        return ""
    parts = _as_text_list(row.get(column))
    return parts[0] if parts else ""


def action_prompt(tokenizer: Any, question: str, action: str, hint: str) -> str:
    user = f"Problem: {question}"
    if action != "NONE" and hint:
        label = action.lower().capitalize()
        user += f"\n\nMinimal {label} assistance: {hint}\nUse it only as guidance."
    return chat_prompt(
        tokenizer,
        "Please reason step by step, and put your final answer within \\boxed{}.",
        user,
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_variant[record["variant"]].append(record)
        paired[(record["id"], record["sample_index"])][record["variant"]] = record

    variants = {}
    for name, rows in sorted(by_variant.items()):
        action_counts = Counter(row["action"] for row in rows)
        variants[name] = {
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "average_hint_tokens": sum(row["hint_tokens"] for row in rows) / len(rows),
            "action_histogram": dict(action_counts),
            "invalid_action_rate": sum(row["invalid_action"] for row in rows) / len(rows),
        }

    no_help = {key: item.get("no_help") for key, item in paired.items()}
    for name in variants:
        if name == "no_help":
            continue
        rescues = harms = eligible = 0
        for key, item in paired.items():
            base = no_help[key]
            current = item.get(name)
            if not base or not current:
                continue
            if not base["correct"]:
                eligible += 1
                rescues += int(bool(current["correct"]))
            harms += int(bool(base["correct"]) and not bool(current["correct"]))
        variants[name]["rescue_rate_on_no_help_failures"] = rescues / eligible if eligible else 0.0
        variants[name]["harms_on_no_help_successes"] = harms
    return {"variants": variants}


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, eligible = load_probe_rows(args.data, args.limit, args.seed, args.max_source_accuracy)
    generator = LocalGenerator(args)
    tokenizer = generator.tokenizer
    verifier = build_math_verifier()
    jobs = []
    for row in rows:
        question = str(row["question"])
        source_id = str(row["id"])
        reference = str(row["reward_model"]["ground_truth"])
        for sample_index in range(args.samples_per_variant):
            jobs.append(
                {
                    "id": stable_item_id(source_id, question),
                    "source_id": source_id,
                    "question": question,
                    "reference": reference,
                    "row": row,
                    "sample_index": sample_index,
                }
            )

    def solve(variant: str, actions: list[str], invalid: list[bool]) -> list[dict[str, Any]]:
        hints = [assistance(job["row"], action) for job, action in zip(jobs, actions)]
        prompts = [
            action_prompt(tokenizer, job["question"], action, hint)
            for job, action, hint in zip(jobs, actions, hints)
        ]
        generated = generator.generate(
            prompts, args.max_new_tokens, args.seed + len(records) + 100, args.stop_after_boxed
        )
        return [
            {
                "id": job["id"],
                "source_id": job["source_id"],
                "question": job["question"],
                "reference": job["reference"],
                "sample_index": job["sample_index"],
                "variant": variant,
                "action": action,
                "invalid_action": bad,
                "hint_tokens": len(tokenizer(hint, add_special_tokens=False)["input_ids"]),
                "correct": verifier(result["text"], job["reference"]),
                **result,
            }
            for job, action, bad, hint, result in zip(jobs, actions, invalid, hints, generated)
        ]

    records = solve("no_help", ["NONE"] * len(jobs), [False] * len(jobs))

    selector_outputs = generator.generate(
        [prompt_for_action(tokenizer, job["question"]) for job in jobs],
        args.action_max_new_tokens,
        args.seed + 500,
    )
    parsed = [parse_action(item["text"]) for item in selector_outputs]
    policy_actions = [item or "NONE" for item in parsed]
    records += solve("policy_action", policy_actions, [item is None for item in parsed])

    confidence_outputs = generator.generate(
        [prompt_for_confidence(tokenizer, job["question"]) for job in jobs],
        args.action_max_new_tokens,
        args.seed + 1000,
    )
    parsed = [parse_action(item["text"]) for item in confidence_outputs]
    confidence_actions = [item or "NONE" for item in parsed]
    records += solve("confidence_action", confidence_actions, [item is None for item in parsed])

    rng = random.Random(args.seed)
    records += solve(
        "random_action",
        [rng.choice(ACTIONS) for _ in jobs],
        [False] * len(jobs),
    )

    raw_path = args.output_dir / "raw_results.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = summarize(records)
    summary.update(
        {
            "experiment": "Constrained MetaAsk action-selection probe",
            "selected_questions": len(rows),
            "eligible_questions": eligible,
            "limitation": (
                "This is a constrained action-selection test. It does not yet train the "
                "selector with RL and does not test free-form proposition verification."
            ),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--samples-per-variant", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--action-max-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-accuracy", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-after-boxed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
