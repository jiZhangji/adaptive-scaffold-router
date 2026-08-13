#!/usr/bin/env python3
"""Generate resumable, scaffold-grounded subproblem candidates with DeepSeek."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value if str(item).strip())
    return str(value)


def ground_truth(row: dict[str, Any]) -> str:
    reward = row.get("reward_model")
    if isinstance(reward, dict):
        return as_text(reward.get("ground_truth")).strip()
    return ""


def reference_solution(row: dict[str, Any], reference: str) -> str:
    for key in ("solution", "model_think", "model_answer"):
        value = as_text(row.get(key)).strip()
        if value:
            return value
    return reference


def normalize_answer(value: str) -> str:
    value = re.sub(r"\\boxed\s*\{(.*?)\}", r"\1", value, flags=re.S)
    return re.sub(r"[\s{}$]", "", value).lower()


def stable_id(source_id: str, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"{source_id}::{digest}"


def build_prompt(row: dict[str, Any], question: str, reference: str) -> str:
    knowledge = as_text(row.get("knowledge_components_parts"))[:5000]
    planning = as_text(row.get("planning_skeleton_parts"))[:5000]
    breakdown = as_text(row.get("solution_breakdown_parts"))[:7000]
    solution = reference_solution(row, reference)[:7000]
    return f"""
Create exactly one easier prerequisite subproblem for the original math problem.

Rules:
1. The subproblem must be self-contained and independently solvable.
2. Its answer must be genuinely used in a solution of the original problem.
3. It must be easier than the original problem.
4. It must have one short, exact answer suitable for automatic checking.
5. It must not ask for, copy, or reveal the original final answer.
6. Prefer turning an intermediate computation from the supplied scaffold into a question.
7. Do not merely restate the original problem and do not provide several subproblems.
8. Return one JSON object with exactly these string fields:
   "subproblem", "answer", "relation", "source_step", "verification".
9. "verification" must briefly derive the subproblem answer, not the original answer.

Original problem:
{question}

Original final answer (for leakage avoidance only):
{reference}

Knowledge scaffold:
{knowledge}

Planning scaffold:
{planning}

Solution-breakdown scaffold:
{breakdown}

Reference solution:
{solution}
""".strip()


def parse_candidate(content: str) -> dict[str, str]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    allowed = ("subproblem", "answer", "relation", "source_step", "verification")
    result = {key: as_text(value.get(key)).strip() for key in allowed}
    if not result["subproblem"] or not result["answer"]:
        raise ValueError("empty subproblem or answer")
    if len(result["subproblem"]) > 1200:
        raise ValueError("subproblem is too long")
    if len(result["answer"]) > 400:
        raise ValueError("answer is too long")
    if len(result["verification"]) > 2000:
        raise ValueError("verification is too long")
    return result


def request_candidate(
    *, api_key: str, api_url: str, model: str, prompt: str, timeout: int
) -> tuple[dict[str, str], dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You construct verifiable mathematical prerequisite subproblems. "
                    "Return valid JSON only and follow every constraint."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_value = json.loads(response.read().decode("utf-8"))
    choice = response_value["choices"][0]
    content = choice["message"].get("content") or ""
    return parse_candidate(content), response_value.get("usage", {}), choice.get(
        "finish_reason", ""
    )


def validate_candidate(candidate: dict[str, str], reference: str) -> None:
    if normalize_answer(candidate["answer"]) == normalize_answer(reference):
        raise ValueError("subproblem answer equals the original final answer")
    if normalize_answer(candidate["subproblem"]) == normalize_answer(reference):
        raise ValueError("subproblem text equals the original final answer")


def read_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["id"]))
    return completed


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def source_accuracy(row: dict[str, Any]) -> float:
    try:
        return float(row.get("accuracy", 0.0))
    except (TypeError, ValueError):
        return 0.0


def select_rows(
    rows: list[dict[str, Any]], limit: int, seed: int, max_source_accuracy: float
) -> tuple[list[dict[str, Any]], int]:
    eligible = [
        row
        for row in rows
        if as_text(row.get("question")).strip()
        and ground_truth(row)
        and source_accuracy(row) <= max_source_accuracy
    ]
    random.Random(seed).shuffle(eligible)
    return eligible[:limit] if limit > 0 else eligible, len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scaffold-grounded subproblem candidates using DeepSeek JSON mode."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit", type=int, default=64, help="0 means all eligible rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-accuracy", type=float, default=0.9)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-seconds", type=float, default=10.0)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = args.errors or args.output.with_name(args.output.stem + "_errors.jsonl")
    summary_path = args.summary or args.output.with_name(args.output.stem + "_summary.json")

    import pyarrow.parquet as pq

    all_rows = pq.read_table(args.data).to_pylist()
    rows, eligible_count = select_rows(
        all_rows, args.limit, args.seed, args.max_source_accuracy
    )
    completed = read_completed(args.output)
    completed_before_run = len(completed)
    success = skipped = failure = 0
    reasons: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()

    print(
        f"eligible={eligible_count} selected={len(rows)} already_completed={len(completed)} "
        f"model={args.model}",
        flush=True,
    )
    for index, row in enumerate(rows, 1):
        question = as_text(row.get("question")).strip()
        source_id = as_text(row.get("id")).strip()
        reference = ground_truth(row)
        item_id = stable_id(source_id, question)
        if item_id in completed:
            skipped += 1
            print(f"[{index}/{len(rows)}] skip {item_id}", flush=True)
            continue

        prompt = build_prompt(row, question, reference)
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                candidate, usage, finish_reason = request_candidate(
                    api_key=api_key,
                    api_url=args.api_url,
                    model=args.model,
                    prompt=prompt,
                    timeout=args.timeout,
                )
                validate_candidate(candidate, reference)
                record = {
                    "id": item_id,
                    "source_id": source_id,
                    "question": question,
                    "reference": reference,
                    "subproblem": candidate["subproblem"],
                    "subproblem_answer": candidate["answer"],
                    "relation": candidate["relation"],
                    "source_step": candidate["source_step"],
                    "verification": candidate["verification"],
                    "generator_model": args.model,
                    "finish_reason": finish_reason,
                    "usage": usage,
                }
                append_jsonl(args.output, record)
                for key, value in usage.items():
                    if isinstance(value, int):
                        usage_totals[key] += value
                completed.add(item_id)
                success += 1
                print(
                    f"[{index}/{len(rows)}] OK {item_id}: "
                    f"{candidate['subproblem'][:100]}",
                    flush=True,
                )
                break
            except Exception as exc:  # API errors must be logged and retried.
                last_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[{index}/{len(rows)}] attempt {attempt}/{args.retries} failed: "
                    f"{last_error}",
                    flush=True,
                )
                if attempt < args.retries:
                    time.sleep(args.retry_seconds * attempt)
        else:
            failure += 1
            reason = last_error.split(":", 1)[0] if last_error else "unknown"
            reasons[reason] += 1
            append_jsonl(
                errors,
                {
                    "id": item_id,
                    "source_id": source_id,
                    "question": question,
                    "error": last_error,
                    "generator_model": args.model,
                },
            )

        summary = {
            "generator_model": args.model,
            "data": str(args.data.resolve()),
            "output": str(args.output.resolve()),
            "eligible_rows": eligible_count,
            "selected_rows": len(rows),
            "completed_before_run": completed_before_run,
            "success_this_run": success,
            "skipped_this_run": skipped,
            "failed_this_run": failure,
            "total_output_rows": len(completed),
            "failure_reasons": dict(reasons),
            "usage_this_run": dict(usage_totals),
            "seed": args.seed,
            "max_source_accuracy": args.max_source_accuracy,
            "important_limitation": (
                "These are teacher-generated candidates, not validated training examples. "
                "They still require leakage checks, answer verification, student q calibration, "
                "and a matched relevance control."
            ),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if not rows:
        summary = {
            "generator_model": args.model,
            "data": str(args.data.resolve()),
            "output": str(args.output.resolve()),
            "eligible_rows": eligible_count,
            "selected_rows": 0,
            "completed_before_run": completed_before_run,
            "success_this_run": 0,
            "skipped_this_run": 0,
            "failed_this_run": 0,
            "total_output_rows": len(completed),
            "failure_reasons": {},
            "usage_this_run": {},
            "seed": args.seed,
            "max_source_accuracy": args.max_source_accuracy,
            "important_limitation": (
                "These are teacher-generated candidates, not validated training examples. "
                "They still require leakage checks, answer verification, student q calibration, "
                "and a matched relevance control."
            ),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
