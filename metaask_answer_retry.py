from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier_probe import SYSTEM_PROMPT, build_math_verifier
from metaask_probe import LocalGenerator, analyze_records, chat_prompt


def run(args: argparse.Namespace) -> None:
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = [row for row in records if row["variant"] == "no_help"]
    if not baseline:
        raise ValueError("Input contains no no_help trajectories")

    generator = LocalGenerator(args)
    verifier = build_math_verifier()
    prompts = []
    oracle_answers = []
    for row in baseline:
        oracle_answer = "YES" if row["correct"] else "NO"
        oracle_answers.append(oracle_answer)
        previous = str(row["text"])[-args.max_previous_characters :]
        prompts.append(
            chat_prompt(
                generator.tokenizer,
                SYSTEM_PROMPT,
                f"Problem: {row['question']}\n\n"
                f"Your previous attempt was:\n{previous}\n\n"
                f"A verifier answered whether that final answer was correct: {oracle_answer}.\n"
                "If the response is NO, identify a different approach and solve again from scratch. "
                "If it is YES, provide the verified answer. Do not ask for more information.",
            )
        )

    generated = generator.generate(
        prompts,
        args.max_new_tokens,
        args.seed,
        stop_after_boxed=args.stop_after_boxed,
    )
    retry_records = []
    for source, oracle_answer, result in zip(baseline, oracle_answers, generated):
        retry_records.append(
            {
                "id": source["id"],
                "source_id": source["source_id"],
                "question": source["question"],
                "reference": source["reference"],
                "sample_index": source["sample_index"],
                "variant": "answer_verification_retry",
                "correct": verifier(result["text"], source["reference"]),
                "oracle_answer": oracle_answer,
                "external_information_tokens": 1,
                "total_generated_tokens": int(source["output_tokens"]) + 1 + result["output_tokens"],
                "previous_correct": source["correct"],
                **result,
            }
        )

    merged = records + retry_records
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "raw_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = analyze_records(merged)
    summary["controlled_verification_note"] = (
        "The environment verifies the student's previous final answer with one bit. "
        "This isolates feedback utility but is not yet a learned self-asked proposition policy."
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-previous-characters", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3042)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-after-boxed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
