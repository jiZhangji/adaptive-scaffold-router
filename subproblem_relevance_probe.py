from __future__ import annotations

import argparse
import copy
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from frontier_probe import (
    SYSTEM_PROMPT,
    build_math_verifier,
    load_probe_rows,
    stable_item_id,
)
from metaask_probe import LocalGenerator, _reference_solution, chat_prompt


def parse_subproblem(text: str) -> tuple[str, str] | None:
    """Parse a teacher candidate without requiring one brittle output format."""
    question_match = re.search(r"<subproblem>(.*?)</subproblem>", text, re.I | re.S)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
    if question_match and answer_match:
        question = " ".join(question_match.group(1).split())
        answer = " ".join(answer_match.group(1).split())
    else:
        # Instruction-tuned teachers sometimes wrap the requested object in a
        # Markdown JSON fence even when XML tags were requested.
        object_match = re.search(r"\{.*\}", text, re.S)
        parsed = None
        if object_match:
            try:
                value = json.loads(object_match.group(0))
                if isinstance(value, dict):
                    parsed = value
            except json.JSONDecodeError:
                parsed = None
        if parsed is not None:
            question = " ".join(
                str(parsed.get("subproblem", parsed.get("question", ""))).split()
            )
            answer = " ".join(
                str(parsed.get("answer", parsed.get("target", ""))).split()
            )
        else:
            question_line = re.search(
                r"(?:subproblem|question)\s*:\s*(.+?)(?=\n\s*(?:answer|target)\s*:|$)",
                text,
                re.I | re.S,
            )
            answer_line = re.search(r"(?:answer|target)\s*:\s*([^\n]+)", text, re.I)
            if not question_line or not answer_line:
                return None
            question = " ".join(question_line.group(1).split())
            answer = " ".join(answer_line.group(1).split())
    if not question or not answer:
        return None
    return question, answer


def _normalise_answer(text: str) -> str:
    value = re.sub(r"\\boxed\s*\{(.*)\}", r"\1", str(text), flags=re.S)
    return re.sub(r"[\s{}$]", "", value).lower()


def _validate_q_band(q_low: float, q_high: float) -> None:
    if not 0.0 <= q_low <= q_high <= 1.0:
        raise ValueError("q band must satisfy 0 <= q_low <= q_high <= 1")


def analyze_records(
    candidates: list[dict[str, Any]],
    root_records: list[dict[str, Any]],
    subproblem_records: list[dict[str, Any]],
    q_low: float = 0.25,
    q_high: float = 0.75,
) -> dict[str, Any]:
    _validate_q_band(q_low, q_high)
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for record in root_records:
        by_variant[record["variant"]].append(record)
        paired[(record["id"], int(record["sample_index"]))][record["variant"]] = bool(
            record["correct"]
        )

    variants: dict[str, dict[str, float | int]] = {}
    for name, rows in sorted(by_variant.items()):
        variants[name] = {
            "trajectories": len(rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "average_external_information_tokens": sum(
                int(row.get("external_information_tokens", 0)) for row in rows
            )
            / len(rows),
        }

    def rescue_rate(name: str) -> float:
        eligible = [row for row in paired.values() if not row.get("no_help", False) and name in row]
        return sum(row[name] for row in eligible) / len(eligible) if eligible else 0.0

    no_help = float(variants.get("no_help", {}).get("accuracy", 0.0))
    question_only = float(variants.get("question_only", {}).get("accuracy", 0.0))
    relevant = float(variants.get("relevant_subproblem", {}).get("accuracy", 0.0))
    random_control = float(variants.get("random_subproblem", {}).get("accuracy", 0.0))
    subproblem_accuracy = (
        sum(bool(row["correct"]) for row in subproblem_records) / len(subproblem_records)
        if subproblem_records
        else 0.0
    )
    subproblem_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subproblem_records:
        subproblem_by_id[str(row["id"])].append(row)
    q_by_id = {
        item_id: sum(bool(row["correct"]) for row in rows) / len(rows)
        for item_id, rows in subproblem_by_id.items()
    }
    trainable = {
        item_id: q for item_id, q in q_by_id.items() if q_low <= q <= q_high
    }
    return {
        "experiment": "causal subproblem relevance probe",
        "candidate_generation": {
            "valid_candidates": len(candidates),
        },
        "subproblem_solve_accuracy": subproblem_accuracy,
        "q_calibration": {
            "q_low": q_low,
            "q_high": q_high,
            "candidate_count": len(q_by_id),
            "trainable_candidate_count": len(trainable),
            "trainable_candidate_rate": (
                len(trainable) / len(q_by_id) if q_by_id else 0.0
            ),
            "mean_q": sum(q_by_id.values()) / len(q_by_id) if q_by_id else 0.0,
            "q_by_id": q_by_id,
        },
        "variants": variants,
        "causal_checks": {
            "relevant_gain_over_no_help": relevant - no_help,
            "relevant_gain_over_random": relevant - random_control,
            "answer_value_beyond_question_only": relevant - question_only,
            "relevant_rescue_rate_on_no_help_failures": rescue_rate("relevant_subproblem"),
            "random_rescue_rate_on_no_help_failures": rescue_rate("random_subproblem"),
            "rescue_advantage_over_random": (
                rescue_rate("relevant_subproblem") - rescue_rate("random_subproblem")
            ),
        },
        "interpretation": (
            "A useful prerequisite should be independently solvable and should improve root "
            "accuracy more than an equally formatted subproblem drawn from another root."
        ),
    }


def run(args: argparse.Namespace) -> None:
    _validate_q_band(args.q_low, args.q_high)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, eligible = load_probe_rows(args.data, args.limit, args.seed, args.max_source_accuracy)
    verifier = build_math_verifier()

    teacher_args = copy.copy(args)
    teacher_args.model = args.teacher_model or args.model
    teacher_args.device = args.teacher_device or args.device
    teacher_args.temperature = args.teacher_temperature
    teacher_args.top_p = args.teacher_top_p
    teacher_generator = LocalGenerator(teacher_args)
    tokenizer = teacher_generator.tokenizer

    teacher_prompts = []
    row_meta = []
    for row in rows:
        question = str(row["question"])
        reference = str(row["reward_model"]["ground_truth"])
        solution = _reference_solution(row, reference)
        teacher_prompts.append(
            chat_prompt(
                tokenizer,
                "Extract one prerequisite; do not answer the original problem.",
                "Create one self-contained, independently checkable mathematical subproblem "
                "whose result is genuinely used by the reference solution. It must be easier "
                "than the original and must not simply ask for the original final answer. "
                "The answer must be a short exact mathematical value that can be checked by "
                "a final-answer verifier. Do not include a proof in <answer>. Return exactly "
                "<subproblem>...</subproblem><answer>...</answer>.\n\n"
                f"Original problem: {question}\nReference solution: {solution}",
            )
        )
        row_meta.append((row, question, reference))

    teacher_outputs = teacher_generator.generate(
        teacher_prompts, args.teacher_max_new_tokens, args.seed + 5000
    )

    # A base model often echoes the requested format. Retry only malformed
    # candidates with a shorter repair prompt rather than discarding the root.
    for retry in range(args.teacher_retries):
        failed = [index for index, output in enumerate(teacher_outputs) if parse_subproblem(output["text"]) is None]
        if not failed:
            break
        repair_prompts = [
            chat_prompt(
                tokenizer,
                "You format mathematical training data. Output XML only.",
                "Rewrite the candidate below as exactly one self-contained easier question and "
                "one short exact answer. Output only "
                "<subproblem>...</subproblem><answer>...</answer>.\n\n"
                f"Candidate:\n{teacher_outputs[index]['text']}",
            )
            for index in failed
        ]
        repaired = teacher_generator.generate(
            repair_prompts, args.teacher_max_new_tokens, args.seed + 5100 + retry
        )
        for index, output in zip(failed, repaired):
            teacher_outputs[index] = output

    if (args.teacher_model or args.model) == args.model and teacher_args.device == args.device:
        generator = teacher_generator
        generator.args = args
    else:
        del teacher_generator
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        generator = LocalGenerator(args)
        tokenizer = generator.tokenizer
    candidates = []
    rejected = []
    for (row, question, reference), generated in zip(row_meta, teacher_outputs):
        parsed = parse_subproblem(generated["text"])
        if parsed is None:
            rejected.append({"question": question, "reason": "parse_failure", "raw": generated["text"]})
            continue
        subproblem, answer = parsed
        leaks = _normalise_answer(answer) == _normalise_answer(reference)
        candidate = {
            "id": stable_item_id(str(row["id"]), question),
            "source_id": str(row["id"]),
            "question": question,
            "reference": reference,
            "subproblem": subproblem,
            "subproblem_answer": answer,
            "answer_leaks_root": leaks,
            "teacher_output_tokens": generated["output_tokens"],
        }
        if leaks:
            rejected.append({**candidate, "reason": "subproblem_answer_equals_root_answer"})
            continue
        candidates.append(candidate)

    if len(candidates) < 2:
        for filename, records in (
            ("candidates.jsonl", candidates),
            ("rejected_candidates.jsonl", rejected),
        ):
            with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = analyze_records(candidates, [], [], args.q_low, args.q_high)
        summary.update(
            {
                "eligible_questions": eligible,
                "selected_questions": len(rows),
                "valid_candidate_rate": len(candidates) / len(rows),
                "samples_per_variant": args.samples_per_variant,
                "model": args.model,
                "data": str(args.data),
                "seed": args.seed,
                "warning": (
                    "Fewer than two valid non-leaking subproblems were generated, so the "
                    "matched random control could not run."
                ),
            }
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    expanded = [
        {**candidate, "sample_index": sample_index}
        for candidate in candidates
        for sample_index in range(args.samples_per_variant)
    ]
    for candidate in candidates:
        support_text = candidate["subproblem"] + " " + candidate["subproblem_answer"]
        candidate["support_tokens"] = len(
            tokenizer(support_text, add_special_tokens=False)["input_ids"]
        )
    random_by_id = {
        source["id"]: min(
            (candidate for candidate in candidates if candidate["id"] != source["id"]),
            key=lambda candidate: abs(candidate["support_tokens"] - source["support_tokens"]),
        )
        for source in candidates
    }
    root_records: list[dict[str, Any]] = []

    def solve_variant(name: str, user_prompts: list[str], token_costs: list[int]) -> None:
        prompts = [chat_prompt(tokenizer, SYSTEM_PROMPT, user) for user in user_prompts]
        outputs = generator.generate(
            prompts,
            args.max_new_tokens,
            args.seed + 6000 + len(root_records),
            stop_after_boxed=args.stop_after_boxed,
        )
        for job, output, cost in zip(expanded, outputs, token_costs):
            root_records.append(
                {
                    "id": job["id"],
                    "source_id": job["source_id"],
                    "question": job["question"],
                    "reference": job["reference"],
                    "subproblem": job["subproblem"],
                    "sample_index": job["sample_index"],
                    "variant": name,
                    "correct": verifier(output["text"], job["reference"]),
                    "external_information_tokens": cost,
                    **output,
                }
            )

    zero_costs = [0] * len(expanded)
    solve_variant("no_help", [job["question"] for job in expanded], zero_costs)
    solve_variant(
        "question_only",
        [
            f"{job['question']}\n\nPotential prerequisite question: {job['subproblem']}\n"
            "No answer to the prerequisite is provided. Solve the original problem yourself."
            for job in expanded
        ],
        [len(tokenizer(job["subproblem"], add_special_tokens=False)["input_ids"]) for job in expanded],
    )
    solve_variant(
        "relevant_subproblem",
        [
            f"{job['question']}\n\nVerified prerequisite: {job['subproblem']}\n"
            f"Prerequisite answer: {job['subproblem_answer']}\n"
            "Use this fact only if it is relevant, then solve the original problem yourself."
            for job in expanded
        ],
        [
            len(
                tokenizer(
                    job["subproblem"] + " " + job["subproblem_answer"],
                    add_special_tokens=False,
                )["input_ids"]
            )
            for job in expanded
        ],
    )
    random_prompts, random_costs = [], []
    for job in expanded:
        control = random_by_id[job["id"]]
        text = control["subproblem"] + " " + control["subproblem_answer"]
        random_costs.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        random_prompts.append(
            f"{job['question']}\n\nA separate verified mathematical fact is: "
            f"{control['subproblem']} Answer: {control['subproblem_answer']}\n"
            "Use it only if relevant, then solve the original problem yourself."
        )
    solve_variant("random_subproblem", random_prompts, random_costs)

    subproblem_prompts = [
        chat_prompt(tokenizer, SYSTEM_PROMPT, job["subproblem"]) for job in expanded
    ]
    subproblem_outputs = generator.generate(
        subproblem_prompts,
        args.subproblem_max_new_tokens,
        args.seed + 7000,
        stop_after_boxed=args.stop_after_boxed,
    )
    subproblem_records = []
    for job, output in zip(expanded, subproblem_outputs):
        subproblem_records.append(
            {
                "id": job["id"],
                "sample_index": job["sample_index"],
                "subproblem": job["subproblem"],
                "reference": job["subproblem_answer"],
                "correct": verifier(output["text"], job["subproblem_answer"]),
                **output,
            }
        )

    for filename, records in (
        ("candidates.jsonl", candidates),
        ("rejected_candidates.jsonl", rejected),
        ("root_results.jsonl", root_records),
        ("subproblem_results.jsonl", subproblem_records),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = analyze_records(
        candidates, root_records, subproblem_records, args.q_low, args.q_high
    )
    q_by_id = summary["q_calibration"]["q_by_id"]
    trainable_candidates = []
    for candidate in candidates:
        q = q_by_id.get(candidate["id"])
        candidate["success_probability"] = q
        candidate["trainable"] = q is not None and args.q_low <= q <= args.q_high
        if candidate["trainable"]:
            trainable_candidates.append(candidate)
    with (args.output_dir / "training_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in trainable_candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    summary.update(
        {
            "eligible_questions": eligible,
            "selected_questions": len(rows),
            "valid_candidate_rate": len(candidates) / len(rows),
            "samples_per_variant": args.samples_per_variant,
            "model": args.model,
            "data": str(args.data),
            "seed": args.seed,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether generated verifiable subproblems causally help their roots."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--teacher-model",
        default=None,
        help="Instruction-following decomposition model; defaults to --model.",
    )
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--samples-per-variant", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=256)
    parser.add_argument("--teacher-retries", type=int, default=2)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--subproblem-max-new-tokens", type=int, default=512)
    parser.add_argument("--q-low", type=float, default=0.25)
    parser.add_argument("--q-high", type=float, default=0.60)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-accuracy", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-after-boxed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
