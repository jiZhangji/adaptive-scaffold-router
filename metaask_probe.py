from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from frontier_probe import (
    SYSTEM_PROMPT,
    _as_text_list,
    build_math_verifier,
    has_complete_boxed_answer,
    load_probe_rows,
    stable_item_id,
)


ASSISTANCE_COLUMNS = {
    "knowledge_min": "knowledge_components_parts",
    "planning_min": "planning_skeleton_parts",
    "solution_min": "solution_breakdown_parts",
}


def extract_verification(text: str) -> tuple[str, str]:
    state_match = re.search(r"<state>(.*?)</state>", text, flags=re.I | re.S)
    verify_match = re.search(r"<verify>(.*?)</verify>", text, flags=re.I | re.S)
    state = " ".join(state_match.group(1).split()) if state_match else ""
    question = " ".join(verify_match.group(1).split()) if verify_match else ""
    if not question:
        question = "Is the most uncertain intermediate assumption valid?"
    if not question.endswith("?"):
        question += "?"
    return state, question


def parse_oracle_answer(text: str) -> str:
    match = re.search(r"\b(YES|NO|UNKNOWN)\b", text.upper())
    return match.group(1) if match else "UNKNOWN"


def chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\nResponse:"


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_variant[record["variant"]].append(record)
        by_key[(record["id"], int(record["sample_index"]))][record["variant"]] = record

    variants = {}
    for name, rows in sorted(by_variant.items()):
        variants[name] = {
            "trajectories": len(rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "avg_external_information_tokens": sum(
                int(row.get("external_information_tokens", 0)) for row in rows
            ) / len(rows),
            "avg_total_generated_tokens": sum(
                int(row.get("total_generated_tokens", row.get("output_tokens", 0))) for row in rows
            ) / len(rows),
        }

    rescue = defaultdict(int)
    eligible = defaultdict(int)
    for variant_rows in by_key.values():
        baseline = variant_rows.get("no_help")
        if not baseline or baseline["correct"]:
            continue
        for variant, row in variant_rows.items():
            if variant == "no_help":
                continue
            eligible[variant] += 1
            rescue[variant] += int(bool(row["correct"]))
    for variant in variants:
        if variant == "no_help":
            continue
        variants[variant]["rescue_rate_on_no_help_failures"] = (
            rescue[variant] / eligible[variant] if eligible[variant] else 0.0
        )

    return {
        "experiment": "MetaAsk minimal-information mechanism probe",
        "important_limitation": (
            "The privileged oracle is the same base model conditioned on the reference solution. "
            "This tests the interaction mechanism, not a final learned ASK policy or a trusted oracle."
        ),
        "variants": variants,
    }


class LocalGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        ).to(args.device)
        self.model.eval()

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        seed: int,
        stop_after_boxed: bool = False,
    ) -> list[dict[str, Any]]:
        torch = self.torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        outputs: list[dict[str, Any]] = []
        for start in range(0, len(prompts), self.args.batch_size):
            batch_prompts = prompts[start : start + self.args.batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_input_tokens,
            )
            input_counts = inputs["attention_mask"].sum(dim=1).tolist()
            inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]
            torch.manual_seed(seed + start)
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": self.args.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.args.temperature > 0:
                kwargs.update(temperature=self.args.temperature, top_p=self.args.top_p)
            if stop_after_boxed:
                tokenizer = self.tokenizer

                class BoxedAnswerStoppingCriteria(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        del scores, kwargs
                        finished = []
                        for sequence in input_ids:
                            generated = sequence[prompt_length:]
                            text = tokenizer.decode(generated, skip_special_tokens=True)
                            finished.append(has_complete_boxed_answer(text))
                        return torch.tensor(finished, device=input_ids.device, dtype=torch.bool)

                kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [BoxedAnswerStoppingCriteria()]
                )
            begun = time.perf_counter()
            with torch.inference_mode():
                sequences = self.model.generate(**inputs, **kwargs)
            elapsed = time.perf_counter() - begun
            for sequence, input_tokens in zip(sequences, input_counts):
                ids = sequence[prompt_length:]
                stop_ids = {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}
                length = len(ids)
                for index, token_id in enumerate(ids.tolist()):
                    if token_id in stop_ids:
                        length = index
                        break
                text = self.tokenizer.decode(ids[:length], skip_special_tokens=True)
                outputs.append(
                    {
                        "text": text,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(length),
                        "latency_seconds": elapsed / len(batch_prompts),
                    }
                )
        return outputs


def _reference_solution(row: dict[str, Any], reference: str) -> str:
    for key in ("solution", "model_think", "model_answer"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return reference


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
                    "reference_solution": _reference_solution(row, reference),
                    "row": row,
                    "sample_index": sample_index,
                }
            )

    records: list[dict[str, Any]] = []

    def solve_variant(variant: str, users: list[str], external_tokens: list[int] | None = None) -> None:
        prompts = [chat_prompt(tokenizer, SYSTEM_PROMPT, user) for user in users]
        generated = generator.generate(
            prompts,
            args.max_new_tokens,
            args.seed + len(records) + 10,
            stop_after_boxed=args.stop_after_boxed,
        )
        for index, (job, result) in enumerate(zip(jobs, generated)):
            info_tokens = external_tokens[index] if external_tokens else 0
            records.append(
                {
                    "id": job["id"],
                    "source_id": job["source_id"],
                    "question": job["question"],
                    "reference": job["reference"],
                    "sample_index": job["sample_index"],
                    "variant": variant,
                    "correct": verifier(result["text"], job["reference"]),
                    "external_information_tokens": info_tokens,
                    "total_generated_tokens": result["output_tokens"],
                    **result,
                }
            )

    solve_variant("no_help", [job["question"] for job in jobs])

    rng = random.Random(args.seed)
    random_bits = [rng.choice(("YES", "NO")) for _ in jobs]
    solve_variant(
        "random_bit",
        [
            f"{job['question']}\n\nAn unrelated one-bit observation is: {bit}. "
            "Solve the problem independently."
            for job, bit in zip(jobs, random_bits)
        ],
        [1] * len(jobs),
    )

    for variant, column in ASSISTANCE_COLUMNS.items():
        users, token_costs = [], []
        for job in jobs:
            parts = _as_text_list(job["row"].get(column))
            hint = parts[0] if parts else ""
            token_costs.append(len(tokenizer(hint, add_special_tokens=False)["input_ids"]))
            users.append(
                f"{job['question']}\n\nMinimal external assistance: {hint}\n"
                "Use it only as guidance and complete the reasoning yourself."
            )
        solve_variant(variant, users, token_costs)

    query_prompts = [
        chat_prompt(
            tokenizer,
            "Identify uncertainty; do not give the final answer.",
            f"Problem: {job['question']}\n"
            "Write a very short tentative plan in <state>...</state>, then ask exactly one "
            "critical yes/no verification question in <verify>...</verify>.",
        )
        for job in jobs
    ]
    query_results = generator.generate(query_prompts, args.query_max_new_tokens, args.seed + 1000)
    parsed = [extract_verification(result["text"]) for result in query_results]

    oracle_prompts = []
    for job, (_, verify_question) in zip(jobs, parsed):
        oracle_prompts.append(
            chat_prompt(
                tokenizer,
                "You are a privileged verifier. Return exactly YES, NO, or UNKNOWN and nothing else.",
                f"Problem: {job['question']}\nReference solution: {job['reference_solution']}\n"
                f"Claim to verify: {verify_question}",
            )
        )
    oracle_results = generator.generate(oracle_prompts, args.oracle_max_new_tokens, args.seed + 2000)
    oracle_answers = [parse_oracle_answer(result["text"]) for result in oracle_results]
    oracle_costs = [
        len(tokenizer(answer, add_special_tokens=False)["input_ids"]) for answer in oracle_answers
    ]
    continuation_users = []
    for job, (state, verify_question), oracle_answer in zip(jobs, parsed, oracle_answers):
        continuation_users.append(
            f"Problem: {job['question']}\n\nYour tentative state: {state or '(not provided)'}\n"
            f"You asked: {verify_question}\nVerifier response: {oracle_answer}\n\n"
            "Continue the reasoning yourself. Do not ask again."
        )
    before = len(records)
    solve_variant("self_asked_verification", continuation_users, oracle_costs)
    for offset, record in enumerate(records[before:]):
        record["verification_question"] = parsed[offset][1]
        record["tentative_state"] = parsed[offset][0]
        record["oracle_answer"] = oracle_answers[offset]
        record["query_output_tokens"] = query_results[offset]["output_tokens"]
        record["oracle_output_tokens"] = oracle_results[offset]["output_tokens"]
        record["total_generated_tokens"] += (
            query_results[offset]["output_tokens"] + oracle_results[offset]["output_tokens"]
        )

    raw_path = args.output_dir / "raw_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = analyze_records(records)
    summary.update({"eligible_questions": eligible, "selected_questions": len(rows)})
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
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--samples-per-variant", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--query-max-new-tokens", type=int, default=128)
    parser.add_argument("--oracle-max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-accuracy", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-after-boxed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
