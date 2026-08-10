from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


SCAFFOLD_COLUMNS = {
    "knowledge": "knowledge_components_parts",
    "planning": "planning_skeleton_parts",
    "solution": "solution_breakdown_parts",
}
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class ScaffoldArm:
    name: str
    kind: str
    strength: float
    selected_parts: int
    total_parts: int
    hint: str
    order: int


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()]


def _strength_label(strength: float) -> str:
    return str(int(round(strength * 100)))


def stable_item_id(source_id: str, question: str) -> str:
    question_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"{source_id}::{question_hash}"


def build_scaffold_arms(row: dict[str, Any], strengths: Iterable[float]) -> list[ScaffoldArm]:
    arms = [
        ScaffoldArm(
            name="none",
            kind="none",
            strength=0.0,
            selected_parts=0,
            total_parts=0,
            hint="",
            order=0,
        )
    ]
    order = 1
    for kind, column in SCAFFOLD_COLUMNS.items():
        parts = _as_text_list(row.get(column))
        if not parts:
            continue
        for strength in strengths:
            selected_count = min(len(parts), max(1, math.ceil(len(parts) * strength)))
            selected = parts[:selected_count]
            hint = "\n".join(f"{index}. {part}" for index, part in enumerate(selected, 1))
            arms.append(
                ScaffoldArm(
                    name=f"{kind}@{_strength_label(strength)}",
                    kind=kind,
                    strength=strength,
                    selected_parts=selected_count,
                    total_parts=len(parts),
                    hint=hint,
                    order=order,
                )
            )
            order += 1
    return arms


def build_prompt(tokenizer: Any, question: str, arm: ScaffoldArm) -> str:
    user_prompt = question
    if arm.kind != "none":
        user_prompt += (
            f"\n\nOptional {arm.kind} scaffold ({_strength_label(arm.strength)}%):\n"
            f"{arm.hint}\n\nUse the scaffold only as guidance and complete the reasoning yourself."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nSolution:"


def build_math_verifier() -> Callable[[str, str], bool]:
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError as exc:
        raise RuntimeError("frontier_probe.py requires the optional math-verify package") from exc

    verify = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )

    def is_correct(model_output: str, ground_truth: str) -> bool:
        try:
            score, _ = verify([f"\\boxed{{{ground_truth}}}"], [model_output])
            return bool(score)
        except Exception:
            return False

    return is_correct


def has_complete_boxed_answer(text: str) -> bool:
    search_end = len(text)
    while True:
        boxed_index = text.rfind("\\boxed", 0, search_end)
        if boxed_index < 0:
            return False
        open_index = text.find("{", boxed_index + len("\\boxed"))
        if open_index < 0:
            return False
        depth = 0
        for character in text[open_index:]:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return True
        search_end = boxed_index


def load_probe_rows(
    path: Path,
    limit: int,
    seed: int,
    max_source_accuracy: float | None,
) -> tuple[list[dict[str, Any]], int]:
    import pandas as pd

    frame = pd.read_parquet(path)
    if max_source_accuracy is not None and "accuracy" in frame.columns:
        frame = frame[frame["accuracy"] <= max_source_accuracy]
    eligible_count = len(frame)
    if eligible_count == 0:
        raise ValueError("No rows remain after source-accuracy filtering")
    sample_count = min(limit, eligible_count)
    frame = frame.sample(n=sample_count, random_state=seed)
    frame = frame.sort_values("id")
    return frame.to_dict(orient="records"), eligible_count


def make_jobs(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    strengths: list[float],
    samples_per_arm: int,
    seed: int,
) -> list[dict[str, Any]]:
    jobs = []
    for row in rows:
        source_id = str(row["id"])
        question = str(row["question"])
        item_id = stable_item_id(source_id, question)
        reference = str(row["reward_model"]["ground_truth"])
        for arm in build_scaffold_arms(row, strengths):
            prompt = build_prompt(tokenizer, question, arm)
            hint_tokens = len(tokenizer(arm.hint, add_special_tokens=False)["input_ids"])
            for sample_index in range(samples_per_arm):
                jobs.append(
                    {
                        "id": item_id,
                        "source_id": source_id,
                        "question": question,
                        "reference": reference,
                        "arm": arm,
                        "prompt": prompt,
                        "hint_tokens": hint_tokens,
                        "sample_index": sample_index,
                    }
                )
    random.Random(seed).shuffle(jobs)
    return jobs


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _load_completed_keys(path: Path) -> set[tuple[str, str, int]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add((record["id"], record["arm_name"], record["sample_index"]))
    return completed


def run_generation(args: argparse.Namespace) -> tuple[Path, int, int]:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.jsonl"
    if args.overwrite and raw_path.exists():
        raw_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows, eligible_count = load_probe_rows(
        args.data, args.limit, args.seed, args.max_source_accuracy
    )
    jobs = make_jobs(rows, tokenizer, args.strengths, args.samples_per_arm, args.seed)
    completed = _load_completed_keys(raw_path)
    jobs = [
        job
        for job in jobs
        if (job["id"], job["arm"].name, job["sample_index"]) not in completed
    ]

    print(
        f"eligible={eligible_count} selected={len(rows)} remaining_generations={len(jobs)}",
        flush=True,
    )
    if not jobs:
        return raw_path, len(rows), eligible_count

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()
    verifier = build_math_verifier()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    class BoxedAnswerStoppingCriteria(StoppingCriteria):
        def __init__(self, prompt_length: int) -> None:
            self.prompt_length = prompt_length

        def __call__(self, input_ids, scores, **kwargs):
            del scores, kwargs
            finished = []
            for sequence in input_ids:
                generated = sequence[self.prompt_length :]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                finished.append(has_complete_boxed_answer(text))
            return torch.tensor(finished, device=input_ids.device, dtype=torch.bool)

    processed = 0
    total_batches = math.ceil(len(jobs) / args.batch_size)
    with raw_path.open("a", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(_chunks(jobs, args.batch_size), 1):
            prompts = [job["prompt"] for job in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            )
            input_token_counts = inputs["attention_mask"].sum(dim=1).tolist()
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            padded_input_length = inputs["input_ids"].shape[1]
            if args.stop_after_boxed:
                generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [BoxedAnswerStoppingCriteria(padded_input_length)]
                )
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)
            elapsed = time.perf_counter() - start

            for job, output, input_tokens in zip(batch, outputs, input_token_counts):
                generated_ids = output[padded_input_length:]
                stop_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
                effective_length = len(generated_ids)
                for token_index, token_id in enumerate(generated_ids.tolist()):
                    if token_id in stop_ids:
                        effective_length = token_index
                        break
                effective_ids = generated_ids[:effective_length]
                text = tokenizer.decode(effective_ids, skip_special_tokens=True)
                arm: ScaffoldArm = job["arm"]
                record = {
                    "id": job["id"],
                    "source_id": job["source_id"],
                    "question": job["question"],
                    "reference": job["reference"],
                    "arm_name": arm.name,
                    "arm_kind": arm.kind,
                    "arm_strength": arm.strength,
                    "arm_order": arm.order,
                    "selected_parts": arm.selected_parts,
                    "total_parts": arm.total_parts,
                    "sample_index": job["sample_index"],
                    "correct": verifier(text, job["reference"]),
                    "input_tokens": int(input_tokens),
                    "hint_tokens": int(job["hint_tokens"]),
                    "output_tokens": int(effective_length),
                    "hit_max_new_tokens": effective_length >= args.max_new_tokens,
                    "latency_seconds": elapsed / len(batch),
                    "text": text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            processed += len(batch)
            print(
                f"batch={batch_index}/{total_batches} processed={processed}/{len(jobs)} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )
    return raw_path, len(rows), eligible_count


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_records(
    records: list[dict[str, Any]],
    target_success: float,
    band_low: float,
    band_high: float,
    lambda_cost: float,
    mu_dependence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_item_arm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_trajectory: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    arm_meta: dict[str, dict[str, Any]] = {}
    questions: dict[str, str] = {}
    references: dict[str, str] = {}
    for record in records:
        key = (record["id"], record["arm_name"])
        by_item_arm[key].append(record)
        by_arm[record["arm_name"]].append(record)
        by_trajectory[(record["id"], record["sample_index"])][record["arm_name"]] = record
        arm_meta[record["arm_name"]] = {
            "kind": record["arm_kind"],
            "strength": record["arm_strength"],
            "order": record["arm_order"],
        }
        questions[record["id"]] = record["question"]
        references[record["id"]] = record["reference"]

    arm_order = sorted(arm_meta, key=lambda name: arm_meta[name]["order"])
    item_ids = sorted(questions)
    item_frontiers = []
    utility_histogram: Counter[str] = Counter()
    threshold_histogram: Counter[str] = Counter()
    band_histogram: Counter[str] = Counter()
    rescued = 0

    for item_id in item_ids:
        stats = []
        no_hint_records = by_item_arm.get((item_id, "none"), [])
        no_hint_p = _mean([float(record["correct"]) for record in no_hint_records])
        for arm_name in arm_order:
            arm_records = by_item_arm.get((item_id, arm_name), [])
            if not arm_records:
                continue
            p_correct = _mean([float(record["correct"]) for record in arm_records])
            hint_tokens = _mean([float(record["hint_tokens"]) for record in arm_records])
            total_tokens = _mean(
                [float(record["input_tokens"] + record["output_tokens"]) for record in arm_records]
            )
            dependence = max(0.0, p_correct - no_hint_p)
            utility = (
                p_correct * (1.0 - p_correct)
                - lambda_cost * (hint_tokens / 100.0)
                - mu_dependence * dependence
            )
            stats.append(
                {
                    "arm_name": arm_name,
                    **arm_meta[arm_name],
                    "p_correct": p_correct,
                    "pass_at_k": any(record["correct"] for record in arm_records),
                    "hint_tokens": hint_tokens,
                    "total_tokens": total_tokens,
                    "utility": utility,
                }
            )

        utility_choice = max(stats, key=lambda item: (item["utility"], -item["hint_tokens"]))
        threshold_candidates = [item for item in stats if item["p_correct"] >= target_success]
        threshold_choice = (
            min(threshold_candidates, key=lambda item: (item["hint_tokens"], item["order"]))
            if threshold_candidates
            else max(stats, key=lambda item: item["p_correct"])
        )
        band_candidates = [
            item for item in stats if band_low <= item["p_correct"] <= band_high
        ]
        midpoint = (band_low + band_high) / 2.0
        band_choice = (
            min(band_candidates, key=lambda item: (item["hint_tokens"], item["order"]))
            if band_candidates
            else min(
                stats,
                key=lambda item: (abs(item["p_correct"] - midpoint), item["hint_tokens"]),
            )
        )
        if no_hint_p == 0 and any(item["p_correct"] > 0 for item in stats if item["kind"] != "none"):
            rescued += 1
        utility_histogram[utility_choice["arm_name"]] += 1
        threshold_histogram[threshold_choice["arm_name"]] += 1
        band_histogram[band_choice["arm_name"]] += 1
        item_frontiers.append(
            {
                "id": item_id,
                "question": questions[item_id],
                "reference": references[item_id],
                "no_hint_p": no_hint_p,
                "utility_choice": utility_choice["arm_name"],
                "threshold_choice": threshold_choice["arm_name"],
                "band_choice": band_choice["arm_name"],
                "arms": stats,
            }
        )

    arm_summary = {}
    for arm_name in arm_order:
        arm_records = by_arm[arm_name]
        item_pass = []
        for item_id in item_ids:
            rows = by_item_arm.get((item_id, arm_name), [])
            if rows:
                item_pass.append(any(row["correct"] for row in rows))
        arm_summary[arm_name] = {
            **arm_meta[arm_name],
            "rollout_accuracy": _mean([float(record["correct"]) for record in arm_records]),
            "pass_at_k": _mean([float(value) for value in item_pass]),
            "average_hint_tokens": _mean([float(record["hint_tokens"]) for record in arm_records]),
            "average_input_tokens": _mean([float(record["input_tokens"]) for record in arm_records]),
            "average_output_tokens": _mean([float(record["output_tokens"]) for record in arm_records]),
            "generation_limit_rate": _mean(
                [float(record.get("hit_max_new_tokens", False)) for record in arm_records]
            ),
        }

    def simulate_progressive(order: list[str]) -> dict[str, float]:
        costs = []
        calls_per_trajectory = []
        successes = []
        for trajectory in by_trajectory.values():
            cumulative_cost = 0
            calls = 0
            success = False
            for arm_name in order:
                if arm_name not in trajectory:
                    continue
                record = trajectory[arm_name]
                cumulative_cost += record["input_tokens"] + record["output_tokens"]
                calls += 1
                if record["correct"]:
                    success = True
                    break
            costs.append(float(cumulative_cost))
            calls_per_trajectory.append(float(calls))
            successes.append(float(success))
        return {
            "accuracy": _mean(successes),
            "average_total_tokens": _mean(costs),
            "average_calls": _mean(calls_per_trajectory),
        }

    full_lattice_progressive = simulate_progressive(arm_order)
    public_scaf_order = [
        name
        for name in arm_order
        if arm_meta[name]["kind"] in {"none", "knowledge", "solution"}
    ]
    public_scaf_progressive = simulate_progressive(public_scaf_order)

    def simulate_min_cost_oracle(order: list[str]) -> dict[str, float]:
        all_costs = []
        success_costs = []
        successes = []
        for trajectory in by_trajectory.values():
            available = [trajectory[name] for name in order if name in trajectory]
            correct_records = [record for record in available if record["correct"]]
            successes.append(float(bool(correct_records)))
            if correct_records:
                selected_cost = float(
                    min(
                        record["input_tokens"] + record["output_tokens"]
                        for record in correct_records
                    )
                )
                success_costs.append(selected_cost)
            else:
                selected_cost = float(
                    min(
                        record["input_tokens"] + record["output_tokens"]
                        for record in available
                    )
                )
            all_costs.append(selected_cost)
        return {
            "accuracy": _mean(successes),
            "average_total_tokens": _mean(all_costs),
            "average_total_tokens_on_success": _mean(success_costs),
        }

    full_lattice_oracle = simulate_min_cost_oracle(arm_order)
    public_scaf_oracle = simulate_min_cost_oracle(public_scaf_order)
    oracle_costs = []
    oracle_success = []
    strongest_solution = max(
        (name for name in arm_order if arm_meta[name]["kind"] == "solution"),
        key=lambda name: arm_meta[name]["strength"],
        default=arm_order[-1],
    )
    strongest_costs = []
    strongest_success = []
    for trajectory in by_trajectory.values():
        correct_records = [record for record in trajectory.values() if record["correct"]]
        oracle_success.append(float(bool(correct_records)))
        if correct_records:
            oracle_costs.append(
                float(min(record["input_tokens"] + record["output_tokens"] for record in correct_records))
            )

        if strongest_solution in trajectory:
            record = trajectory[strongest_solution]
            strongest_costs.append(float(record["input_tokens"] + record["output_tokens"]))
            strongest_success.append(float(record["correct"]))

    summary = {
        "num_examples": len(item_ids),
        "num_records": len(records),
        "arms": arm_summary,
        "rescued_zero_accuracy_examples": rescued,
        "rescue_rate": rescued / len(item_ids) if item_ids else 0.0,
        "utility_choice_histogram": dict(utility_histogram),
        "threshold_choice_histogram": dict(threshold_histogram),
        "learning_band_choice_histogram": dict(band_histogram),
        "cost_simulation": {
            "progressive_accuracy": full_lattice_progressive["accuracy"],
            "progressive_average_total_tokens": full_lattice_progressive[
                "average_total_tokens"
            ],
            "progressive_average_calls": full_lattice_progressive["average_calls"],
            "full_lattice_progressive": full_lattice_progressive,
            "public_scaf_progressive": public_scaf_progressive,
            "full_lattice_min_cost_oracle": full_lattice_oracle,
            "public_scaf_min_cost_oracle": public_scaf_oracle,
            "public_scaf_oracle_token_saving_fraction": (
                1.0
                - public_scaf_oracle["average_total_tokens"]
                / public_scaf_progressive["average_total_tokens"]
                if public_scaf_progressive["average_total_tokens"]
                else 0.0
            ),
            "planning_absolute_accuracy_gain": (
                full_lattice_oracle["accuracy"] - public_scaf_oracle["accuracy"]
            ),
            "oracle_min_cost_accuracy": _mean(oracle_success),
            "oracle_min_cost_average_total_tokens_on_success": _mean(oracle_costs),
            "strongest_solution_accuracy": _mean(strongest_success),
            "strongest_solution_average_total_tokens": _mean(strongest_costs),
        },
        "objective": {
            "target_success": target_success,
            "learning_band": [band_low, band_high],
            "lambda_cost_per_100_hint_tokens": lambda_cost,
            "mu_dependence": mu_dependence,
        },
    }
    return summary, item_frontiers


def analyze_file(args: argparse.Namespace, raw_path: Path, selected: int, eligible: int) -> None:
    with raw_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        record.setdefault(
            "hit_max_new_tokens", record["output_tokens"] >= args.max_new_tokens
        )
    summary, frontiers = analyze_records(
        records,
        target_success=args.target_success,
        band_low=args.band_low,
        band_high=args.band_high,
        lambda_cost=args.lambda_cost,
        mu_dependence=args.mu_dependence,
    )
    summary.update(
        {
            "model": args.model,
            "data": str(args.data),
            "eligible_examples": eligible,
            "selected_examples": selected,
            "samples_per_arm": args.samples_per_arm,
            "strengths": args.strengths,
            "max_new_tokens": args.max_new_tokens,
            "stop_after_boxed": args.stop_after_boxed,
            "temperature": args.temperature,
            "seed": args.seed,
        }
    )
    summary_path = args.output_dir / "summary.json"
    frontiers_path = args.output_dir / "item_frontiers.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with frontiers_path.open("w", encoding="utf-8") as handle:
        for frontier in frontiers:
            handle.write(json.dumps(frontier, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved={summary_path}", flush=True)


def _parse_strengths(value: str) -> list[float]:
    strengths = [float(item) for item in value.split(",")]
    if not strengths or any(value <= 0 or value > 1 for value in strengths):
        raise argparse.ArgumentTypeError("strengths must be comma-separated values in (0, 1]")
    return sorted(set(strengths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the instance-level, capability-dependent scaffold frontier."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--samples-per-arm", type=int, default=4)
    parser.add_argument("--strengths", type=_parse_strengths, default=[0.25, 0.5, 1.0])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-accuracy", type=float, default=0.0)
    parser.add_argument("--target-success", type=float, default=0.5)
    parser.add_argument("--band-low", type=float, default=0.25)
    parser.add_argument("--band-high", type=float, default=0.75)
    parser.add_argument("--lambda-cost", type=float, default=0.02)
    parser.add_argument("--mu-dependence", type=float, default=0.05)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-after-boxed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.samples_per_arm < 1 or args.batch_size < 1:
        raise ValueError("limit, samples-per-arm, and batch-size must be positive")
    if not 0 <= args.band_low <= args.band_high <= 1:
        raise ValueError("learning band must satisfy 0 <= low <= high <= 1")
    raw_path, selected, eligible = run_generation(args)
    analyze_file(args, raw_path, selected, eligible)


if __name__ == "__main__":
    main()
