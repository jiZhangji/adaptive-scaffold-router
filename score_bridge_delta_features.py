#!/usr/bin/env python3
"""Score system-preserving bridge deltas for candidate subproblems.

The script reuses root-only responses already stored by the replicated RCST
probe.  For every response it compares the normalized response log-probability
under the original prompt and under the identical prompt plus one candidate
subproblem.  No generation or parameter update is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from bridge_prompt_utils import BRIDGE_PROMPT_VERSION, build_bridge_messages
from frontier_probe import SYSTEM_PROMPT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def root_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(question)},
    ]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    ordered = sorted(values)
    return {
        "mean": mean,
        "std": std,
        "standard_error": std / math.sqrt(len(values)),
        "median": statistics.median(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "positive_rate": sum(value > 0 for value in values) / len(values),
    }


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def score_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    responses: list[str],
    batch_size: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
    device: str,
) -> tuple[list[float], list[int]]:
    import torch
    import torch.nn.functional as F

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have equal length")
    examples: list[tuple[list[int], int, int]] = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_tokens
        )
        response_ids = tokenizer.encode(
            str(response),
            add_special_tokens=False,
            truncation=True,
            max_length=max_response_tokens,
        )
        if not response_ids:
            response_ids = [tokenizer.eos_token_id]
        examples.append((prompt_ids + response_ids, len(prompt_ids), len(response_ids)))

    all_scores: list[float] = []
    all_lengths: list[int] = []
    pad = int(tokenizer.pad_token_id)
    for chunk in batched(examples, batch_size):
        width = max(len(ids) for ids, _, _ in chunk)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        for row_index, (ids, _, _) in enumerate(chunk):
            offset = width - len(ids)
            input_ids[row_index, offset:] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row_index, offset:] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
            targets = input_ids[:, 1:]
            token_nll = F.cross_entropy(
                logits.transpose(1, 2), targets, reduction="none"
            )
        for row_index, (ids, prompt_length, response_length) in enumerate(chunk):
            offset = width - len(ids)
            start = max(offset + prompt_length - 1, 0)
            stop = start + response_length
            score = -token_nll[row_index, start:stop].float().mean()
            all_scores.append(float(score.cpu()))
            all_lengths.append(response_length)
        del input_ids, attention_mask, logits, targets, token_nll
    return all_scores, all_lengths


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    candidates = read_jsonl(args.candidates)
    candidate_by_id = {str(row["id"]): row for row in candidates}
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_root[str(row["root_id"])].append(row)

    probe_rows: list[dict[str, Any]] = []
    for path in args.probes:
        probe_rows.extend(read_jsonl(path))
    response_sets: dict[tuple[str, int], dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    for row in probe_rows:
        root_id = str(row["root_id"])
        candidate_id = str(row["candidate_id"])
        candidate_ids.add(candidate_id)
        key = (root_id, int(row["probe_seed"]))
        payload = {
            "texts": list(row["baseline_texts"]),
            "correct": [bool(value) for value in row["baseline_correct"]],
        }
        previous = response_sets.get(key)
        if previous is not None and previous != payload:
            raise ValueError(f"inconsistent root-only responses for {key}")
        response_sets[key] = payload

    missing = sorted(candidate_ids - set(candidate_by_id))
    if missing:
        raise ValueError(f"{len(missing)} probe candidates are absent from candidate file")
    eligible_roots = sorted({root_id for root_id, _ in response_sets})
    shard_roots = [
        root_id
        for index, root_id in enumerate(eligible_roots)
        if index % args.num_shards == args.shard_index
    ]
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = {str(row["candidate_id"]) for row in existing}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    model.eval()

    for root_index, root_id in enumerate(shard_roots, start=1):
        root_candidates = sorted(by_root[root_id], key=lambda row: str(row["dimension"]))
        pending = [row for row in root_candidates if str(row["id"]) not in completed]
        if not pending:
            continue
        if len(root_candidates) != 3:
            raise ValueError(f"{root_id} has {len(root_candidates)} candidates, expected 3")
        question = str(root_candidates[0]["question"])
        seeds = sorted(seed for candidate_root, seed in response_sets if candidate_root == root_id)
        responses: list[str] = []
        correct: list[bool] = []
        sample_seeds: list[int] = []
        for seed in seeds:
            payload = response_sets[(root_id, seed)]
            responses.extend(payload["texts"])
            correct.extend(payload["correct"])
            sample_seeds.extend([seed] * len(payload["texts"]))
        messages = root_messages(question)
        root_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        root_scores, response_lengths = score_responses(
            model,
            tokenizer,
            [root_prompt] * len(responses),
            responses,
            args.batch_size,
            args.max_prompt_tokens,
            args.max_response_tokens,
            args.device,
        )
        for candidate in pending:
            bridge_messages = build_bridge_messages(
                messages, str(candidate["subproblem"]), str(candidate["subproblem_answer"])
            )
            bridge_prompt = tokenizer.apply_chat_template(
                bridge_messages, tokenize=False, add_generation_prompt=True
            )
            bridge_scores, _ = score_responses(
                model,
                tokenizer,
                [bridge_prompt] * len(responses),
                responses,
                args.batch_size,
                args.max_prompt_tokens,
                args.max_response_tokens,
                args.device,
            )
            deltas = [bridge - root for bridge, root in zip(bridge_scores, root_scores)]

            subproblem_messages = root_messages(str(candidate["subproblem"]))
            subproblem_prompt = tokenizer.apply_chat_template(
                subproblem_messages, tokenize=False, add_generation_prompt=True
            )
            answer_score, answer_lengths = score_responses(
                model,
                tokenizer,
                [subproblem_prompt],
                [str(candidate["subproblem_answer"])],
                1,
                args.max_prompt_tokens,
                args.max_response_tokens,
                args.device,
            )
            stats = summarize(deltas)
            record = {
                "root_id": root_id,
                "candidate_id": str(candidate["id"]),
                "dimension": str(candidate.get("dimension", "")),
                "prompt_version": BRIDGE_PROMPT_VERSION,
                "rollout_count": len(deltas),
                "probe_seeds": seeds,
                "sample_seeds": sample_seeds,
                "root_baseline_accuracy": sum(correct) / len(correct),
                "root_logprob_mean": statistics.fmean(root_scores),
                "bridge_logprob_mean": statistics.fmean(bridge_scores),
                "delta_mean": stats["mean"],
                "delta_std": stats["std"],
                "delta_standard_error": stats["standard_error"],
                "delta_lcb": stats["mean"] - args.delta_confidence_z * stats["standard_error"],
                "delta_median": stats["median"],
                "delta_min": stats["minimum"],
                "delta_max": stats["maximum"],
                "delta_positive_rate": stats["positive_rate"],
                "response_length_mean": statistics.fmean(response_lengths),
                "response_length_std": (
                    statistics.stdev(response_lengths) if len(response_lengths) > 1 else 0.0
                ),
                "subproblem_answer_logprob": answer_score[0],
                "subproblem_answer_tokens": answer_lengths[0],
                "subproblem_tokens": len(
                    tokenizer.encode(str(candidate["subproblem"]), add_special_tokens=False)
                ),
                "sample_deltas": deltas,
            }
            append_jsonl(args.output, record)
            completed.add(str(candidate["id"]))
        print(
            f"[{root_index}/{len(shard_roots)}] {root_id}: "
            f"scored {len(pending)} candidates on {len(responses)} responses",
            flush=True,
        )
        torch.cuda.empty_cache()

    rows = read_jsonl(args.output)
    summary = {
        "prompt_version": BRIDGE_PROMPT_VERSION,
        "eligible_roots": len(eligible_roots),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_roots": len(shard_roots),
        "completed_candidates_in_output": len(rows),
        "definition": "mean token log p(response|root+subproblem) - mean token log p(response|root)",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--probes", type=Path, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-response-tokens", type=int, default=768)
    parser.add_argument("--delta-confidence-z", type=float, default=1.0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    return args


if __name__ == "__main__":
    run(parse_args())
