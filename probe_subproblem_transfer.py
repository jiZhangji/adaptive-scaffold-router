#!/usr/bin/env python3
"""Estimate same-root subproblem-to-root transfer with temporary LoRA updates."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from calibrate_helpful_subproblems import read_jsonl
from frontier_probe import SYSTEM_PROMPT, build_math_verifier


def indexed_root_shard(
    roots: list[str], num_shards: int, shard_index: int
) -> list[tuple[int, str]]:
    """Return a deterministic shard while retaining each root's global index."""
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [
        (global_index, root_id)
        for global_index, root_id in enumerate(roots)
        if global_index % num_shards == shard_index
    ]


def chat_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def train_example(tokenizer: Any, question: str, answer: str, max_length: int) -> dict[str, Any]:
    prompt = chat_prompt(tokenizer, question)
    completion = f"{answer}\n"
    full = prompt + completion
    encoded = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)[
        "input_ids"
    ]
    labels = encoded["input_ids"].clone()
    labels[:, : min(prompt_ids.shape[1], labels.shape[1])] = -100
    encoded["labels"] = labels
    return encoded


def generate_rollouts(
    model: Any,
    tokenizer: Any,
    question: str,
    samples: int,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
) -> list[str]:
    import torch

    prompt = chat_prompt(tokenizer, question)
    encoded = tokenizer(
        [prompt] * samples,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    torch.manual_seed(seed)
    model.eval()
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        output = model.generate(**encoded, **generation)
    prompt_length = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)


def adapter_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }


def restore_adapter(model: Any, state: dict[str, Any]) -> None:
    for name, value in model.named_parameters():
        if name in state:
            value.data.copy_(state[name].to(device=value.device, dtype=value.dtype))


def run(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.candidates)
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_root[str(row["root_id"])].append(row)
    eligible = [root for root, values in by_root.items() if len(values) >= 2]
    random.Random(args.selection_seed).shuffle(eligible)
    if args.root_limit > 0:
        eligible = eligible[: args.root_limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = {str(row["candidate_id"]) for row in existing}

    indexed_roots = indexed_root_shard(eligible, args.num_shards, args.shard_index)
    indexed_roots = [
        (global_index, root_id)
        for global_index, root_id in indexed_roots
        if any(str(candidate["id"]) not in completed for candidate in by_root[root_id])
    ]
    print(
        f"Shard {args.shard_index}/{args.num_shards}: "
        f"{len(indexed_roots)} incomplete of {len(eligible)} eligible roots",
        flush=True,
    )

    if not indexed_roots:
        summary = {
            "eligible_roots": len(eligible),
            "shard_roots": 0,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "completed_candidates": len(existing),
            "mean_proxy_transfer_gain": (
                sum(float(row["proxy_transfer_gain"]) for row in existing) / len(existing)
                if existing
                else 0.0
            ),
            "definition": "temporary LoRA SFT update followed by same-root no-hint evaluation",
            "warning": "This is a low-cost proxy, not an unbiased causal estimate of full GRPO transfer.",
        }
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, config)
    initial = adapter_state(model)
    verifier = build_math_verifier()

    for local_index, (global_index, root_id) in enumerate(indexed_roots):
        candidates = sorted(by_root[root_id], key=lambda row: str(row["dimension"]))
        question = str(candidates[0]["question"])
        reference = str(candidates[0]["reference"])
        restore_adapter(model, initial)
        baseline_texts = generate_rollouts(
            model,
            tokenizer,
            question,
            args.root_samples,
            args.max_input_tokens,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.seed + global_index * 1000,
            args.device,
        )
        baseline_correct = [bool(verifier(text, reference)) for text in baseline_texts]
        baseline_p = sum(baseline_correct) / len(baseline_correct)

        for candidate_index, candidate in enumerate(candidates):
            candidate_id = str(candidate["id"])
            if candidate_id in completed:
                continue
            restore_adapter(model, initial)
            model.train()
            optimizer = torch.optim.AdamW(
                [value for value in model.parameters() if value.requires_grad],
                lr=args.learning_rate,
                weight_decay=0.0,
            )
            losses: list[float] = []
            example = train_example(
                tokenizer,
                str(candidate["subproblem"]),
                str(candidate["subproblem_answer"]),
                args.max_train_tokens,
            )
            example = {key: value.to(args.device) for key, value in example.items()}
            for _ in range(args.probe_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = model(**example).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [value for value in model.parameters() if value.requires_grad], 1.0
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.config.use_cache = True
            post_texts = generate_rollouts(
                model,
                tokenizer,
                question,
                args.root_samples,
                args.max_input_tokens,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.seed + global_index * 1000,
                args.device,
            )
            post_correct = [bool(verifier(text, reference)) for text in post_texts]
            post_p = sum(post_correct) / len(post_correct)
            record = {
                "root_id": root_id,
                "candidate_id": candidate_id,
                "dimension": str(candidate.get("dimension", "")),
                "probe_seed": args.seed,
                "selection_seed": args.selection_seed,
                "baseline_probability": baseline_p,
                "post_update_probability": post_p,
                "proxy_transfer_gain": post_p - baseline_p,
                "root_samples": args.root_samples,
                "probe_steps": args.probe_steps,
                "learning_rate": args.learning_rate,
                "losses": losses,
                "baseline_correct": baseline_correct,
                "post_update_correct": post_correct,
                "baseline_texts": baseline_texts,
                "post_update_texts": post_texts,
            }
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            completed.add(candidate_id)
            print(
                f"[shard {local_index + 1}/{len(indexed_roots)}; "
                f"global {global_index + 1}/{len(eligible)}] {candidate_id} "
                f"baseline={baseline_p:.3f} post={post_p:.3f} "
                f"gain={post_p - baseline_p:+.3f}",
                flush=True,
            )
            del optimizer
            torch.cuda.empty_cache()

    all_results = read_jsonl(args.output)
    summary = {
        "eligible_roots": len(eligible),
        "shard_roots": len(indexed_roots),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "completed_candidates": len(all_results),
        "mean_proxy_transfer_gain": (
            sum(float(row["proxy_transfer_gain"]) for row in all_results) / len(all_results)
            if all_results
            else 0.0
        ),
        "definition": "temporary LoRA SFT update followed by same-root no-hint evaluation",
        "warning": "This is a low-cost proxy, not an unbiased causal estimate of full GRPO transfer.",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-limit", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--root-samples", type=int, default=4)
    parser.add_argument("--probe-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-train-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
