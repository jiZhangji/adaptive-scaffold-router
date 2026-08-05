from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Protocol


LEVELS = ("none", "knowledge", "planning", "solution")
SYSTEM_PROMPT = (
    "Solve the math problem carefully. Give a concise explanation and end with "
    "exactly one line in the form FINAL_ANSWER: <answer>."
)


@dataclass
class Generation:
    text: str
    output_tokens: int
    latency_seconds: float


class Backend(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> Generation:
        ...


class TransformersBackend:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has no CUDA-enabled PyTorch."
            )

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = self.temperature > 0
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)

        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        latency = time.perf_counter() - start
        generated_ids = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return Generation(text=text, output_tokens=len(generated_ids), latency_seconds=latency)


class ServerBackend:
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        api_key: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=headers, method="POST"
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model server request failed: {exc}") from exc
        latency = time.perf_counter() - start
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage") or {}
        output_tokens = int(usage.get("completion_tokens") or len(text.split()))
        return Generation(text=text, output_tokens=output_tokens, latency_seconds=latency)


def load_examples(path: Path, limit: int | None) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def build_messages(example: dict, level: str) -> list[dict[str, str]]:
    user_text = example["question"]
    if level != "none":
        hint = example["hints"][level]
        user_text += (
            f"\n\nOptional {level} scaffold:\n{hint}\n"
            "Use the scaffold as guidance, but complete the reasoning yourself."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def extract_number(text: str) -> Fraction | None:
    matches = re.findall(r"FINAL_ANSWER\s*:\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    candidate = matches[-1] if matches else text
    candidate = candidate.replace(",", "").replace("$", "").strip()
    fraction_match = re.search(r"[-+]?\d+\s*/\s*[-+]?\d+", candidate)
    if fraction_match:
        numerator, denominator = fraction_match.group(0).split("/")
        denominator_value = int(denominator.strip())
        if denominator_value == 0:
            return None
        return Fraction(int(numerator.strip()), denominator_value)
    number_match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", candidate)
    if not number_match:
        return None
    try:
        return Fraction(Decimal(number_match.group(0)))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def is_correct(text: str, reference: str) -> bool:
    predicted = extract_number(text)
    expected = extract_number(f"FINAL_ANSWER: {reference}")
    return predicted is not None and expected is not None and predicted == expected


def attempt_level(
    backend: Backend,
    example: dict,
    level: str,
    samples_per_level: int,
) -> dict:
    generations = []
    for _ in range(samples_per_level):
        generation = backend.generate(build_messages(example, level))
        generations.append(
            {
                **asdict(generation),
                "correct": is_correct(generation.text, example["answer"]),
            }
        )
    return {
        "level": level,
        "correct": any(item["correct"] for item in generations),
        "generations": generations,
        "output_tokens": sum(item["output_tokens"] for item in generations),
        "latency_seconds": sum(item["latency_seconds"] for item in generations),
    }


def run_example(backend: Backend, example: dict, samples_per_level: int) -> dict:
    attempts = []
    selected_level = None
    for level_index, level in enumerate(LEVELS):
        attempt = attempt_level(backend, example, level, samples_per_level)
        attempts.append(attempt)
        if attempt["correct"]:
            selected_level = level_index
            break

    no_hint = attempts[0]
    if selected_level == 3:
        full_hint = attempts[-1]
    else:
        full_hint = attempt_level(backend, example, "solution", samples_per_level)

    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "no_hint_correct": no_hint["correct"],
        "full_hint_correct": full_hint["correct"],
        "progressive_correct": selected_level is not None,
        "selected_level": selected_level,
        "selected_level_name": LEVELS[selected_level] if selected_level is not None else "failed",
        "scaffold_recovered": not no_hint["correct"] and selected_level is not None,
        "progressive_calls": len(attempts) * samples_per_level,
        "progressive_output_tokens": sum(item["output_tokens"] for item in attempts),
        "progressive_latency_seconds": sum(item["latency_seconds"] for item in attempts),
        "full_hint_output_tokens": full_hint["output_tokens"],
        "full_hint_latency_seconds": full_hint["latency_seconds"],
        "attempts": attempts,
        "full_hint_attempt": full_hint,
    }


def summarize(results: list[dict]) -> dict:
    count = len(results)
    if count == 0:
        raise ValueError("No examples were evaluated.")
    histogram = {level: 0 for level in LEVELS}
    histogram["failed"] = 0
    for result in results:
        histogram[result["selected_level_name"]] += 1
    successful_levels = [
        result["selected_level"]
        for result in results
        if result["selected_level"] is not None
    ]
    return {
        "num_examples": count,
        "no_hint_accuracy": sum(item["no_hint_correct"] for item in results) / count,
        "full_hint_accuracy": sum(item["full_hint_correct"] for item in results) / count,
        "progressive_accuracy": sum(item["progressive_correct"] for item in results) / count,
        "scaffold_recovery_count": sum(item["scaffold_recovered"] for item in results),
        "average_selected_level_on_success": (
            sum(successful_levels) / len(successful_levels) if successful_levels else None
        ),
        "selected_level_histogram": histogram,
        "progressive_calls": sum(item["progressive_calls"] for item in results),
        "progressive_output_tokens": sum(
            item["progressive_output_tokens"] for item in results
        ),
        "full_hint_output_tokens": sum(item["full_hint_output_tokens"] for item in results),
        "progressive_latency_seconds": sum(
            item["progressive_latency_seconds"] for item in results
        ),
        "full_hint_latency_seconds": sum(
            item["full_hint_latency_seconds"] for item in results
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive scaffold proof-of-concept.")
    parser.add_argument("--backend", choices=("transformers", "server"), default="transformers")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "toy_math.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "adaptive_scaffold")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-level", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_level < 1:
        raise ValueError("--samples-per-level must be at least 1")
    random.seed(args.seed)

    if args.backend == "transformers":
        backend: Backend = TransformersBackend(
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        backend = ServerBackend(
            endpoint=args.endpoint,
            model_name=args.model,
            api_key=args.api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    examples = load_examples(args.data, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    results = []
    with results_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            print(f"[{index}/{len(examples)}] {example['id']}", flush=True)
            result = run_example(backend, example, args.samples_per_level)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"  no_hint={result['no_hint_correct']} "
                f"progressive={result['progressive_correct']} "
                f"level={result['selected_level_name']}",
                flush=True,
            )

    summary = summarize(results)
    summary.update(
        {
            "backend": args.backend,
            "model": args.model,
            "samples_per_level": args.samples_per_level,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
