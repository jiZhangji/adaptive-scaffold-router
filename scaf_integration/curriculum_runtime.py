"""Optional capability-matched regeneration for the Scaf-GRPO trainer.

This module deliberately lives outside the upstream trainer.  The integration
patch calls it only when ``trainer.curriculum_manifest`` is configured, so the
original Scaf-GRPO behavior remains the default.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from capability_scaffold import stable_question_key
from scaf_curriculum_adapter import (
    apply_sequence_weights,
    build_curriculum_prompt,
    load_curriculum_manifest,
    sequence_importance_weights,
)


_HINT_LABELS = {
    "knowledge": "Knowledge Hints",
    "planning": "Planning Hints",
    "solution": "Solution Hints",
}

_HINT_LEVELS = {
    "knowledge": 1,
    "planning": 5,
    "solution": 9,
}


def load_optional_manifest(path: str | None) -> dict[str, dict[str, Any]] | None:
    if path is None or not str(path).strip():
        return None
    return load_curriculum_manifest(path)


def guided_rollout_count(rollouts: int, visible_fraction: float) -> int:
    """Convert the fading schedule into deterministic scaffold dropout.

    A fraction of each failed root's regenerated rollouts keeps the minimal
    scaffold; the rest are generated from the naked root prompt.  This keeps
    the target test distribution present throughout root-aligned training.
    """
    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    if not 0.0 <= visible_fraction <= 1.0:
        raise ValueError("visible_fraction must be in [0, 1]")
    return min(rollouts, max(0, int(math.floor(rollouts * visible_fraction + 0.5))))


def _as_parts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _row_value(batch: Any, key: str, index: int, default: Any) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None:
        return default
    return values[index]


def _build_prompt(tokenizer: Any, question: str, kind: str | None, hints: tuple[str, ...]) -> str:
    content = f"Question: {question}"
    if kind and hints:
        content += f"\n{_HINT_LABELS.get(kind, 'Hints')}: {' '.join(hints)}"
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_curriculum_new_message(
    new_gen_batch: Any,
    tokenizer: Any,
    manifest: dict[str, dict[str, Any]],
    step: int,
    fade_start: int,
    fade_end: int,
    rollouts: int = 4,
    max_length: int = 4096,
    truncation: str = "error",
):
    """Build one learner-matched scaffold arm per failed root question.

    Each arm is repeated ``rollouts`` times so its empirical success rate can be
    measured.  Non-guided phases receive an unguided probe; they are not silently
    upgraded to the upstream exhaustive eight-arm search.
    """
    from verl import DataProto
    import verl.utils.torch_functional as verl_F

    if rollouts < 1:
        raise ValueError("curriculum rollouts must be positive")

    questions = new_gen_batch.non_tensor_batch["question"]
    original_uids = new_gen_batch.non_tensor_batch["uid"]
    reward_models = new_gen_batch.non_tensor_batch["reward_model"]
    data_sources = new_gen_batch.non_tensor_batch["data_source"]

    all_input_ids: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_position_ids: list[torch.Tensor] = []
    non_tensors: dict[str, list[Any]] = {
        "uid": [],
        "hint_level": [],
        "raw_prompt": [],
        "reward_model": [],
        "data_source": [],
        "curriculum_phase": [],
        "scaffold_name": [],
        "question_key": [],
        "visible_fraction": [],
    }

    for index, question_value in enumerate(questions):
        question = str(question_value)
        question_key = stable_question_key(question)
        decision = manifest.get(question_key)
        parts_by_kind = {
            "knowledge": _as_parts(
                _row_value(new_gen_batch, "knowledge_components_parts", index, [])
            ),
            "planning": _as_parts(
                _row_value(new_gen_batch, "planning_skeleton_parts", index, [])
            ),
            "solution": _as_parts(
                _row_value(new_gen_batch, "solution_breakdown_parts", index, [])
            ),
        }
        request = build_curriculum_prompt(
            question=question,
            parts_by_kind=parts_by_kind,
            decision=decision,
            step=step,
            fade_start=fade_start,
            fade_end=fade_end,
        )
        kind = request.get("scaffold_kind")
        hints = tuple(request.get("hint_parts") or ())
        visible_fraction = float(request.get("visible_fraction", 0.0))
        guided_count = (
            guided_rollout_count(rollouts, visible_fraction)
            if kind and hints
            else 0
        )
        encoded: dict[bool, tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for rollout_index in range(rollouts):
            use_scaffold = rollout_index < guided_count
            if use_scaffold not in encoded:
                rollout_kind = kind if use_scaffold else None
                rollout_hints = hints if use_scaffold else ()
                raw_prompt = _build_prompt(tokenizer, question, rollout_kind, rollout_hints)
                model_inputs = tokenizer(
                    raw_prompt, return_tensors="pt", add_special_tokens=False
                )
                input_ids, attention_mask = verl_F.postprocess_data(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    max_length=max_length,
                    pad_token_id=tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=truncation,
                )
                position_ids = torch.clip(
                    torch.cumsum(attention_mask, dim=-1) - 1, min=0
                )
                encoded[use_scaffold] = (
                    raw_prompt,
                    input_ids[0],
                    attention_mask[0],
                    position_ids[0],
                )
            raw_prompt, input_ids, attention_mask, position_ids = encoded[use_scaffold]
            hint_level = _HINT_LEVELS.get(str(kind), 0) if use_scaffold else 0
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_position_ids.append(position_ids)
            non_tensors["uid"].append(original_uids[index])
            non_tensors["hint_level"].append(hint_level)
            non_tensors["raw_prompt"].append(raw_prompt)
            non_tensors["reward_model"].append(reward_models[index])
            non_tensors["data_source"].append(data_sources[index])
            non_tensors["curriculum_phase"].append(
                request["phase"] if use_scaffold else "unguided_dropout"
            )
            non_tensors["scaffold_name"].append(
                request.get("scaffold_name") if use_scaffold else None
            )
            non_tensors["question_key"].append(question_key)
            non_tensors["visible_fraction"].append(
                visible_fraction if use_scaffold else 0.0
            )

    if not all_input_ids:
        raise ValueError("No failed root questions were available for curriculum regeneration")

    tensors = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_masks),
        "position_ids": torch.stack(all_position_ids),
    }
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info={"new_gen": True, "capability_matched_curriculum": True},
        num_batch_dims=1,
        auto_padding=False,
    )
