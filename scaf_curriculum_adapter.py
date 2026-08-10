from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from capability_scaffold import stable_question_key


def load_curriculum_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            decision = json.loads(line)
            key = str(decision["question_key"])
            if key in decisions:
                raise ValueError(f"Duplicate question_key in curriculum manifest: {key}")
            decisions[key] = decision
    if not decisions:
        raise ValueError("Curriculum manifest is empty")
    return decisions


def resolve_curriculum_decision(
    question: str, manifest: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    return manifest.get(stable_question_key(question))


def parse_scaffold_name(name: str) -> tuple[str, float]:
    match = re.fullmatch(r"([A-Za-z_]+)@([0-9]+(?:\.[0-9]+)?)", str(name))
    if not match:
        raise ValueError(f"Expected scaffold name like planning@25, got {name!r}")
    return match.group(1), float(match.group(2)) / 100.0


def build_curriculum_prompt(
    question: str,
    parts_by_kind: dict[str, list[str]],
    decision: dict[str, Any] | None,
    step: int,
    fade_start: int,
    fade_end: int,
) -> dict[str, Any]:
    """Build one regeneration request from a compiled decision.

    The returned `phase` is intentionally explicit. A trainer can use
    `phase == "guided_root"` to replace the original failed rollout, and
    send all other phases to its ordinary curriculum path.
    """
    from capability_scaffold import fade_scaffold, visible_scaffold_fraction

    if decision is None:
        return {
            "phase": "unguided_root",
            "scaffold_name": None,
            "hint_parts": (),
            "question": question,
        }
    phase = str(decision.get("phase", "decompose_or_defer"))
    scaffold_name = decision.get("selected_scaffold") if phase == "guided_root" else None
    if not scaffold_name:
        return {
            "phase": phase,
            "scaffold_name": None,
            "hint_parts": (),
            "question": question,
        }

    kind, strength = parse_scaffold_name(str(scaffold_name))
    all_parts = [str(item) for item in parts_by_kind.get(kind, []) if str(item).strip()]
    selected_count = max(1, math.ceil(len(all_parts) * strength)) if all_parts else 0
    selected_parts = all_parts[:selected_count]
    visible_fraction = visible_scaffold_fraction(step, fade_start, fade_end)
    hint_parts = fade_scaffold(selected_parts, visible_fraction)
    return {
        "phase": phase,
        "scaffold_name": scaffold_name,
        "scaffold_kind": kind,
        "scaffold_strength": strength,
        "visible_fraction": visible_fraction,
        "hint_parts": hint_parts,
        "question": question,
    }


def build_regeneration_requests(
    rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    step: int,
    fade_start: int,
    fade_end: int,
) -> list[dict[str, Any]]:
    requests = []
    for row in rows:
        question = str(row["question"])
        request = build_curriculum_prompt(
            question,
            {
                "knowledge": list(row.get("knowledge_components_parts") or []),
                "planning": list(row.get("planning_skeleton_parts") or []),
                "solution": list(row.get("solution_breakdown_parts") or []),
            },
            resolve_curriculum_decision(question, manifest),
            step,
            fade_start,
            fade_end,
        )
        requests.append({**row, **request})
    return requests


def sequence_importance_weights(
    target_log_probs: Any,
    behavior_log_probs: Any,
    response_mask: Any,
    clip_low: float = 0.2,
    clip_high: float = 5.0,
    length_normalize: bool = False,
):
    """Compute clipped off-context weights for generated response sequences.

    `target_log_probs` are evaluated under the original question, while
    `behavior_log_probs` are evaluated under the scaffolded question. Shapes are
    `[batch, response_tokens]`; padded positions are removed by `response_mask`.
    """
    if target_log_probs.shape != behavior_log_probs.shape:
        raise ValueError("Target and behavior log-probability tensors must have equal shape")
    if response_mask.shape != target_log_probs.shape:
        raise ValueError("response_mask must match the log-probability tensors")
    if clip_low <= 0 or clip_high < clip_low:
        raise ValueError("Importance clipping must satisfy 0 < clip_low <= clip_high")

    mask = response_mask.to(dtype=target_log_probs.dtype)
    log_ratio = ((target_log_probs - behavior_log_probs) * mask).sum(dim=-1)
    if length_normalize:
        lengths = mask.sum(dim=-1).clamp_min(1.0)
        log_ratio = log_ratio / lengths
    weights = log_ratio.clamp(min=math.log(clip_low), max=math.log(clip_high)).exp()
    return weights


def apply_sequence_weights(advantages: Any, weights: Any, response_mask: Any):
    """Apply one off-context weight to every valid token of a trajectory."""
    if advantages.shape != response_mask.shape:
        raise ValueError("advantages and response_mask must have equal shape")
    if weights.ndim != 1 or weights.shape[0] != advantages.shape[0]:
        raise ValueError("weights must have shape [batch]")
    return advantages * weights.unsqueeze(-1) * response_mask.to(advantages.dtype)


def attach_curriculum_metadata(batch: Any, decisions: list[dict[str, Any]]) -> Any:
    """Attach controller outputs to a verl DataProto-like non-tensor batch."""
    if len(decisions) != len(batch):
        raise ValueError("One curriculum decision is required per batch row")
    batch.non_tensor_batch["curriculum_phase"] = [item["phase"] for item in decisions]
    batch.non_tensor_batch["selected_scaffold"] = [
        item.get("selected_scaffold") for item in decisions
    ]
    batch.non_tensor_batch["active_subproblem_ids"] = [
        item.get("active_subproblem_ids", []) for item in decisions
    ]
    return batch
