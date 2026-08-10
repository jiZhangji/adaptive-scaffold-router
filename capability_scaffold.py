from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def empirical_success(rewards: Sequence[float]) -> float:
    if not rewards:
        raise ValueError("At least one rollout reward is required")
    return sum(float(reward > 0) for reward in rewards) / len(rewards)


def stable_question_key(question: str) -> str:
    normalized = " ".join(str(question).split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def informative_group_probability(success_probability: float, group_size: int) -> float:
    """Probability that a binary-reward group contains both outcomes."""
    _validate_probability(success_probability, "success_probability")
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    return 1.0 - success_probability**group_size - (1.0 - success_probability) ** group_size


@dataclass(frozen=True)
class ControllerConfig:
    band_low: float = 0.25
    band_high: float = 0.60
    mastery_threshold: float = 0.75
    prerequisite_coverage: float = 0.80
    graduation_threshold: float = 0.75
    graduation_patience: int = 2
    group_size: int = 8
    max_hint_tokens: int = 96

    def __post_init__(self) -> None:
        for name in (
            "band_low",
            "band_high",
            "mastery_threshold",
            "prerequisite_coverage",
            "graduation_threshold",
        ):
            _validate_probability(float(getattr(self, name)), name)
        if self.band_low > self.band_high:
            raise ValueError("band_low cannot exceed band_high")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if self.graduation_patience < 1:
            raise ValueError("graduation_patience must be positive")
        if self.max_hint_tokens < 0:
            raise ValueError("max_hint_tokens cannot be negative")


@dataclass(frozen=True)
class RolloutArm:
    name: str
    rewards: tuple[float, ...]
    hint_tokens: int = 0
    strength: float = 0.0
    kind: str = "none"
    text: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RolloutArm":
        return cls(
            name=str(value["name"]),
            rewards=tuple(float(item) for item in value["rewards"]),
            hint_tokens=int(value.get("hint_tokens", 0)),
            strength=float(value.get("strength", 0.0)),
            kind=str(value.get("kind", "none")),
            text=str(value.get("text", "")),
        )

    @property
    def success_probability(self) -> float:
        return empirical_success(self.rewards)


@dataclass(frozen=True)
class SubproblemState:
    id: str
    question: str
    answer: str
    rewards: tuple[float, ...]
    depth: int = 1
    parent_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SubproblemState":
        return cls(
            id=str(value["id"]),
            question=str(value["question"]),
            answer=str(value["answer"]),
            rewards=tuple(float(item) for item in value["rewards"]),
            depth=int(value.get("depth", 1)),
            parent_id=(str(value["parent_id"]) if value.get("parent_id") is not None else None),
        )

    @property
    def success_probability(self) -> float:
        return empirical_success(self.rewards)


@dataclass(frozen=True)
class RootState:
    id: str
    question: str
    answer: str
    root_rewards: tuple[float, ...]
    subproblems: tuple[SubproblemState, ...] = ()
    scaffolds: tuple[RolloutArm, ...] = ()
    consecutive_unguided_mastery: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RootState":
        return cls(
            id=str(value["id"]),
            question=str(value["question"]),
            answer=str(value["answer"]),
            root_rewards=tuple(float(item) for item in value["root_rewards"]),
            subproblems=tuple(
                SubproblemState.from_dict(item) for item in value.get("subproblems", [])
            ),
            scaffolds=tuple(RolloutArm.from_dict(item) for item in value.get("scaffolds", [])),
            consecutive_unguided_mastery=int(value.get("consecutive_unguided_mastery", 0)),
        )


@dataclass(frozen=True)
class CurriculumDecision:
    root_id: str
    question_key: str
    phase: str
    reason: str
    root_success: float
    informative_probability: float
    selected_scaffold: str | None = None
    selected_scaffold_success: float | None = None
    active_subproblem_ids: tuple[str, ...] = ()
    unresolved_subproblem_ids: tuple[str, ...] = ()
    prerequisite_mastery: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _in_band(probability: float, config: ControllerConfig) -> bool:
    return config.band_low <= probability <= config.band_high


def select_scaffold(
    scaffolds: Iterable[RolloutArm], config: ControllerConfig
) -> RolloutArm | None:
    candidates = [
        arm
        for arm in scaffolds
        if arm.kind != "none" and arm.hint_tokens <= config.max_hint_tokens
    ]
    if not candidates:
        return None

    in_band = [arm for arm in candidates if _in_band(arm.success_probability, config)]
    if in_band:
        return min(
            in_band,
            key=lambda arm: (
                arm.hint_tokens,
                arm.strength,
                -informative_group_probability(arm.success_probability, config.group_size),
                arm.name,
            ),
        )

    above_floor = [arm for arm in candidates if arm.success_probability >= config.band_low]
    if above_floor:
        return min(
            above_floor,
            key=lambda arm: (arm.hint_tokens, arm.strength, arm.success_probability, arm.name),
        )
    return max(
        candidates,
        key=lambda arm: (
            arm.success_probability,
            -arm.hint_tokens,
            -arm.strength,
            arm.name,
        ),
    )


def decide_curriculum(root: RootState, config: ControllerConfig) -> CurriculumDecision:
    root_success = empirical_success(root.root_rewards)
    root_information = informative_group_probability(root_success, config.group_size)

    if (
        root_success >= config.graduation_threshold
        and root.consecutive_unguided_mastery >= config.graduation_patience
    ):
        return CurriculumDecision(
            root_id=root.id,
            question_key=stable_question_key(root.question),
            phase="graduated",
            reason="Repeated unguided evaluation exceeds the graduation threshold.",
            root_success=root_success,
            informative_probability=root_information,
            prerequisite_mastery=1.0,
        )

    if _in_band(root_success, config):
        return CurriculumDecision(
            root_id=root.id,
            question_key=stable_question_key(root.question),
            phase="unguided_root",
            reason="The original problem already produces informative mixed rewards.",
            root_success=root_success,
            informative_probability=root_information,
            prerequisite_mastery=1.0,
        )

    if root_success > config.band_high:
        return CurriculumDecision(
            root_id=root.id,
            question_key=stable_question_key(root.question),
            phase="unguided_probe",
            reason="The root is above the learning band but has not passed repeated graduation checks.",
            root_success=root_success,
            informative_probability=root_information,
            prerequisite_mastery=1.0,
        )

    subproblem_probabilities = {
        item.id: item.success_probability for item in root.subproblems
    }
    mastered = [
        item_id
        for item_id, probability in subproblem_probabilities.items()
        if probability >= config.mastery_threshold
    ]
    prerequisite_mastery = (
        len(mastered) / len(root.subproblems) if root.subproblems else 0.0
    )
    active = [
        item.id
        for item in root.subproblems
        if _in_band(item.success_probability, config)
    ]
    unresolved = [
        item.id
        for item in root.subproblems
        if item.success_probability < config.band_low
    ]

    scaffold = select_scaffold(root.scaffolds, config)
    scaffold_is_trainable = scaffold is not None and scaffold.success_probability >= config.band_low
    prerequisites_ready = prerequisite_mastery >= config.prerequisite_coverage

    if scaffold_is_trainable and (prerequisites_ready or not root.subproblems):
        return CurriculumDecision(
            root_id=root.id,
            question_key=stable_question_key(root.question),
            phase="guided_root",
            reason="Prerequisites are ready and the selected scaffold restores a trainable root group.",
            root_success=root_success,
            informative_probability=root_information,
            selected_scaffold=scaffold.name,
            selected_scaffold_success=scaffold.success_probability,
            active_subproblem_ids=tuple(active),
            unresolved_subproblem_ids=tuple(unresolved),
            prerequisite_mastery=prerequisite_mastery,
            diagnostics={
                "scaffold_informative_probability": informative_group_probability(
                    scaffold.success_probability, config.group_size
                )
            },
        )

    if active:
        return CurriculumDecision(
            root_id=root.id,
            question_key=stable_question_key(root.question),
            phase="subproblem_curriculum",
            reason="Trainable prerequisite subproblems exist while the root remains below the band.",
            root_success=root_success,
            informative_probability=root_information,
            selected_scaffold=(scaffold.name if scaffold else None),
            selected_scaffold_success=(scaffold.success_probability if scaffold else None),
            active_subproblem_ids=tuple(active),
            unresolved_subproblem_ids=tuple(unresolved),
            prerequisite_mastery=prerequisite_mastery,
        )

    return CurriculumDecision(
        root_id=root.id,
        question_key=stable_question_key(root.question),
        phase="decompose_or_defer",
        reason="Neither the root, an available scaffold, nor a subproblem is currently trainable.",
        root_success=root_success,
        informative_probability=root_information,
        selected_scaffold=(scaffold.name if scaffold else None),
        selected_scaffold_success=(scaffold.success_probability if scaffold else None),
        unresolved_subproblem_ids=tuple(unresolved),
        prerequisite_mastery=prerequisite_mastery,
    )


def visible_scaffold_fraction(
    step: int,
    fade_start: int,
    fade_end: int,
    initial_fraction: float = 1.0,
    final_fraction: float = 0.0,
) -> float:
    if fade_start < 0 or fade_end <= fade_start:
        raise ValueError("fade schedule must satisfy 0 <= fade_start < fade_end")
    _validate_probability(initial_fraction, "initial_fraction")
    _validate_probability(final_fraction, "final_fraction")
    if step <= fade_start:
        return initial_fraction
    if step >= fade_end:
        return final_fraction
    progress = (step - fade_start) / (fade_end - fade_start)
    return initial_fraction + progress * (final_fraction - initial_fraction)


def fade_scaffold(parts: Sequence[str], visible_fraction: float) -> tuple[str, ...]:
    _validate_probability(visible_fraction, "visible_fraction")
    clean_parts = tuple(str(part).strip() for part in parts if str(part).strip())
    if not clean_parts or visible_fraction == 0.0:
        return ()
    visible_count = max(1, math.ceil(len(clean_parts) * visible_fraction))
    return clean_parts[:visible_count]


def clipped_importance_weight(
    target_token_logprobs: Sequence[float],
    behavior_token_logprobs: Sequence[float],
    clip_low: float = 0.2,
    clip_high: float = 5.0,
    length_normalize: bool = False,
) -> float:
    if len(target_token_logprobs) != len(behavior_token_logprobs):
        raise ValueError("Target and behavior log-probability sequences must have equal length")
    if not target_token_logprobs:
        raise ValueError("At least one generated token is required")
    if clip_low <= 0 or clip_high < clip_low:
        raise ValueError("Importance clipping must satisfy 0 < clip_low <= clip_high")
    log_ratio = sum(
        float(target) - float(behavior)
        for target, behavior in zip(target_token_logprobs, behavior_token_logprobs)
    )
    if length_normalize:
        log_ratio /= len(target_token_logprobs)
    log_ratio = min(math.log(clip_high), max(math.log(clip_low), log_ratio))
    return math.exp(log_ratio)


def effective_sample_size(weights: Sequence[float]) -> float:
    if not weights:
        return 0.0
    if any(weight < 0 for weight in weights):
        raise ValueError("Importance weights cannot be negative")
    denominator = sum(weight * weight for weight in weights)
    return (sum(weights) ** 2 / denominator) if denominator else 0.0


def compile_manifest(
    roots: Iterable[RootState], config: ControllerConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decisions = [decide_curriculum(root, config) for root in roots]
    histogram: dict[str, int] = {}
    for decision in decisions:
        histogram[decision.phase] = histogram.get(decision.phase, 0) + 1
    rows = [asdict(decision) for decision in decisions]
    summary = {
        "num_roots": len(rows),
        "phase_histogram": histogram,
        "mean_root_success": (
            sum(row["root_success"] for row in rows) / len(rows) if rows else 0.0
        ),
        "config": asdict(config),
    }
    return rows, summary


def _load_roots(path: Path) -> list[RootState]:
    with path.open("r", encoding="utf-8") as handle:
        return [RootState.from_dict(json.loads(line)) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile capability-matched subproblem/scaffold curriculum decisions."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band-low", type=float, default=0.25)
    parser.add_argument("--band-high", type=float, default=0.60)
    parser.add_argument("--mastery-threshold", type=float, default=0.75)
    parser.add_argument("--prerequisite-coverage", type=float, default=0.80)
    parser.add_argument("--graduation-threshold", type=float, default=0.75)
    parser.add_argument("--graduation-patience", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-hint-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ControllerConfig(
        band_low=args.band_low,
        band_high=args.band_high,
        mastery_threshold=args.mastery_threshold,
        prerequisite_coverage=args.prerequisite_coverage,
        graduation_threshold=args.graduation_threshold,
        graduation_patience=args.graduation_patience,
        group_size=args.group_size,
        max_hint_tokens=args.max_hint_tokens,
    )
    rows, summary = compile_manifest(_load_roots(args.input), config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "curriculum.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
