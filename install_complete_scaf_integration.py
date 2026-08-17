#!/usr/bin/env python3
"""Idempotently install the complete curriculum integration into Scaf-GRPO."""

from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path


RUNTIME_IMPORTS = (
    "apply_sequence_weights",
    "build_curriculum_new_message",
    "load_optional_manifest",
    "sequence_importance_weights",
)


def insert_after(lines: list[str], needle: str, block: list[str]) -> None:
    for index, line in enumerate(lines):
        if needle in line:
            lines[index + 1 : index + 1] = block
            return
    raise ValueError(f"Required trainer anchor not found: {needle}")


def ensure_runtime_imports(lines: list[str]) -> None:
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == "from hint_mix_grpo.curriculum_runtime import ("
        ),
        None,
    )
    if start is None:
        block = ["from hint_mix_grpo.curriculum_runtime import ("]
        block.extend(f"    {name}," for name in RUNTIME_IMPORTS)
        block.extend([")", ""])
        insert_after(lines, "from verl.utils.tracking import ValidationGenerationsLogger", block)
        return

    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip() == ")"),
        None,
    )
    if end is None:
        raise ValueError("Unterminated curriculum_runtime import block")
    existing = "\n".join(lines[start : end + 1])
    missing = [name for name in RUNTIME_IMPORTS if name not in existing]
    lines[end:end] = [f"    {name}," for name in missing]


def replace_build_new_message(lines: list[str]) -> None:
    if any("if self.curriculum_enabled:" in line for line in lines):
        return
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if "new_gen_batch = build_new_message(" in line
        ),
        None,
    )
    if start is None:
        raise ValueError("build_new_message call was not found")
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    if lines[start].rstrip().endswith(")"):
        end = start
    else:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].strip() == ")"
            ),
            None,
        )
        if end is None:
            raise ValueError("Unterminated build_new_message call")
    block = [
        f"{indent}if self.curriculum_enabled:",
        f"{indent}    new_gen_batch = build_curriculum_new_message(",
        f"{indent}        new_gen_batch,",
        f"{indent}        self.tokenizer,",
        f"{indent}        manifest=self.curriculum_manifest,",
        f"{indent}        step=self.global_steps,",
        f"{indent}        fade_start=self.curriculum_fade_start,",
        f"{indent}        fade_end=self.curriculum_fade_end,",
        f"{indent}        rollouts=self.curriculum_rollouts,",
        f"{indent}        max_length=self.config.data.max_prompt_length,",
        f"{indent}        truncation=self.config.data.truncation,",
        f"{indent}    )",
        f"{indent}else:",
        f"{indent}    new_gen_batch = build_new_message(",
        f"{indent}        new_gen_batch,",
        f"{indent}        self.tokenizer,",
        f"{indent}    )",
    ]
    lines[start : end + 1] = block


def insert_hinted_behavior_log_probs(lines: list[str]) -> None:
    marker = 'new_gen_batch_output.batch["rollout_log_probs"] = ('
    if any(marker in line for line in lines):
        return
    unpad_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "new_gen_batch_output = unpad_dataproto(" in line
        ),
        None,
    )
    if unpad_index is None:
        raise ValueError("Hinted generation unpad call was not found")
    indent = lines[unpad_index][
        : len(lines[unpad_index]) - len(lines[unpad_index].lstrip())
    ]
    block = [
        f"{indent}if self.curriculum_off_context:",
        f"{indent}    hinted_log_prob = self.actor_rollout_wg.compute_log_prob(",
        f"{indent}        new_gen_batch_output",
        f"{indent}    )",
        f'{indent}    if "old_log_probs" not in hinted_log_prob.batch:',
        f"{indent}        raise ValueError(",
        f'{indent}            "Actor log-prob output is missing old_log_probs"',
        f"{indent}        )",
        f'{indent}    new_gen_batch_output.batch["rollout_log_probs"] = (',
        f'{indent}        hinted_log_prob.batch["old_log_probs"]',
        f"{indent}    )",
    ]
    lines[unpad_index + 1 : unpad_index + 1] = block


def insert_off_context_replacement(lines: list[str]) -> None:
    complete_marker = 'if "rollout_log_probs" not in batch.batch:'
    if any(complete_marker in line for line in lines):
        return
    legacy_assignment = next(
        (
            index
            for index, line in enumerate(lines)
            if 'batch.batch["rollout_log_probs"][orig_idx] = (' in line
        ),
        None,
    )
    if legacy_assignment is not None:
        # Older versions of the integration copied hinted behavior log probs
        # into a batch field that had never been allocated.  Upgrade that
        # partial patch in place instead of treating the mask assignment as a
        # signal that the complete integration is already installed.
        indent = lines[legacy_assignment][
            : len(lines[legacy_assignment]) - len(lines[legacy_assignment].lstrip())
        ]
        lines[legacy_assignment:legacy_assignment] = [
            f'{indent}if "rollout_log_probs" not in batch.batch:',
            f'{indent}    batch.batch["rollout_log_probs"] = torch.zeros_like(',
            f'{indent}        batch.batch["responses"],',
            f'{indent}        dtype=new_gen_batch_output.batch["rollout_log_probs"].dtype,',
            f"{indent}    )",
        ]
        return
    replace_start = next(
        (index for index, line in enumerate(lines) if "if self.replace_num == 1:" in line),
        None,
    )
    if replace_start is None:
        raise ValueError("replace_num == 1 branch was not found")
    position_index = next(
        (
            index
            for index in range(replace_start, len(lines))
            if 'batch.batch["position_ids"][orig_idx] = compute_position_id_with_mask' in lines[index]
        ),
        None,
    )
    if position_index is None:
        raise ValueError("Hint replacement position_ids update was not found")
    indent = lines[position_index][: len(lines[position_index]) - len(lines[position_index].lstrip())]
    block = [
        f"{indent}if self.curriculum_off_context:",
        f'{indent}    if "rollout_log_probs" not in new_gen_batch_output.batch:',
        f"{indent}        raise ValueError(",
        f'{indent}            "Off-context correction requires rollout_log_probs from hinted generation"',
        f"{indent}        )",
        f'{indent}    if "rollout_log_probs" not in batch.batch:',
        f'{indent}        batch.batch["rollout_log_probs"] = torch.zeros_like(',
        f'{indent}            batch.batch["responses"],',
        f'{indent}            dtype=new_gen_batch_output.batch["rollout_log_probs"].dtype,',
        f"{indent}        )",
        f'{indent}    batch.batch["rollout_log_probs"][orig_idx] = (',
        f'{indent}        new_gen_batch_output.batch["rollout_log_probs"][new_idx].to(',
        f'{indent}            batch.batch["responses"].device',
        f"{indent}        )",
        f"{indent}    )",
        f'{indent}    batch.batch["curriculum_off_context_mask"][orig_idx] = True',
    ]
    lines[position_index + 1 : position_index + 1] = block


def insert_importance_correction(lines: list[str]) -> None:
    if any('"curriculum/is_weight_mean"' in line for line in lines):
        return
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if "batch = compute_advantage(" in line
        ),
        None,
    )
    if start is None:
        raise ValueError("compute_advantage call was not found")
    depth = 0
    end = None
    for index in range(start, len(lines)):
        depth += lines[index].count("(") - lines[index].count(")")
        if index > start and depth <= 0:
            end = index
            break
    if end is None:
        raise ValueError("Unterminated compute_advantage call")
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    block = [
        f"{indent}if (",
        f"{indent}    self.curriculum_off_context",
        f'{indent}    and "curriculum_off_context_mask" in batch.batch',
        f'{indent}    and batch.batch["curriculum_off_context_mask"].any()',
        f"{indent}):",
        f'{indent}    active = batch.batch["curriculum_off_context_mask"].bool()',
        f"{indent}    behavior_log_probs = torch.where(",
        f"{indent}        active.unsqueeze(-1),",
        f'{indent}        batch.batch["rollout_log_probs"],',
        f'{indent}        batch.batch["old_log_probs"],',
        f"{indent}    )",
        f"{indent}    is_weights = sequence_importance_weights(",
        f'{indent}        target_log_probs=batch.batch["old_log_probs"],',
        f"{indent}        behavior_log_probs=behavior_log_probs,",
        f'{indent}        response_mask=batch.batch["response_mask"],',
        f"{indent}        clip_low=self.curriculum_is_clip_low,",
        f"{indent}        clip_high=self.curriculum_is_clip_high,",
        f"{indent}        length_normalize=self.curriculum_is_length_normalize,",
        f"{indent}    )",
        f"{indent}    is_weights = torch.where(",
        f"{indent}        active, is_weights, torch.ones_like(is_weights)",
        f"{indent}    )",
        f'{indent}    batch.batch["advantages"] = apply_sequence_weights(',
        f'{indent}        batch.batch["advantages"],',
        f"{indent}        is_weights,",
        f'{indent}        batch.batch["response_mask"],',
        f"{indent}    )",
        f"{indent}    active_weights = is_weights[active]",
        f"{indent}    metrics.update(",
        f"{indent}        {{",
        f'{indent}            "curriculum/is_weight_mean": active_weights.mean().detach().item(),',
        f'{indent}            "curriculum/is_weight_min": active_weights.min().detach().item(),',
        f'{indent}            "curriculum/is_weight_max": active_weights.max().detach().item(),',
        f'{indent}            "curriculum/off_context_count": int(active.sum().detach().item()),',
        f"{indent}        }}",
        f"{indent}    )",
    ]
    lines[end + 1 : end + 1] = block


def repair_hint_level_statistics(lines: list[str]) -> None:
    for index, line in enumerate(lines):
        if line.strip() == "solve_with_hint = solve_any_second":
            indent = line[: len(line) - len(line.lstrip())]
            lines[index : index + 1] = [
                f"{indent}solve_with_hint = sum(solved_by_hint_level.values())",
                f"{indent}dynamic_unguided_solutions = max(",
                f"{indent}    0, solve_any_second - solve_with_hint",
                f"{indent})",
            ]
            break
    for index, line in enumerate(lines):
        if line.strip() == "solve_without_hint = solve_any_first":
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = (
                f"{indent}solve_without_hint = "
                "solve_any_first + dynamic_unguided_solutions"
            )
            break

    marker = "counted_hint_solutions = ("
    if any(marker in line for line in lines):
        return
    assertion_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "assert solved_by_hint_level[1]"
            in line
            and "solve_with_hint" in line
        ),
        None,
    )
    if assertion_index is None:
        return
    indent = lines[assertion_index][
        : len(lines[assertion_index]) - len(lines[assertion_index].lstrip())
    ]
    lines[assertion_index : assertion_index + 1] = [
        f"{indent}counted_hint_solutions = (",
        f"{indent}    sum(solved_by_hint_level.values())",
        f'{indent}    if hasattr(solved_by_hint_level, "values")',
        f"{indent}    else sum(solved_by_hint_level)",
        f"{indent})",
        f"{indent}assert counted_hint_solutions == solve_with_hint",
    ]


def patch_trainer_source(source: str) -> str:
    lines = source.splitlines()
    ensure_runtime_imports(lines)

    if not any("self.curriculum_manifest = load_optional_manifest" in line for line in lines):
        insert_after(
            lines,
            'self.warmup_steps = config.trainer.get("warmup_steps", 0)',
            [
                "        self.curriculum_manifest = load_optional_manifest(",
                '            config.trainer.get("curriculum_manifest", None)',
                "        )",
                '        self.curriculum_fade_start = int(config.trainer.get("curriculum_fade_start", 0))',
                '        self.curriculum_fade_end = int(config.trainer.get("curriculum_fade_end", 1000000000))',
                '        self.curriculum_rollouts = int(config.trainer.get("curriculum_rollouts", 4))',
                "        self.curriculum_enabled = self.curriculum_manifest is not None",
            ],
        )

    if not any("self.curriculum_off_context = bool" in line for line in lines):
        insert_after(
            lines,
            'self.replace_num = config.trainer.get("replace_num", 1)',
            [
                "        self.curriculum_off_context = bool(",
                '            config.trainer.get("curriculum_off_context", False)',
                "        )",
                "        self.curriculum_is_clip_low = float(",
                '            config.trainer.get("curriculum_is_clip_low", 0.2)',
                "        )",
                "        self.curriculum_is_clip_high = float(",
                '            config.trainer.get("curriculum_is_clip_high", 5.0)',
                "        )",
                "        self.curriculum_is_length_normalize = bool(",
                '            config.trainer.get("curriculum_is_length_normalize", True)',
                "        )",
                "        if self.curriculum_off_context and not self.replace_hint_prompt:",
                '            raise ValueError("curriculum_off_context requires trainer.replace_hint_prompt=true")',
            ],
        )

    replace_build_new_message(lines)
    insert_hinted_behavior_log_probs(lines)

    if not any('batch.batch["curriculum_off_context_mask"] = torch.zeros' in line for line in lines):
        insert_after(
            lines,
            'assert torch.equal(reward_tensor_first, reward_tensor_debug)',
            [
                "                        if self.curriculum_off_context:",
                '                            batch.batch["curriculum_off_context_mask"] = torch.zeros(',
                '                                batch.batch["responses"].shape[0], dtype=torch.bool,',
                '                                device=batch.batch["responses"].device,',
                "                            )",
            ],
        )

    insert_off_context_replacement(lines)
    repair_hint_level_statistics(lines)
    insert_importance_correction(lines)
    text = "\n".join(lines) + "\n"
    text = text.replace(
        "len(batch.batch), dtype=torch.bool",
        'batch.batch["responses"].shape[0], dtype=torch.bool',
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scaf-repo", type=Path, required=True)
    args = parser.parse_args()

    trainer = args.scaf_repo / "hint_mix_grpo" / "trainer" / "ray_trainer.py"
    if not trainer.exists():
        raise FileNotFoundError(trainer)

    shutil.copy2(args.project_root / "capability_scaffold.py", args.scaf_repo / "capability_scaffold.py")
    shutil.copy2(args.project_root / "scaf_curriculum_adapter.py", args.scaf_repo / "scaf_curriculum_adapter.py")
    shutil.copy2(
        args.project_root / "scaf_integration" / "curriculum_runtime.py",
        args.scaf_repo / "hint_mix_grpo" / "curriculum_runtime.py",
    )

    original = trainer.read_text(encoding="utf-8")
    patched = patch_trainer_source(original)
    if patched != original:
        backup = trainer.with_suffix(".py.before_complete_repair")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        trainer.write_text(patched, encoding="utf-8")
        print(f"Installed complete curriculum integration: {trainer}")
    else:
        print(f"Complete curriculum integration already installed: {trainer}")

    reject = Path(str(trainer) + ".rej")
    if reject.exists():
        reject.unlink()

    for path in (
        args.scaf_repo / "capability_scaffold.py",
        args.scaf_repo / "scaf_curriculum_adapter.py",
        args.scaf_repo / "hint_mix_grpo" / "curriculum_runtime.py",
        trainer,
    ):
        py_compile.compile(str(path), doraise=True)
    print("Complete Scaf/gradient curriculum integration is ready")


if __name__ == "__main__":
    main()
