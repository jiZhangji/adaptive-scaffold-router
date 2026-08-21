import unittest

from install_complete_scaf_integration import (
    patch_trainer_source,
    repair_invalid_dataloader_resume,
)


PARTIALLY_PATCHED = '''from verl.utils.tracking import ValidationGenerationsLogger

from hint_mix_grpo.curriculum_runtime import (
    build_curriculum_new_message,
    load_optional_manifest,
)

class RayPPOTrainer:
    def __init__(self, config):
        self.warmup_steps = config.trainer.get("warmup_steps", 0)
        self.curriculum_manifest = load_optional_manifest(
            config.trainer.get("curriculum_manifest", None)
        )
        self.curriculum_fade_start = int(config.trainer.get("curriculum_fade_start", 0))
        self.curriculum_fade_end = int(config.trainer.get("curriculum_fade_end", 1000000000))
        self.curriculum_rollouts = int(config.trainer.get("curriculum_rollouts", 4))
        self.curriculum_enabled = self.curriculum_manifest is not None
        self.replace_num = config.trainer.get("replace_num", 1)

    def fit(self):
        assert len(new_gen_batch.non_tensor_batch['uid']) == solve_none_first
        new_gen_batch = build_new_message(new_gen_batch, self.tokenizer)
        new_gen_batch_output = unpad_dataproto(new_gen_batch_output_padded, pad_size=pad_size)
        new_gen_batch_output.non_tensor_batch["responses_text"] = []
        reward_tensor_debug, reward_extra_infos_dict_debug = compute_reward(batch, self.reward_fn)
        assert torch.equal(reward_tensor_first, reward_tensor_debug), "Reward tensors do not match after debug"
        if new_data_map:
            if self.replace_num == 1:
                batch.batch["position_ids"][orig_idx] = compute_position_id_with_mask(batch.batch["attention_mask"][orig_idx])
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
        )
        # update critic
'''


LEGACY_OFF_CONTEXT_PATCHED = PARTIALLY_PATCHED.replace(
    'batch.batch["position_ids"][orig_idx] = compute_position_id_with_mask(batch.batch["attention_mask"][orig_idx])',
    '''batch.batch["position_ids"][orig_idx] = compute_position_id_with_mask(batch.batch["attention_mask"][orig_idx])
                if self.curriculum_off_context:
                    batch.batch["rollout_log_probs"][orig_idx] = (
                        new_gen_batch_output.batch["rollout_log_probs"][new_idx]
                    )
                    batch.batch["curriculum_off_context_mask"][orig_idx] = True''',
).replace(
    "        batch = compute_advantage(",
    "        solve_with_hint = solve_any_second\n"
    "        solve_without_hint = solve_any_first\n"
    "        assert solved_by_hint_level[1] + solved_by_hint_level[2] + solved_by_hint_level[3] == solve_with_hint\n"
    "        batch = compute_advantage(",
)


class CompleteScafInstallerTest(unittest.TestCase):
    def test_recovers_incompatible_cross_stage_dataloader_cursor(self):
        lines = [
            "        for epoch in range(10):",
            "            for batch_dict in self.train_dataloader:",
            "                consume(batch_dict)",
        ]
        repair_invalid_dataloader_resume(lines)
        text = "\n".join(lines)
        self.assertIn("train_iterator = iter(self.train_dataloader)", text)
        self.assertIn("except StopIteration:", text)
        self.assertIn("self.train_dataloader.next_iter_state = None", text)
        self.assertIn("for batch_dict in train_iterator:", text)
        repaired = list(lines)
        repair_invalid_dataloader_resume(repaired)
        self.assertEqual(repaired, lines)

    def test_repairs_partial_patch_and_is_idempotent(self):
        patched = patch_trainer_source(PARTIALLY_PATCHED)
        self.assertIn("apply_sequence_weights", patched)
        self.assertIn("if self.curriculum_enabled:", patched)
        self.assertIn("self.curriculum_off_context = bool", patched)
        self.assertIn("hinted_log_prob = self.actor_rollout_wg.compute_log_prob", patched)
        self.assertIn('hinted_log_prob.batch["old_log_probs"]', patched)
        self.assertIn('if "rollout_log_probs" not in batch.batch', patched)
        self.assertIn("behavior_log_probs = torch.where", patched)
        self.assertIn("curriculum_off_context_mask", patched)
        self.assertIn('"curriculum/is_weight_mean"', patched)
        self.assertEqual(patch_trainer_source(patched), patched)

    def test_upgrades_legacy_off_context_replacement(self):
        patched = patch_trainer_source(LEGACY_OFF_CONTEXT_PATCHED)
        self.assertIn('if "rollout_log_probs" not in batch.batch:', patched)
        self.assertIn(
            'batch.batch["rollout_log_probs"] = torch.zeros_like(',
            patched,
        )
        self.assertEqual(patched.count('batch.batch["curriculum_off_context_mask"][orig_idx] = True'), 1)
        self.assertIn("counted_hint_solutions = (", patched)
        self.assertIn("sum(solved_by_hint_level.values())", patched)
        self.assertIn(
            "solve_with_hint = sum(solved_by_hint_level.values())",
            patched,
        )
        self.assertIn("dynamic_unguided_solutions = max(", patched)
        self.assertIn(
            "solve_without_hint = solve_any_first + dynamic_unguided_solutions",
            patched,
        )
        self.assertNotIn("solve_with_hint = solve_any_second", patched)
        self.assertNotIn(
            "assert solved_by_hint_level[1] + solved_by_hint_level[2] + solved_by_hint_level[3]",
            patched,
        )
        self.assertEqual(patch_trainer_source(patched), patched)


if __name__ == "__main__":
    unittest.main()
