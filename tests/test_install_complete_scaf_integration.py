import unittest

from install_complete_scaf_integration import patch_trainer_source


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


class CompleteScafInstallerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
