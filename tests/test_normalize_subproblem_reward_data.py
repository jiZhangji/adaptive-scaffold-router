import unittest

from normalize_subproblem_reward_data import normalize_data_source


class NormalizeSubproblemRewardDataTest(unittest.TestCase):
    def test_removes_only_legacy_subproblem_suffix(self):
        self.assertEqual(
            normalize_data_source(
                "deepscaler-clean-39k_except-still/math-verify::verifiable_subproblem"
            ),
            "deepscaler-clean-39k_except-still/math-verify",
        )
        self.assertEqual(normalize_data_source("AIME24/math-verify"), "AIME24/math-verify")
        self.assertIsNone(normalize_data_source(None))


if __name__ == "__main__":
    unittest.main()
