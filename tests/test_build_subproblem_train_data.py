import unittest

from build_subproblem_train_data import build_mixed_rows


class BuildSubproblemTrainDataTests(unittest.TestCase):
    def test_builds_equal_root_and_subproblem_rows(self):
        root = {
            "id": "r1",
            "question": "Hard question",
            "prompt": [{"role": "user", "content": "Hard question"}],
            "reward_model": {"ground_truth": "42", "style": "rule"},
            "data_source": "math",
            "extra_info": {},
            "knowledge_components_parts": ["private hint"],
        }
        candidate = {
            "source_id": "r1",
            "question": "Hard question",
            "subproblem": "Compute 2+3.",
            "subproblem_answer": "5",
            "success_probability": 0.5,
            "trainable": True,
        }
        rows, summary = build_mixed_rows([root], [candidate], seed=1, max_pairs=None)
        self.assertEqual(summary["mix_ratio"], "1:1")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question"], "Hard question")
        self.assertEqual(rows[1]["question"], "Compute 2+3.")
        self.assertEqual(rows[1]["data_source"], "math")
        self.assertEqual(rows[1]["reward_model"]["ground_truth"], "5")
        self.assertEqual(rows[1]["knowledge_components_parts"], [])
        self.assertTrue(rows[1]["extra_info"]["is_subproblem"])


if __name__ == "__main__":
    unittest.main()
