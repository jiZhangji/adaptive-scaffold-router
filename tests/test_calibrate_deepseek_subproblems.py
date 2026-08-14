import unittest

from calibrate_deepseek_subproblems import (
    choose_training_candidates,
    select_complete_roots,
)


class DeepSeekCalibrationTest(unittest.TestCase):
    def test_selects_only_complete_roots(self):
        rows = []
        for dimension in ("knowledge", "planning", "calculation"):
            rows.append({"id": f"a::{dimension}", "root_id": "a", "dimension": dimension})
        rows.append({"id": "b::knowledge", "root_id": "b", "dimension": "knowledge"})
        selected = select_complete_roots(rows, root_limit=10, seed=42)
        self.assertEqual({row["root_id"] for row in selected}, {"a"})
        self.assertEqual(len(selected), 3)

    def test_chooses_one_in_band_candidate_per_root(self):
        candidates = [
            {
                "id": "a::knowledge",
                "root_id": "a",
                "dimension": "knowledge",
                "subproblem": "A longer candidate question",
            },
            {
                "id": "a::planning",
                "root_id": "a",
                "dimension": "planning",
                "subproblem": "Short",
            },
            {
                "id": "a::calculation",
                "root_id": "a",
                "dimension": "calculation",
                "subproblem": "Too easy",
            },
        ]
        chosen = choose_training_candidates(
            candidates,
            {"a::knowledge": 0.25, "a::planning": 0.5, "a::calculation": 1.0},
            0.25,
            0.60,
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["id"], "a::planning")
        self.assertEqual(chosen[0]["success_probability"], 0.5)
        self.assertTrue(chosen[0]["trainable"])


if __name__ == "__main__":
    unittest.main()
