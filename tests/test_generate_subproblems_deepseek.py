import json
import unittest

from generate_subproblems_deepseek import (
    build_prompt,
    normalize_answer,
    parse_candidate,
    select_rows,
    stable_id,
    validate_candidate,
)


class GenerateSubproblemsDeepSeekTest(unittest.TestCase):
    def test_parses_expected_json(self):
        result = parse_candidate(
            json.dumps(
                {
                    "subproblem": "Compute 7+5.",
                    "answer": "12",
                    "relation": "ones-place sum",
                    "source_step": "add the ones",
                    "verification": "7+5=12",
                }
            )
        )
        self.assertEqual(result["subproblem"], "Compute 7+5.")
        self.assertEqual(result["answer"], "12")

    def test_rejects_root_answer_leak(self):
        with self.assertRaisesRegex(ValueError, "original final answer"):
            validate_candidate(
                {
                    "subproblem": "Compute a prerequisite.",
                    "answer": "\\boxed{42}",
                },
                "42",
            )

    def test_prompt_uses_existing_scaffolds(self):
        row = {
            "knowledge_components_parts": ["slope formula"],
            "planning_skeleton_parts": ["find the slope"],
            "solution_breakdown_parts": ["slope equals 2"],
            "solution": "First calculate the slope.",
        }
        prompt = build_prompt(row, "Find the intercept.", "-3/2")
        self.assertIn("slope formula", prompt)
        self.assertIn("slope equals 2", prompt)
        self.assertIn("must not ask for, copy, or reveal", prompt)

    def test_selection_is_stable_and_filters_easy_rows(self):
        rows = [
            {"id": i, "question": f"q{i}", "accuracy": accuracy,
             "reward_model": {"ground_truth": str(i)}}
            for i, accuracy in enumerate((0.0, 0.5, 1.0))
        ]
        first, eligible = select_rows(rows, limit=2, seed=7, max_source_accuracy=0.9)
        second, _ = select_rows(rows, limit=2, seed=7, max_source_accuracy=0.9)
        self.assertEqual(eligible, 2)
        self.assertEqual(first, second)
        self.assertNotIn("q2", {row["question"] for row in first})

    def test_normalization_and_id(self):
        self.assertEqual(normalize_answer("$\\boxed{12}$"), "12")
        self.assertEqual(stable_id("x", "question"), stable_id("x", "question"))


if __name__ == "__main__":
    unittest.main()
