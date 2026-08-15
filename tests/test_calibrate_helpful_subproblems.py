import unittest

from calibrate_helpful_subproblems import choose_candidate, minimal_plan


class HelpfulSubproblemCalibrationTest(unittest.TestCase):
    def test_plan_is_short_and_rejects_answer_leak(self):
        candidate = {
            "source_step": "Use the Pythagorean theorem to determine the missing side before substitution",
            "subproblem_answer": "5",
            "reference": "13",
        }
        plan = minimal_plan(candidate, 6)
        self.assertLessEqual(len(plan.split()), 6)
        candidate["source_step"] = "The missing side is 5"
        self.assertEqual(minimal_plan(candidate, 12), "")

    def test_q_is_root_success_and_must_beat_controls(self):
        candidate = {"id": "c", "dimension": "planning", "minimal_plan": "factor first"}
        records = []
        for index, correct in enumerate((0, 0, 0, 0)):
            records.append({"variant": "no_help", "correct": correct, "sample_index": index})
        for variant, values in (("relevant_plan", (1, 1, 0, 0)), ("random_plan", (0, 0, 0, 0))):
            for index, correct in enumerate(values):
                records.append(
                    {
                        "variant": variant,
                        "candidate_id": "c",
                        "correct": correct,
                        "sample_index": index,
                    }
                )
        chosen = choose_candidate([candidate], records, 0.25, 0.60, 0.0)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["success_probability"], 0.5)

    def test_does_not_stop_before_minimum_samples(self):
        candidate = {"id": "c", "dimension": "planning", "minimal_plan": "factor first"}
        records = [
            {"variant": "no_help", "correct": 0, "sample_index": 0},
            {"variant": "relevant_plan", "candidate_id": "c", "correct": 1, "sample_index": 0},
            {"variant": "random_plan", "candidate_id": "c", "correct": 0, "sample_index": 0},
        ]
        self.assertIsNone(choose_candidate([candidate], records, 0.25, 1.0, 0.0, 4))


if __name__ == "__main__":
    unittest.main()
