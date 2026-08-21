import unittest
from pathlib import Path

from calibrate_helpful_subproblems import (
    RootCalibrationState,
    build_pending_jobs,
    choose_candidate,
    completed_sample_prefix,
    expected_sample_keys,
    minimal_plan,
)


class HelpfulSubproblemCalibrationTest(unittest.TestCase):
    def test_boxed_stopping_uses_interval_batched_decode(self):
        source = Path("metaask_probe.py").read_text(encoding="utf-8")
        self.assertIn("stop_check_interval", source)
        self.assertIn("tokenizer.batch_decode", source)

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

    def test_plan_tries_safe_relation_after_leaking_source_step(self):
        candidate = {
            "source_step": "The missing side is 5",
            "relation": "Use the Pythagorean theorem before substitution",
            "subproblem_answer": "5",
            "reference": "13",
        }
        self.assertEqual(
            minimal_plan(candidate, 12),
            "Use the Pythagorean theorem before substitution",
        )

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

    def test_resume_fills_partial_sample_holes_before_new_indices(self):
        candidates = [
            {"id": "c", "dimension": "planning", "minimal_plan": "factor first"}
        ]
        completed = set()
        completed.update(expected_sample_keys("r", candidates, 0))
        completed.add(("r", "", "no_help", 1))
        self.assertEqual(completed_sample_prefix("r", candidates, completed, 12), 1)

        state = RootCalibrationState("r", candidates, "Question", "4", [])
        jobs, end_sample = build_pending_jobs(
            state,
            completed,
            {("r", "c"): "try substitution"},
            min_samples=4,
            max_samples=12,
            sample_batch=2,
        )
        self.assertEqual(end_sample, 4)
        keys = {
            (root, candidate, variant, sample)
            for root, candidate, variant, sample, _ in jobs
        }
        self.assertNotIn(("r", "", "no_help", 1), keys)
        self.assertIn(("r", "c", "relevant_plan", 1), keys)
        self.assertIn(("r", "", "no_help", 2), keys)
        self.assertNotIn(("r", "", "no_help", 4), keys)


if __name__ == "__main__":
    unittest.main()
