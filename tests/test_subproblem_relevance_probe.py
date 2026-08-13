import unittest

from subproblem_relevance_probe import analyze_records, parse_subproblem


class SubproblemRelevanceProbeTests(unittest.TestCase):
    def test_parses_strict_subproblem_tags(self):
        parsed = parse_subproblem(
            "<subproblem>Compute 2+3.</subproblem><answer>5</answer>"
        )
        self.assertEqual(parsed, ("Compute 2+3.", "5"))
        self.assertEqual(
            parse_subproblem('{"subproblem": "Compute 2+3.", "answer": "5"}'),
            ("Compute 2+3.", "5"),
        )
        self.assertEqual(
            parse_subproblem("Subproblem: Compute 2+3.\nAnswer: 5"),
            ("Compute 2+3.", "5"),
        )
        self.assertIsNone(parse_subproblem("Compute 2+3 = 5"))

    def test_reports_relevant_gain_against_random_control(self):
        candidates = [{"id": "q"}]
        root_records = [
            {"id": "q", "sample_index": 0, "variant": "no_help", "correct": False,
             "external_information_tokens": 0},
            {"id": "q", "sample_index": 0, "variant": "question_only", "correct": False,
             "external_information_tokens": 4},
            {"id": "q", "sample_index": 0, "variant": "random_subproblem", "correct": False,
             "external_information_tokens": 8},
            {"id": "q", "sample_index": 0, "variant": "relevant_subproblem", "correct": True,
             "external_information_tokens": 8},
        ]
        subproblem_records = [{"id": "q", "correct": True}]
        result = analyze_records(candidates, root_records, subproblem_records)
        self.assertEqual(result["causal_checks"]["relevant_gain_over_random"], 1.0)
        self.assertEqual(result["causal_checks"]["rescue_advantage_over_random"], 1.0)

    def test_calibrates_trainable_subproblems_by_per_candidate_q(self):
        subproblem_records = [
            {"id": "a", "correct": True},
            {"id": "a", "correct": False},
            {"id": "b", "correct": True},
            {"id": "b", "correct": True},
        ]
        result = analyze_records(
            [{"id": "a"}, {"id": "b"}], [], subproblem_records, q_low=0.25, q_high=0.75
        )
        self.assertEqual(result["q_calibration"]["q_by_id"], {"a": 0.5, "b": 1.0})
        self.assertEqual(result["q_calibration"]["trainable_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
