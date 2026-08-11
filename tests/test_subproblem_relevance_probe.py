import unittest

from subproblem_relevance_probe import analyze_records, parse_subproblem


class SubproblemRelevanceProbeTests(unittest.TestCase):
    def test_parses_strict_subproblem_tags(self):
        parsed = parse_subproblem(
            "<subproblem>Compute 2+3.</subproblem><answer>5</answer>"
        )
        self.assertEqual(parsed, ("Compute 2+3.", "5"))
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
        subproblem_records = [{"correct": True}]
        result = analyze_records(candidates, root_records, subproblem_records)
        self.assertEqual(result["causal_checks"]["relevant_gain_over_random"], 1.0)
        self.assertEqual(result["causal_checks"]["rescue_advantage_over_random"], 1.0)


if __name__ == "__main__":
    unittest.main()
