import unittest

from metaask_probe import analyze_records, extract_verification, parse_oracle_answer


class MetaAskProbeTests(unittest.TestCase):
    def test_extracts_state_and_verification_question(self):
        state, question = extract_verification(
            "<state>Check parity first.</state><verify>Is n even</verify>"
        )
        self.assertEqual(state, "Check parity first.")
        self.assertEqual(question, "Is n even?")

    def test_oracle_output_is_strictly_parsed(self):
        self.assertEqual(parse_oracle_answer("Answer: NO."), "NO")
        self.assertEqual(parse_oracle_answer("perhaps"), "UNKNOWN")

    def test_summary_reports_information_cost_and_rescue(self):
        records = [
            {"id": "q", "sample_index": 0, "variant": "no_help", "correct": False,
             "external_information_tokens": 0, "total_generated_tokens": 10},
            {"id": "q", "sample_index": 0, "variant": "self_asked_verification", "correct": True,
             "external_information_tokens": 1, "total_generated_tokens": 20},
        ]
        summary = analyze_records(records)
        result = summary["variants"]["self_asked_verification"]
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["avg_external_information_tokens"], 1.0)
        self.assertEqual(result["rescue_rate_on_no_help_failures"], 1.0)


if __name__ == "__main__":
    unittest.main()
