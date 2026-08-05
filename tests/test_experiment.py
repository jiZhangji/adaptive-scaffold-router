import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiment import Generation, extract_number, is_correct, run_example, summarize


class FakeBackend:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages):
        del messages
        return Generation(next(self.responses), output_tokens=5, latency_seconds=0.1)


class ExperimentTests(unittest.TestCase):
    def test_extract_number(self):
        self.assertEqual(str(extract_number("FINAL_ANSWER: 1,275")), "1275")
        self.assertEqual(str(extract_number("FINAL_ANSWER: 3/8")), "3/8")
        self.assertEqual(str(extract_number("FINAL_ANSWER: $86.40")), "432/5")

    def test_correctness(self):
        self.assertTrue(is_correct("Reasoning\nFINAL_ANSWER: 3/8", "3/8"))
        self.assertFalse(is_correct("FINAL_ANSWER: 2/8", "3/8"))

    def test_progressive_stops_at_minimal_success(self):
        example = {
            "id": "x",
            "question": "question",
            "answer": "10",
            "hints": {"knowledge": "k", "planning": "p", "solution": "s"},
        }
        backend = FakeBackend(
            [
                "FINAL_ANSWER: 8",
                "FINAL_ANSWER: 10",
                "FINAL_ANSWER: 10",
            ]
        )
        result = run_example(backend, example, samples_per_level=1)
        self.assertEqual(result["selected_level_name"], "knowledge")
        self.assertTrue(result["scaffold_recovered"])
        self.assertEqual(result["progressive_calls"], 2)

    def test_summary(self):
        results = [
            {
                "no_hint_correct": True,
                "full_hint_correct": True,
                "progressive_correct": True,
                "selected_level": 0,
                "selected_level_name": "none",
                "scaffold_recovered": False,
                "progressive_calls": 1,
                "progressive_output_tokens": 5,
                "full_hint_output_tokens": 5,
                "progressive_latency_seconds": 0.1,
                "full_hint_latency_seconds": 0.1,
            },
            {
                "no_hint_correct": False,
                "full_hint_correct": True,
                "progressive_correct": True,
                "selected_level": 2,
                "selected_level_name": "planning",
                "scaffold_recovered": True,
                "progressive_calls": 3,
                "progressive_output_tokens": 15,
                "full_hint_output_tokens": 5,
                "progressive_latency_seconds": 0.3,
                "full_hint_latency_seconds": 0.1,
            },
        ]
        summary = summarize(results)
        self.assertEqual(summary["progressive_accuracy"], 1.0)
        self.assertEqual(summary["scaffold_recovery_count"], 1)
        self.assertEqual(summary["average_selected_level_on_success"], 1.0)


if __name__ == "__main__":
    unittest.main()
