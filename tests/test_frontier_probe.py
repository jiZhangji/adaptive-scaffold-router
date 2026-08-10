import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier_probe import (
    analyze_records,
    build_scaffold_arms,
    has_complete_boxed_answer,
    stable_item_id,
)


class FrontierProbeTests(unittest.TestCase):
    def test_builds_two_dimensional_scaffold_lattice(self):
        row = {
            "knowledge_components_parts": ["k1", "k2", "k3", "k4"],
            "planning_skeleton_parts": ["p1", "p2", "p3", "p4"],
            "solution_breakdown_parts": ["s1", "s2", "s3", "s4"],
        }
        arms = build_scaffold_arms(row, [0.25, 0.5, 1.0])
        self.assertEqual(len(arms), 10)
        self.assertEqual(arms[0].name, "none")
        self.assertEqual(arms[1].name, "knowledge@25")
        self.assertEqual(arms[1].selected_parts, 1)
        self.assertEqual(arms[2].selected_parts, 2)
        self.assertEqual(arms[3].selected_parts, 4)

    def test_analysis_finds_rescue_and_cost_gap(self):
        records = []
        arms = [
            ("none", "none", 0.0, 0, False, 0),
            ("knowledge@25", "knowledge", 0.25, 1, True, 10),
            ("solution@100", "solution", 1.0, 2, True, 40),
        ]
        for sample_index in range(2):
            for name, kind, strength, order, correct, hint_tokens in arms:
                records.append(
                    {
                        "id": "x",
                        "question": "q",
                        "reference": "1",
                        "arm_name": name,
                        "arm_kind": kind,
                        "arm_strength": strength,
                        "arm_order": order,
                        "sample_index": sample_index,
                        "correct": correct,
                        "input_tokens": 20 + hint_tokens,
                        "hint_tokens": hint_tokens,
                        "output_tokens": 10,
                    }
                )
        summary, frontiers = analyze_records(records, 0.5, 0.25, 0.75, 0.02, 0.05)
        self.assertEqual(summary["rescued_zero_accuracy_examples"], 1)
        self.assertEqual(frontiers[0]["threshold_choice"], "knowledge@25")
        self.assertGreater(
            summary["cost_simulation"]["progressive_average_total_tokens"],
            summary["cost_simulation"]["oracle_min_cost_average_total_tokens_on_success"],
        )
        self.assertEqual(
            summary["cost_simulation"]["public_scaf_progressive"]["accuracy"],
            summary["cost_simulation"]["public_scaf_min_cost_oracle"]["accuracy"],
        )

    def test_detects_complete_nested_boxed_answer(self):
        self.assertTrue(has_complete_boxed_answer("answer \\boxed{\\frac{1}{2}}"))
        self.assertFalse(has_complete_boxed_answer("answer \\boxed{\\frac{1}{2}"))

    def test_stable_id_disambiguates_reused_source_ids(self):
        self.assertNotEqual(stable_item_id("x", "question one"), stable_item_id("x", "question two"))


if __name__ == "__main__":
    unittest.main()
