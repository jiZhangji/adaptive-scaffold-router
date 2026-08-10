import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_router import evaluate_routing, expected_calibration_error, split_item_ids


class TrainRouterTests(unittest.TestCase):
    def test_group_split_is_disjoint(self):
        train, validation, test = split_item_ids(
            [f"q{index}" for index in range(20)], 42, 0.6, 0.2
        )
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(len(train | validation | test), 20)

    def test_ece_is_zero_for_calibrated_bins(self):
        labels = [0, 0, 1, 1]
        probabilities = [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(expected_calibration_error(labels, probabilities, bins=2), 0.0)

    def test_routing_evaluates_each_rollout_trajectory_separately(self):
        records = []
        probabilities = []
        for sample_index in range(2):
            for order, (name, kind, correct) in enumerate(
                [("none", "none", False), ("knowledge@25", "knowledge", True)]
            ):
                records.append(
                    {
                        "id": "q",
                        "item_key": "q::hash",
                        "sample_index": sample_index,
                        "arm_name": name,
                        "arm_kind": kind,
                        "arm_strength": 0.25 if order else 0.0,
                        "arm_order": order,
                        "hint_tokens": 10 * order,
                        "input_tokens": 20,
                        "output_tokens": 10,
                        "correct": correct,
                    }
                )
                probabilities.append(0.8 if correct else 0.1)
        metrics, _ = evaluate_routing(records, probabilities, 0.5, 1)
        self.assertEqual(metrics["num_test_examples"], 1)
        self.assertEqual(metrics["num_test_trajectories"], 2)
        self.assertEqual(metrics["public_scaf_progressive"]["average_calls"], 2.0)


if __name__ == "__main__":
    unittest.main()
