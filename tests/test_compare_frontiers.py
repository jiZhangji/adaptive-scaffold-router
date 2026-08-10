import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_frontiers import compare_frontiers


def make_row(item_id, no_hint_p, choice, hint_tokens):
    return {
        "id": item_id,
        "no_hint_p": no_hint_p,
        "threshold_choice": choice,
        "arms": [
            {
                "arm_name": choice,
                "hint_tokens": hint_tokens,
            }
        ],
    }


class CompareFrontiersTests(unittest.TestCase):
    def test_detects_capability_dependent_frontier_shift(self):
        weaker = {"x": make_row("x", 0.0, "solution@100", 100)}
        stronger = {"x": make_row("x", 0.5, "knowledge@25", 20)}
        result = compare_frontiers(weaker, stronger, "threshold_choice")
        self.assertEqual(result["choice_changed_fraction"], 1.0)
        self.assertEqual(result["stronger_uses_fewer_hint_tokens_fraction"], 1.0)
        self.assertEqual(result["stronger_average_no_hint_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
