import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier_to_curriculum import build_root_states


class FrontierToCurriculumTests(unittest.TestCase):
    def test_groups_rollouts_into_root_and_scaffold_arms(self):
        records = []
        for sample_index in range(2):
            for name, kind, strength, correct, hint_tokens in (
                ("none", "none", 0.0, False, 0),
                ("planning@25", "planning", 0.25, sample_index == 0, 12),
            ):
                records.append(
                    {
                        "id": "root",
                        "question": "question",
                        "reference": "answer",
                        "arm_name": name,
                        "arm_kind": kind,
                        "arm_strength": strength,
                        "sample_index": sample_index,
                        "correct": correct,
                        "hint_tokens": hint_tokens,
                    }
                )
        roots = build_root_states(records)
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].root_rewards, (0.0, 0.0))
        self.assertEqual(roots[0].scaffolds[0].success_probability, 0.5)
