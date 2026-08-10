import sys
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ScafCurriculumAdapterTests(unittest.TestCase):
    def test_manifest_resolves_by_stable_question_key(self):
        from capability_scaffold import stable_question_key
        from scaf_curriculum_adapter import load_curriculum_manifest, resolve_curriculum_decision

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curriculum.jsonl"
            row = {
                "question_key": stable_question_key("question text"),
                "phase": "guided_root",
                "selected_scaffold": "planning@25",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest = load_curriculum_manifest(path)
            self.assertEqual(
                resolve_curriculum_decision("question   text", manifest)["selected_scaffold"],
                "planning@25",
            )

    def test_sequence_weights_and_advantage_application(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional in the local probe environment")

        from scaf_curriculum_adapter import apply_sequence_weights, sequence_importance_weights

        target = torch.tensor([[-1.0, -1.0, -9.0], [-2.0, -2.0, -2.0]])
        behavior = torch.tensor([[-2.0, -2.0, -3.0], [-2.0, -2.0, -2.0]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        weights = sequence_importance_weights(target, behavior, mask, 0.2, 5.0)
        self.assertAlmostEqual(float(weights[0]), 5.0, places=5)
        self.assertAlmostEqual(float(weights[1]), 1.0, places=5)

        advantages = torch.ones_like(target)
        weighted = apply_sequence_weights(advantages, weights, mask)
        self.assertEqual(float(weighted[0, 2]), 0.0)
        self.assertAlmostEqual(float(weighted[0, 0]), 5.0, places=5)

    def test_builds_guided_request_and_fades_selected_parts(self):
        from scaf_curriculum_adapter import build_curriculum_prompt

        decision = {
            "phase": "guided_root",
            "selected_scaffold": "planning@100",
        }
        request = build_curriculum_prompt(
            "q",
            {"planning": ["p1", "p2", "p3", "p4"]},
            decision,
            step=5,
            fade_start=0,
            fade_end=10,
        )
        self.assertEqual(request["scaffold_kind"], "planning")
        self.assertEqual(request["hint_parts"], ("p1", "p2"))
        self.assertEqual(request["visible_fraction"], 0.5)

    def test_non_guided_phase_has_no_hint(self):
        from scaf_curriculum_adapter import build_curriculum_prompt

        request = build_curriculum_prompt(
            "q", {}, {"phase": "decompose_or_defer", "selected_scaffold": "plan@25"}, 1, 0, 10
        )
        self.assertEqual(request["hint_parts"], ())
        self.assertIsNone(request["scaffold_name"])


if __name__ == "__main__":
    unittest.main()
