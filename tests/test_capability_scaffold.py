import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capability_scaffold import (
    ControllerConfig,
    RootState,
    RolloutArm,
    SubproblemState,
    clipped_importance_weight,
    decide_curriculum,
    fade_scaffold,
    informative_group_probability,
    select_scaffold,
    stable_question_key,
    visible_scaffold_fraction,
)


class CapabilityScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.config = ControllerConfig(
            band_low=0.25,
            band_high=0.60,
            mastery_threshold=0.75,
            prerequisite_coverage=0.5,
            group_size=4,
            max_hint_tokens=64,
        )

    def test_mixed_group_probability_is_maximal_at_half(self):
        middle = informative_group_probability(0.5, 8)
        self.assertGreater(middle, informative_group_probability(0.1, 8))
        self.assertGreater(middle, informative_group_probability(0.9, 8))

    def test_question_key_ignores_whitespace_but_not_content(self):
        self.assertEqual(stable_question_key("a  b\n c"), stable_question_key("a b c"))
        self.assertNotEqual(stable_question_key("a b c"), stable_question_key("a b d"))

    def test_selects_cheapest_scaffold_inside_learning_band(self):
        arms = [
            RolloutArm("weak", (0, 1, 0, 0), 20, 0.25, "plan"),
            RolloutArm("strong", (1, 1, 0, 0), 50, 1.0, "plan"),
            RolloutArm("too-long", (0, 1, 1, 0), 100, 1.0, "solution"),
        ]
        self.assertEqual(select_scaffold(arms, self.config).name, "weak")

    def test_uses_subproblem_curriculum_before_unready_root(self):
        root = RootState(
            id="r1",
            question="hard",
            answer="1",
            root_rewards=(0, 0, 0, 0),
            subproblems=(
                SubproblemState("s1", "trainable", "1", (0, 1, 0, 0)),
                SubproblemState("s2", "unresolved", "1", (0, 0, 0, 0)),
            ),
            scaffolds=(RolloutArm("plan", (0, 1, 0, 0), 30, 0.5, "plan"),),
        )
        decision = decide_curriculum(root, self.config)
        self.assertEqual(decision.phase, "subproblem_curriculum")
        self.assertEqual(decision.active_subproblem_ids, ("s1",))

    def test_activates_guided_root_after_prerequisites_are_mastered(self):
        root = RootState(
            id="r2",
            question="hard",
            answer="1",
            root_rewards=(0, 0, 0, 0),
            subproblems=(
                SubproblemState("s1", "mastered", "1", (1, 1, 1, 0)),
                SubproblemState("s2", "mastered", "1", (1, 1, 1, 1)),
            ),
            scaffolds=(RolloutArm("weak-plan", (0, 1, 0, 0), 24, 0.25, "plan"),),
        )
        decision = decide_curriculum(root, self.config)
        self.assertEqual(decision.phase, "guided_root")
        self.assertEqual(decision.selected_scaffold, "weak-plan")

    def test_fades_scaffold_and_computes_clipped_weight(self):
        self.assertEqual(visible_scaffold_fraction(5, 0, 10), 0.5)
        self.assertEqual(fade_scaffold(["a", "b", "c", "d"], 0.5), ("a", "b"))
        weight = clipped_importance_weight(
            target_token_logprobs=[-1.0, -1.0],
            behavior_token_logprobs=[-2.0, -2.0],
            clip_low=0.2,
            clip_high=5.0,
        )
        self.assertTrue(math.isclose(weight, 5.0))


if __name__ == "__main__":
    unittest.main()
