import unittest

from metaask_constrained_probe import parse_action, summarize


class MetaAskConstrainedProbeTests(unittest.TestCase):
    def test_parses_only_allowed_action(self):
        self.assertEqual(parse_action("PLANNING"), "PLANNING")
        self.assertEqual(parse_action("I choose STEP."), "STEP")
        self.assertIsNone(parse_action("perhaps help"))

    def test_summary_includes_harm_and_rescue(self):
        records = [
            {"id": "a", "sample_index": 0, "variant": "no_help", "correct": False,
             "action": "NONE", "hint_tokens": 0, "invalid_action": False},
            {"id": "a", "sample_index": 0, "variant": "policy_action", "correct": True,
             "action": "STEP", "hint_tokens": 4, "invalid_action": False},
            {"id": "b", "sample_index": 0, "variant": "no_help", "correct": True,
             "action": "NONE", "hint_tokens": 0, "invalid_action": False},
            {"id": "b", "sample_index": 0, "variant": "policy_action", "correct": False,
             "action": "STEP", "hint_tokens": 4, "invalid_action": False},
        ]
        result = summarize(records)["variants"]["policy_action"]
        self.assertEqual(result["rescue_rate_on_no_help_failures"], 1.0)
        self.assertEqual(result["harms_on_no_help_successes"], 1)


if __name__ == "__main__":
    unittest.main()
