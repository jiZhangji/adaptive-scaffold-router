import unittest

from scaf_integration.curriculum_runtime import (
    _as_parts,
    _build_prompt,
    guided_rollout_count,
    load_optional_manifest,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        self.messages = messages
        return messages[-1]["content"]


class CurriculumRuntimeTests(unittest.TestCase):
    def test_optional_manifest_can_be_disabled(self):
        self.assertIsNone(load_optional_manifest(None))
        self.assertIsNone(load_optional_manifest("  "))

    def test_normalizes_hint_parts(self):
        self.assertEqual(_as_parts("plan"), ["plan"])
        self.assertEqual(_as_parts(["a", "", "b"]), ["a", "b"])

    def test_prompt_omits_empty_scaffold(self):
        tokenizer = FakeTokenizer()
        prompt = _build_prompt(tokenizer, "2+2?", None, ())
        self.assertEqual(prompt, "Question: 2+2?")

    def test_prompt_labels_planning_scaffold(self):
        tokenizer = FakeTokenizer()
        prompt = _build_prompt(tokenizer, "2+2?", "planning", ("add", "verify"))
        self.assertIn("Planning Hints: add verify", prompt)

    def test_fading_fraction_controls_scaffold_dropout(self):
        self.assertEqual(guided_rollout_count(8, 1.0), 8)
        self.assertEqual(guided_rollout_count(8, 0.5), 4)
        self.assertEqual(guided_rollout_count(8, 0.0), 0)


if __name__ == "__main__":
    unittest.main()
