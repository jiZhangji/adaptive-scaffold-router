import json
import tempfile
import unittest
from pathlib import Path

from generate_subproblems_deepseek import (
    build_prompt,
    build_summary,
    normalize_answer,
    parse_candidate,
    parse_candidate_set,
    parse_dimensions,
    read_terminal_failures,
    select_rows,
    stable_id,
    validate_candidate,
)


class GenerateSubproblemsDeepSeekTest(unittest.TestCase):
    def test_parses_expected_json(self):
        result = parse_candidate(
            json.dumps(
                {
                    "subproblem": "Compute 7+5.",
                    "answer": "12",
                    "relation": "ones-place sum",
                    "source_step": "add the ones",
                    "verification": "7+5=12",
                }
            )
        )
        self.assertEqual(result["subproblem"], "Compute 7+5.")
        self.assertEqual(result["answer"], "12")

    def test_parses_all_requested_dimensions(self):
        item = {
            "subproblem": "Compute 7+5.",
            "answer": "12",
            "relation": "ones-place sum",
            "source_step": "add the ones",
            "verification": "7+5=12",
        }
        result = parse_candidate_set(
            json.dumps({"candidates": {"knowledge": item, "calculation": item}}),
            ("knowledge", "calculation"),
        )
        self.assertEqual(set(result), {"knowledge", "calculation"})

    def test_dimension_parser_rejects_unknown_dimension(self):
        self.assertEqual(
            parse_dimensions("knowledge,planning,calculation"),
            ("knowledge", "planning", "calculation"),
        )
        with self.assertRaises(ValueError):
            parse_dimensions("unknown")

    def test_rejects_root_answer_leak(self):
        with self.assertRaisesRegex(ValueError, "original final answer"):
            validate_candidate(
                {
                    "subproblem": "Compute a prerequisite.",
                    "answer": "\\boxed{42}",
                },
                "42",
            )

    def test_prompt_uses_existing_scaffolds(self):
        row = {
            "knowledge_components_parts": ["slope formula"],
            "planning_skeleton_parts": ["find the slope"],
            "solution_breakdown_parts": ["slope equals 2"],
            "solution": "First calculate the slope.",
        }
        prompt = build_prompt(
            row, "Find the intercept.", "-3/2", ("knowledge", "calculation")
        )
        self.assertIn("slope formula", prompt)
        self.assertIn("slope equals 2", prompt)
        self.assertIn("must not ask for, copy, or reveal", prompt)
        self.assertIn('"knowledge"', prompt)

    def test_selection_is_stable_and_filters_easy_rows(self):
        rows = [
            {"id": i, "question": f"q{i}", "accuracy": accuracy,
             "reward_model": {"ground_truth": str(i)}}
            for i, accuracy in enumerate((0.0, 0.5, 1.0))
        ]
        first, eligible = select_rows(rows, limit=2, seed=7, max_source_accuracy=0.9)
        second, _ = select_rows(rows, limit=2, seed=7, max_source_accuracy=0.9)
        self.assertEqual(eligible, 2)
        self.assertEqual(first, second)
        self.assertNotIn("q2", {row["question"] for row in first})

    def test_normalization_and_id(self):
        self.assertEqual(normalize_answer("$\\boxed{12}$"), "12")
        self.assertEqual(stable_id("x", "question"), stable_id("x", "question"))

    def test_summary_counts_dimension_candidates(self):
        from argparse import Namespace
        from collections import Counter
        from pathlib import Path

        summary = build_summary(
            args=Namespace(
                model="teacher",
                data=Path("data.parquet"),
                output=Path("out.jsonl"),
                seed=42,
                max_source_accuracy=0.0,
            ),
            dimensions=("knowledge", "planning", "calculation"),
            eligible_count=10,
            selected_count=10,
            completed_before_run=0,
            completed_count=30,
            success=30,
            skipped=0,
            failure=0,
            reasons=Counter(),
            usage_totals=Counter(total_tokens=100),
        )
        self.assertEqual(summary["requested_candidates"], 30)
        self.assertEqual(summary["total_output_rows"], 30)

    def test_resume_skips_only_terminal_leakage_failures(self):
        rows = [
            {
                "id": "leak",
                "error": "ValueError: subproblem answer equals the original final answer",
            },
            {
                "id": "paid",
                "error": "HTTPError: HTTP Error 402: Payment Required",
            },
            {
                "id": "json",
                "error": "JSONDecodeError: Unterminated string",
            },
            {"id": "explicit", "error": "other", "terminal": True},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errors.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual(read_terminal_failures(path), {"leak", "explicit"})


if __name__ == "__main__":
    unittest.main()
