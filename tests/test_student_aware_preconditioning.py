import sys
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_student_aware_preconditioning_experiment import (
    candidate_diagnostics,
    needs_preconditioning,
    run as build_experiment,
)
from probe_selected_subproblem_learnability import aggregate_learnability


class StudentAwarePreconditioningTests(unittest.TestCase):
    def test_combines_subproblem_contrast_and_root_relevance(self):
        candidate = {
            "success_probability": 0.25,
            "no_help_probability": 0.0,
            "random_plan_probability": 0.0,
        }
        diagnostics = candidate_diagnostics(
            candidate,
            {"p_sub": 0.5},
            group_size=8,
        )
        self.assertAlmostEqual(diagnostics["relevance_score"], 0.25)
        self.assertGreater(diagnostics["contrast_score"], 0.99)
        self.assertGreater(diagnostics["final_score"], 0.24)

    def test_only_low_usability_candidates_precondition(self):
        low = {"q_help": 0.25, "p_sub": 0.5, "contrast_score": 0.9}
        ready = {"q_help": 0.50, "p_sub": 0.5, "contrast_score": 0.9}
        dead_subproblem = {"q_help": 0.25, "p_sub": 0.0, "contrast_score": 0.0}
        self.assertTrue(needs_preconditioning(low, 0.50, 0.5))
        self.assertFalse(needs_preconditioning(ready, 0.50, 0.5))
        self.assertFalse(needs_preconditioning(dead_subproblem, 0.50, 0.0))

    def test_aggregates_subproblem_success_and_contrast(self):
        candidates = [{"id": "c", "root_id": "r", "dimension": "planning"}]
        records = [
            {"candidate_id": "c", "correct": value}
            for value in (True, False, True, False)
        ]
        result = aggregate_learnability(candidates, records, group_size=8)[0]
        self.assertEqual(result["p_sub"], 0.5)
        self.assertGreater(result["contrast_score"], 0.99)

    def test_builds_three_stage_data_from_existing_pair(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is optional in the local test environment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            candidates = root / "candidates.jsonl"
            learnability = root / "learnability.jsonl"
            output = root / "out"
            pd.DataFrame(
                [
                    {
                        "id": 1,
                        "question": "What is 2+2?",
                        "prompt": [{"role": "user", "content": "What is 2+2?"}],
                        "reward_model": {"ground_truth": "4", "style": "rule"},
                        "data_source": "math",
                        "extra_info": {"seed": 0},
                        "knowledge_components_parts": [],
                        "planning_skeleton_parts": [],
                        "solution_breakdown_parts": [],
                    }
                ]
            ).to_parquet(source, index=False)
            candidate = {
                "id": "r::planning",
                "root_id": "r",
                "source_id": "1",
                "dimension": "planning",
                "question": "What is 2+2?",
                "reference": "4",
                "subproblem": "What is 1+1?",
                "subproblem_answer": "2",
                "minimal_plan": "add the two values",
                "success_probability": 0.25,
                "no_help_probability": 0.0,
                "random_plan_probability": 0.0,
            }
            candidates.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
            learnability.write_text(
                json.dumps({"candidate_id": candidate["id"], "p_sub": 0.5}) + "\n",
                encoding="utf-8",
            )
            build_experiment(
                Namespace(
                    source_data=source,
                    candidates=candidates,
                    learnability=learnability,
                    output_dir=output,
                    scaffold_ready_threshold=0.5,
                    contrast_min=0.0,
                    group_size=8,
                    max_plan_words=12,
                )
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["precondition_roots"], 1)
            self.assertTrue((output / "precondition_train.parquet").is_file())
            self.assertTrue((output / "root_scaffold_train.parquet").is_file())
            self.assertTrue((output / "root_only_train.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
