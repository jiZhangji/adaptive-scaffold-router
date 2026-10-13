import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_simple_rcst import build_rows, review_template, unique_records
from rcst_source_join import attach_sources


def fixture():
    sources, candidates, learns = [], [], []
    for i in range(2):
        question = f"What is {i}+2?"
        sources.append({
            "id": 7, "question": question,
            "prompt": [{"role": "system", "content": "Box the answer."},
                       {"role": "user", "content": question}],
            "reward_model": {"style": "rule", "ground_truth": str(i + 2)},
            "data_source": "math", "extra_info": {"bridge_enabled": True},
            "planning_skeleton_parts": ["OLD HINT"], "solution": f"solution {i}",
        })
        candidates.append({
            "id": f"c{i}", "root_id": f"r{i}", "source_id": "7",
            "question": question, "reference": str(i + 2),
            "subproblem": f"What is {i}+1?", "subproblem_answer": str(i + 1),
            "transfer_probe": {"gain_lcb": 0.1, "replicates": 2},
        })
        learns.append({"candidate_id": f"c{i}", "root_id": f"r{i}",
                       "num_samples": 8, "correct_samples": 4, "p_sub": 0.5})
    return sources, candidates, learns


def reviewed(candidates, learns):
    rows = review_template(candidates, unique_records(learns, "candidate_id"))
    for row in rows:
        row.update(root_answer_verified=True, subproblem_and_reward_verified=True,
                   probe_and_learnability_use_reviewed_data=True,
                   evidence="synthetic arithmetic fixture; not production approval")
    return rows


class SourceJoinTests(unittest.TestCase):
    def test_id_collision_uses_full_question_and_preserves_inputs(self):
        sources, candidates, _ = fixture()
        before = copy.deepcopy((sources, candidates))
        rows = attach_sources(candidates, sources[::-1])
        self.assertEqual([r["reference_solution"] for r in rows], ["solution 0", "solution 1"])
        self.assertEqual((sources, candidates), before)
        self.assertFalse(rows[0]["source_join_audit"]["mathematical_correctness_verified"])

    def test_missing_ambiguous_and_duplicate_roots_fail(self):
        sources, candidates, _ = fixture()
        for roots, source in ((candidates, sources[:1]), (candidates, sources + sources[:1]),
                              (candidates + candidates[:1], sources)):
            with self.subTest(roots=len(roots), source=len(source)), self.assertRaises(ValueError):
                attach_sources(roots, source)


class SimpleRCSTTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.candidates, self.learns = fixture()

    def build(self, reviews=None):
        return build_rows(self.sources, self.candidates, self.learns,
                          reviewed(self.candidates, self.learns) if reviews is None else reviews,
                          steps=5, mixed_steps=3, batch_size=8, sub_per_batch=2)

    def test_exact_schedule_no_hint_unique_ids_and_shared_root_slots(self):
        original = copy.deepcopy((self.sources, self.candidates, self.learns))
        vanilla, method, summary = self.build()
        self.assertEqual(len(vanilla), 40)
        self.assertEqual(summary["subproblem_slots"], 6)
        self.assertEqual([r["id"] for r in method], list(range(40)))
        for step in range(5):
            rows = method[step * 8:(step + 1) * 8]
            self.assertEqual(sum(r["extra_info"]["is_subproblem"] for r in rows), 2 if step < 3 else 0)
        for v, r in zip(vanilla, method):
            self.assertEqual(r["prompt"][0], self.sources[0]["prompt"][0])
            self.assertEqual(r["prompt"][-1]["content"], r["question"])
            self.assertEqual(r["planning_skeleton_parts"], [])
            self.assertNotIn("bridge_enabled", r["extra_info"])
            if not r["extra_info"]["is_subproblem"]:
                self.assertEqual(v, r)
        self.assertEqual((self.sources, self.candidates, self.learns), original)
        self.assertEqual((vanilla, method, summary), self.build())

    def test_missing_or_stale_reviews_and_source_answer_mismatch_fail(self):
        with self.assertRaises(ValueError):
            self.build([])
        rows = reviewed(self.candidates, self.learns)
        self.candidates[0]["subproblem_answer"] = "changed"
        with self.assertRaises(ValueError):
            self.build(rows)
        self.sources[0]["reward_model"]["ground_truth"] = "wrong"
        with self.assertRaises(ValueError):
            self.build()

    def test_unreviewed_children_fall_back_without_dropping_roots(self):
        checks = reviewed(self.candidates, self.learns)
        for r in checks:
            r["subproblem_and_reward_verified"] = False
        vanilla, method, summary = self.build(checks)
        self.assertEqual(vanilla, method)
        self.assertEqual(summary["root_count"], 2)
        self.assertEqual(summary["accepted_subproblems"], 0)

    def test_abstention_unknown_and_uninformative_learnability(self):
        for scenario in ("abstain", "quarantine", "untrainable", "missing", "zero", "all", "negative", "one_probe"):
            with self.subTest(scenario=scenario):
                self.sources, self.candidates, self.learns = fixture()
                c, l = self.candidates[0], self.learns[0]
                if scenario == "abstain": c["rcst_abstained"] = True
                if scenario == "quarantine": c["eligible_for_training"] = False
                if scenario == "untrainable": c["trainable"] = False
                if scenario == "missing": self.learns = self.learns[1:]
                if scenario == "zero": l.update(correct_samples=0, p_sub=0.0)
                if scenario == "all": l.update(correct_samples=8, p_sub=1.0)
                if scenario == "negative": c["transfer_probe"]["gain_lcb"] = -0.1
                if scenario == "one_probe": c["transfer_probe"]["replicates"] = 1
                _, method, summary = self.build()
                self.assertEqual(summary["accepted_subproblems"], 1)
                self.assertFalse(any(r["extra_info"]["candidate_id"] == "c0" for r in method))

    def test_ambiguous_question_inconsistent_counts_and_hint_prompt_fail(self):
        self.sources.append(copy.deepcopy(self.sources[0]))
        with self.assertRaises(ValueError): self.build()
        self.sources, self.candidates, self.learns = fixture()
        self.learns[0]["p_sub"] = 0.2
        with self.assertRaises(ValueError): self.build()
        self.sources, self.candidates, self.learns = fixture()
        self.sources[0]["prompt"][-1]["content"] += " Here is a hint."
        with self.assertRaises(ValueError): self.build()

    def test_parquet_cli_and_training_command_without_model_execution(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.parquet"
            pd.DataFrame(self.sources).to_parquet(source, index=False)
            for name, rows in (("selected", self.candidates), ("learns", self.learns),
                               ("reviews", reviewed(self.candidates, self.learns))):
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            data = root / "data"
            command = [sys.executable, str(ROOT / "prepare_simple_rcst.py"),
                       "--source-data", str(source), "--candidates", str(root / "selected.jsonl"),
                       "--learnability", str(root / "learns.jsonl"), "--reviews", str(root / "reviews.jsonl"),
                       "--output-dir", str(data), "--steps", "5", "--mixed-steps", "3",
                       "--batch-size", "8", "--sub-per-batch", "2"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            rows = pd.read_parquet(data / "method_train.parquet")
            self.assertEqual(len(rows), 40)
            self.assertEqual(sum(r["is_subproblem"] for r in rows.extra_info), 6)
            env = dict(os.environ, DATA_DIR=str(data), SCAF_REPO=str(root), MODEL_PATH="fixture-model",
                       RUN_DIR=str(root / "run"), PYTHON_BIN=sys.executable, DRY_RUN="1")
            launcher = ["bash", str(ROOT / "scripts/run_simple_rcst_1gpu.sh")]
            result = subprocess.run(launcher, env=env, check=True, capture_output=True, text=True)
            self.assertIn("steps=5 batch=8 shuffle=false hints=false bridge=false", result.stdout)
            self.assertFalse((root / "run").exists())

            # Capture the actual trainer argv. No torch/verl/model/GPU is invoked.
            stub = root / "python-stub"
            stub.write_text(f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
                            f"Path({str(root / 'argv.json')!r}).write_text(json.dumps(sys.argv[1:]))\n")
            stub.chmod(0o755)
            arm_env = dict(env, PLAIN_GRPO="1", TRAIN_SHUFFLE="false", FILTER_OVERLONG_PROMPTS="false",
                           PYTHON_BIN=str(stub), OUTPUT_DIR=str(root / "stub-run"),
                           TRAIN_DATA=str(data / "method_train.parquet"), BRIDGE_ENABLED="1")
            subprocess.run(["bash", str(ROOT / "scripts/run_fixed_budget_grpo_arm_1gpu.sh")],
                           env=arm_env, check=True, capture_output=True, text=True)
            argv = json.loads((root / "argv.json").read_text())
            for required in ("data.shuffle=false", "data.filter_overlong_prompts=false",
                             "+algorithm.bridge.enabled=0", "data.custom_cls.path=null",
                             "data.custom_cls.name=null"):
                self.assertIn(required, argv)
            self.assertNotIn("data.custom_cls.name=RCSTBridgeDataset", argv)
            legacy_env = dict(arm_env, PLAIN_GRPO="0", TRAIN_SHUFFLE="true", FILTER_OVERLONG_PROMPTS="true")
            subprocess.run(["bash", str(ROOT / "scripts/run_fixed_budget_grpo_arm_1gpu.sh")],
                           env=legacy_env, check=True, capture_output=True, text=True)
            legacy_argv = json.loads((root / "argv.json").read_text())
            self.assertIn("data.custom_cls.name=RCSTBridgeDataset", legacy_argv)
            self.assertIn("+algorithm.bridge.enabled=1", legacy_argv)
            self.assertIn("data.shuffle=true", legacy_argv)
            with (data / "method_train.parquet").open("ab") as stream: stream.write(b"modified")
            result = subprocess.run(launcher, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Prepared data changed", result.stderr)


if __name__ == "__main__":
    unittest.main()
