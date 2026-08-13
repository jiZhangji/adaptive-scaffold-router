import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_paper_eval import DATASETS, PAPER, read_pass_at_one


class PaperEvalSummaryTest(unittest.TestCase):
    def test_reads_nested_pass_at_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            metric = Path(tmp) / "metric.json"
            metric.write_text(
                json.dumps({"MATH-500/math-verify": {"pass@1": 0.536}}),
                encoding="utf-8",
            )
            self.assertAlmostEqual(read_pass_at_one(metric), 53.6)

    def test_has_all_seven_paper_datasets(self):
        self.assertEqual(len(DATASETS), 7)
        self.assertIn("GaoKao2023en", DATASETS)

    def test_one_point_five_b_paper_averages(self):
        self.assertAlmostEqual(sum(PAPER["1.5b"]["base"]) / 7, 18.7, places=1)
        self.assertAlmostEqual(sum(PAPER["1.5b"]["vanilla"]) / 7, 37.6, places=1)
        self.assertAlmostEqual(sum(PAPER["1.5b"]["scaf"]) / 7, 41.5, places=1)

    def test_paper_rows_share_the_unified_dataset_order(self):
        for model_rows in PAPER.values():
            for row in model_rows.values():
                self.assertEqual(len(row), len(DATASETS))


if __name__ == "__main__":
    unittest.main()
