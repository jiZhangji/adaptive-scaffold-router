import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_complete_four_way import DATASETS, collect


class CompleteFourWaySummaryTest(unittest.TestCase):
    def test_collects_all_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in DATASETS:
                path = root / "eval_vanilla" / dataset
                path.mkdir(parents=True)
                (path / "metric.json").write_text(
                    json.dumps({"math-verify": {"pass@1": 0.25}}), encoding="utf-8"
                )
            self.assertEqual(set(collect(root, "vanilla")), set(DATASETS))


if __name__ == "__main__":
    unittest.main()
