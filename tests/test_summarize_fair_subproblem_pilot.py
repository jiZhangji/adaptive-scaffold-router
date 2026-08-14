import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from summarize_fair_subproblem_pilot import DATASETS, collect


class FairPilotSummaryTests(unittest.TestCase):
    def test_collects_all_seven_pass_at_one_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for index, dataset in enumerate(DATASETS):
                metric_dir = run_dir / dataset
                metric_dir.mkdir(parents=True)
                payload = {
                    f"{dataset}/math-verify": {"pass@1": index / 10.0}
                }
                (metric_dir / "metric.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            metrics = collect(run_dir)

        self.assertEqual(tuple(metrics), DATASETS)
        self.assertEqual(metrics["AIME24"], 0.0)
        self.assertEqual(metrics["GaoKao2023en"], 0.6)


if __name__ == "__main__":
    unittest.main()
