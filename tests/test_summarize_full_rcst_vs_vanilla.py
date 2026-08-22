import json
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_full_rcst_vs_vanilla import DATASETS, run


def write_summary(root, value):
    root.mkdir()
    payload = {
        "scores_percent": {dataset: value for dataset in DATASETS},
        "macro_average_percent": value,
    }
    (root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_writes_per_dataset_and_macro_delta(tmp_path):
    vanilla = tmp_path / "vanilla"
    rcst = tmp_path / "rcst"
    output = tmp_path / "output"
    write_summary(vanilla, 30.0)
    write_summary(rcst, 32.5)

    run(
        Namespace(
            vanilla_eval=vanilla,
            rcst_eval=rcst,
            output_dir=output,
            training_steps=440,
            training_roots=1866,
        )
    )

    payload = json.loads((output / "comparison.json").read_text())
    assert payload["macro"]["delta_pp"] == 2.5
    assert len(payload["rows"]) == 7
    assert "+2.5 pp" in (output / "comparison.md").read_text()
