import json
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_full_rcst_inputs import run


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_prepares_complete_three_candidate_pool_and_safe_fallbacks(tmp_path):
    import pandas as pd

    source = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {"question": "Question one?", "answer": "1"},
            {"question": "Question two?", "answer": "2"},
            {"question": "Unused question?", "answer": "3"},
        ]
    ).to_parquet(source, index=False)
    raw = tmp_path / "candidates.jsonl"
    rows = []
    for root_id, question in (("r1", "Question one?"), ("r2", "Question two?")):
        for dimension in ("knowledge", "planning", "calculation"):
            rows.append(
                {
                    "id": f"{root_id}:{dimension}",
                    "root_id": root_id,
                    "dimension": dimension,
                    "question": question,
                    "reference": "1",
                    "subproblem": f"{dimension} subproblem",
                    "subproblem_answer": "1",
                    "source_step": f"Use a {dimension} fact before solving the root.",
                }
            )
    write_jsonl(raw, rows)
    output = tmp_path / "prepared"

    run(
        Namespace(
            source_data=source,
            raw_candidates=raw,
            output_dir=output,
            expected_roots=2,
        )
    )

    candidates = [json.loads(line) for line in (output / "candidate_sets.jsonl").read_text().splitlines()]
    anchors = [json.loads(line) for line in (output / "fallback_anchors.jsonl").read_text().splitlines()]
    roots = pd.read_parquet(output / "vanilla_root_train.parquet")
    summary = json.loads((output / "summary.json").read_text())
    assert len(candidates) == 6
    assert len(anchors) == 2
    assert len(roots) == 2
    assert all(anchor["selection_policy"] == "synthetic_root_only_fallback" for anchor in anchors)
    assert all(anchor["success_probability"] == 0.0 for anchor in anchors)
    assert summary["complete_roots"] == 2
    assert summary["candidate_rows"] == 6
