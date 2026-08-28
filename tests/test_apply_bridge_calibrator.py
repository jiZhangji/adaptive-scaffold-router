import numpy as np

from apply_bridge_calibrator import select_frozen


def candidate(root: str, index: int) -> dict:
    return {
        "root_id": root,
        "candidate_id": f"{root}-c{index}",
        "dimension": ("calculation", "knowledge", "planning")[index],
        "candidate": {
            "id": f"{root}-c{index}",
            "root_id": root,
            "dimension": ("calculation", "knowledge", "planning")[index],
        },
    }


def test_crossfit_roots_are_preserved_and_unseen_roots_use_frozen_lcb():
    rows = [candidate("known", i) for i in range(3)] + [candidate("new", i) for i in range(3)]
    mean = np.asarray([0.0, 0.9, 0.1, -0.1, 0.2, 0.4])
    std = np.zeros(6)
    crossfit = [
        {
            "id": "known-c0",
            "root_id": "known",
            "dimension": "calculation",
            "rcst_abstained": True,
            "bridge_calibration": {"predicted_gain_lcb": -0.1},
        }
    ]
    selected, audit, summary = select_frozen(
        rows, mean, std, crossfit, confidence_z=1.0, uncertainty_floor=0.01, min_score=0.0
    )
    by_root = {row["root_id"]: row for row in selected}
    assert by_root["known"]["id"] == "known-c0"
    assert by_root["known"]["rcst_abstained"] is True
    assert by_root["new"]["id"] == "new-c2"
    assert by_root["new"]["rcst_abstained"] is False
    assert len(audit) == 2
    assert summary["selection_sources"] == {
        "root_grouped_crossfit_212": 1,
        "frozen_212_bootstrap_ensemble": 1,
    }
    assert summary["uses_full_pool_transfer_labels"] is False
