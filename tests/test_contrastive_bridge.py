from __future__ import annotations

from build_contrastive_bridge_data import select_same_root_controls


def test_selects_lowest_lcb_unselected_candidate_as_control() -> None:
    candidates = [
        {"id": "r::calculation", "root_id": "r"},
        {"id": "r::knowledge", "root_id": "r"},
        {"id": "r::planning", "root_id": "r"},
    ]
    aggregates = [
        {"candidate_id": "r::calculation", "gain_lcb": 0.2},
        {"candidate_id": "r::knowledge", "gain_lcb": -0.4},
        {"candidate_id": "r::planning", "gain_lcb": -0.1},
    ]
    positives = {"r": candidates[0]}
    controls = select_same_root_controls(candidates, aggregates, positives)
    assert controls["r"]["id"] == "r::knowledge"
