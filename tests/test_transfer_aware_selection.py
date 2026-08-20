import json
from pathlib import Path

from generate_transfer_candidates_local import extract_json


def test_extract_json_from_fence():
    value = extract_json('```json\n{"candidates": {"knowledge": {}}}\n```')
    assert "candidates" in value


def test_selector_prefers_transfer_gain(tmp_path: Path, monkeypatch):
    from select_transfer_aware_candidates import run

    anchors = tmp_path / "anchors.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    probes = tmp_path / "probes.jsonl"
    output = tmp_path / "selected.jsonl"
    anchor = {
        "root_id": "q1", "id": "q1::knowledge", "dimension": "knowledge",
        "no_help_probability": 0.125, "random_plan_probability": 0.25,
    }
    planning = {"root_id": "q1", "id": "q1::planning", "dimension": "planning"}
    anchors.write_text(json.dumps(anchor) + "\n")
    candidates.write_text(json.dumps(anchor) + "\n" + json.dumps(planning) + "\n")
    probes.write_text(
        json.dumps({
            "root_id": "q1", "candidate_id": "q1::knowledge",
            "proxy_transfer_gain": 0.0, "post_update_probability": 0.0,
            "losses": [1.0], "baseline_probability": 0.0,
            "root_samples": 4, "probe_steps": 2, "learning_rate": 2e-4,
        }) + "\n" +
        json.dumps({
            "root_id": "q1", "candidate_id": "q1::planning",
            "proxy_transfer_gain": 0.5, "post_update_probability": 0.5,
            "losses": [1.2], "baseline_probability": 0.0,
            "root_samples": 4, "probe_steps": 2, "learning_rate": 2e-4,
        }) + "\n"
    )
    args = type("Args", (), {
        "anchors": anchors, "candidates": candidates,
        "transfer_results": probes, "output": output,
    })()
    run(args)
    selected = json.loads(output.read_text().strip())
    assert selected["id"] == "q1::planning"
    assert selected["selection_policy"] == "max_same_root_proxy_transfer_gain"
    assert selected["success_probability"] == 0.0
    assert selected["no_help_probability"] == 0.125
    assert selected["random_plan_probability"] == 0.25
    assert selected["transfer_selected_for_preconditioning"] is True
