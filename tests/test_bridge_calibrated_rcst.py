from __future__ import annotations

import numpy as np

from bridge_prompt_utils import BRIDGE_PROMPT_VERSION, build_bridge_messages
from fit_bridge_calibrated_rcst import assemble_rows, select_crossfit


def candidate(root: str, dimension: str) -> dict:
    return {
        "id": f"{root}::{dimension}",
        "root_id": root,
        "dimension": dimension,
        "question": f"question {root}",
        "subproblem": f"subproblem {dimension}",
        "subproblem_answer": "1",
    }


def delta(root: str, dimension: str, value: float) -> dict:
    return {
        "root_id": root,
        "candidate_id": f"{root}::{dimension}",
        "dimension": dimension,
        "prompt_version": BRIDGE_PROMPT_VERSION,
        "delta_mean": value,
        "delta_std": 0.1,
        "delta_lcb": value - 0.025,
        "delta_median": value,
        "delta_positive_rate": float(value > 0),
        "root_logprob_mean": -1.5,
        "root_baseline_accuracy": 0.0,
        "response_length_mean": 100.0,
        "response_length_std": 10.0,
        "subproblem_answer_logprob": -0.5,
        "subproblem_answer_tokens": 2,
        "subproblem_tokens": 12,
    }


def aggregate(root: str, dimension: str, gain: float) -> dict:
    return {
        "root_id": root,
        "candidate_id": f"{root}::{dimension}",
        "dimension": dimension,
        "mean_transfer_gain": gain,
        "gain_lcb": gain - 0.05,
    }


def test_bridge_messages_preserve_system_and_original() -> None:
    original = [
        {"role": "system", "content": "system instruction"},
        {"role": "user", "content": "root question"},
    ]
    bridged = build_bridge_messages(original, "small question", "42")
    assert original[1]["content"] == "root question"
    assert bridged[0] == original[0]
    assert bridged[1]["content"].startswith("root question\n\nVerified prerequisite")
    assert "small question" in bridged[1]["content"]
    assert "Prerequisite answer: 42" in bridged[1]["content"]


def test_assemble_and_select_cover_every_root_without_label_leakage() -> None:
    dimensions = ("calculation", "knowledge", "planning")
    candidates = [candidate(root, dim) for root in ("r1", "r2") for dim in dimensions]
    aggregates = [
        aggregate(root, dim, (index - 1) * 0.1)
        for root in ("r1", "r2")
        for index, dim in enumerate(dimensions)
    ]
    deltas = [
        delta(root, dim, (index - 1) * 0.05)
        for root in ("r1", "r2")
        for index, dim in enumerate(dimensions)
    ]
    rows = assemble_rows(candidates, aggregates, deltas)
    predictions = np.asarray([-0.1, 0.0, 0.2, -0.1, -0.05, -0.02])
    uncertainty = np.zeros(6)
    folds = np.asarray([0, 0, 0, 1, 1, 1])
    selected, audit, summary = select_crossfit(
        rows,
        predictions,
        uncertainty,
        folds,
        confidence_z=1.0,
        uncertainty_floor=0.01,
        min_score=0.0,
    )
    assert len(selected) == len(audit) == 2
    assert summary["accepted_roots"] == 1
    assert summary["abstained_roots"] == 1
    assert selected[0]["id"] == "r1::planning"
    assert selected[0]["rcst_abstained"] is False
    assert selected[1]["rcst_abstained"] is True
    assert "diagnostic_actual_gain" not in selected[0]
