from aggregate_rcst_probes import aggregate_all, select_candidates


def probe(root, candidate, seed, before, after):
    return {
        "root_id": root,
        "candidate_id": candidate,
        "dimension": candidate,
        "probe_seed": seed,
        "baseline_correct": before,
        "post_update_correct": after,
        "losses": [1.0],
    }


def candidate(root, name):
    return {
        "root_id": root,
        "id": name,
        "question": f"question {root}",
        "subproblem": name,
        "subproblem_answer": "answer",
        "dimension": name,
        "success_probability": 0.0,
        "no_help_probability": 0.0,
        "random_plan_probability": 0.0,
    }


def test_rcst_selects_positive_candidate_and_abstains_on_nonpositive_root():
    rows = [
        probe("r1", "a", 1, [False, False], [True, False]),
        probe("r1", "a", 2, [False, False], [True, True]),
        probe("r1", "b", 1, [False, False], [False, False]),
        probe("r1", "b", 2, [False, False], [False, False]),
        probe("r2", "c", 1, [True, False], [False, False]),
        probe("r2", "c", 2, [True, False], [True, False]),
        probe("r2", "d", 1, [False, False], [False, False]),
        probe("r2", "d", 2, [False, False], [False, False]),
    ]
    aggregates = aggregate_all(rows, confidence_z=0.0, min_replicates=2)
    candidates = {name: candidate(root, name) for root, name in [("r1", "a"), ("r1", "b"), ("r2", "c"), ("r2", "d")]}
    anchors = [candidates["b"], candidates["d"]]
    selected, audit, summary = select_candidates(
        anchors, candidates, aggregates, policy="mean_positive", min_score=0.0
    )
    assert selected[0]["id"] == "a"
    assert selected[0]["transfer_probe"]["proxy_transfer_gain"] == 0.75
    assert selected[1]["id"] == "d"
    assert selected[1]["rcst_abstained"] is True
    assert [row["accepted"] for row in audit] == [True, False]
    assert summary["accepted_roots"] == 1
