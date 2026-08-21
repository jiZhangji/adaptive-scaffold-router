import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from prepare_td_grokking_candidates import prompt_question, run


def test_prompt_question_prefers_last_user_message() -> None:
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert prompt_question(prompt) == "question"


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    anchors = tmp_path / "anchors.jsonl"
    roots = tmp_path / "root_only.parquet"
    subs = tmp_path / "sub_only.parquet"
    anchor = {
        "id": "r1::planning",
        "root_id": "r1",
        "question": "What is 2+2?",
        "reference": "4",
        "minimal_plan": "add the values",
        "success_probability": 0.25,
        "no_help_probability": 0.0,
        "random_plan_probability": 0.0,
    }
    anchors.write_text(json.dumps(anchor) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "problem_id": "td1",
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "extra_info": {"root_problem_id": "td1", "train_variant": "origin_only"},
                "reward_model": {"ground_truth": "4"},
            }
        ]
    ).to_parquet(roots, index=False)
    pd.DataFrame(
        [
            {
                "problem_id": f"td1_subproblem_{index}",
                "prompt": [{"role": "user", "content": question}],
                "extra_info": {
                    "root_problem_id": "td1",
                    "subproblem_id": f"subproblem_{index}",
                    "train_variant": "sub_only",
                },
                "reward_model": {"ground_truth": answer},
            }
            for index, (question, answer) in enumerate(
                (("What is 1+1?", "2"), ("What is 3+1?", "4")), start=1
            )
        ]
    ).to_parquet(subs, index=False)
    return anchors, roots, subs


def test_aligns_td_children_to_existing_root(tmp_path: Path) -> None:
    anchors, roots, subs = write_fixture(tmp_path)
    output = tmp_path / "candidates.jsonl"
    run(
        Namespace(
            anchors=anchors,
            root_only=roots,
            sub_only=subs,
            output=output,
            require_matched_roots=1,
            min_candidates_per_root=2,
            exclude_same_answer=False,
        )
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["root_id"] for row in rows} == {"r1"}
    assert all(row["candidate_origin"] == "td_grokking_released_subproblem" for row in rows)


def test_refuses_incomplete_root_coverage(tmp_path: Path) -> None:
    anchors, roots, subs = write_fixture(tmp_path)
    with pytest.raises(ValueError, match="covers only 1 of 1"):
        run(
            Namespace(
                anchors=anchors,
                root_only=roots,
                sub_only=subs,
                output=tmp_path / "out.jsonl",
                require_matched_roots=2,
                min_candidates_per_root=2,
                exclude_same_answer=False,
            )
        )
