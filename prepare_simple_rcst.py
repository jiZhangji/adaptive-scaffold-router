#!/usr/bin/env python3
"""Prepare an ordered RCST stream: independent subproblems, then naked roots.

No model is loaded. A content-bound quality review is mandatory; this builder
does not claim to verify mathematical correctness or turn old data into clean data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

from rcst_source_join import question_key, source_index, unique_source

HINT_FIELDS = ("knowledge_components_parts", "planning_skeleton_parts", "solution_breakdown_parts")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_records(rows: list[dict], key: str) -> dict:
    result = {}
    for row in rows:
        identity = str(row[key]).strip()
        if row[key] is None or not identity or identity in result:
            raise ValueError(f"Missing or duplicate {key}")
        result[identity] = row
    return result


def answer_text(value) -> str:
    if value is None:
        raise ValueError("Missing reference answer")
    value = str(value).strip()
    if value.startswith("\\boxed{") and value.endswith("}"):
        value = value[7:-1].strip()
    if not value:
        raise ValueError("Empty reference answer")
    return value


def review_fingerprint(candidate: dict, learnability: dict | None) -> str:
    payload = {key: candidate.get(key) for key in (
        "root_id", "id", "question", "reference", "subproblem", "subproblem_answer",
        "transfer_probe", "rcst_abstained", "eligible_for_training", "trainable",
    )}
    payload["learnability"] = learnability
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def review_template(candidates: list[dict], learnability: dict) -> list[dict]:
    return [{
        "candidate_id": str(c["id"]),
        "payload_sha256": review_fingerprint(c, learnability.get(str(c["id"]))),
        "root_answer_verified": False,
        "subproblem_and_reward_verified": False,
        "probe_and_learnability_use_reviewed_data": False,
        "evidence": "",
    } for c in candidates]


def naked_row(source: dict, question: str, answer: str, root_id: str, candidate_id=None) -> dict:
    row = copy.deepcopy(source)
    prompt = row.get("prompt")
    prompt = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Source prompt must be a nonempty chat message list")
    users = [m for m in prompt if m.get("role") == "user"]
    if (len(users) != 1 or prompt[-1].get("role") != "user"
            or any(m.get("role") not in ("system", "user") for m in prompt)
            or question_key(users[0]["content"]) != question_key(source["question"])):
        raise ValueError("Source prompt must contain only system messages and the naked root question")
    row["prompt"] = [copy.deepcopy(m) for m in prompt if m["role"] == "system"]
    row["prompt"].append({"role": "user", "content": question})
    row["question"] = question
    row["reward_model"] = {"style": "rule", "ground_truth": answer}
    # Rebuild metadata so old hint/Bridge fields cannot reach the new dataset.
    row["extra_info"] = {
        "root_id": root_id, "candidate_id": candidate_id,
        "source_id": str(source.get("id", "")), "is_subproblem": candidate_id is not None,
    }
    for key in HINT_FIELDS:
        if key in row:
            row[key] = []
    for key in ("solution", "model_think", "model_answer", "parsed_output"):
        if key in row:
            row[key] = None
    return row


def build_rows(source_rows, candidates, learnability_rows, reviews, *, steps=50,
               mixed_steps=40, batch_size=32, sub_per_batch=4, seed=42):
    if not (0 < mixed_steps < steps and 0 < sub_per_batch < batch_size):
        raise ValueError("Require 0 < mixed_steps < steps and 0 < sub_per_batch < batch_size")
    unique_records(candidates, "root_id")
    unique_records(candidates, "id")
    learns = unique_records(learnability_rows, "candidate_id")
    checks = unique_records(reviews, "candidate_id")
    index, roots, subs, decisions, seen_questions = source_index(source_rows), [], [], [], set()
    for candidate in sorted(candidates, key=lambda c: str(c["root_id"])):
        cid, rid = str(candidate["id"]), str(candidate["root_id"])
        key = question_key(candidate["question"])
        if key in seen_questions:
            raise ValueError("The same root question occurs under multiple root IDs")
        seen_questions.add(key)
        source = unique_source(index, candidate["question"])
        reference = answer_text(candidate["reference"])
        if reference != answer_text(source["reward_model"]["ground_truth"]):
            raise ValueError(f"Root reference disagrees with source reward: {rid}")
        learn, review = learns.get(cid), checks.get(cid, {})
        if (review.get("payload_sha256") != review_fingerprint(candidate, learn)
                or review.get("root_answer_verified") is not True
                or not str(review.get("evidence", "")).strip()):
            raise ValueError(f"Missing, stale, or incomplete root quality review: {rid}")
        roots.append(naked_row(source, source["question"], reference, rid))
        probe = candidate.get("transfer_probe") or {}
        score = probe.get("gain_lcb")
        reason = "accepted"
        if (candidate.get("rcst_abstained") is True
                or candidate.get("eligible_for_training") is False
                or candidate.get("trainable") is False):
            reason = "upstream_abstention_or_quarantine"
        elif (review.get("subproblem_and_reward_verified") is not True
                or review.get("probe_and_learnability_use_reviewed_data") is not True):
            reason = "subproblem_or_measurements_not_reviewed"
        elif (not isinstance(score, (int, float)) or not math.isfinite(score)
                or score <= 0 or probe.get("replicates", 0) < 2):
            reason = "no_positive_replicated_transfer_score"
        elif not learn or learn.get("num_samples", 0) <= 0:
            reason = "learnability_unknown"
        else:
            total, correct, p_sub = learn["num_samples"], learn.get("correct_samples"), learn.get("p_sub")
            if (not isinstance(total, int) or not isinstance(correct, int)
                    or not 0 <= correct <= total or not isinstance(p_sub, (int, float))
                    or not math.isfinite(p_sub) or not math.isclose(p_sub, correct / total)):
                raise ValueError(f"Invalid learnability counts: {cid}")
            if learn.get("root_id", rid) != rid:
                raise ValueError(f"Learnability root mismatch: {cid}")
            if not 0 < correct < total:
                reason = "no_observed_reward_contrast"
        if reason == "accepted":
            question_key(candidate["subproblem"])
            subs.append(naked_row(source, candidate["subproblem"],
                                  answer_text(candidate["subproblem_answer"]), rid, cid))
        decisions.append({"root_id": rid, "candidate_id": cid, "reason": reason})
    if not roots:
        raise ValueError("No roots supplied")

    def cycle(rows, rng):
        while True:
            order = list(range(len(rows)))
            rng.shuffle(order)
            for i in order:
                yield rows[i]

    root_stream = cycle(roots, random.Random(seed))
    sub_stream = cycle(subs, random.Random(seed + 1)) if subs else None
    position_rng = random.Random(seed + 2)
    vanilla, method = [], []
    for step in range(steps):
        positions = set(position_rng.sample(range(batch_size), sub_per_batch)) if step < mixed_steps and subs else set()
        for position in range(batch_size):
            root = next(root_stream)
            chosen = next(sub_stream) if position in positions else root
            for output, sample in ((vanilla, root), (method, chosen)):
                sample = copy.deepcopy(sample)
                sample["id"] = len(output)
                sample["extra_info"]["index"] = len(output)
                output.append(sample)
    summary = {
        "method": "simple_rcst_no_hint_v1", "status": "implementation_only_no_performance_claim",
        "steps": steps, "mixed_steps": mixed_steps, "root_only_steps": steps - mixed_steps,
        "batch_size": batch_size, "sub_per_mixed_batch": sub_per_batch if subs else 0,
        "data_order_seed": seed, "root_count": len(roots), "accepted_subproblems": len(subs),
        "rows_per_arm": len(method), "subproblem_slots": mixed_steps * sub_per_batch if subs else 0,
        "shuffle": False, "filter_overlong_prompts": False, "bridge_enabled": False,
        "guided_root": False, "decisions": decisions,
        "budget_note": "Equal prompt slots and global steps; generated tokens and probe costs may differ.",
    }
    return vanilla, method, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--learnability", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review-template", type=Path)
    mode.add_argument("--reviews", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--mixed-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sub-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    candidates, learns = read_jsonl(args.candidates), read_jsonl(args.learnability)
    if args.review_template:
        rows = review_template(candidates, unique_records(learns, "candidate_id"))
        with args.review_template.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("Review template created with all checks FALSE; it does not authorize training.")
        return
    if args.output_dir is None:
        parser.error("--output-dir is required with --reviews")
    import pandas as pd

    vanilla, method, summary = build_rows(
        pd.read_parquet(args.source_data).to_dict(orient="records"), candidates, learns,
        read_jsonl(args.reviews), steps=args.steps, mixed_steps=args.mixed_steps,
        batch_size=args.batch_size, sub_per_batch=args.sub_per_batch, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, rows in (("vanilla_train.parquet", vanilla), ("method_train.parquet", method)):
        path = args.output_dir / name
        pd.DataFrame(rows).to_parquet(path, index=False)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary["file_sha256"] = files
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(method)} slots per arm; accepted {summary['accepted_subproblems']} subproblems.")


if __name__ == "__main__":
    main()
