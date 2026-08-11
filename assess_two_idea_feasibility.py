from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _accuracy(summary: dict[str, Any], variant: str) -> float:
    return float(summary.get("variants", {}).get(variant, {}).get("accuracy", 0.0))


def assess(
    capability: dict[str, Any],
    subproblem: dict[str, Any],
    metaask: dict[str, Any],
    metaask_diagnostics: dict[str, Any],
    controlled: dict[str, Any],
) -> dict[str, Any]:
    rescue_rate = float(capability.get("rescue_rate", 0.0))
    token_saving = float(
        capability.get("cost_simulation", {}).get(
            "public_scaf_oracle_token_saving_fraction", 0.0
        )
    )
    relevance_gain = float(
        subproblem.get("causal_checks", {}).get("relevant_gain_over_random", 0.0)
    )
    rescue_advantage = float(
        subproblem.get("causal_checks", {}).get("rescue_advantage_over_random", 0.0)
    )
    candidate_rate = float(subproblem.get("valid_candidate_rate", 0.0))
    subproblem_accuracy = float(subproblem.get("subproblem_solve_accuracy", 0.0))

    scaffold_gate = rescue_rate >= 0.15
    cost_gate = token_saving >= 0.30
    relevance_gate = relevance_gain >= 0.05 and rescue_advantage >= 0.0
    construction_gate = candidate_rate >= 0.70
    if relevance_gate and scaffold_gate and construction_gate:
        scheme1_status = "promising"
    elif (scaffold_gate and cost_gate) or relevance_gate:
        scheme1_status = "mixed_evidence"
    else:
        scheme1_status = "not_supported_yet"

    no_help = _accuracy(metaask, "no_help")
    random_bit = _accuracy(metaask, "random_bit")
    self_asked = _accuracy(metaask, "self_asked_verification")
    minimal_assistance = max(
        _accuracy(metaask, "knowledge_min"),
        _accuracy(metaask, "planning_min"),
        _accuracy(metaask, "solution_min"),
    )
    controlled_retry = _accuracy(controlled, "answer_verification_retry")
    self_ask_gain = self_asked - max(no_help, random_bit)
    controlled_gain = controlled_retry - no_help
    assistance_gain = minimal_assistance - no_help
    paired = metaask_diagnostics.get("paired_comparison", {})
    rescues = int(paired.get("metaask_rescues", 0))
    harms = int(paired.get("metaask_harms", 0))
    self_ask_gate = self_ask_gain >= 0.05 and rescues > harms
    feedback_gate = controlled_gain >= 0.05
    information_gate = assistance_gain >= 0.10
    if self_ask_gate and feedback_gate:
        scheme2_status = "promising"
    elif feedback_gate or information_gate:
        scheme2_status = "mixed_evidence"
    else:
        scheme2_status = "not_supported_yet"

    examples = min(
        int(capability.get("selected_examples", capability.get("num_examples", 0))),
        int(subproblem.get("selected_questions", 0)),
        int(metaask.get("selected_questions", 0)),
    )
    warnings = []
    if examples < 32:
        warnings.append(
            f"Only {examples} shared-scale examples were tested; use at least 32 before a go/no-go decision."
        )
    if float(capability.get("arms", {}).get("none", {}).get("generation_limit_rate", 0.0)) > 0.20:
        warnings.append("More than 20% of no-hint generations hit the token limit.")
    if metaask_diagnostics.get("fallback_question_count", 0):
        warnings.append("Some MetaAsk queries failed strict parsing and used the fallback question.")
    if subproblem.get("warning"):
        warnings.append(str(subproblem["warning"]))

    return {
        "scope": "mechanism feasibility only; this is not an RL training result",
        "scheme1_capability_matched_subproblem_curriculum": {
            "status": scheme1_status,
            "metrics": {
                "scaffold_rescue_rate": rescue_rate,
                "hindsight_token_saving_vs_public_scaf": token_saving,
                "valid_subproblem_candidate_rate": candidate_rate,
                "subproblem_solve_accuracy": subproblem_accuracy,
                "relevant_gain_over_random_subproblem": relevance_gain,
                "rescue_advantage_over_random_subproblem": rescue_advantage,
            },
            "gates": {
                "scaffold_rescue_at_least_0.15": scaffold_gate,
                "token_saving_at_least_0.30": cost_gate,
                "causal_relevance_gain_at_least_0.05": relevance_gate,
                "valid_candidate_rate_at_least_0.70": construction_gate,
            },
        },
        "scheme2_metaask": {
            "status": scheme2_status,
            "metrics": {
                "no_help_accuracy": no_help,
                "random_bit_accuracy": random_bit,
                "self_asked_accuracy": self_asked,
                "self_asked_gain_over_best_control": self_ask_gain,
                "controlled_answer_verification_gain": controlled_gain,
                "minimal_assistance_gain": assistance_gain,
                "paired_metaask_rescues": rescues,
                "paired_metaask_harms": harms,
            },
            "gates": {
                "self_asked_gain_and_more_rescues_than_harms": self_ask_gate,
                "controlled_feedback_gain_at_least_0.05": feedback_gate,
                "minimal_information_gain_at_least_0.10": information_gate,
            },
        },
        "warnings": warnings,
        "recommended_next_step": (
            "Proceed to a small RL smoke only for schemes marked promising. For mixed evidence, "
            "increase the probe sample or improve the failed mechanism before RL."
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    first = report["scheme1_capability_matched_subproblem_curriculum"]
    second = report["scheme2_metaask"]
    lines = [
        "# Two-Idea Feasibility Screen",
        "",
        f"- Scheme 1 status: **{first['status']}**",
        f"- Scheme 2 status: **{second['status']}**",
        "",
        "## Scheme 1 metrics",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in first["metrics"].items())
    lines.extend(["", "## Scheme 2 metrics", ""])
    lines.extend(f"- {key}: {value}" for key, value in second["metrics"].items())
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["scope"],
            "",
            report["recommended_next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    report = assess(
        capability=_read(root / "capability_frontier" / "summary.json"),
        subproblem=_read(root / "subproblem_relevance" / "summary.json"),
        metaask=_read(root / "metaask" / "summary.json"),
        metaask_diagnostics=_read(root / "metaask" / "diagnostics.json"),
        controlled=_read(root / "metaask_controlled" / "summary.json"),
    )
    json_path = root / "feasibility_report.json"
    markdown_path = root / "feasibility_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report_json={json_path}")
    print(f"report_markdown={markdown_path}")


if __name__ == "__main__":
    main()
