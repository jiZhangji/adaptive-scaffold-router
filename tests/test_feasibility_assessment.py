import unittest

from assess_two_idea_feasibility import assess


class FeasibilityAssessmentTests(unittest.TestCase):
    def test_marks_strong_scheme1_and_weak_self_ask_separately(self):
        capability = {
            "selected_examples": 32,
            "rescue_rate": 0.25,
            "cost_simulation": {"public_scaf_oracle_token_saving_fraction": 0.5},
            "arms": {"none": {"generation_limit_rate": 0.0}},
        }
        subproblem = {
            "selected_questions": 32,
            "valid_candidate_rate": 0.8,
            "subproblem_solve_accuracy": 0.5,
            "causal_checks": {
                "relevant_gain_over_random": 0.1,
                "rescue_advantage_over_random": 0.1,
            },
        }
        metaask = {
            "selected_questions": 32,
            "variants": {
                "no_help": {"accuracy": 0.3},
                "random_bit": {"accuracy": 0.3},
                "self_asked_verification": {"accuracy": 0.3},
                "knowledge_min": {"accuracy": 0.5},
                "planning_min": {"accuracy": 0.4},
                "solution_min": {"accuracy": 0.5},
            },
        }
        diagnostics = {"paired_comparison": {"metaask_rescues": 1, "metaask_harms": 1}}
        controlled = {"variants": {"answer_verification_retry": {"accuracy": 0.3}}}
        report = assess(capability, subproblem, metaask, diagnostics, controlled)
        self.assertEqual(
            report["scheme1_capability_matched_subproblem_curriculum"]["status"],
            "promising",
        )
        self.assertEqual(report["scheme2_metaask"]["status"], "mixed_evidence")


if __name__ == "__main__":
    unittest.main()
