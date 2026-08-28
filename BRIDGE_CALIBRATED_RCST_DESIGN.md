# Bridge-Calibrated RCST

## Motivation

The old bridge score directly rewarded responses whose likelihood increased
after adding a subproblem. The 212-root result shows that this contextual
influence is not equivalent to post-update no-hint transfer. In addition, the
old bridge prompt dropped the root system message, so its delta also measured a
chat-template change.

## Definition

For the same root-only response `o`, score two prompts that differ only by the
added verified subproblem:

```text
delta(q, s, o) = mean log p(o | system, q + s) - mean log p(o | system, q)
```

The system message, original user question, response, tokenizer and response
normalization are identical in both terms.

## Calibration protocol

1. Reuse the two historical RCST probes for 212 roots and 636 candidates.
2. Compute 16 corrected delta samples per candidate from the saved root-only
   responses, plus learnability and root-difficulty features.
3. Fit a root-grouped five-fold ridge ensemble to predict the replicated RCST
   mean transfer gain.
4. For every 212-root training choice, train the calibrator on the other folds
   only. This prevents a root's own transfer label from selecting its candidate.
5. Select the candidate with the largest predicted lower confidence bound and
   abstain when that bound is not positive.
6. Feed the selected file into the original RCST curriculum. Delta is not used
   as a GRPO advantage or correctness reward.

The model fitted on all 212 roots is saved only for application to unseen roots;
it must not be used to report cross-fitted 212-root selection performance.

## Main files

- `bridge_prompt_utils.py`: system-preserving prompt construction.
- `score_bridge_delta_features.py`: length-normalized offline delta scoring.
- `fit_bridge_calibrated_rcst.py`: grouped cross-fitting, uncertainty and abstention.
- `scripts/run_bridge_calibrated_rcst_212_f.sh`: two-GPU scoring and calibration launcher.
- `tests/test_bridge_calibrated_rcst.py`: prompt and leakage-safety tests.

## Runtime outputs on F

```text
outputs/bridge_calibrated_rcst_212/
  assets/
  delta_system_preserving_v2/
  calibration_crossfit_v1/
  logs/
```

Training should start only after inspecting `calibration_crossfit_v1/diagnostics.json`.
