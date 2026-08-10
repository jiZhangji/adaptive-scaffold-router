# DSFL Experimental Plan

## Research question

Can a capability-aware policy choose a low-cost semantic scaffold without
enumerating every hint, while preserving the reward variance needed by RLVR
and improving the final no-hint reasoning ability?

The contribution is not merely dynamic hinting. Scaf-GRPO already searches
hierarchical hints, and SEELE already adapts solution-prefix length to current
rollout accuracy. DSFL must demonstrate a non-stationary, cost-aware frontier
over both scaffold type and strength, followed by explicit scaffold retirement.

## Phase A: frontier diagnostic

Use the official Scaf-GRPO data and its knowledge, planning, and solution
components. Evaluate each component at multiple cumulative strengths with
several stochastic rollouts. Run the identical selected questions on two model
capability levels.

Primary measurements:

1. Per-item success probability for every scaffold arm.
2. Prompt tokens, output tokens, calls, and latency.
3. Lowest-cost arm reaching target success probability.
4. Lowest-cost arm in the useful rollout-accuracy band.
5. Cost-aware utility `p(1-p) - lambda*C - mu*D`.
6. Public Scaf-GRPO search cost versus a minimum-cost oracle.

Go/no-go evidence for continuing to online training:

- The selected frontier contains at least three materially used arms.
- The selected arm changes on at least 30% of common questions across model
  capability levels.
- The stronger model uses fewer hint tokens on at least 25% of questions.
- The minimum-cost oracle reduces total generation tokens by at least 25%
  relative to the public Scaf-GRPO enumeration path at comparable success.
- Knowledge or planning scaffolds are selected often enough to justify a
  semantic lattice rather than solution-prefix length alone.

These are decision thresholds for the initial probe, not paper claims.

## Phase B: uncertainty-aware frontier router

Train on disjoint questions. The router predicts success probability for each
scaffold arm from the question representation, autonomous-rollout uncertainty,
model checkpoint/stage, and scaffold cost. Calibrate the probabilities on a
held-out split.

At inference time:

- choose the lowest-cost arm whose predicted accuracy lies in a useful
  learning band;
- explore only adjacent arms when the confidence interval crosses a decision
  boundary;
- update the router with the newly observed rollout outcomes.

Baselines:

- no hint;
- strongest solution hint;
- Scaf-GRPO progressive enumeration;
- solution-prefix-only adaptation in the style of SEELE;
- static four-class router;
- uncertainty-aware DSFL router.

Router metrics include success, total tokens, calls, latency, calibration
error, constraint violations, and regret against the full-information oracle.

## Phase C: RLVR and scaffold retirement

Insert the router before hinted regeneration in the official Scaf-GRPO/verl
pipeline. Begin with Qwen2.5-Math-1.5B on two GPUs and a controlled subset.
Every previously rescued example is periodically retried with a weaker
scaffold. It graduates only after repeated no-hint success.

Final measurements:

- held-out no-hint pass@1 and pass@k;
- training reward and zero-advantage sample rate;
- total rollout, prompt-token, and GPU-hour cost;
- hinted versus no-hint accuracy gap;
- scaffold graduation rate and regression rate;
- results at matched compute and matched wall-clock budgets.

Required ablations remove online updates, uncertainty exploration, cost terms,
semantic hint types, and retirement independently.

## Current execution

- Hardware: two RTX 3090 GPUs on the selected remote server.
- Dataset: official `Qwen2.5-Math-1.5B.parquet`, 12,880 examples.
- Initial model: Qwen2.5-Math-1.5B.
- Initial probe: 24 source-zero-accuracy examples, 10 scaffold arms, four
  rollouts per arm, for 960 generations.
- A Qwen2.5-Math-7B run on the same examples is used as the second capability
  level after model download completes.
