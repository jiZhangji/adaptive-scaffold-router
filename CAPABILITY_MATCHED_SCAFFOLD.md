# Capability-Matched Scaffold Curriculum

## Motivation

Guidance cannot create a useful RL update merely by making a hard answer more
likely. The learner must already possess enough prerequisite skill to execute
the guidance, the guided success rate must remain inside a reward-diverse
learning region, and the guided trajectory must transfer to the original
no-guidance policy.

This implementation separates those requirements into three mechanisms:

1. **Prerequisite curriculum**: train independently verifiable subproblems
   while the original root remains below the learnable region.
2. **Capability-matched scaffolding**: select the cheapest scaffold whose
   empirical success rate falls in a configurable learning band.
3. **Scaffold fading and off-context correction**: progressively hide scaffold
   parts and weight guided trajectories by their likelihood under the original
   no-scaffold context.

For binary rewards and a group of size `G`, a group contains a non-zero
relative learning signal with probability

```text
P(informative | p, G) = 1 - p^G - (1 - p)^G.
```

The implementation reports this quantity, but does not treat a fixed empirical
band such as `[0.25, 0.60]` as a theorem. The band remains an experiment
parameter.

## Controller phases

`capability_scaffold.py` assigns every root to one phase:

- `subproblem_curriculum`: train root-derived, self-contained subproblems that
  already yield mixed rewards;
- `decompose_or_defer`: recursively decompose unresolved prerequisites or wait
  for later checkpoints;
- `guided_root`: prerequisites are ready and a low-cost scaffold makes the root
  trainable;
- `unguided_root`: the original root itself yields mixed rewards;
- `unguided_probe`: the root appears easy but still requires repeated no-hint
  confirmation;
- `graduated`: repeated no-hint evaluations meet the mastery threshold.

The controller never gives gradient credit to an all-zero root merely because
it was mixed into a batch. Such roots are probes until they become informative.

## Input contract

The input is JSONL with one row per root. See
`data/capability_curriculum_example.jsonl`. Each row contains:

- `root_rewards`: current-checkpoint no-hint rollout rewards;
- `subproblems`: self-contained question, answer, and rollout rewards;
- `scaffolds`: scaffold text, token cost, strength, and rollout rewards;
- `consecutive_unguided_mastery`: successful no-hint evaluation rounds.

Subproblems and scaffold candidates can be constructed from the official
Scaf-GRPO parquet columns. Their answers must be checked with the same final
answer verifier used by RLVR. Generation and validation are data-construction
steps; they are not trusted as reward labels by themselves.

## Compile a curriculum manifest

```bash
python capability_scaffold.py \
  --input data/capability_curriculum_example.jsonl \
  --output-dir outputs/capability_curriculum \
  --group-size 8 \
  --band-low 0.25 \
  --band-high 0.60 \
  --max-hint-tokens 96
```

The command writes `curriculum.jsonl` and `summary.json`. These artifacts can
be joined to a verl batch by stable root ID.

## Scaf-GRPO integration

`scaf_curriculum_adapter.py` provides the tensor operations required by the
trainer:

1. Generate a response under the selected scaffold and retain its behavior
   token log probabilities.
2. Re-evaluate the same response under the original question to obtain target
   token log probabilities.
3. Compute a clipped sequence weight with `sequence_importance_weights`.
4. Apply it to GRPO advantages with `apply_sequence_weights`.
5. Use `build_regeneration_requests` to construct one selected, possibly faded
   hint per all-zero root. Non-guided phases return an empty hint and can be
   routed to the normal root/subproblem scheduler.

The target-context log probability must be recomputed; reusing the guided
prompt log probability would not correct the off-context distribution shift.
The optional patches under `scaf_integration/` implement this path in the
official trainer: the hinted response is placed under the original prompt,
the actor recomputes its target log probability, and a clipped sequence ratio
weights the GRPO advantage. Length-normalized ratios are the stable default and
must be ablated against no correction and other clipping ranges.

Set `CURRICULUM_MANIFEST` when invoking `scripts/run_scaf_smoke_2gpu.sh` to
enable the controller. `CURRICULUM_OFF_CONTEXT=true` also switches on original-
prompt replacement and clipped importance weighting. Without a manifest, the
script and trainer retain the upstream exhaustive Scaf-GRPO path.

## Minimum experiment

Run equal-compute variants on the same root split:

1. vanilla GRPO;
2. Scaf-GRPO;
3. subproblem curriculum only;
4. calibrated scaffold without fading/correction;
5. full method;
6. full method without prerequisite gating;
7. full method without importance correction.

Report final no-hint pass@1/pass@k, informative-group rate, all-zero-root
activation rate, total generated tokens, GPU hours, scaffold reliance, root
graduation rate, and importance-weight effective sample size.
