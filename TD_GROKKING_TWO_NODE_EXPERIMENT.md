# TD-Grokking candidate experiment on two A6000 nodes

## Current artifact status

The public TD-Grokking repository exposes the code, README, and a manifest for
the DeepScaleR-hard views. It does not currently contain the three parquet
files named by that manifest (`root_only.parquet`, `sub_only.parquet`, and
`mixed.parquet`). The fetch script therefore exits with status 42 instead of
silently substituting locally generated candidates.

If a verified archive becomes available, set `TD_GROKKING_ARCHIVE_URL` or copy
the three parquet files into
`outputs/td_grokking_artifact/code/data/DeepScaleR-hard/`.

## Fair comparison protocol

1. Match TD roots to the same 212 zero-reward roots used by the existing
   experiments after normalized-question matching.
2. Require at least two released TD children for every matched root.
3. Probe all candidates and select by same-root no-hint transfer gain.
4. Keep the previous 50 optimizer-step budget, global batch 32, eight
   rollouts, learning rate, model, and seven-dataset greedy pass@1 evaluation.
5. Use two nodes only as one data-parallel training job; do not double the
   global batch or the training-step budget.

The strict coverage requirement is intentional. TD-Grokking selected its hard
roots for a different student checkpoint, so a partial overlap would not be a
fair direct replacement for the current 212-root candidate pool.

## Overnight variants

- `all_selected`: train the highest-scoring TD child for every root.
- `positive`: use the selected child only when estimated transfer gain is
  positive.
- `conservative`: additionally require transfer gain at least 0.25 and
  post-update root success probability at least 0.5.

The main question is whether higher-quality children improve the existing
Positive-Gated result (35.2 macro) and approach or exceed the Student-Aware
Step-50 result (35.5 macro), without sacrificing AMC23 or AIME24.

## Commands

Validate or fetch the artifact:

```bash
cd /home/powerleader/project/adaptive-scaffold-router
bash scripts/fetch_td_grokking_artifact.sh
```

Start the guarded overnight queue after the parquet files are present:

```bash
cd /home/powerleader/project/adaptive-scaffold-router
nohup bash scripts/run_td_grokking_overnight_a6000.sh \
  > outputs/td_grokking_overnight_launcher.log 2>&1 < /dev/null &
```

Validate two-node distributed training without a full evaluation:

```bash
cd /home/powerleader/project/adaptive-scaffold-router
bash scripts/run_two_node_distributed_smoke_a6000.sh
```
