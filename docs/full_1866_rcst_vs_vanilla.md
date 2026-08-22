# Full 1,866-root RCST LCB-Positive vs Vanilla GRPO

This is the largest currently candidate-complete comparison. It uses all 1,866
zero-reward roots for which the DeepSeek artifact contains exactly three
standalone candidates (knowledge, planning, and calculation).

## Fixed protocol

- Base model: Qwen2.5-Math-1.5B
- Training roots per arm: 1,866
- RCST candidates: 5,598
- RCST probes: two independent seeds, eight paired no-hint root rollouts each
- Batch size: 32
- GRPO rollouts: 8
- Fixed optimizer updates per arm: 440
- RCST boundaries: 88 (precondition), 308 (fading), 440 (root-only)
- Evaluation: unified greedy pass@1 on the same seven benchmarks

RCST accepts a candidate only when its replicated transfer-gain lower confidence
bound is positive. Otherwise the root falls back to naked-root training.

## One-command launch

Run on Server A. It coordinates Server B over the existing LAN SSH key:

```bash
cd /home/powerleader/project/adaptive-scaffold-router
nohup bash scripts/run_full_1866_rcst_vs_vanilla_two_node_a6000.sh \
  > outputs/full_1866_suite_launcher.log 2>&1 < /dev/null &
echo $!
```

The launcher is resumable. Running the same command with an explicit existing
`RUN_ROOT` reuses completed probe rows, checkpoints, and evaluation metrics:

```bash
RUN_ROOT=/home/powerleader/project/adaptive-scaffold-router/outputs/full_1866_rcst_vs_vanilla_YYYYMMDD_HHMMSS \
  bash scripts/run_full_1866_rcst_vs_vanilla_two_node_a6000.sh
```

The current run path is written to:

```text
outputs/latest_full_1866_rcst_vs_vanilla.txt
```

The final table is written to:

```text
<RUN_ROOT>/final_comparison/comparison.md
```
