# Idea: Adaptive Minimal-Scaffold Routing

## Motivation

Scaf-GRPO restores the learning signal of all-zero-reward samples by adding
hierarchical hints. Its progressive search is effective, but it may require
multiple generations before finding a useful hint. Meanwhile, the earlier
reasoning papers show that more computation or more explicit reasoning is not
always beneficial.

## Core idea

For each problem, use the weakest intervention that is sufficient:

1. Let the model solve the problem without help.
2. If it fails, add a knowledge hint.
3. If it still fails, add a planning hint.
4. Finally, add a concrete solution-step hint.
5. Record the first successful level as the minimal effective scaffold.

The resulting labels can later train a lightweight router that predicts one of
`none`, `knowledge`, `planning`, or `solution`, replacing repeated scaffold
search with one routing decision.

## Hypothesis

Progressive scaffolding should improve accuracy over no-hint reasoning while
using less hint information than always giving the strongest hint. If minimal
effective levels are predictable from the question and the model's initial
response, a learned router can reduce rollout cost further.

## What this experiment tests

This first experiment does not train GRPO. It tests the prerequisite empirical
claim and creates router labels:

- Does any scaffold recover initially failed examples?
- Which hint level is minimally sufficient?
- How much extra generation cost does progressive search introduce?
- How does progressive accuracy compare with always using the strongest hint?

## Next step

After the proof of concept, replace the hand-written questions with a filtered
GSM8K/DART-Math subset, generate hints offline, and train a cost-sensitive
router. The router can then be inserted before Scaf-GRPO batch augmentation.

