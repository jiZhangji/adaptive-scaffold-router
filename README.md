# Adaptive Scaffold Experiment

## Current focus: verifiable subproblem GRPO

The current research target is the training-time subproblem idea, not the
older dynamic-hint router.  The fair first comparison is:

1. Vanilla GRPO on a matched set of hard roots.
2. Mixed GRPO on the same roots plus their calibrated, self-contained,
   verifiable subproblems at a 1:1 ratio.

The decomposition teacher is used offline only.  The policy model estimates
each candidate's success rate `q`; only candidates in the configurable
learning band are retained.  A relevant candidate must also beat a
length-matched random-subproblem control before RL is launched.

Download the small instruction-tuned decomposition teacher on the networked
instance:

```bash
ENV_NAME=scaf-grpo bash scripts/download_subproblem_teacher.sh
```

Then run the feasibility gate and the two sequential one-step H100 training
checks on the offline instance:

```bash
conda activate scaf-grpo
ENV_NAME=scaf-grpo LIMIT=64 SAMPLES=4 \
  bash scripts/run_vanilla_then_subproblem_smoke_2h100.sh
```

This smoke run uses zero DataLoader subprocesses to avoid the CPU-memory kill
seen in the earlier baseline logs.  It is a mechanism check, not a final paper
result.  Scale both arms to identical steps/tokens only after the feasibility
gate passes.

On a network-isolated two-H100/H200 instance that shares storage with the
download instance, the following command waits for both 1.5B policy and teacher
weights, performs an offline safetensors/tokenizer check, and starts training
automatically:

```bash
ENV_NAME=scaf-grpo LIMIT=64 SAMPLES=4 TRAIN_STEPS=10 \
  bash scripts/wait_for_1_5b_and_train_2gpu.sh
```

## Qwen2.5-Math-1.5B paper reproduction

The paper-facing sequence is Base, Vanilla GRPO, Scaf-GRPO, and then the new
method, all evaluated on the same seven benchmarks. Start with the 1.5B model;
its Table 1 macro averages are 18.7%, 37.6%, and 41.5%, respectively. Prepare
the model and matching model-specific training parquet on the networked machine:

```bash
SCAF_REPO=/absolute/path/to/Scaf-GRPO ENV_NAME=scaf-grpo \
  STAGE=prepare bash scripts/run_qwen_math_1_5b_reproduction.sh
```

The preparation step resumes interrupted downloads, validates all model
shards and official datasets, and applies the context configuration documented
by Scaf-GRPO (`sliding_window=null`, `rope_theta=15000`, and
`max_position_embeddings=6144`). The original Hugging Face config is retained
as `config.huggingface-original.json`.

If download and evaluation happen on the same two-GPU machine, use
`STAGE=prepare-base` to perform both steps automatically. With separate
networked and offline instances, use `STAGE=prepare` on the networked instance
and `STAGE=base` on the H100 instance after the shared files are visible.

On the offline two-H100 machine, first reproduce the Base greedy pass@1 row:

```bash
conda activate scaf-grpo
SCAF_REPO=/absolute/path/to/Scaf-GRPO \
  STAGE=base bash scripts/run_qwen_math_1_5b_reproduction.sh
```

This uses one Ray job at a time and both GPUs, avoiding interference between
independent Ray clusters. It writes `paper_comparison.md` against the
Qwen2.5-Math-1.5B Base row in Table 1. Do not start expensive RL runs until
this row is reasonably close to the paper's 18.7% macro average.

The official Vanilla GRPO and Scaf-GRPO configurations can then be reproduced
sequentially. The full mode preserves the paper's 10 epochs, global batch 256,
eight rollouts, PPO mini-batch 64, learning rate 1e-6, and 2048 response tokens.
Micro-batching is reduced for two H100 GPUs and FlashAttention-dependent
optimizations are disabled; these change throughput, not the learning target.
The public training launcher differs from the paper in some defaults (notably
100 versus 10 epochs), so this reproduction follows the paper text and reports
the difference explicitly.

```bash
CONFIRM_FULL_REPRO=YES METHOD=vanilla MODE=paper \
  SCAF_REPO=/absolute/path/to/Scaf-GRPO \
  STAGE=vanilla-paper bash scripts/run_qwen_math_1_5b_reproduction.sh

CONFIRM_FULL_REPRO=YES METHOD=scaf MODE=paper \
  SCAF_REPO=/absolute/path/to/Scaf-GRPO \
  STAGE=scaf-paper bash scripts/run_qwen_math_1_5b_reproduction.sh
```

Use `MODE=smoke` without the confirmation flag for a one-step pipeline check.
`STAGE=baseline-smokes` runs both one-step checks sequentially.
The paper selects the best validation checkpoint, so checkpoint merging and
the final seven-benchmark evaluation must use that checkpoint rather than
silently assuming the final checkpoint is best.

??????? [`ideas/README.md`](ideas/README.md)????????????????????????????????? MetaAsk-GRPO ???????

?????????????? [`results/TWO_IDEA_PRELIMINARY_RESULTS_zh.md`](results/TWO_IDEA_PRELIMINARY_RESULTS_zh.md)?

## Quick server setup

Clone this public repository over HTTPS; no GitHub SSH key is required:

```bash
git clone https://github.com/jiZhangji/adaptive-scaffold-router.git
cd adaptive-scaffold-router
```

## Two-GPU feasibility screen

The recommended first decision experiment is a mechanism screen rather than a
full RL run. It tests both ideas in parallel on two GPUs:

- GPU 0: scaffold frontier plus a causal subproblem-relevance control. A root
  receives either its own verified prerequisite or an equally formatted
  prerequisite from another root.
- GPU 1: MetaAsk, random-bit, minimal-assistance, and controlled answer-
  verification comparisons.

Download the exact Qwen2.5-Math-1.5B model and the official model-specific
Scaf-GRPO parquet split into repository-relative directories, then validate
both assets:

```bash
ENV_NAME=scaf-grpo MAX_WORKERS=8 \
  bash scripts/download_probe_assets.sh
```

The resolved locations are always printed. By default they are:

```text
models/Qwen2.5-Math-1.5B
data/DeepScaleR/Qwen2.5-Math-1.5B.parquet
```

Run a 32-question preliminary screen on two visible H200 GPUs:

```bash
ENV_NAME=scaf-grpo GPU0=0 GPU1=1 LIMIT=32 \
  bash scripts/run_feasibility_screen_2gpu.sh
```

The two independent branches run concurrently, one heavy model process per
GPU. The launcher writes separate logs and then produces:

```text
outputs/feasibility_screen_n32/feasibility_report.md
outputs/feasibility_screen_n32/feasibility_report.json
```

The report labels each idea as `promising`, `mixed_evidence`, or
`not_supported_yet`. These labels use explicit rescue, cost, causal relevance,
active-query, and control thresholds. They are a go/no-go screen for a later
RL smoke test, not a final training result.

Download the exact Scaf-GRPO training split used by the current Qwen2.5-Math-
1.5B experiments. The script downloads the official Hugging Face artifact and
checks its published SHA256 digest:

```bash
bash scripts/download_scaf_data.sh
```

While the model is downloading in another shell, the following command can
wait for both assets and automatically validate them using the named Conda
environment `scaf-grpo`:

```bash
ENV_NAME=scaf-grpo bash scripts/wait_and_validate_downloads.sh
```

It verifies the model shards and safetensors headers without loading the full
model onto a GPU, checks the official dataset SHA256, row count and required
columns, then runs the repository tests and a Scaf-GRPO integration compile
check. The JSON report is written to `outputs/preflight_validation.json`.

To automatically run the two first-stage research tests after validation:

```bash
ENV_NAME=scaf-grpo LIMIT=8 DEVICE=cuda:0 \
  bash scripts/run_two_idea_probes.sh
```

The launcher also waits until the selected GPU has at least 10 GB free. Override
this guard with `MIN_FREE_GPU_MB` when necessary.

The first probe measures the capability-dependent scaffold frontier and
compiles a curriculum manifest. The second compares no help, random one-bit
feedback, self-asked binary verification, and minimal knowledge/planning/
solution assistance. The MetaAsk oracle is the same base model conditioned on
the reference solution, so this is explicitly a mechanism test rather than a
final learned ASK policy result.

This creates `data/DeepScaleR/Qwen2.5-Math-1.5B.parquet` (12,880 problems).
The 126 MB parquet is intentionally not duplicated in GitHub. To fetch the
whole official data repository instead, run:

```bash
git clone https://huggingface.co/datasets/hkuzxc/scaf-grpo-dataset \
  data/DeepScaleR
```

The current probe and RL smoke use `Qwen/Qwen2.5-Math-1.5B`; the 7B model is
used only for the capability-frontier comparison. Download model weights on
the GPU server, not on a local laptop.

### DeepSeek subproblem candidate generation

The local 1.5B teacher is not reliable enough for strict subproblem generation.
The resumable DeepSeek generator instead grounds each candidate in the existing
Scaf-GRPO knowledge, planning and solution-breakdown fields. Enter the API key
without placing it in shell history, then run a 64-example quality screen:

```bash
read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
LIMIT=64 bash scripts/generate_subproblems_deepseek.sh
unset DEEPSEEK_API_KEY
```

The run directory is recorded in
`outputs/latest_deepseek_subproblems.txt`. It contains `candidates.jsonl`,
`errors.jsonl`, `summary.json`, and a resumable log. Generated candidates are
not yet training examples: they must still pass answer verification, leakage
checks, student success-probability calibration and the matched random-
subproblem relevance control.

After the 64-root screen passes, generate three candidates for every zero-
reward root (`accuracy == 0`): knowledge application, planning target, and
intermediate calculation. The confirmation guard first prints the exact number
of paid API calls and requested candidates.

```bash
bash scripts/generate_all_zero_reward_subproblems_deepseek.sh
CONFIRM_FULL_GENERATION=YES \
  bash scripts/generate_all_zero_reward_subproblems_deepseek.sh
```

The full run uses one API request per root to obtain all three dimensions and
resumes in the fixed directory
`outputs/deepseek_zero_reward_subproblems_all`.

For the RL trainer, clone the official baseline next to this repository and
install the optional curriculum hook:

```bash
git clone https://github.com/JIA-Lab-research/Scaf-GRPO.git ../Scaf-GRPO
SCAF_REPO="$(realpath ../Scaf-GRPO)" bash scripts/install_scaf_integration.sh
```

This is a small proof-of-concept for progressive, minimal scaffolding. It runs
three strategies on a compact math set:

- `no_hint`: one autonomous answer.
- `full_hint`: always provide the strongest solution-step hint.
- `progressive`: try no hint, knowledge, planning, and solution hints until a
  verifier accepts an answer.

The experiment writes per-problem JSONL results and an aggregate `summary.json`.

## Quick start

Install a CUDA-enabled PyTorch build separately if GPU inference is required,
then install the remaining packages:

```powershell
python -m pip install -r requirements.txt
```

Run a two-problem CPU smoke experiment with a small model:

```powershell
python run_experiment.py --backend transformers --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --limit 2 --max-new-tokens 160
```

Run all examples on CUDA after installing CUDA-enabled PyTorch:

```powershell
python run_experiment.py --backend transformers --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --max-new-tokens 256
```

An OpenAI-compatible local server such as vLLM or LM Studio can also be used:

```powershell
python run_experiment.py --backend server --model local-model --endpoint http://127.0.0.1:1234/v1/chat/completions
```

Useful options:

- `--samples-per-level 1`: generations attempted at each scaffold level.
- `--temperature 0`: greedy decoding; use a positive value for sampling.
- `--limit N`: run only the first `N` examples.
- `--output-dir outputs/run_name`: select the result directory.

## Outputs

`summary.json` reports:

- no-hint, full-hint, and progressive accuracy;
- average selected scaffold level;
- scaffold recovery count;
- progressive calls and generated-token cost;
- the distribution of minimal effective scaffold levels.

The toy data only checks whether the mechanism works. It is not evidence for a
paper-level claim. A serious experiment should use a held-out benchmark and
teacher-generated hints whose quality is independently checked.

## Dynamic frontier probe

`frontier_probe.py` runs the first paper-facing diagnostic on the official
Scaf-GRPO parquet data. It evaluates a two-dimensional scaffold lattice:

- type: `knowledge`, `planning`, or `solution`;
- strength: 25%, 50%, or 100% of the available scaffold parts.

Every arm receives multiple rollouts. The output reports per-arm accuracy,
prompt and generation cost, the lowest-cost arm reaching a target success
rate, the arm inside a useful learning-accuracy band, and a cost-aware utility
choice. It also reports how often generations hit the configured token limit
and simulates the token overhead of Scaf-GRPO-style progressive
search against a hindsight minimum-cost policy. The report separates the full
three-type lattice from the current public Scaf-GRPO implementation path,
which enumerates knowledge and solution hints while its planning stage is
disabled in code.

Install CUDA-enabled PyTorch separately, then install the probe dependencies:

```bash
python -m pip install -r requirements-probe.txt
```

Example remote run:

```bash
python frontier_probe.py \
  --model /path/to/Qwen2.5-Math-1.5B \
  --data /path/to/Qwen2.5-Math-1.5B.parquet \
  --output-dir outputs/frontier_qwen_math_1_5b \
  --limit 32 --samples-per-arm 4 --strengths 0.25,0.5,1.0
```

For a long generation budget, `--stop-after-boxed` stops each sequence after a
balanced `\boxed{...}` answer is complete. This preserves the answer used by
the verifier while avoiding trailing generation after the final result.

This is a diagnostic rather than a GRPO training result. Its purpose is to
test whether the minimal useful scaffold varies by instance and whether a
cost-aware frontier has enough headroom to justify online router training.

After running the same selected examples with two model capability levels,
compare their frontiers with:

```bash
python compare_frontiers.py \
  --weaker outputs/frontier_1_5b/item_frontiers.jsonl \
  --stronger outputs/frontier_7b/item_frontiers.jsonl \
  --choice-key threshold_choice \
  --output outputs/capability_shift.json
```

The comparison reports how often the selected scaffold changes, whether the
stronger model needs fewer hint tokens, and how no-hint accuracy changes.

## Calibrated router baseline

After collecting a larger full-information probe, train the first realizable
cost-aware router with a question-grouped split:

```bash
python train_router.py \
  --input outputs/router_labels/raw_results.jsonl \
  --output-dir outputs/router_baseline
```

The router uses question text together with scaffold type, strength, and cost.
Calibration uses a separate validation-question split. The report compares a
single routed action and bounded failure fallback against public Scaf-GRPO
progressive enumeration, the strongest solution hint, and a full-information
oracle. This lightweight router is a baseline for a later hidden-state router.

## Capability-matched scaffold curriculum

The newer controller in `capability_scaffold.py` implements the training logic
for prerequisite subproblems, calibrated scaffold activation, explicit no-hint
graduation, and scaffold fading. It addresses a stronger question than direct
routing: whether the current learner has enough prerequisite capability for a
guided rollout to produce useful and transferable credit.

```bash
python capability_scaffold.py \
  --input data/capability_curriculum_example.jsonl \
  --output-dir outputs/capability_curriculum
```

`scaf_curriculum_adapter.py` contains the clipped sequence-level off-context
weight used when the same guided response is optimized for the original
unguided context. See `CAPABILITY_MATCHED_SCAFFOLD.md` for the data contract,
trainer integration, and required ablations.
it is not presented as the final DSFL architecture.

## Two-GPU Scaf training smoke test

`scripts/run_scaf_smoke_2gpu.sh` scales the official eight-GPU launch down to
a one-step, two-GPU environment check. It disables remove-padding and chunked
prefill so FlashAttention is not required for the first smoke test. Required
paths are supplied through environment variables; no credentials are stored in
the script.

Set `DRY_RUN=1` to print and validate the composed Hydra configuration without
allocating model or GPU workers. A real one-step run should only be launched
after the dry run succeeds and both GPUs are idle.

`scripts/setup_scaf_env.sh` records the tested installation order and pins
packages whose current releases otherwise pull incompatible Transformers 5,
NumPy 2, or CUDA 13 dependencies. FlashAttention is optional because the
two-GPU smoke configuration does not require remove-padding kernels.
