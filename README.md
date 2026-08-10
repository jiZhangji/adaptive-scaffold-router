# Adaptive Scaffold Experiment

中文研究方案见 [`ideas/README.md`](ideas/README.md)：其中分别记录了图片中的能力匹配子问题—脚手架课程方案，以及独立的 MetaAsk-GRPO 主动求助方案。

## Quick server setup

Clone this public repository over HTTPS; no GitHub SSH key is required:

```bash
git clone https://github.com/jiZhangji/adaptive-scaffold-router.git
cd adaptive-scaffold-router
```

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
