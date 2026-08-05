# Adaptive Scaffold Experiment

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

