# Student-Aware A6000 deployment

This bundle installs the Student-Aware GRPO code and tested CUDA environment
under `/home/powerleader/project`, downloads the essential ModelScope
artifacts, restores the model/data/checkpoints, and optionally evaluates
global steps 10, 35, and 50 on the same seven-benchmark protocol.

GitHub stores only code and deployment scripts. Model weights, checkpoints,
datasets, and experiment logs remain in ModelScope or their original data
repositories.

## Requirements

- Ubuntu Linux with an NVIDIA A6000 and a working NVIDIA driver.
- Internet access from the server.
- `sudo` access for system packages.
- A ModelScope access token with permission to read
  `shimian123/student-aware-grpo-migration`.
- At least 150 GiB free disk space for the environment, checkpoints, models,
  and restored experiment artifacts.

## One-click setup

```bash
cd /home/powerleader/project

git clone \
  --branch fix-off-context-logprob \
  https://github.com/jiZhangji/adaptive-scaffold-router.git

cd adaptive-scaffold-router

bash deploy/a6000/one_click_a6000.sh
```

The script asks for the ModelScope token without echoing it. The token is not
written into this repository.

## Setup and immediately evaluate Step 10/35/50

Evaluation is intentionally opt-in because it consumes substantial GPU time.

```bash
cd /home/powerleader/project/adaptive-scaffold-router

RUN_EVAL=1 bash deploy/a6000/one_click_a6000.sh
```

## Run evaluation later

```bash
cd /home/powerleader/project/adaptive-scaffold-router

bash deploy/a6000/evaluate_checkpoints_a6000.sh
```

The resulting table is written to:

```text
/home/powerleader/project/adaptive-scaffold-router/
outputs/a6000_student_aware_checkpoint_eval/checkpoint_comparison.md
```

## Important paths

```text
/home/powerleader/project/adaptive-scaffold-router  project code
/home/powerleader/project/Scaf-GRPO                GRPO/VERL repository
/home/powerleader/project/envs/scaf-grpo           pinned Conda environment
/home/powerleader/project/modelscope               downloaded MS artifacts
/home/powerleader/project/adaptive-scaffold-router/models
/home/powerleader/project/adaptive-scaffold-router/data
/home/powerleader/project/adaptive-scaffold-router/outputs
```

## Configuration overrides

```bash
INSTALL_ROOT=/home/powerleader/project
MS_REPO_ID=shimian123/student-aware-grpo-migration
RUN_ID=student_aware_root_aligned_20260816_183357
GPU=0
```

Example:

```bash
GPU=0 EVAL_BATCH_SIZE=32 \
  bash deploy/a6000/evaluate_checkpoints_a6000.sh
```

Do not train on the AMC23 or AIME24 evaluation files. They are held-out
benchmarks. Any AMC-style replay data must come from a separate training set.
