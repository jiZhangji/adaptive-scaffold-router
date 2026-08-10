#!/usr/bin/env bash
set -euo pipefail

: "${ENV_PREFIX:?Set ENV_PREFIX to an environment path on a data disk}"
: "${SCAF_REPO:?Set SCAF_REPO to the official Scaf-GRPO checkout}"

CONDA_BIN="${CONDA_BIN:-conda}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "$ENV_PREFIX")/pip-cache}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$(dirname "$ENV_PREFIX")/conda-pkgs}"
export PIP_CACHE_DIR CONDA_PKGS_DIRS

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create -p "$ENV_PREFIX" python=3.10 pip -y
fi

pip_bin="$ENV_PREFIX/bin/pip"

"$pip_bin" install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

"$pip_bin" install \
  vllm==0.8.5.post1 tensordict==0.6.2 torchdata

# These pins prevent newer package releases from silently moving the tested
# CUDA 12.4 stack to Transformers 5, NumPy 2, or CUDA 13 dependencies.
"$pip_bin" install \
  transformers==4.51.3 huggingface-hub==0.36.2 tokenizers==0.21.4 \
  numpy==1.26.4 cupy-cuda12x==13.3.0 opencv-python-headless==4.11.0.86

"$pip_bin" install --no-deps -e "$SCAF_REPO"
"$pip_bin" install \
  accelerate codetiming datasets==4.0.0 hydra-core liger-kernel pandas \
  peft pyarrow==19.0.1 pybind11 pylatexenc pre-commit packaging uvicorn \
  fastapi latex2sympy2_extended math-verify
"$pip_bin" install protobuf==4.25.9 wandb==0.19.10

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  wheel_dir="$(dirname "$ENV_PREFIX")/wheels"
  wheel_name="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  mkdir -p "$wheel_dir"
  curl -L --fail --retry 3 --continue-at - \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$wheel_name" \
    -o "$wheel_dir/$wheel_name"
  "$pip_bin" install "$wheel_dir/$wheel_name"
fi

"$pip_bin" check
