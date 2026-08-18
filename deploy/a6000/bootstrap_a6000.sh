#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/home/powerleader/project}"
SCAF_REPO="${SCAF_REPO:-$INSTALL_ROOT/Scaf-GRPO}"
ENV_PREFIX="${ENV_PREFIX:-$INSTALL_ROOT/envs/scaf-grpo}"
CONDA_ROOT="${CONDA_ROOT:-$INSTALL_ROOT/miniconda3}"
CACHE_ROOT="${CACHE_ROOT:-$INSTALL_ROOT/.cache/student-aware}"

mkdir -p "$INSTALL_ROOT" "$INSTALL_ROOT/envs" "$CACHE_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda-pkgs"
export HF_HOME="$CACHE_ROOT/huggingface"

echo "===== GPU ====="
command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is unavailable; install the NVIDIA driver first." >&2
  exit 2
}
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader

echo
echo "===== SYSTEM PACKAGES ====="
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git curl wget zstd rsync build-essential
elif [[ "${EUID:-$(id -u)}" == "0" ]]; then
  apt-get update
  apt-get install -y git curl wget zstd rsync build-essential
else
  echo "sudo/root is required to install git, curl, zstd, and build tools." >&2
  exit 3
fi

echo
echo "===== CONDA ====="
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    installer="$CACHE_ROOT/Miniconda3-latest-Linux-x86_64.sh"
    curl -L --fail --retry 3 \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -o "$installer"
    bash "$installer" -b -p "$CONDA_ROOT"
  fi
  CONDA_BIN="$CONDA_ROOT/bin/conda"
fi
echo "Conda: $CONDA_BIN"

echo
echo "===== SCAF-GRPO ====="
if [[ ! -d "$SCAF_REPO/.git" ]]; then
  git clone https://github.com/JIA-Lab-research/Scaf-GRPO.git "$SCAF_REPO"
else
  echo "Existing checkout retained: $SCAF_REPO"
fi

echo
echo "===== CUDA/PYTHON ENVIRONMENT ====="
ENV_PREFIX="$ENV_PREFIX" SCAF_REPO="$SCAF_REPO" CONDA_BIN="$CONDA_BIN" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" \
  bash "$PROJECT_ROOT/scripts/setup_scaf_env.sh"

"$ENV_PREFIX/bin/pip" install -U modelscope
"$ENV_PREFIX/bin/pip" install -r "$PROJECT_ROOT/requirements.txt"

echo
echo "===== INSTALL TRAINER INTEGRATION ====="
PATH="$ENV_PREFIX/bin:$PATH" SCAF_REPO="$SCAF_REPO" \
  bash "$PROJECT_ROOT/scripts/install_complete_scaf_integration.sh"

echo
echo "===== OFFICIAL TRAINING PARQUET ====="
(
  cd "$PROJECT_ROOT"
  DATA_DIR="$PROJECT_ROOT/data/DeepScaleR" \
    bash scripts/download_scaf_data.sh
)

cat > "$INSTALL_ROOT/a6000_paths.env" <<EOF
export INSTALL_ROOT="$INSTALL_ROOT"
export PROJECT_ROOT="$PROJECT_ROOT"
export SCAF_REPO="$SCAF_REPO"
export ENV_PREFIX="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:\$PATH"
export HF_HOME="$HF_HOME"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS"
EOF

echo
echo "Bootstrap complete."
echo "Environment: $ENV_PREFIX"
echo "Paths file:  $INSTALL_ROOT/a6000_paths.env"
