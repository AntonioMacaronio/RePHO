#!/usr/bin/env bash
# Build the RePHO_tracking conda env on THIS machine (H200 / CUDA 12.1).
# Mirrors README "2. Install Isaac Gym Environment for Motion Tracking".
set -o pipefail

CONDA=/home/sky/miniconda3
ENV_NAME=RePHO_tracking
REPO=/home/sky/sky_workdir/RePHO

export PATH="$CONDA/bin:$PATH"
# CUDA_HOME for any source builds (pytorch3d/isaacgym extensions)
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"

echo "==================== [1/5] create env ($(date)) ===================="
if "$CONDA/bin/conda" env list | grep -qE "^${ENV_NAME}\s"; then
    echo "env $ENV_NAME already exists, skipping create"
else
    "$CONDA/bin/conda" create -n "$ENV_NAME" -c conda-forge python=3.8 -y || { echo "!! create FAILED"; exit 11; }
fi

source "$CONDA/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "!! activate failed"; exit 12; }
echo "python: $(which python)  ($(python --version 2>&1))"

echo "==================== [2/5] torch 2.4.1 cu121 ($(date)) ===================="
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121 || { echo "!! torch FAILED"; exit 13; }

echo "==================== [3/5] requirement.txt ($(date)) ===================="
cd "$REPO"
pip install -r requirement.txt || echo "!! requirement.txt had errors (continuing)"

echo "==================== [4/5] pytorch3d 0.7.8 (py38 cu121 pyt241) ($(date)) ===================="
conda install -y -c conda-forge \
    https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py38_cu121_pyt241.tar.bz2 \
    || echo "!! pytorch3d install had errors (continuing)"

echo "==================== [5/5] isaacgym -e ($(date)) ===================="
cd "$REPO/isaacgym/python"
pip install -e . || echo "!! isaacgym install had errors (continuing)"

echo "==================== verify ($(date)) ===================="
cd "$REPO"
python - <<'PY'
import importlib
for m in ["torch","torchvision","numpy","rl_games","trimesh","open3d","pytorch3d","smpl_sim","smplx"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"FAIL {m:14s} {type(e).__name__}: {e}")
import torch
print("torch.cuda.is_available:", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
PY
echo "############ RePHO_tracking BUILD DONE ($(date)) ############"
