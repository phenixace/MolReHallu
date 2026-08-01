#!/bin/bash
#PBS -N install_easyr1
#PBS -P <your-project-code>
#PBS -q auto
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=96gb:ngpus=1
#PBS -j oe
#PBS -o logs/$PBS_JOBNAME.out

# ---------------------------------------------------------------------------
# Site configuration. Every path below was specific to the machine this ran on;
# set them for yours. Defaults assume you are in the EasyR1 checkout with this
# repository's `reward/` copied to examples/reward_function/.
: "${PROJECT_DIR:=$PWD}"                 # this repository's checkout
: "${EASYR1_DIR:=$PWD}"                  # the patched EasyR1 checkout (see ../verl_patch)
: "${CONDA_SH:=$HOME/miniconda3/etc/profile.d/conda.sh}"
: "${EASYR1_ENV:=easyr1}"                # conda env for training
: "${EVAL_PYTHON:=python}"               # interpreter for the evaluation scripts
: "${HF_HOME:=$HOME/.cache/huggingface}"
: "${CKPT_ROOT:=$PWD/checkpoints}"       # where GRPO checkpoints are written
# ---------------------------------------------------------------------------

set -eo pipefail
module load cuda12.4/toolkit/12.4.1

source "$CONDA_SH"

ENV=$EASYR1_ENV
EASYR1=$EASYR1_DIR

echo "=========================================="
echo "EasyR1 env install on $(hostname) @ $(date)"
echo "=========================================="

# Fresh env (python 3.11 matches the flash-attn wheel below)
if [ ! -d "$ENV" ]; then
    conda create -y -p "$ENV" python=3.11
fi
conda activate "$ENV"
python -V
pip install --upgrade pip

# vLLM 0.8.4 pins torch 2.6.0 + cu124 and pulls transformers>=4.51 — the exact
# combo EasyR1's Dockerfile targets. Install it first so torch is fixed.
pip install "vllm==0.8.4"

# flash-attn: use the prebuilt wheel matching torch2.6/cu12/py311/cxx11abiFALSE
# (compiling from source takes 30-60 min). Fall back to source build if the
# wheel URL ever 404s.
FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install "$FA_WHL" || pip install flash-attn==2.7.4.post1 --no-build-isolation

# Remaining EasyR1 deps (transformers/vllm already satisfied by vllm install).
pip install \
    "transformers>=4.51.0" \
    accelerate codetiming datasets liger-kernel mathruler numpy omegaconf \
    pandas "pyarrow>=15.0.0" pillow pylatexenc qwen-vl-utils "ray[default]" \
    tensordict torchdata wandb peft

# Install EasyR1 (verl) itself without re-resolving deps.
cd "$EASYR1"
pip install -e . --no-deps

echo "=== Verify imports ==="
python -c "
import torch, vllm, transformers, ray, tensordict, flash_attn
print('torch', torch.__version__, '| cuda', torch.version.cuda)
print('vllm', vllm.__version__, '| transformers', transformers.__version__)
print('ray', ray.__version__, '| flash_attn', flash_attn.__version__)
import verl; print('verl OK')
"
# Confirm our reward functions import (they pull rdkit-free logic from main proj)
pip install rdkit scikit-learn
python -c "
import sys; sys.path.insert(0,'$EASYR1')
import os; os.environ['MOLLM_PROJECT_DIR']='$PROJECT_DIR'
from examples.reward_function import chem_process_ours, chem_empo_ours
print('reward functions import OK')
"

echo "=========================================="
echo "DONE @ $(date)  env=$ENV"
echo "=========================================="
