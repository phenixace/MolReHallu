#!/bin/bash
#PBS -N easyr1_merged_v8_coupled
#PBS -P <your-project-code>
#PBS -q auto
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=32:mpiprocs=1:ompthreads=32:mem=300gb:ngpus=4
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
conda activate "$EASYR1_ENV"
export MOLLM_PROJECT_DIR=$PROJECT_DIR
export HF_HOME=$HF_HOME
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export COUPLED=1
if [[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]]; then
    NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    [[ "$NGPU" -gt 0 ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU-1)))
fi
cd "$MOLLM_PROJECT_DIR/EasyR1-main"
echo "=== merged_v8_coupled (COUPLED=$COUPLED) reward=chem_merged_v8_ours.py on $(hostname) @ $(date) ==="
python3 -m verl.trainer.main config=examples/config_merged_v8_coupled.yaml
echo "=== DONE @ $(date) ==="
