#!/bin/bash
#PBS -N merge-and-eval
#PBS -P <your-project-code>
#PBS -q auto
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=128gb:ngpus=1
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

export HF_HOME=$HF_HOME
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export MOLLM_PROJECT_DIR=$PROJECT_DIR

if [[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]]; then
    NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    [[ "$NGPU" -gt 0 ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU-1)))
fi

PROJ=$PROJECT_DIR
EASYR1=$PROJ/EasyR1-main
CKPT_BASE=$CKPT_ROOT/chem-r-faithful
# Release display name. The internal training codename for this run was
# Chem-R-v8-coupled; data/raw/README.md maps the two.
NAME=Chem-R-Faithful
GEN="cap2mol mol2cap retrosynthesis"
S2="s2_MolCustom_AtomNum s2_MolCustom_BondNum s2_MolCustom_FunctionalGroup s2_MolEdit_AddComponent s2_MolEdit_DelComponent s2_MolEdit_SubComponent s2_MolOpt_LogP s2_MolOpt_MR s2_MolOpt_QED"

STEP=$(cat "$CKPT_BASE/latest_global_step.txt" 2>/dev/null || ls "$CKPT_BASE" | grep -oE 'global_step_[0-9]+' | sed 's/global_step_//' | sort -n | tail -1)
CKPT="$CKPT_BASE/global_step_$STEP/actor"
echo "=== Merge + eval Chem-R-Faithful step $STEP on $(hostname) @ $(date) ==="
echo "CKPT=$CKPT"

conda activate "$EASYR1_ENV"
cd "$EASYR1"
if ls "$CKPT"/huggingface/*.safetensors >/dev/null 2>&1; then echo "[skip] merged"; else
    python3 scripts/model_merger.py --local_dir "$CKPT"; fi

conda deactivate
SEPY=$EVAL_PYTHON
cd "$PROJ"

$SEPY run_multitask_se.py --model_id "$CKPT/huggingface" --model_name "$NAME" \
    --backend vllm --tasks $GEN --n_samples 1 --temperature 0.8 \
    --tensor_parallel 1 --gpu_memory_utilization 0.90 --max_model_len 4096 \
    --max_samples 6000 --output_dir "$PROJ/se_results/$NAME" --resume

$SEPY run_multitask_se.py --model_id "$CKPT/huggingface" --model_name "$NAME" \
    --backend vllm --tasks $S2 --n_samples 1 --temperature 0.8 \
    --tensor_parallel 1 --gpu_memory_utilization 0.90 --max_model_len 4096 \
    --max_samples 500 --output_dir "$PROJ/se_results/$NAME" --resume

for t in $GEN $S2; do
    out="$PROJ/se_results/$NAME/$t/output.json"
    [ -f "$out" ] && $SEPY diagnose_multitask.py --task "$t" \
        --model_output "$out" --model_name "$NAME" \
        --output_dir "$PROJ/results/$NAME/$t" --verbose || echo "[warn] no output $t"
done

echo ""
echo "=== decoupling comparison: baseline vs verification-grounded ==="
$SEPY - <<'PY'
import sys; sys.path.insert(0, "eval")
import metrics as MX
for m, lbl in [("Chem-R", "baseline"), ("Chem-R-Faithful", "verification-grounded")]:
    for fam, ts in MX.FAMILIES:
        s = MX.family_stats(m, ts)
        if s: print(f"  {lbl:13s} {fam:7s} perf={s['perf']:5.1f} ER={s['ER']:5.2f} overall={s['overall']:5.2f} %ER0={s['pct_er0']:4.0f}")
PY
echo "DONE @ $(date)"
