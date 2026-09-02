# Training — verification-grounded GRPO

Everything needed to reproduce the RL stage that turns the released Chem-R checkpoint into
Chem-R-Faithful.

The reward itself is `../reward/verification_grounded_reward.py`; it is the only part of this that runs
without a GPU, and `../README.md` shows how to exercise the ER=0 gate on CPU in a few seconds.

## The run

`configs/chem_r_faithful.yaml` with `COUPLED=1` is the run that produced
**Chem-R-Faithful**: the same process reward as an ordinary run, except that the accuracy term is
paid only when the trace is clean (ER = 0). Setting `COUPLED` to anything else removes the gate,
which is how the ablation in the paper was produced.

The internal training codename for this run was `Chem-R-v8-coupled`; `data/raw/README.md` keeps
that mapping so the shipped records, the checkpoints and the logs still line up with each other.

## Layout

```
training/
  configs/         the GRPO config (grpo, lr 1e-6, kl_coef 1e-2, rollout n=5, 4 GPUs, 3 epochs)
  jobs/            the submission script as run, with site paths replaced by shell variables
  dataset/         parquet builders + the exact train/test parquets the runs consumed
  format_prompt/   blank_format.jinja, referenced by both configs
  verl_patch/      three EasyR1 files that must replace the upstream ones -- see below
  model_merger.py  FSDP shards -> a HuggingFace directory, after training
  runtime_env.yaml ray runtime env
  requirements.txt EasyR1's own dependency pins (the repo root has the analysis-side ones)
```

`dataset/train.parquet` (12 MB) is the actual training data and `dataset/test.parquet` the
file the trainer expects alongside it, so the run is reproducible without rebuilding them. `dataset/make_*_parquet.py` are the builders, kept
so the construction is inspectable; they read the benchmark corpora, which are not shipped as
corpora anywhere in this repository.

Note that these two parquets are the one place the release does carry corpus content: the
prompts and targets inside them come from the ChEBI-20 and USPTO-50k training splits and from
OpenMolIns. `LICENSE`, THIRD-PARTY MATERIAL, states the amounts and whose licence governs
their use.

### What the training set is made of

80,000 prompts, four equal families of 20,000, mixed by `make_merged_parquet.py` with
`--seed 42` and shuffled:

| family | rows | source | built by |
|---|---|---|---|
| cap2mol | 20,000 | ChEBI-20 **train** split | `make_task_parquet.py` / `make_chebi_parquet.py` |
| mol2cap | 20,000 | ChEBI-20 **train** split | `make_task_parquet.py` |
| retrosynthesis | 20,000 | USPTO-50k **train** split (rows before the derived valid/test tail) | `make_task_parquet.py` |
| the nine `s2_*` subtasks | 20,000 total, 2,170-2,288 each | **OpenMolIns** (`phenixace/OpenMolIns-*`, 45k rows, TOMG-shaped), *not* S²-Bench | `make_openmolins_parquet.py` |

The equal 20,000 cap is deliberate: uncapped, retrosynthesis and S² would swamp the two
ChEBI tasks. Within the S² block the nine subtasks land unevenly because the 20,000 are
sampled from the merged OpenMolIns pool rather than quota-filled per subtask.

Each row is `problem` (system prompt + the task template from `prompts.py` with the
instruction substituted), `answer`, and `task`. For the S² subtasks `answer` is JSON
carrying `{task, metadata, gt}`, because the reward's EO dimension needs the parsed
numeric constraints or source molecule; `make_openmolins_parquet.py` recovers those from
the instruction prose, since OpenMolIns ships only `[SubTask, Instruction, molecule]`.
For the other three `answer` is the plain target. `data.task_key: task` is what routes
each row to the right branch of the reward, which is why the verl patch is required.

`dataset/test.parquet` (800 rows, 200 per family) is the file EasyR1 expects at
`data.val_files`. The name is that loader's filename convention, not a reference to the
paper's test set -- the evaluation set is a different thing entirely, described in
`data/CORPORA.md`. Its rows only ever fed the trainer's periodic monitoring generations,
never the training signal, and validation is off in the shipped config, so it is not read
at all; the path still has to resolve, because the patched data loader builds the
validation dataset unconditionally and asserts it is non-empty.

## The verl patch — required, not optional

The reward function dispatches per task:

```python
compute_score(predicts, ground_truths, tasks=..., prompts=...)
```

Upstream EasyR1 has no `task_key`, so it never passes `tasks`, and the reward silently falls back
to `["cap2mol"] * n` — every S²-Bench and retrosynthesis response would then be scored with the
caption-to-molecule verifier. Nothing raises; the run simply optimises the wrong objective.

`verl_patch/` holds the three files that add the plumbing. Copy them over the upstream ones:

```
verl_patch/utils_dataset.py        -> verl/utils/dataset.py
verl_patch/trainer_config.py       -> verl/trainer/config.py
verl_patch/trainer_data_loader.py  -> verl/trainer/data_loader.py
```

They carry a `task_key` from the config through the dataset and the data loader into the reward
call. `data.task_key: task` in the config is what selects the column.

## Running

```bash
# 1. EasyR1 checkout, patched
git clone <easyr1>  && cd EasyR1
cp <this repo>/training/verl_patch/utils_dataset.py       verl/utils/dataset.py
cp <this repo>/training/verl_patch/trainer_config.py      verl/trainer/config.py
cp <this repo>/training/verl_patch/trainer_data_loader.py verl/trainer/data_loader.py

# 2. the reward has to sit where the config points
mkdir -p examples/reward_function
cp <this repo>/reward/verification_grounded_reward.py examples/reward_function/
cp <this repo>/s2_success.py <this repo>/s2_official_eval.py examples/reward_function/
cp <this repo>/diagnose_hallucination.py <this repo>/diagnose_multitask.py .

# 3. data and config
mkdir -p data/merged_4task && cp <this repo>/training/dataset/*.parquet data/merged_4task/
cp -r <this repo>/training/format_prompt examples/
cp <this repo>/training/configs/chem_r_faithful.yaml examples/

# 4. train (COUPLED=1 is what makes it Chem-R-Faithful)
COUPLED=1 MOLLM_PROJECT_DIR=$PWD python3 -m verl.trainer.main config=examples/chem_r_faithful.yaml

# 5. FSDP shards -> HuggingFace format
python3 scripts/model_merger.py --local_dir checkpoints/chem-r-faithful/global_step_936/actor
```

`jobs/*.sh` are the same steps as they were actually submitted (PBS). Their header block lists
every path that was site-specific — `PROJECT_DIR`, `EASYR1_DIR`, `CONDA_SH`, `EASYR1_ENV`,
`EVAL_PYTHON`, `HF_HOME`, `CKPT_ROOT` — so they read as documentation of the real invocation
rather than as scripts that only ran on one cluster.

## What is not here

The trained weights. **Chem-R-Faithful is at `phenixace/Chem-R-Faithful`.** The two-stage SFT
lineage behind the ladder's SFT rung is at `slayertear/llama-3.1-8b-stage1` and
`slayertear/llama-3.1-8b-stage2`; stage2 is the one the paper evaluates. The config points at the
public `weidawang/Chem-R-8B`, which is what this run actually began from, so it is reproducible
end to end.

Compute: 936 steps on 4 NVIDIA H200 GPUs, 26.6 hours wall-clock (about 106 GPU-hours),
measured from the job's start timestamp to the mtime of `global_step_936`.
