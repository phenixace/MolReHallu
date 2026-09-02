# Benchmark corpora — what to download and where to put it

The three benchmark corpora are **not shipped here as corpora**; each stays under its own
upstream licence (see `LICENSE`, THIRD-PARTY MATERIAL). Everything derived from them
*is* shipped — the prompts as sent, the model responses, and the diagnosis records — so
every number in the paper is auditable without downloading anything.

One file does carry corpus content: `training/dataset/train.parquet` embeds the ChEBI-20 and
USPTO-50k training-split rows the GRPO run consumed, rewritten as prompt/answer pairs, plus
OpenMolIns instructions. `LICENSE` gives the amounts. Nothing in `data/` does this, and the
evaluation splits are not reconstructible from what ships here.

You need these corpora only to regenerate model responses from scratch, or to call
`data_loaders.load_task()`. Put them under `data/` in the layout below and the loaders
find them with no configuration.

## ChEBI-20 — `cap2mol`, `mol2cap`

Distributed with MolT5 (Edwards et al. 2022), `github.com/blender-nlp/MolT5`, directory
`ChEBI-20_data/`. Copy the three files in:

```
data/chebi-20/train.txt
data/chebi-20/validation.txt
data/chebi-20/test.txt
```

Tab-separated, one header line, then `CID <TAB> SMILES <TAB> description`. The paper uses
the **full test split**; `load_task(..., split=...)` also accepts `train` and `val`.

## USPTO-50k — `retrosynthesis`

Liu et al. 2017. Any mirror works as long as it is a CSV with columns `id`, `class` and
`reactions`, where `reactions` is `reactants>>product`:

```
data/uspto50k/USPTO_50K.csv
```

The split is derived in code, not read from the file: the **last 5,007 rows are test**,
the 5,007 before that are validation, the rest is train (`USPTO50K_DEFAULT_TEST_SIZE`
in `data_loaders.py`). This is the conventional retrosynthesis layout, and it is
deterministic — no seed, no shuffle — so the test set is reproducible from the CSV alone.
The task is retro: the product becomes the input and the reactants are the target.

## S²-Bench (TOMG-Bench) — the nine `s2_*` subtasks

Li et al. 2024. One CSV per subtask, named exactly:

```
data/s2-bench/MolCustom_AtomNum.csv        data/s2-bench/MolEdit_AddComponent.csv
data/s2-bench/MolCustom_BondNum.csv        data/s2-bench/MolEdit_DelComponent.csv
data/s2-bench/MolCustom_FunctionalGroup.csv  data/s2-bench/MolEdit_SubComponent.csv
data/s2-bench/MolOpt_LogP.csv              data/s2-bench/MolOpt_MR.csv
data/s2-bench/MolOpt_QED.csv
```

Take these from **`phenixace/S2-TOMG-Bench`** (5,000 rows per subtask). The nine files in
this study's corpus directory are byte-identical to that repository's, verified by sha256.

**Which 500 prompts.** Each subtask is evaluated on the **first 500 rows of its CSV, in the
order the file is distributed**, skipping any row with an empty `Instruction`
(`load_s2bench` slices `data[:500]`). There is no sampling, no shuffle and no seed
anywhere in `data_loaders.py`, so the subset is fixed by the file itself. Verified for all
nine subtasks: the ids and the prompt strings in `data/responses/*/s2_*/output.json.gz`
match that rule exactly, in order.

Two details make the subset recoverable without guessing. MolEdit and MolOpt rows carry an
`index` column holding the source dataset's own identifier, which is what the shipped
sample ids use, so those rows can be selected from any copy of the CSV. MolCustom has no
`index` column, so its ids are row positions (0-499) and do depend on file order --
which is why the release also ships the prompt text itself, all 500 per subtask, in
`output.json.gz`. The evaluated set is therefore reconstructible from this repository alone.

**Not the mini release.** `phenixace/S2-TOMG-Bench-mini` is also 500 rows per subtask, but
it is a *different* 500: its rows are drawn from across the full 5,000, and only 63 of them
fall in the first 500 that this study evaluates. Downloading the mini release and running
this code would score a different subset. Use the full repository and the head-500 rule
above.

`data_loaders.py` reads `source_molecule` out of the MolEdit/MolOpt rows into each
sample's `metadata`. That field is load-bearing: the detector needs it to run the
input-grounding half of the ER check, and S² performance scoring needs it too. The
release therefore also ships it directly, as
`data/responses/<model>/<task>/metadata.json.gz`, so neither the detector nor the
metrics require the corpus to be present.

## Not used in the paper

`data_loaders.py` also registers MoleculeNet classification (`bace`, `bbbp`, `hiv`,
`tox21`, `clintox`, expecting `data/moleculenet/<name>/<name>.csv` plus
`ogb_splits_<name>.json`) and forward `reaction_prediction`. These are capabilities of
the code that the study does not report; nothing in the paper or in `data/` depends on
them.
