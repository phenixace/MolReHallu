# MolReHallu

Reasoning hallucination in chemical reasoning models, made structurally checkable. A chemical
chain-of-thought states functional-group claims that can be decided against the molecular graph,
so the rationale can be audited claim by claim rather than judged as fluent text.

This repository holds the detector, the analysis pipeline, the verification-grounded training
recipe, and the data behind every figure and number in the paper.

## What the paper finds, and where each finding lives here

The study evaluates four reasoning model families on twelve chemistry task variants
(16,107 responses per model) and reports four things. Each maps to code and shipped data:

| finding | evidence in this repo |
|---|---|
| Fabrication is **widespread**: chemical traces routinely assert functional groups that are in neither the input nor the answer molecule. | `diagnose_hallucination.py` decides each claim against the molecular graph by RDKit SMARTS; `data/results/` holds the per-response verdicts, `eval/metrics.py` aggregates them. |
| It is **largely decoupled from correctness**: a right answer often arrives through a fabricating trace. | `Diagnosis_model_task` / `Diagnosis_family` in `data/source_data.xlsx`, plotted as Figures 2 and 3. |
| The trace is still **not inert**. Corrupting a verified functional-group *claim* barely moves the answer, but corrupting the *drafted SMILES* degrades it — a scratchpad, in model-specific form. | the perturbation and attribution probes in `eval/` (`cot_drift.py`, `cot_info_gain.py`, `attention_attribution.py`, `attr_probe.py`), Figures 4-6. |
| Fabrication **originates before chemistry fine-tuning** and answer-only RL does not remove it; a reward that pays accuracy only on a verified trace does. | `eval/stage_ladder_metrics.py` over one lineage (base -> SFT -> Chem-R -> Chem-R-Faithful); the reward is `reward/verification_grounded_reward.py`, the run is `training/`. |

Reproducing the reported numbers needs nothing but this repository. The benchmark corpora
are not shipped as corpora (`data/CORPORA.md` says where to get them); you need them only to
regenerate responses from scratch.

---

## Two things you can verify in under a minute, on a CPU, with no data and no weights

```bash
pip install -r requirements.txt

# 1. the detector: a correct answer reached through a fabricating trace
python - <<'PY'
from diagnose_multitask import diagnose_one
A = "CC(=O)Oc1ccccc1C(=O)O"                                  # aspirin, the correct answer
t = "<think>It contains an azide and a sulfonamide.</think><answer>%s</answer>" % A
d = diagnose_one("cap2mol", {"answer": t, "question": "aspirin", "gt": A, "metadata": {}}, verbose=True)
print(d["exact_match"], d["hallucination_scores"]["ER_factual_fabrication"],
      d["details"]["ER"]["fabricated_fgs"])
# -> True 75.0 ['azide', 'sulfonamide']   (set order varies) — answer perfect, trace fabricating
PY

# 2. the reward that fixes it: accuracy is paid only when the trace is clean
COUPLED=1 python -c "
import sys; sys.path[:0]=['.','reward']
from verification_grounded_reward import compute_score
A='CC(=O)Oc1ccccc1C(=O)O'
s=compute_score(['<think>It contains an azide.</think><answer>%s</answer>'%A],[A],['cap2mol'],['Description: aspirin'])[0]
print(s['accuracy'], s['accuracy_raw'])
# -> 0.0 1.0    the answer is right, so accuracy_raw is 1.0, but the gate pays nothing
"
```

---

## The pipeline

```
  benchmark corpora                    ChEBI-20 · USPTO-50k · S²-Bench
  (not shipped as corpora)             cited in Methods, fetched from their own sources
          |
          |  data_loaders.py        task registry, splits, 18 tasks
          |  prompts.py             per-task instruction templates
          v
  [1] GENERATION                     run_multitask_se.py            GPU + model weights
          |                          samples n responses per prompt, writes the traces
          |                          semantic_entropy.py does the sampling + clustering
          v
      responses  ------------------->  data/responses/<model>/<task>/output.json.gz   SHIPPED
          |                            + metadata.json.gz  (S² constraints, needed for success)
          |
          |  diagnose_multitask.py   per-task dispatch
          |  diagnose_hallucination.py  the 2x2 detector (IR / IO / ER / EO)
          |  s2_success.py + s2_official_eval.py   official S²-TOMG success
          v
  [2] DIAGNOSIS                      CPU only, no weights needed
          |
          v
      claim records  --------------->  data/results/<model>/<task>/*_details.jsonl.gz  SHIPPED
          |
          |  eval/metrics.py         the single source of truth for every metric
          |  eval/export_stats.py    aggregates to per-(model,task) and per-family
          |  eval/stage_ladder_metrics.py   the origin ladder
          v
  [3] METRICS  ------------------->   data/source_data.xlsx  (21 sheets)              SHIPPED
          |
          |                          [4] MECHANISM PROBES        GPU + weights
          |                          eval/cot_drift.py        perturb the trace, regenerate
          |                          eval/cot_condsent.py     conditional answer entropy
          |                          eval/cot_info_gain.py    information gain of the CoT
          |                          eval/attr_probe.py       gradient x input saliency
          |                          eval/attention_attribution.py  region attention, Dlogp
          |                                  |
          |                                  v
          |                              data/raw/{drift,condsent,gradattr,region}_*.json  SHIPPED
          |                                  |
          |  eval/pull_fullvol.py  <---------+  aggregates the R5 tables
          |  eval/verify_paper_metric.py <---+  re-derives the R2/R3 headline numbers
          v
  [5] FIGURES                        figures/make_nmi_figures.py    CPU, workbook only
                                     -> the six main-display figures
```

Separately, the training branch that produces Chem-R-Faithful:

```
  training/dataset/make_*_parquet.py  ->  training/dataset/{train,test}.parquet   SHIPPED
                                              |
  reward/verification_grounded_reward.py  ------------>|  GRPO with COUPLED=1
  training/verl_patch/  (required)  --------->|  EasyR1 + 4 GPUs, 936 steps
                                              v
                                        Chem-R-Faithful
```

---

## What each piece is

### The detector
| file | what it does |
|---|---|
| `diagnose_hallucination.py` | The 2×2 taxonomy. IR self-contradiction, IO answer invalidity, **ER fabricated reasoning claims**, EO output deviation. Extracts functional-group claims from the trace and checks each against the molecular graph with RDKit. CLI: `--model_output ... --output_dir ...` |
| `diagnose_multitask.py` | Per-task dispatch on top of it — caption-to-molecule, molecule-to-caption, retrosynthesis, the nine S² subtasks, classification. `diagnose_one(task, record)` is the single-record entry point. |
| `s2_success.py`, `s2_official_eval.py` | Official S²-TOMG success criterion, ported verbatim, used as the S² accuracy signal. |
| `prompts.py` | The instruction templates every model saw. |

### Generation
| file | what it does |
|---|---|
| `run_multitask_se.py` | vLLM harness. Samples responses per prompt and writes `output.json`. Needs a GPU and the model weights. |
| `data_loaders.py` | Task registry: which corpus, which split, how a prompt is built. 18 tasks (the twelve reported, plus five MoleculeNet classification tasks and forward reaction prediction, which the paper does not report). Splits are derived in code and deterministic; see `data/CORPORA.md` for the corpora it expects and where they go. |
| `semantic_entropy.py` | Multi-sample sampling and task-aware clustering for the semantic-entropy measure. |

### Analysis
| file | what it does |
|---|---|
| `eval/metrics.py` | **Source of truth.** `task_stats(model, task)` returns n, performance, ER, %ER=0 and the rest, recomputed from the released records. |
| `eval/cot_drift.py` | Perturb the trace (corrupt a claim, swap the whole trace, mask or corrupt a drafted SMILES), regenerate the answer, measure flip-to-wrong. |
| `eval/cot_condsent.py`, `eval/cot_info_gain.py` | Metric-free tests: how much does the trace reduce uncertainty about the answer? |
| `eval/attr_probe.py` | Gradient×input saliency of the answer with respect to each trace token, by token type. |
| `eval/attention_attribution.py` | Region attention (input span vs trace span), teacher-forced Δlog p, and the matched-token control. |
| `eval/stage_ladder_metrics.py` | The origin ladder: base → SFT → answer-only GRPO → verification-grounded. |
| `eval/pull_fullvol.py`, `eval/verify_paper_metric.py` | Recompute the R5 and R2/R3 headline numbers from `data/raw/` and print them. They write their own output (`fullvol.txt`, `verify_metric.txt`, `data/token_examples/r5_*.csv`) but never touch anything shipped, so they are safe to run against a fresh clone. |
| `eval/redx_all.py` | Runs the detector again over every released response and compares each record against the shipped one. `--verify` is the reproducibility check: 84 (model, task) pairs, 112,749 records, writes nothing, exits non-zero on any mismatch. `--out DIR` writes fresh records instead; `--models a,b` restricts the run. |
| `eval/data_inventory.py` | Regenerates `data/DATA_INVENTORY.md`: which model was measured on which dataset, at what volume, in which experiment. |
| `io_utils.py` | Reads the released data whether it is gzipped or not, and whether it sits under `se_results/` or `data/responses/`. |

### Figures and training
| file | what it does |
|---|---|
| `figures/make_nmi_figures.py` | `main()` regenerates all six main-display figures from `data/source_data.xlsx` alone. The rendered output is committed alongside it — `figures/*.pdf` are the exact files in the paper, `figures/*.png` are previews. |
| `reward/verification_grounded_reward.py` | The process reward. `0.1·format + 0.4·accuracy + 0.4·(1−hallucination) + 0.2·grounded`, and with `COUPLED=1` **the accuracy term is paid only when ER = 0**. |
| `training/` | Both GRPO configs, the submission scripts as run, the dataset builders, the exact parquets, and the three EasyR1 files that must be patched. See `training/README.md`. |

---

## Data

`data/source_data.xlsx` is the single source for every shipped table — 22 sheets, described in
`data/SOURCE_DATA.md`. Alongside it:

```
data/results/<model>/<task>/*_hallucination_details.jsonl.gz   per-response claim records
data/responses/<model>/<task>/output.json.gz                   the model generations
data/responses/<model>/<task>/metadata.json.gz                 S² constraints (S² tasks only)
data/raw/{drift,condsent,gradattr,region}_<model>.json         mechanism-probe outputs
```

Seven models × twelve task variants. `data/raw/README.md` maps the release display names to the
internal training codenames, `data/DATA_INVENTORY.md` is the full coverage table, and
`data/CORPORA.md` says where to obtain the three benchmark corpora, which are not shipped
here as corpora.

### The fields

**`data/responses/<model>/<task>/output.json.gz`** — one record per prompt, the generation as it
was produced:

| field | meaning |
|---|---|
| `id` | instance id, stable across every artefact below |
| `question` | the prompt the model saw (already instruction-formatted) |
| `gt` | reference answer from the benchmark |
| `answer` | the model's full response, `<think>…</think><answer>…</answer>` |
| `model`, `task` | provenance |

`metadata.json.gz` sits next to it for the nine S² subtasks only: `{id: {constraints…}}` with the
instance's constraint set and source molecule. `s2_success` needs it; without it every S²
performance number is 0.

**`data/results/<model>/<task>/*_hallucination_details.jsonl.gz`** — one record per response, the
detector's verdict:

| field | meaning |
|---|---|
| `pred_smiles`, `pred_valid` | the parsed answer and whether RDKit accepts it |
| `exact_match`, `tanimoto` | answer-level correctness |
| `hallucination_scores` | the 2×2: `IR_self_contradiction`, `IO_structural_invalidity`, `ER_factual_fabrication`, `EO_phantom_structure`, each 0–100 |
| `overall_hallucination_score` | the weighted aggregate |
| `details.ER` | `claimed_fgs`, `verified_fgs`, `fabricated_fgs`, and the `penalties`/`checks` the ER score is computed from |
| `details.IO` | validity and the molecular formula |
| `reasoning_length` | characters inside `<think>` |

**`data/raw/drift_<model>.json`** — the perturbation experiment. `per_example` has one record per
instance with `uid`, `task`, `er`, `base_correct` (was the unperturbed answer right?), `n_draft`
(did the trace draft a SMILES?), and then for each condition a `_drift` flag (did the answer
change at all?) and a `_dperf` (−1 means it turned wrong). Conditions: `syn_cot` synonym control,
`wrong_cot` one claim corrupted, `all_wrong_cot` every claim corrupted, `drop_cot` empty trace,
`swap_cot` another molecule's trace, `mask_draft` and `corrupt_draft` for the drafted SMILES, and
`wrong_input` — the same corruption applied to the *input* instead of the trace, which is the
positive control the paper contrasts against.
**The paper's flip-to-wrong is `dperf == −1` among `base_correct == 1`**, not the `_drift` flag.

**`data/raw/condsent_<model>.json`** — conditional entropy. `per_example` gives `H_realCoT`,
`H_noCoT`, `H_corrCoT`, `H_swapCoT` (answer entropy under each trace condition) and the
information gains `ig_presence`, `ig_content`, `ig_swap`. `per_task` holds the per-task means;
the paper plots the **unweighted mean over tasks**.

**`data/raw/region_<model>.json`** — attention and the causal perturbation. `region_attr` is
full-volume, one record per response: `input` and `trace` attention mass, `draft_copy` (did the
answer copy a SMILES the trace drafted?), plus `task`, `er`, `exact`. `matched` is the
matched-token control and `perturb` the teacher-forced Δlog p, both on cap2mol only, both stored
**per layer** — read index `L//2`, the middle layer, as `eval/export_stats.py` does.

**`data/raw/gradattr_<model>.json`** — gradient×input saliency, aggregated rather than
per-response: `n`, `n_by_stratum` and `skipped_long`/`skipped_fmt` for the volume, then
`trace_saliency_frac` and `token_counts` each keyed by stratum (`all`, `er0`, `erpos`) and within
that by token type — SMILES fragment, functional-group word, position digit, other word,
punctuation, space. Enrichment is the saliency share of a type divided by its share of tokens, so
above 1 means that type carries more answer-sensitivity per token than the average trace token.

**Workbook columns.** `Diagnosis_model_task` is per (model, task): `n`, `perf`, `perf_er0` and
`perf_erpos` (performance on clean vs fabricating traces), `pct_er0` (share of clean traces), the
four taxonomy scores, `overall`, `gc` grounded claims, `cp` claim precision, `length`, `validity`,
`tanimoto`, `semantic_entropy`. `R1_stage_ladder` adds `claims_per_resp`, `perclaim_fab_rate`
(fabricated over verified+fabricated **specific** groups, per task then averaged),
`hedge_rate`, `abstain_rate` and `fab_position` (where in the trace the first fabrication appears,
0 = start, 1 = end). Every other sheet is described in `data/SOURCE_DATA.md`.

Three scripts **rewrite** files in `data/` and need the full evaluation tree, so they refuse to
run unless you set `MOLREHALLU_REGEN=1`: `eval/export_stats.py`, `eval/stage_ladder_metrics.py`,
`eval/build_source_data.py`. Against the released subset they would silently replace the
submitted numbers with incomplete ones.

## What you cannot reproduce from this repository alone

- **Generation and the mechanism probes** need GPUs and the model weights. Which weights you can
  get differs by model:

  | model | weights |
  |---|---|
  | Chem-R | `weidawang/Chem-R-8B` |
  | ChemDFM-R | `OpenDFM/ChemDFM-R` |
  | ether-0 | `futurehouse/ether0` |
  | DeepSeek-R1-Distill | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
  | base-a (pre-SFT) | `meta-llama/Llama-3.1-8B-Instruct` |
  | **Chem-R-Faithful** | `phenixace/Chem-R-Faithful` |
  | **SFT** | `slayertear/llama-3.1-8b-stage2` |

  Every model the mechanism probes accept can now be fetched by id, so each probe is
  re-runnable given the GPUs.
- **The benchmark corpora** are not shipped as corpora; `data_loaders.PATHS` shows the layout
  expected and `data/CORPORA.md` where to get each one. `training/dataset/*.parquet` is the one
  file that does carry corpus content — see `LICENSE`, THIRD-PARTY MATERIAL.
- **The RL run** needs an EasyR1 checkout with `training/verl_patch/` applied. Without that patch
  the reward silently scores every task with the caption-to-molecule verifier.

Everything downstream of the released records — metrics, the workbook, all six figures, the
human-eval numbers — reproduces from this repository with no GPU.

## License
CC BY 4.0 (`SPDX-License-Identifier: CC-BY-4.0`), see `LICENSE`. Model weights are not
redistributed. The benchmark corpora are not shipped as corpora, with one stated exception —
`training/dataset/*.parquet` — and everything upstream remains under its own licence; `LICENSE`,
THIRD-PARTY MATERIAL, spells this out.
