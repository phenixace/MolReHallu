# MolReHallu

Reasoning hallucination in chemical reasoning models, made structurally checkable. A chemical
chain-of-thought states functional-group claims that can be decided against the molecular graph,
so the rationale can be audited claim by claim rather than judged as fluent text.

This repository holds the detector, the analysis pipeline, the verification-grounded training
recipe, and the data behind every figure and number in the paper.

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
from chem_merged_v8_ours import compute_score
A='CC(=O)Oc1ccccc1C(=O)O'
s=compute_score(['<think>It contains an azide.</think><answer>%s</answer>'%A],[A],['cap2mol'],['Description: aspirin'])[0]
print(s['accuracy'], s['accuracy_raw'])
# -> 0.0 1.0    the answer is right, so accuracy_raw is 1.0, but the gate pays nothing
"
```

---

## The pipeline

```
  benchmark corpora                    ChEBI-20 · USPTO-50k · S²-Bench · MoleculeNet
  (not redistributed)                  cited in Methods, fetched from their own sources
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
  reward/chem_merged_v8_ours.py  ------------>|  GRPO with COUPLED=1
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
| `data_loaders.py` | Task registry: which corpus, which split, how a prompt is built. 18 tasks. |
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
| `eval/data_inventory.py` | Regenerates `data/DATA_INVENTORY.md`: which model was measured on which dataset, at what volume, in which experiment. |
| `io_utils.py` | Reads the released data whether it is gzipped or not, and whether it sits under `se_results/` or `data/responses/`. |

### Figures and training
| file | what it does |
|---|---|
| `figures/make_nmi_figures.py` | `main()` regenerates all six main-display figures from `data/source_data.xlsx` alone. The rendered PDFs are not stored here; they are in the paper. |
| `reward/chem_merged_v8_ours.py` | The process reward. `0.1·format + 0.4·accuracy + 0.4·(1−hallucination) + 0.2·grounded`, and with `COUPLED=1` **the accuracy term is paid only when ER = 0**. |
| `training/` | Both GRPO configs, the submission scripts as run, the dataset builders, the exact parquets, and the three EasyR1 files that must be patched. See `training/README.md`. |
| `human_eval/` | The chemist validation: per-claim audit (`human_eval/score_claims.py` reproduces the 97.3% extraction precision) and the blind forced-choice arena. |

---

## Data

`data/source_data.xlsx` is the single source for every shipped table — 21 sheets, described in
`data/SOURCE_DATA.md`. Alongside it:

```
data/results/<model>/<task>/*_hallucination_details.jsonl.gz   per-response claim records
data/responses/<model>/<task>/output.json.gz                   the model generations
data/responses/<model>/<task>/metadata.json.gz                 S² constraints (S² tasks only)
data/raw/{drift,condsent,gradattr,region}_<model>.json         mechanism-probe outputs
```

Eight models × twelve task variants. `data/raw/README.md` maps the release display names to the
internal training codenames. `data/DATA_INVENTORY.md` is the full coverage table.

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
  | **Chem-R-Faithful** | released with the paper |
  | **SFT** | not released — internal checkpoint |
  | **+process** | not released — an ablation arm, not reported in the paper |

  So the five public models and Chem-R-Faithful can be re-probed; the SFT and `+process` rungs
  cannot. Their measured outputs are still here (`data/raw/`, `data/results/`,
  `data/responses/`), so every number that depends on them is auditable even though the probe
  cannot be re-run.
- **The benchmark corpora** are not redistributed. `data_loaders.PATHS` shows the layout expected.
- **The RL run** needs an EasyR1 checkout with `training/verl_patch/` applied. Without that patch
  the reward silently scores every task with the caption-to-molecule verifier.

Everything downstream of the released records — metrics, the workbook, all six figures, the
human-eval numbers — reproduces from this repository with no GPU.

## License
CC BY 4.0 (`SPDX-License-Identifier: CC-BY-4.0`), see `LICENSE`. Upstream corpora and model
weights are not redistributed and remain under their own licenses.
