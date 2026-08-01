# data/raw/ — per-model raw JSONs backing the source-data sheets

Each file backs a sheet in `../source_data.xlsx` (see `../SOURCE_DATA.md`). Filenames use the
release **display** name; the internal training codename is given only for provenance.

| display token (filename) | sheet label (`model` column) | internal codename |
|---|---|---|
| `Chem-R` | Chem-R | `Chem-R` |
| `Chem-R-Faithful` | Chem-R-Faithful | `Chem-R-v8-coupled` |
| `ChemDFM-R` | ChemDFM-R | `ChemDFM-R-14B` |
| `ether-0` | ether-0 | `ether-0` |
| `process` | +process | `Chem-R-v8` |
| `SFT` | SFT | `Chem-R-SFT` |
| `base-a` | base-a (pre-SFT) | `Llama-3.1-8B-Instruct-base` |

| file pattern | backs sheet | probe script |
|---|---|---|
| `gradattr_<model>.json` | `R5_grad_enrichment` | `eval/attr_probe.py` |
| `region_<model>.json` (`region_attr`) | `R5_region_attention` | `eval/attention_attribution.py` |
| `drift_<model>.json` (`per_example`, `mask_draft`/`corrupt_draft`) | `R2_drift`, `R2_drift_by_task`, `R2_draft_perturbation` | `eval/cot_drift.py` |
| `condsent_<model>.json` | `R3_condentropy`, `R3_condentropy_task` | `eval/cot_condsent.py` |

Coverage: R5 (gradattr/region) = 4 mechanism models; R2/R3 (drift/condsent) = 7 models
(mechanism + origin-ladder: base-a, SFT, +process).

## `+process` is retained here but is not reported in the paper

`+process` (internal `Chem-R-v8`) is the ablation of the released training recipe with the
accuracy gate switched off — the same process reward as Chem-R-Faithful, but paying the accuracy
term whether or not the trace is clean. It is what isolates the gate as the active ingredient:
the two runs differ in `COUPLED` and in nothing else (`../../training/configs/`).

It appears in the shipped statistics (`Diagnosis_model_task`, `Diagnosis_family`, `R2_drift`,
`R2_drift_by_task`, `R3_condentropy*`, `mitigation.csv`, `stage_ladder.csv`) and has its own
`drift_process.json` and `condsent_process.json` here, but **no figure or number in the
manuscript is computed from it**, and it is not plotted anywhere. It is kept because it is a real
arm of the study and because removing it would leave `mitigation.csv` — whose entire structure is
baseline vs `+process` vs `+coupled` — without its middle column.

Note that `+process` is *not* the retired process-reward variant from the earlier detector
generation; that arm is not part of this release at all.
