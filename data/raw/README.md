# data/raw/ — per-model raw JSONs backing the source-data sheets

Each file backs a sheet in `../source_data.xlsx` (see `../SOURCE_DATA.md`). Filenames use the
release **display** name; the internal training codename is given only for provenance.

| display token (filename) | sheet label (`model` column) | internal codename |
|---|---|---|
| `Chem-R` | Chem-R | `Chem-R` |
| `Chem-R-Faithful` | Chem-R-Faithful | `Chem-R-v8-coupled` |
| `ChemDFM-R` | ChemDFM-R | `ChemDFM-R-14B` |
| `ether-0` | ether-0 | `ether-0` |
| `SFT` | SFT | `Chem-R-SFT` |
| `base-a` | base-a (pre-SFT) | `Llama-3.1-8B-Instruct-base` |

| file pattern | backs sheet | probe script |
|---|---|---|
| `gradattr_<model>.json` | `R5_grad_enrichment` | `eval/attr_probe.py` |
| `region_<model>.json` (`region_attr`) | `R5_region_attention` | `eval/attention_attribution.py` |
| `drift_<model>.json` (`per_example`, `mask_draft`/`corrupt_draft`) | `R2_drift`, `R2_drift_by_task`, `R2_draft_perturbation` | `eval/cot_drift.py` |
| `condsent_<model>.json` | `R3_condentropy`, `R3_condentropy_task` | `eval/cot_condsent.py` |

Coverage: R5 (gradattr/region) = 4 mechanism models; R2/R3 (drift/condsent) = 7 models
(mechanism + origin-ladder: base-a, SFT).

