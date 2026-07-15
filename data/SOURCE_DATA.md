# Source data bundle (download this whole `data/` folder)

Everything needed to reproduce every figure/table is in this folder (~73 MB). Verified 2026-07-04.

## Top-level deliverable
- **`source_data.xlsx`** — NMI-style Source Data workbook, one sheet per figure/result (12 sheets).
  Sheet `README` inside it lists every sheet + metric definitions + N + regenerate command.

## Layout
```
data/
  source_data.xlsx            # << the workbook (all figures, editable)
  RESULTS.md                  # verified results narrative (R1–R5) + caveats
  EXPERIMENTS.md STORYLINE.md DATA_STATUS.md   # planning / manifest
  stats_per_model_task.csv    # diagnosis: per (model,task)
  stats_per_family.csv        # diagnosis: per (model,family)
  stage_ladder.csv            # R1 origin-of-hallucination ladder
  mitigation.csv              # R4 per-family mitigation
  attention_perturbation.csv  # legacy Δlogp perturbation (superseded by raw/region_*)
  human_eval_agreement.json   # detector↔chemist agreement (n=400)
  token_examples/             # R5 per-token heatmap bundle
    token_examples.html       #   open in a browser — colored token heatmaps (12 examples)
    token_examples.json       #   per-token sal/attn (machine-readable)
    token_examples.csv        #   flat, one row per token (edit in Excel)
    r5_gradient.csv           #   R5 gradient enrichment (model × stratum × token-type)
    r5_region.csv             #   R5 region attention (model × stratum × region)
  raw/                        # raw per-model JSONs backing R2/R3/R5 (see raw/README.md)
```

## Figure → workbook sheet → raw file
| result | sheet in source_data.xlsx | raw source |
|---|---|---|
| Diagnosis (perf/ER/2×2) | `Diagnosis_model_task`, `Diagnosis_family` | `stats_per_*.csv` |
| R1 origin ladder | `R1_stage_ladder` | `stage_ladder.csv` |
| R2 FG-claim drift (flip-to-wrong among originally-correct) | `R2_drift`, `R2_drift_by_task` | `raw/drift_<model>.json` (`per_example`) |
| R2b draft-SMILES perturbation (mask/corrupt vs FG-name/swap; direct draft-channel test) | `R2_draft_perturbation` | `raw/drift_<model>.json` (`mask_draft`/`corrupt_draft`, `n_draft`) |
| R3 metric-free entropy (ig_presence/content/swap) | `R3_condentropy`, `R3_condentropy_task` | `raw/condsent_<model>.json` |
| R4 mitigation | `R4_mitigation` | `mitigation.csv` |
| R5 gradient saliency by token type | `R5_grad_enrichment` | `raw/gradattr_<model>.json` |
| R5 region attention + draft-copy | `R5_region_attention` | `raw/region_<model>.json` (`region_attr`) |
| R5 per-token heatmap examples | `R5_token_examples` | `token_examples/token_examples.json` |

## Two metric notes that matter (matched to the paper)
- **R2 flip_to_wrong** = among originally-correct responses (`base_correct=1`), fraction that turn
  incorrect (`dperf=-1`) after perturbing the reasoning. This is NOT the JSON `summary.drift_rate`
  (which is *any* answer change over *all* examples — a different, larger quantity). Recomputed from
  `per_example`; reproduces RESULTS.md R2 (e.g. Chem-R translate all_wrong 7.1%, drop 25.8%, swap 32.7%).
- **R3** plots the **mean-over-tasks** aggregation of the info-gains (not the pooled-over-all-examples
  value). The workbook gives both, clearly labeled, plus per-task granular rows.

## Regenerate
```
PY=python
$PY eval/export_stats.py            # diagnosis + mitigation CSVs
$PY eval/stage_ladder_metrics.py    # R1
# R2/R3 raw:  qsub -v MODEL=<m> jobs/eval/{drift_one,condsent_one}.sh
# R5 raw:     qsub -v MODEL=<m> jobs/eval/{gradattr_one,attn_regions}.sh ; then eval/pull_fullvol.py
$PY eval/token_examples.py --model <m> ; $PY eval/token_examples_render.py   # R5 heatmaps
$PY eval/build_source_data.py       # assembles source_data.xlsx from all of the above
```
