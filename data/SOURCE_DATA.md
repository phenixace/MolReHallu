# Source data bundle (download this whole `data/` folder)

Everything needed to reproduce every **figure** is in this folder (~84 MB). Verified 2026-07-30:
all plotted figures regenerate with byte-identical drawing operators from this folder alone
(see `../figures/`).

## Top-level deliverable
- **`source_data.xlsx`** — NMI-style Source Data workbook, **18 sheets**: 8 back a plotted figure
  panel, the rest are supporting data behind statements in the text, plus a `README` index sheet.
  Sheet `README` inside it lists every sheet + metric definitions + N.

## Layout
```
data/
  source_data.xlsx            # << the single source: every shipped table, 21 sheets
  DATA_INVENTORY.md           # model x dataset x experiment coverage, with volumes
  results/                    # per-response diagnosis records, 8 models x 12 tasks (.jsonl.gz)
  responses/                  # the model generations themselves, same grid (.json.gz)
  raw/                        # per-model JSONs backing R2/R3/R5 (see raw/README.md)
```

The loose CSVs that used to sit here (`stats_per_model_task.csv`, `stats_per_family.csv`,
`stage_ladder.csv`, `mitigation.csv`, `attention_perturbation.csv`, `human_eval_agreement.json`)
were duplicates of workbook sheets and have been folded in: keeping the same numbers in two
places is how the semantic-entropy column once drifted between the CSV and the sheet. The
regeneration scripts still write them as intermediates when rebuilding from the full evaluation
tree; they are simply not shipped.

## Figure -> workbook sheet -> raw file
| result | sheet in source_data.xlsx | raw source in this folder |
|---|---|---|
| Diagnosis (perf/ER/2x2) | `Diagnosis_model_task`, `Diagnosis_family` | terminal (rebuilt from `results/`) |
| R1 origin ladder | `R1_stage_ladder` | terminal (rebuilt from `results/` + `responses/`) |
| R2 FG-claim drift (flip-to-wrong among originally-correct) | `R2_drift`, `R2_drift_by_task` | `raw/drift_<model>.json` (`per_example`) |
| R2b draft-SMILES perturbation (mask/corrupt vs FG-name/swap) | `R2_draft_perturbation` | `raw/drift_<model>.json` (`mask_draft`/`corrupt_draft`, `n_draft`) |
| R3 metric-free entropy (ig_presence/content/swap) | `R3_condentropy`, `R3_condentropy_task` | `raw/condsent_<model>.json` |
| R4 mitigation | `R4_mitigation` | a re-slice of `Diagnosis_family` |
| R5 gradient saliency by token type | `R5_grad_enrichment` | `raw/gradattr_<model>.json` |
| R5 region attention + draft-copy | `R5_region_attention` | `raw/region_<model>.json` (`region_attr`) |
| R5 per-token heatmap examples | `R5_token_examples` | terminal — the sheet is the per-token data; the rendered heatmap bundle is an intermediate and is not shipped |
| Fig 4a,c causal perturbation | `Fig4_attention_perturbation` | `raw/region_<model>.json` (`perturb`, `matched`) |

Every sheet in the first eight rows above has been **recomputed from `raw/` and matches the shipped
values exactly** (max deviation 0.0). `attention_perturbation.csv` likewise reproduces for the four
models that have a `raw/region_*.json`.

### Sheets with no producer in this repo (terminal data)
`R5_token_examples`, `R2_restated_derived`, `R2b_flip_by_draftcopy`, `Fig3_decoupling_perresponse`,
`R2_paired_flips`, `Fig1_hallucination_by_model`, and the four `Diagnosis_*`/`R1`/`R4` tables ship as
final values. They are correct and self-consistent, but regenerating them needs the full diagnosed
evaluation tree (`results/`, `se_results/`), which is not part of this release.

## Two metric notes that matter (matched to the paper)
- **R2 flip_to_wrong** = among originally-correct responses (`base_correct=1`), fraction that turn
  incorrect (`dperf=-1`) after perturbing the reasoning. This is NOT the JSON `summary.drift_rate`
  (which is *any* answer change over *all* examples — a different, larger quantity). Recomputed from
  `per_example` (e.g. Chem-R translate all_wrong 7.1%, drop 25.8%, swap 32.7%).
- **R3** plots the **mean-over-tasks** aggregation of the info-gains (not the pooled-over-all-examples
  value). The workbook gives both, clearly labeled, plus per-task granular rows.
- **Fig 4a,c** use the **middle** transformer layer (`_mid`) and the `cap2mol` subset only. Averaging
  over all layers instead gives a materially different ratio and will not match the published number.

## Regenerate
```
PY=python

# From raw/ alone — safe, no extra inputs, reproduces the shipped files byte-identically:
$PY eval/pull_fullvol.py            # recomputes the R5 tables from raw/ and prints them
$PY eval/verify_paper_metric.py     # re-derives the R2/R3 headline numbers and prints them

# Figures — reproduces the six shipped main-display PDFs/PNGs from source_data.xlsx:
cd figures && $PY -c "import make_nmi_figures as m; m.main()"
```
The three scripts that **rewrite** files in this folder (`eval/export_stats.py`,
`eval/stage_ladder_metrics.py`, `eval/build_source_data.py`) require the full diagnosed evaluation
tree and therefore refuse to run by default: against this release's subset they would silently
replace the submitted numbers with incomplete ones. They are gated behind `MOLREHALLU_REGEN=1` for
anyone who has rebuilt that tree.
