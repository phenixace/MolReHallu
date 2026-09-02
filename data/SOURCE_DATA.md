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

### Human validation
`Human_eval_agreement` holds the detector-versus-chemist forced-choice agreement (n=400) and
`Human_eval_extraction` the per-claim extraction audit — 300 claims, 292 confirmed, the 97.3%
pooled precision quoted in the Limitations. Only these two tables ship: the annotation records and
the scoring scripts are not part of the release.

### Sheets with no producer in this repo (terminal data)
`R5_token_examples`, `R2_restated_derived`, `R2b_flip_by_draftcopy` and `R2_paired_flips` ship as
final values: regenerating them needs the GPU probes that produced them.

Everything else recomputes from the released data alone, and does:

| sheet | recomputed from | result |
|---|---|---|
| `Diagnosis_model_task` | `eval/metrics.py` over `data/results/` | 840/840 cells |
| `Diagnosis_family` | same | 260/260 cells |
| `R1_stage_ladder` | `eval/stage_ladder_metrics.py` | 40/40 cells |
| `R4_mitigation` | `eval/metrics.py`, per family | 48/48 cells |
| `Fig1_hallucination_by_model` | `Diagnosis_model_task`, mean over the twelve tasks; `overall` is the weighted class sum | 50/50 cells |
| `Fig3_decoupling_perresponse` | `data/results/`, per response | 11,607/11,607 records |
| `R2_drift`, `R2_drift_by_task` | `data/raw/drift_<model>.json` | 60/60 and 190/190 rows |
| `R3_condentropy`, `R3_condentropy_task` | `data/raw/condsent_<model>.json` | 106/106 and 218/218 cells |
| `R5_grad_enrichment`, `R5_region_attention` | `data/raw/{gradattr,region}_<model>.json` | 288/288 and 120/120 cells |

The `wrong_input` rows of `R2_drift` are computed separately and are not in `per_example`, so they
are outside that check. `R3_condentropy`'s agreement means the sheet matches the probe output it
came from; the ER strata inside that probe output are the stale ones noted below.

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

### Correction, 2026-09-02: `R1_stage_ladder`, base-a and SFT rows

The S² diagnosis records for `SFT` and `base-a` were originally produced without the S²
instance metadata. Without it the detector cannot run the input-grounding half of the ER
check — a claim that names a group present in the source molecule was scored as
fabrication — and EO collapsed onto two fallback values. The five other models were
unaffected: their records were regenerated by the full re-diagnosis that followed the
2026-06-26 input-grounding fix, whereas these two lineage rungs were added afterwards and
scored by the per-run path, which reads only `output.json`.

Those 18 files (2 models × 9 S² tasks) have been re-diagnosed with the metadata present
and replaced. `perf`, `n_resp`, `claims_per_resp`, `hedge_rate` and `abstain_rate` are
unchanged, as are all four `Chem-R` and `Chem-R-Faithful` values. What moved:

| row | ER | %ER=0 | per-claim fabrication |
|---|---|---|---|
| base-a (pre-SFT) | 16.88 -> **12.94** | 44.9 -> **51.4** | 51.8% -> **34.7%** |
| SFT | 11.61 -> **10.26** | 38.3 -> **42.5** | 28.2% -> **22.1%** |

`eval/redx_all.py --verify` now reproduces every shipped record exactly. One sheet is
knowingly left stale: the `pooled_ER=0` / `pooled_ER>0` rows of `R3_condentropy` for
`SFT` stratify by ER and so shifted with it. They are supporting data — neither plotted
nor cited — and regenerating them needs the GPU conditional-entropy probe.
