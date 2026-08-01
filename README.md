# MolReHallu

Structure-grounded study of **reasoning hallucination in chemical reasoning models**:
a chemical chain-of-thought (CoT) states functional-group claims that can be verified
directly against the molecular graph, so its rationale can be audited claim by claim.
This repo holds the essential code and the data results behind the figures.

## Layout
- `diagnose_hallucination.py`, `diagnose_multitask.py`, `s2_success.py` — the detector
  (2×2 IR/IO/ER/EO taxonomy; **ER** = extrinsic reasoning fabrication) and per-task wrappers.
- `run_multitask_se.py` — vLLM generation harness (needs `data_loaders.py` /
  `semantic_entropy.py`, which are not part of this release; kept here for reference).
- `eval/` — `metrics.py` (metric source of truth) plus the mechanism analyses: drift
  (`cot_drift.py`), conditional entropy (`cot_info_gain.py`, `cot_condsent.py`),
  gradient/attention attribution (`attention_attribution.py`, `attr_probe.py`),
  stage ladder, and `build_source_data.py` (assembles `data/source_data.xlsx`; needs the full
  diagnosed evaluation tree, so it is gated behind `MOLREHALLU_REGEN=1` — see `data/SOURCE_DATA.md`).
- `reward/chem_merged_v8_ours.py` — process reward for Chem-R-Faithful (format + accuracy +
  anti-hallucination + grounded; the **accuracy term is gated on ER=0**). Plugs into EasyR1/verl.
- `training/` — the GRPO configs, submission scripts, dataset builders and the exact train/test
  parquets for both runs, plus the three EasyR1 files that must be patched for the reward to be
  dispatched per task at all. See `training/README.md`.
- `figures/make_nmi_figures.py` (+ `nature_figures.py`) — regenerates the main figures from
  `data/`.  Reproduce:  `cd figures && python -c "import make_nmi_figures as m; m.main()"`.
- `human_eval/` — per-claim reliability set builder, blind-annotation scorer, arena, κ.
- `data/` — `source_data.xlsx` and CSVs (the plotted series), `SOURCE_DATA.md` (the bundle's
  own manifest), `DATA_INVENTORY.md` (which model was measured on which dataset, at what
  volume, in which experiment), `raw/` (the per-model JSONs the R2/R3/R5 sheets are computed
  from), `results/` (per-response diagnosis records) and `responses/` (the model generations,
  both gzipped — `io_utils.py` reads them transparently).

## Reproducing
`pip install -r requirements.txt` covers everything; the figure path alone needs only
`matplotlib numpy pandas openpyxl pillow`. Two things are verifiable end-to-end on CPU with no
data and no model weights: the detector (`diagnose_multitask.diagnose_one`) and the ER-gated
reward (`reward/chem_merged_v8_ours.compute_score`, with `COUPLED=1`). The mechanism probes and
the RL run are not re-runnable from this release — they need the full generation tree and, for
the three trained arms, weights released separately.

## License
CC BY 4.0 (`SPDX-License-Identifier: CC-BY-4.0`) — see `LICENSE`. Share and adapt freely, including
commercially, with attribution. Upstream benchmark corpora and model weights are not redistributed
here and remain under their own licenses.

## Data version
All numbers use the current (V8) detector. Chem-R cap2mol ER = 6.66; the decoupling
figure/stats are on V8 (means 6.4 vs 6.8; 13% of responses right-yet-fabricating,
Pearson ≈ −0.02). Superseded plotting scripts and legacy diagnosis copies are
intentionally excluded.
