# MolReHallu

Structure-grounded study of **reasoning hallucination in chemical reasoning models**:
a chemical chain-of-thought (CoT) states functional-group claims that can be verified
directly against the molecular graph, so its rationale can be audited claim by claim.
This repo holds the essential code and the data results behind the figures.

## Layout
- `diagnose_hallucination.py`, `diagnose_multitask.py`, `s2_success.py` — the detector
  (2×2 IR/IO/ER/EO taxonomy; **ER** = extrinsic reasoning fabrication) and per-task wrappers.
- `run_multitask_se.py` — vLLM generation harness.
- `eval/` — `metrics.py` (metric source of truth) plus the mechanism analyses: drift
  (`cot_drift.py`), conditional entropy (`cot_info_gain.py`, `cot_condsent.py`),
  gradient/attention attribution (`attention_attribution.py`, `attr_probe.py`),
  stage ladder, and `build_source_data.py` (rebuilds `data/source_data.xlsx`).
- `reward/chem_merged_v8_ours.py` — process reward for Chem-R-Faithful (format + accuracy +
  anti-hallucination + grounded; the **accuracy term is gated on ER=0**). Plugs into EasyR1/verl.
- `figures/make_nmi_figures.py` (+ `nature_figures.py`) — regenerates the main figures from
  `data/`.  Reproduce:  `cd figures && python -c "import make_nmi_figures as m; m.main()"`.
- `human_eval/` — per-claim reliability set builder, blind-annotation scorer, arena, κ.
- `data/` — `source_data.xlsx` and CSVs (the plotted series), `RESULTS.md` (cross-checked
  numbers), `SOURCE_DATA.md`/`EXPERIMENTS.md`, and `data/results/…` diagnosis details used
  by the figures.

## Data version
All numbers use the current (V8) detector. Chem-R cap2mol ER = 6.66; the decoupling
figure/stats are on V8 (means 6.4 vs 6.8; 13% of responses right-yet-fabricating,
Pearson ≈ −0.02). Superseded plotting scripts and legacy diagnosis copies are
intentionally excluded.
