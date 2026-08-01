# IMPLEMENTATION — paper result → code map

Companion to the manuscript at
`/nfs/home/svu/t0937992/MolLM_R_Hallucination_v1/paper/MolHallu/nmi/sections/{results,methods,limitations}.tex`.

Every claim below is traced to the code that actually computes it. Constants and formulas are
read out of the source, not out of the paper; where the two disagree, the disagreement is stated
explicitly (§12) rather than smoothed over. `file.py:N` is always relative to the release root
`/nfs/home/svu/t0937992/MolReHallu/` unless the path starts with `MolLM_R_Hallucination_v1/`,
which means the private working repo `/nfs/home/svu/t0937992/MolLM_R_Hallucination_v1/`.

Line numbers are as of the files' state on 2026-08-02. (`figures/make_nmi_figures.py` was edited
at 00:11 on 2026-08-02, after most of this document was drafted; every citation to it has been
re-verified against the current 806-line file. Only the region after line 785 moved — see
§12.5.)

Python used throughout: `/scratch/t0937992/envs/se_vllm/bin/python` (rdkit + nltk + vllm +
transformers + openpyxl). Scratch to `/scratch/t0937992/tmp`, never `/tmp`.

---

## 0. One-page map

| Paper claim / panel | Script | Function (lines) | Lands in |
|---|---|---|---|
| Fig 1b framework, ER definition | `diagnose_hallucination.py` | `score_er_factual_fabrication` 666–747 | `data/results/<m>/<t>/*_hallucination_details.jsonl` → `details.ER` |
| Fig 1b IR / IO / EO | `diagnose_hallucination.py` | 489–544 / 549–661 / 752–802 | same JSONL → `hallucination_scores` |
| Fig 1b aggregate score (+ renormalization) | `diagnose_hallucination.py`, `diagnose_multitask.py` | 807–812 & 867–897; `_aggregate` 73–81 | `overall_hallucination_score` |
| Fig 1c worked example | hand-drawn (`figures/fig1_framework_white.pdf`) | — | not code-generated |
| Fig 1d hallucination vs performance | `figures/make_nmi_figures.py` | `fig1_measures` 254–306 | `figures/fig1_measures.pdf`; sheet `Diagnosis_model_task` |
| Fig 1e IR/IO/ER/EO composition | `figures/make_nmi_figures.py` | `fig1_measures` 289–305 | same |
| Fig 2a ER heatmap, 5 models × 12 tasks | `figures/make_nmi_figures.py` | `fig2_widespread` 146–169 | `figures/fig2_widespread.pdf`; sheet `Diagnosis_model_task` |
| Fig 2b semantic entropy vs overall | `MolLM_R_Hallucination_v1/semantic_entropy.py` → `eval/export_stats.py:44–51` | `sample_completions` 418–444, `cluster_by_similarity` 159–181, `entropy_from_clusters` 188–197; `se()` | `se_summary.json` → `data/stats_per_model_task.csv` col `semantic_entropy` |
| prompt construction (all tasks) | `prompts.py` | `build_messages` 163–169, `PROMPTS` 141–160 | consumed by every generation/probe script |
| test-set construction | `MolLM_R_Hallucination_v1/data_loaders.py` | `PATHS` 35–42, `TASK_REGISTRY` 366–382, `load_s2bench` 333–359 | `se_results/<m>/<t>/{completions,output}.json` |
| Fig 2c per-claim fabrication rate | `eval/metrics.py` | `task_stats` 102–140 (`cp`, lines 124–125, 135) | `stats_per_model_task.csv` col `cp` |
| Fig 2d length vs ER | `eval/metrics.py` | `task_stats` (`length`, line 118/137) | `stats_per_model_task.csv` col `length` |
| Fig 2e / ED Fig 1 training-stage ladder | `eval/stage_ladder_metrics.py` | `mechanism_stats` 103–141 | `data/stage_ladder.csv` → sheet `R1_stage_ladder` |
| Fig 3a ER distribution, correct vs wrong | `figures/make_nmi_figures.py` | `_chemr_cap2mol` 312–317, `fig3_accuracy_gap` 320–351 | reads `data/results/Chem-R/cap2mol/*.jsonl` directly |
| Fig 3b joint mosaic | `figures/make_nmi_figures.py` | `fig3_accuracy_gap` 355–381 | same JSONL |
| Fig 3c perf(ER=0) vs perf(ER>0) | `eval/metrics.py` + `eval/export_stats.py` | `family_stats` 143–157; export 70–83 | `data/stats_per_family.csv` → sheet `Diagnosis_family` |
| Fig 3d clean-trace rate | same | `pct_er0` (`metrics.py:132`) | same |
| Fig 4a teacher-forced Δlogp | `eval/attention_attribution.py` | `perturb` 179–241, `answer_logprob` 153–164 | `data/raw/region_<m>.json:perturb` → `data/attention_perturbation.csv` |
| Fig 4b behavioural flip-to-wrong | `eval/cot_drift.py` | `main` 265–425 | `data/raw/drift_<m>.json:per_example` → sheet `R2_drift` |
| Fig 4c matched-token attention | `eval/attention_attribution.py` | `matched_attention` 292–358 | `region_<m>.json:matched` → `attention_perturbation.csv` |
| restated vs derived Δlogp (text, §res-mechanism) | `eval/attention_attribution.py` | `perturb` 195–208 (`syn_in` present/absent) | sheet `R2_restated_derived` (**no producer in release**) |
| Fig 5a token-saliency strips | `eval/token_examples.py` | `analyze` 30–91 | `data/token_examples/token_examples.json/.csv` → sheet `R5_token_examples` |
| Fig 5b gradient×input enrichment | `eval/attr_probe.py` + `eval/pull_fullvol.py` | `main` 30–114; pull 23–34, 73–81 | `data/raw/gradattr_<m>.json` → `data/token_examples/r5_gradient.csv` → sheet `R5_grad_enrichment` |
| Fig 5c within-trace attention | `eval/attention_attribution.py --regions_only` | `region_attr` 410–455 | `region_<m>.json:region_attr` → `r5_region.csv` → sheet `R5_region_attention` |
| Fig 6a,b,c draft perturbation | `eval/cot_drift.py` + `eval/pull_draft.py` | `build_conditions` 202–262; `pull_draft.flip` 18–22 | `drift_<m>.json` → sheet `R2_draft_perturbation` |
| Fig 6d draft-copy rate | `eval/attention_attribution.py` | `region_attr` 452–453 | `r5_region.csv` col `draft_copy` |
| Fig 6e partial-only vs early-answer | — | **no producer in release** | sheet `R2b_flip_by_draftcopy` |
| R3 conditional entropy (text §res-draft) | `eval/cot_condsent.py` | `main` 31–147, `entropy` 25–28 | `data/raw/condsent_<m>.json` → sheet `R3_condentropy` |
| R4 mitigation (text, summary ¶) | `eval/export_stats.py` | 85–97 | `data/mitigation.csv` → sheet `R4_mitigation` |
| Verification-grounded reward | `reward/chem_merged_v8_ours.py` | `compute_score` 108–164 (gate at 153) | consumed by EasyR1/verl (not in release) |
| GRPO run that made Chem-R-Faithful | `MolLM_R_Hallucination_v1/jobs/training/easyr1_merged_v8_coupled.sh` + `EasyR1-main/examples/config_merged_v8_coupled.yaml` | — | `global_step_936` checkpoint (released separately) |
| ED Fig 2 human validation | `MolLM_R_Hallucination_v1/make_latex_tables.py` | `_human_agreement` 728–770 | `data/human_eval_agreement.json` |
| Claim-level extraction precision (Limitations, 97.3%) | `human_eval/build_claim_set.py` + `human_eval/score_claims.py` | `main` 107–174; `main` 35–82 | `human_eval/claim_reliability.json` |

---

## 1. The measurement framework (2×2 IR/IO/ER/EO, claim extraction, grounding)

### Claim
`results.tex:18–23`

> "We anchor the analysis on functional-group-level claims, which are frequent in chemical
> rationales and reliably decidable by exact RDKit SMARTS matching. Our molecular
> claim-grounding framework parses a response into a reasoning trace and a final answer,
> extracts the chemical entities and transformations asserted in the trace, and checks each
> against the task-relevant structures (Fig. 1). … The most diagnostic cell is *extrinsic
> reasoning fabrication* (ER): a functional group asserted in the trace that is absent from the
> input, the predicted molecule and the reference alike."

`results.tex:20`: "The framework separates two axes … whether an error appears in the reasoning
or in the output, and whether a claim contradicts an external reference or the model's own
generated content."

### Entry point

- Single response, cap2mol: `diagnose_hallucination.py:815–905` `diagnose_single()`.
- Single response, every other task: `diagnose_multitask.py:675–694` `diagnose_one()`
  (dispatch), which delegates to `diagnose_cap2mol` (100–101, a thin wrapper on
  `diagnose_single`), `diagnose_mol2cap` (116–178), `diagnose_classification` (185–243),
  `diagnose_retrosynthesis` (261–343), `diagnose_reaction_prediction` (346–429),
  `diagnose_s2` (613–668).
- Batch over a whole `(model, task)`: `diagnose_multitask.py:701–783` `evaluate_file()`, or
  the parallel re-diagnosis driver `eval/redx_all.py:35–101` `run_pair()` /
  `eval/redx_all.py:104–131` `main()`.

### Algorithm

**(a) Response parsing.** `diagnose_hallucination.py:238–252` `extract_answer_region()` reads
ether-0's native `<|answer_start|>…<|answer_end|>` **first** (line 246), and only falls back to
`<answer>…</answer>`, taking the **last** pair (`_STD_ANS_RE.findall(text)[-1]`, line 249–251),
so a stray `<answer>` mentioned inside the reasoning does not win. `extract_think()`
(255–267) mirrors it, and if there are no think tags at all it takes everything before the
first answer marker as the trace (264–266). SELFIES answers are detected by
`_SELFIES_RE` (270) and decoded (`_selfies_to_smiles`, 286–296).

**(b) Claim extraction.** `extract_chemical_entities()` (`diagnose_hallucination.py:422–442`)
loops over three libraries:

- `FUNCTIONAL_GROUP_DB` (74–187) — **99 entries**, each `name -> (SMARTS, [synonyms])`.
- `MOLECULAR_CLASS_HINTS` (194–212) — **9 classes** (steroid, sugar, amino_acid, nucleotide,
  fatty_acid, alkaloid, peptide, terpene, flavonoid) with property predicates
  (`min_rings`, `min_carbons`, `requires_N/O/P`).
- inline element patterns (431–438) — nitrogen/oxygen/sulfur/phosphorus/F/Cl/Br. **These are
  extracted but never scored** by any dimension; only `functional_groups` and `mol_classes`
  reach IR/IO/ER/EO.

Matching is whole-word with an optional plural, via a compiled cached pattern
(`_kw_pattern`, 382–387):

```python
pat = re.compile(r"(?<![a-z])" + re.escape(kw.lower()) + r"(?:e?s)?(?![a-z])")
```

This is what stops `nitro` matching inside `nitrogen` (docstring 391–393).

**(c) Derivation context.** `_fg_claimed()` (411–419) with `exclude_derivation=True` keeps a
group only if it has at least one occurrence **outside** a derivation clause. The clause
detector is `_DERIV_RE` (402–407) — `derived from | obtained (from|by) | condens | replac |
convert | precursor | starting material | reactant | formed (by|from) | comes from | originat |
tautomer | conjugate (acid|base) | reacts? with | …` — applied to a **±60-character window**
around the match (`_DERIV_WINDOW = 60`, line 408).

**This flag is not applied uniformly.** It is `True` only in the cap2mol path
(`diagnose_hallucination.py:626` for IO and `:688` for ER). Every other task calls
`extract_chemical_entities(reasoning)` with the default `exclude_derivation=False`
(`diagnose_multitask.py:147`, `:213`, `:300`, `:392`, `:647`). See §12.

**(d) Structural verdict.** `_compiled_fg_patterns()` (448–456) compiles every SMARTS once into
`_SMARTS_CACHE`; `_get_mol_fgs(mol)` (459–467) returns the set of FG names for which
`mol.HasSubstructMatch(pat)` is true. There is no generative judge and no similarity
threshold anywhere in the claim verdict.

**(e) The four dimensions.**

| dim | function | what it scores |
|---|---|---|
| IR | `score_ir_self_contradiction` 489–544 | regex tally of self-correction phrases (503–511, weights 3–6 each), >3 "conclusion attempts" (519–526, +5 each beyond 3), repeated sentences (528–538, +3 each beyond 2), length >10 000 chars (540–542, +min(20,(L−10000)/1000)); `min(100, …)` |
| IO | `score_io_structural_invalidity` 549–661 | class-property violations (+10 each, cap 40; 613–615), description-FGs missing from the prediction (`ratio*30`, 619–622), reasoning→output transmission failure (`ratio*25`, 630–637), trace SMILES fragments not substructures of the prediction (`ratio*15`, 641–659) |
| ER | `score_er_factual_fabrication` 666–747 | see §1 formula below |
| EO | `score_eo_phantom_structure` 752–802 | `(1 − Tanimoto)*100`, or 80.0 if Tanimoto is undefined (777–780) |

**(f) ER, the actual formula (cap2mol).** `diagnose_hallucination.py:699–747`:

```python
pred_fgs = _get_mol_fgs(pred_mol)
gt_fgs   = _get_mol_fgs(gt_mol)
desc_ents = extract_chemical_entities(separate_structural_desc(description))
desc_fgs  = desc_ents.get("functional_groups", set())
real_fgs  = pred_fgs | gt_fgs | desc_fgs                       # line 705
...
checks    = max(len(trace_fgs), 1)                             # line 715
penalties = len(fabricated_fgs) * 5.0                          # line 716
...
if wrong_classes:
    penalties += len(wrong_classes) * 8.0                      # line 740
    checks += len(trace_classes)                               # line 742
score = min(100.0, (penalties / max(checks, 1)) * 15)          # line 744
```

For the pure functional-group case this is exactly

    ER = min(100, 75 · n_fabricated / n_claimed)

i.e. **κ = 75 for cap2mol**. `separate_structural_desc()` (369–376) strips sentences containing
any `BIO_ROLE_KEYWORDS` (214–220: metabolite, inhibitor, agonist, "has a role as", …) before
mining the caption, so bio-role prose does not ground structural claims.

Verified empirically on the shipped `data/results/Chem-R/cap2mol/Chem-R_cap2mol_hallucination_details.jsonl`:
the non-zero ER values are exactly `75·k/m` — 37.5 (=75·1/2), 25.0 (1/3), 18.75 (1/4), 15.0
(1/5), 12.5 (1/6), 10.71 (1/7), 9.375 (1/8), 21.43 (=75·2/7), 30.0 (=75·2/5).

**(g) ER on the other tasks — κ = 60, not 75.** `diagnose_multitask.py`:

| task | line | grounding set (`real_fgs`) |
|---|---|---|
| mol2cap | `160` | input molecule only (`input_fgs`, 144–145); claims mined from `reasoning + pred_caption` (146–149) |
| classification | `221` | input molecule only (211–212) |
| retrosynthesis | `307` | product ∪ every parsed predicted reactant (293–299) |
| reaction_prediction | `399` | reactants ∪ predicted product (384–391) |
| s2_* | `654` | predicted molecule ∪ `metadata.source_molecule` ∪ groups named in the instruction (640–646) |

All five use the identical form
`er_score = min(100.0, len(fabricated) / max(len(claimed), 1) * 60.0)`, so **κ = 60**. This is
a genuine per-task constant, not a rounding artefact: the same `n_fab/n_claimed` produces a
25 % lower ER on retrosynthesis than on cap2mol.

**(h) Grounding set — task-aware, matches Methods.** cap2mol grounds on
input caption ∪ prediction ∪ reference (line 705); mol2cap on the input molecule only;
retrosynthesis on product ∪ predicted reactants (no reference reactants); S²-Bench on source
molecule ∪ instruction-named groups ∪ prediction. Note that retrosynthesis and S² do **not**
consult the reference, contrary to `methods.tex:17` ("absent from the relevant input evidence,
the prediction **and the reference**"). See §12.

**(i) Aggregate score and its renormalization.** Weights: `DIMENSION_WEIGHTS = {"IR":0.15,
"IO":0.25, "ER":0.25, "EO":0.35}` (`diagnose_hallucination.py:807–812`; re-exported at
`diagnose_multitask.py:37`).

The Methods states only the fixed linear form. The code **renormalizes** whenever a dimension
is skipped. cap2mol, `diagnose_hallucination.py:867–896`:

```python
io_skipped = io_det.get("skipped", False)
eo_skipped = eo_det.get("skipped", False)
if io_skipped or eo_skipped:
    active_weights = {"IR": 0.15, "ER": 0.25}
    if not io_skipped: active_weights["IO"] = 0.25
    if not eo_skipped: active_weights["EO"] = 0.35
    total_w = sum(active_weights.values())
    overall = sum(score[k] * (w / total_w) for ...)
else:
    overall = 0.15·IR + 0.25·IO + 0.25·ER + 0.35·EO
```

`skipped` is set when there is no prediction or the SMILES does not parse
(`:571`, `:575` for IO; `:763`, `:767` for EO). So on an invalid-SMILES response the aggregate
becomes `0.375·IR + 0.625·ER` — an invalid molecule contributes **zero**, not 100, to the
aggregate, and the trace dimensions are up-weighted. The multitask path does the same through
`_aggregate()` (`diagnose_multitask.py:73–81`), with per-task skip flags: `ER` is skipped
whenever no claim was extracted (`:172`, `:236`, `:336`, `:427`, `:662`), `EO` when the
prediction set is empty (`:337`, `:427`) or no label was produced (`:237`).

**(j) EO is zeroed on exact match.** cap2mol `diagnose_hallucination.py:771–772`
(canonical equality → `return 0.0, {"exact_match": True}`); mol2cap `:169–170` (Jaccard ≥ 0.85
→ `exact_match`, EO forced to 0); retrosynthesis `:315–318`; classification `:228–230`.
For S² the relation is **inverted**: `diagnose_multitask.py:660` sets
`result["exact_match"] = eo_score == 0.0`, i.e. "exact match" on S² *means* "no constraint
deviation", and is not the official S² success metric. `eval/metrics.py:86–93` therefore
ignores it and calls `s2_success()` instead.

### Parameters
| parameter | value | source |
|---|---|---|
| dimension weights | 0.15 / 0.25 / 0.25 / 0.35 | `diagnose_hallucination.py:807–812` |
| ER scale κ (cap2mol) | 75 (= 5.0 × 15) | `:716`, `:744` |
| ER scale κ (all other tasks) | 60 | `diagnose_multitask.py:160,221,307,399,654` |
| fabricated molecular-class penalty | 8.0 | `diagnose_hallucination.py:740` |
| derivation-clause window | ±60 chars | `:408` |
| Morgan fingerprint for Tanimoto | radius 2, 2048 bits | `:335–336` |
| mol2cap "exact match" threshold | Jaccard ≥ 0.85 | `diagnose_multitask.py:166` |
| IR excessive-length trigger | 10 000 chars, cap +20 | `diagnose_hallucination.py:540–541` |
| FG library size | 99 groups, 6 generic | `:74–187`, `:191–193` |

### Inputs
`se_results/<model>/<task>/output.json` — a JSON list of
`{id, question, gt, answer, model, task}` written by `run_multitask_se.py:35–49`
`_write_outputs_json()`. For S² tasks the constraint metadata lives in
`se_results/<model>/<task>/completions.json` and is merged by
`eval/redx_all.py:28–32` `_meta_map()`.

### Output
`data/results/<model>/<task>/<model>_<task>_hallucination_details.jsonl`, one JSON per response
with `hallucination_scores` (4 dims), `overall_hallucination_score`, `pred_smiles`,
`pred_valid`, `exact_match`, `tanimoto`, `reasoning_length` and (with `--verbose`) `details.ER`
containing `claimed_fgs / verified_fgs / fabricated_fgs`. Plus a
`*_hallucination_summary.json`. Consumed by every downstream analysis.

### Reproduce
```bash
# CPU. Detector on one (model, task):
python diagnose_multitask.py --task cap2mol \
  --model_output se_results/Chem-R/cap2mol/output.json \
  --model_name Chem-R --output_dir data/results/Chem-R/cap2mol --verbose

# CPU, parallel over the whole tree (12 workers by default):
python eval/redx_all.py 12

# Single-response smoke test with no data at all (CPU, seconds):
python -c "
from diagnose_multitask import diagnose_one
print(diagnose_one('cap2mol', {'answer':'<think>It has a carboxylic acid and a pyrrolidine ring.</think><answer>CC(=O)O</answer>',
                               'question':'a molecule with a carboxylic acid','gt':'CC(=O)O'}, verbose=True))"
```
No GPU needed for any of the above.

---

## 2. Figs 1d,e and 2 — prevalence across models; the proxies that fail

### Claim
`results.tex:54–60`

> "At the model level, hallucination does not simply mirror task performance (Fig. 1d). Chem-R
> and ChemDFM-R have nearly identical aggregate hallucination scores despite a ten-point
> performance gap … output-side errors account for most of the aggregate score, but ER remains
> present in every model."

`results.tex:77–85`

> "Among the released models, mean ER across tasks ranges from $7.6$ to $26.4$;
> verification-grounded training lowers Chem-R-Faithful to $2.4$ …
> Semantic entropy … spans only approximately $1.07$–$1.37$ … (Pearson $-0.54$) …
> The per-claim fabrication rate, by contrast, tracks the released-model ranking
> ($14 \to 23 \to 38 \to 55\%$; Fig. 2c) … Chem-R-Faithful … $4\%$."

### Entry point
- Per-(model,task) statistics: `eval/metrics.py:102–140` `task_stats()`.
- CSV export: `eval/export_stats.py:59–67` (per model×task), `:70–83` (per model×family).
- Panels: `figures/make_nmi_figures.py:254–306` `fig1_measures()`,
  `:131–246` `fig2_widespread()`.

### Algorithm

`task_stats()` iterates the details JSONL joined with `output.json` and accumulates:

- `perf` — `perf_one()` (`eval/metrics.py:73–94`): cap2mol/retro `exact_match`;
  mol2cap NLTK `sentence_bleu` with uniform 4-gram weights and `SmoothingFunction().method1`
  (`:29`, `:32`, `:82–83`) — i.e. **BLEU-4**; `s2_*` the official `s2_success()`
  (`:86–93`), deliberately using the diagnoser's `pred_smiles` so hallucination and
  performance score the same molecule.
- `ER/EO/IR/IO/overall` — means of the per-response fields (`:116–117`, `:133`).
- `pct_er0` — `100 · mean(ER == 0)` (`:126`, `:132`). This is Fig 3d.
- `gc` (grounded claims) — `_gc_one()` (`:97–99`): number of **distinct non-generic**
  verified FGs per response.
- `cp` (claim precision) — `100·V/(V+F)` where `V`, `F` are the **pooled** counts of
  verified / fabricated FGs across the task, **with generic groups removed**
  (`:124–125`, `:135`). Fig 2c plots `100 − cp`.
- `length` — mean `reasoning_length` in characters (`:118`).

Generic groups (`GENERIC_FG_NAMES` = `aromatic_ring, ring, heterocycle, halogen, double_bond,
triple_bond`, `diagnose_hallucination.py:191–193`) are imported at `eval/metrics.py:27` and
excluded from `gc` and from **both** the numerator and denominator of `cp`. They are **not**
excluded from the ER score itself.

`family_stats()` (`:143–157`) aggregates a family with **sample-count weighting**
(`w()` at `:150–153`); `FAMILIES` (`:161–162`) is cap2mol / mol2cap / retrosynthesis / s2 (the
9 S² subtasks pooled).

**Fig 2a** (`make_nmi_figures.py:146–169`) is a 5 × 12 `imshow` of the `ER` column restricted
to `SURV5 = ["Chem-R","Chem-R-Faithful","ChemDFM-R","ether-0","DeepSeek-R1-Distill"]` (line
133) and `GEN12` (lines 100–106). The per-model aggregates in panels b–d are the **unweighted
mean over the 12 tasks** (`agg()`, lines 139–140), not sample-weighted — this differs from
`family_stats`.

**Fig 1d/1e** (`fig1_measures`, 254–306) recomputes the aggregate as a **fixed** linear form on
per-model dimension means:

```python
contrib = {m: [w * wmean(m, k) for k, w, _ in W] for m in MODELS}   # line 264
overall = {m: sum(contrib[m]) for m in MODELS}                       # line 265
```

with `W = [("IR",0.15),("IO",0.25),("ER",0.25),("EO",0.35)]` (258–259). This is **not** the
mean of the per-response `overall` column (which uses the per-response renormalization of §1i).
Reproduced values (mean over GEN12): Chem-R 12.571 vs 12.605; ChemDFM-R 12.526 vs 12.609;
ether-0 22.400 vs 22.768; DeepSeek 50.193 vs 50.998; Chem-R-Faithful 8.946 vs 8.965. Panel 1e
plots `100 · contrib[m][j] / overall[m]`, i.e. the weight-scaled share (291–299).

### Semantic entropy (Fig 2b x-axis)

Not computed in the release. `eval/export_stats.py:44–51` `se()` only reads
`se_results/<model>/<task>/se_summary.json → semantic_entropy.mean`. The producer is
`MolLM_R_Hallucination_v1/semantic_entropy.py`, called from `run_multitask_se.py:117`
(`evaluate_semantic_entropy(all_samples, task=se_kind)`). The two `run_multitask_se.py` copies
are byte-identical.

**Sampling** — `MolLM_R_Hallucination_v1/semantic_entropy.py:418–444`:
```python
def sample_completions(backend, messages, n_samples: int = 10, temperature: float = 0.8):
    params = SamplingParams(temperature=temperature, top_p=0.95,
                            max_tokens=... or 4096, n=n_samples)
```
`top_p = 0.95` is **hard-coded** at `:438` and is not a CLI flag — this is the only place the
Methods' top-p appears. **No seed is set** in either `LLM(...)` or `SamplingParams(...)`, so SE
runs are not reproducible by seed.

**Clustering** — greedy single-pass, first member of a cluster is its representative
(`cluster_by_similarity`, `:159–181`, admission test `sim >= threshold` at `:173`). Per-kind
similarity functions (`TASK_SIM_FNS`, `:246–253`) and thresholds (`TASK_THRESHOLDS`,
`:255–262`):

| se_kind | similarity | threshold |
|---|---|---|
| `cap2mol` (and **all 9 S² subtasks**, `data_loaders.py:377–382`) | Morgan Tanimoto, radius 2, 2048 bits (`:106–108`) | **0.85** (`:257`) |
| `retrosynthesis` | `reaction_similarity` — sorted canonical reactant-set equality → 1.0, else mean best-match Tanimoto (`:137–152`) | **0.85** (`:262`) |
| `mol2cap` | `text_embedding_similarity` (`:111–123`) | **0.80** (`:258`) |
| `classification` | exact case-insensitive match | 0.99 (`:260`) |
| `property` | `max(0, 1−|Δ|/0.1)` | 0.90 (`:259`) |

**Entropy** — `entropy_from_clusters` (`:188–197`) is `H = −Σ p log p` with `math.log`, i.e.
**nats**, over the cluster distribution of the *valid extracted* completions
(`n = len(valid_items)`, `:320`). The reported figure is the unnormalised output-layer SE:
`summary["semantic_entropy"]["mean"]` (`:528`). A `normalized_entropy` (÷ `ln(n)`, `:200–204`)
and a dual output/reasoning entropy (`compute_dual_entropy`, `:350–411`) are also computed but
not used by any paper number.

**Two implementation facts that matter for reading Fig 2b.**

1. For `mol2cap` the extractor is
   `"mol2cap": lambda text: extract_reasoning(text) or text` (`:213`) — it clusters the
   **`<think>` reasoning text**, not the generated caption. Combined with the fallback
   similarity (the driver passes no `embedding_model`, so `text_embedding_similarity` degrades
   to lowercased whitespace-token Jaccard, `:119–123`) at threshold 0.80, free-text traces
   essentially never cluster. The shipped `Diagnosis_model_task` sheet confirms it: mol2cap SE
   is exactly **2.3026 = ln(10)** — maximum entropy, ten singleton clusters — for Chem-R,
   Chem-R-Faithful, ChemDFM-R and DeepSeek-R1-Distill, and 2.280 for ether-0. One of the twelve
   tasks is therefore a saturated constant in every model's mean.
2. SE is present for all five surveyed models on all twelve tasks in the sheet (`+process` has
   none). The per-task spread is wide (0.17–2.30); the "1.07–1.37" range of `results.tex:80` is
   the **unweighted mean over the twelve tasks**.

**The `r = -0.54` in Fig 2b is a hard-coded string**, `make_nmi_figures.py:184`:
```python
axb.text(0.97, 0.95, r"$r = -0.54$", transform=axb.transAxes, ...)
```
Recomputing Pearson over the five plotted points gives **−0.533** (against the fixed-form
aggregate) or **−0.535** (against the mean-of-column aggregate). Nothing in the plotting code
computes it.

### Parameters
| parameter | value | source |
|---|---|---|
| surveyed models | 5 (`SURV5`) | `make_nmi_figures.py:133`, `:257`, `:384` |
| task universe | 12 generative variants (`GEN12`) | `make_nmi_figures.py:100–106`, filter at `:137`, `:256` |
| cross-task aggregation in Fig 1/2 | unweighted mean over the 12 tasks | `:139–140`, `:261–262` |
| classification tasks | defined (`CLS`, `export_stats.py:33`) but **excluded** from every paper figure | `make_nmi_figures.py:137` |
| SE samples / temperature | n = 10, T = 0.8 (CLI defaults, and the function defaults) | `run_multitask_se.py:151–152`; `MolLM_R_Hallucination_v1/semantic_entropy.py:421–422` |
| SE top-p | 0.95, hard-coded | `MolLM_R_Hallucination_v1/semantic_entropy.py:438` |
| SE decode length | `max_new_tokens = 4096`, `max_model_len = 4096` | `run_multitask_se.py:153`, `:158` |
| SE seed | **none set** | — |
| SE clustering thresholds | 0.85 / 0.85 / 0.80 (molecule / retro / caption) | `semantic_entropy.py:257`, `:262`, `:258` |
| SE entropy units | nats (`math.log`) | `semantic_entropy.py:196` |
| BLEU | 4-gram uniform, smoothing method1 | `eval/metrics.py:32`, `:82–83` |

### Test-set sizes
From `MolLM_R_Hallucination_v1/data_loaders.py`: ChEBI-20 test = **3300** prompts for cap2mol
and mol2cap (`_read_chebi20`, `:49–57`, one header line skipped); USPTO-50k test = **5007**
(`USPTO50K_DEFAULT_TEST_SIZE = 5007`, `:197`, a deterministic **tail slice** `rows[n-5007:]`
with no shuffle and no seed, `:207–218`); S²-Bench "mini" = **the first 500 rows of each
`<Group>_<Sub>.csv` in file order** — `return data[:max_samples] if max_samples else data`
(`:359`), with `--max_samples 500` supplied by the job scripts. There is no random subsampling
anywhere in the S² selection, contrary to what "500 prompts (mini size) per subtask for
tractability" (`methods.tex:5`) might suggest. Totals: 3300 + 3300 + 5007 + 9×500 = **16 107**
responses per model — exactly the `n_resp` of `data/stage_ladder.csv` (§8).

### Inputs
`data/results/*/*/**hallucination_details.jsonl` + `se_results/*/*/output.json`
(+`completions.json` for S² metadata) + `se_results/*/*/se_summary.json`.

### Output
`data/stats_per_model_task.csv` (17 metric columns + `semantic_entropy`),
`data/stats_per_family.csv` → workbook sheets `Diagnosis_model_task` (92 rows),
`Diagnosis_family` (24 rows) → `figures/fig1_measures.pdf`, `figures/fig2_widespread.pdf`.

### Reproduce
```bash
# CPU. Needs the full diagnosed tree; refuses to run without the env guard:
MOLREHALLU_REGEN=1 python eval/export_stats.py

# CPU, from the shipped workbook only:
cd figures && python -c "import make_nmi_figures as m; m.fig1_measures(); m.fig2_widespread()"
```
`eval/export_stats.py:23–28` raises `SystemExit` unless `MOLREHALLU_REGEN=1`, because it
overwrites `data/*.csv` **at import time** from `data/results/`, which the release ships only a
one-directory subset of.

Semantic entropy itself needs a GPU (vLLM):
```bash
python run_multitask_se.py --model_id weidawang/Chem-R-8B --model_name Chem-R \
  --tasks all --backend vllm --n_samples 10 --temperature 0.8 --output_dir se_results/Chem-R
```

---

## 3. Fig 3 — decoupling of correctness from faithfulness

### Claim
`results.tex:96–104`

> "On the full caption-to-molecule test set for Chem-R ($n=3300$) … the ER distributions of
> exactly correct and incorrect responses are nearly indistinguishable, with mean ER scores of
> $6.4$ and $6.8$ … ($|\mathrm{Pearson}(\mathrm{ER},\text{exact match})|<0.02$) …
> $28\%$ are both exactly correct and clean ($\mathrm{ER}=0$), whereas $13\%$ are exactly
> correct but contain at least one detected fabrication … although the overall exact-match rate
> is $41\%$, approximately $31\%$ of exactly correct responses contain a fabrication."

`results.tex:110`: "Chem-R-Faithful raises the clean-trace rate to approximately $84$–$91\%$
across the four task families."

### Entry point
`figures/make_nmi_figures.py:312–317` `_chemr_cap2mol()` and `:320–421` `fig3_accuracy_gap()`.
Panels c,d read the family sheet produced by `eval/metrics.py:143–157` `family_stats()`.

### Algorithm
Panels a and b bypass the workbook entirely and read the raw per-response JSONL
(`:313`):

```python
p = REPRO / "Chem-R" / "cap2mol" / "Chem-R_cap2mol_hallucination_details.jsonl"
er    = np.array([r["hallucination_scores"]["ER_factual_fabrication"] for r in recs], float)
exact = np.array([1 if r.get("exact_match") else 0 for r in recs])
```

- Panel a: violin + inner quartile box of `er[exact==1]` vs `er[exact==0]` (330–343);
  the Pearson coefficient **is** computed live at line 347
  (`r = np.corrcoef(er, exact)[0,1]`) — unlike Fig 2b's.
- Panel b: a mosaic whose cell areas are the four joint proportions (358–374).
- Panel c: scatter of `perf_er0` against `perf_erpos` from sheet `Diagnosis_family`
  (385, 389–392) with the identity line at 393.
- Panel d: `pct_er0` per (model, family) (413–415).

Reproduced from the shipped JSONL (n = 3300): mean ER correct **6.413**, wrong **6.837**;
Pearson **−0.0185**; right & fabricating **12.8 %**, right & clean **28.2 %**, wrong &
fabricating 21.7 %, wrong & clean 37.3 %; exact-match rate **41.0 %**; share of correct
responses with ER>0 **31.2 %**. All match the paper.

`perf_er0`/`perf_erpos` come from `eval/metrics.py:130–131`:
```python
perf_er0  = 100 * perf[c].mean()   # c = (ER array == 0)
perf_erpos= 100 * perf[~c].mean()
```
with `perf` the official per-task metric of §2. The family-level values are sample-weighted
(`family_stats.w`, `:150–153`), and `decoupling_gap = perf_er0 − perf_erpos` is written by
`eval/export_stats.py:80`.

The mol2cap / retrosynthesis correlations quoted in `results.tex:98` (−0.04, −0.05) have **no
producer in the release** — they are not in any sheet or CSV. See §13.

### Parameters
| parameter | value | source |
|---|---|---|
| model / task for panels a,b | Chem-R, cap2mol, n = 3300 | `make_nmi_figures.py:313` |
| clean-trace definition | `ER == 0` exactly (not a threshold) | `eval/metrics.py:126` |
| y-limit on panel a | −2 to 30 (clipped display) | `make_nmi_figures.py:346` |
| families in c,d | 4 (`cap2mol, mol2cap, retrosynthesis, s2`) | `eval/metrics.py:161–162` |

### Inputs
`data/results/Chem-R/cap2mol/Chem-R_cap2mol_hallucination_details.jsonl` (the only diagnosed
tree shipped) + sheet `Diagnosis_family`.

### Output
`figures/fig3_accuracy_gap.pdf/.png`. The per-response series is also frozen in sheet
`Fig3_decoupling_perresponse` (11 607 rows = cap2mol 3300 + mol2cap 3300 + retro 5007) — that
sheet has no producer in the release.

### Reproduce
```bash
cd figures && python -c "import make_nmi_figures as m; m.fig3_accuracy_gap()"   # CPU
```

---

## 4. Fig 4 — causal perturbation and matched-token attention

### Claim
`results.tex:132–145`

> "Starting from clean caption-to-molecule traces, we selected a functional-group mention that
> was verified to be present in the target molecule and replaced it with a random incorrect
> group. We then measured the teacher-forced log-probability of the original correct answer …
> Corrupting the trace had a negligible effect ($\Delta\log p=-0.0007$ for Chem-R) … corrupting
> the input produced reductions … two to three orders of magnitude larger ($-0.20$ for Chem-R
> and $-0.13$ for ChemDFM-R; Fig. 4a). … corrupting the functional-group claim in the trace
> caused only $7\%$ of the answers to become incorrect … Corrupting the corresponding
> information in the input … approximately $40\%$ (Fig. 4b). Across models and tasks, the
> answer-flip rate was $1$–$11\%$ for trace corruption and $16$–$78\%$ for input corruption …
> ($p<10^{-15}$). The weak effect … persisted both for claims restated from the input
> ($\Delta\log p=-0.0092$ to $-0.0000$) and for claims derived by the model ($-0.0013$ to
> $-0.0004$). … answer tokens assigned $20\times$ more attention to its input occurrence for
> Chem-R and $4\times$ more for ChemDFM-R (Fig. 4c)."

### 4a — teacher-forced Δlogp

**Entry point.** `eval/attention_attribution.py:179–241` `perturb()`, with
`answer_logprob()` at `:153–164` and `_delta()` at `:287–289`. Driver
`main()` `:458–592`, perturbation loop `:529–538`.

**Algorithm.**
1. Split the recorded response at the **last** `<answer>` (`:181–186`): `prefix_orig` is
   everything through `<answer>`, `answer_txt` is the SMILES + `</answer>`.
2. Baseline `lp0 = answer_logprob(model, tok, full_prompt + prefix_orig, answer_txt)`
   (`:212–213`). `answer_logprob` (`:153–164`) does a single forward pass and returns the
   **mean per-token** log-probability over the answer tokens:
   `pos.gather(1, tgt[:,None]).squeeze(1).mean().item()` (`:164`).
3. Choose a specific FG whose synonym appears in the trace, preferring one that also appears
   in the input caption (`:195–208`). `SPECIFIC` (`:63–64`) is `FUNCTIONAL_GROUP_DB` minus the
   6 generic names, restricted to entries with at least one synonym (93 groups).
4. Conditions, each a `Δ = lp_cond − lp0`:
   - `d_wrong_cot` (`:218`) — one trace mention → the first synonym of a **random other**
     specific group (`wrong_fg = random.choice(...)`, `:209–210`).
   - `d_syn_cot` (`:222`) — one trace mention → another synonym of **the same** group
     (negative control).
   - `d_wrong_input` (`:228`) — the same corruption applied to the caption instead; only
     defined when the group is actually named in the input (`:226`).
   - `d_drop_cot` (`:232`) — `<think>` content blanked (`_empty_think`, `:247–253`).
   - `d_all_wrong_cot` (`:238`) — every specific-FG mention corrupted (`_corrupt_all_cot`,
     `:272–284`).
5. Substitution is `re.sub(re.escape(old), new, text, count=1, flags=re.I)` (`replace_ci`,
   `:175–176`) — first occurrence, case-insensitive.

**Scope filters.** The perturbation pool is **clean-and-correct only**:
`clean = [e for e in all_ex if e["er"] == 0 and e["exact"]]` (`:485`). `_is_correct()`
(`:68–72`) is `exact_match` for everything except mol2cap (caption Jaccard ≥ 0.5).

**Aggregation to the published number.** `eval/export_stats.py:120–141`. Two filters matter
and are documented in the code's own comment block at `:100–110`:

```python
CAP = {"cap2mol"}                                  # export_stats.py:112
def _mid(lst):                                     # :113–115
    return lst[len(lst) // 2] if lst else None
def pm(k):
    v = [p[k] for p in d.get("perturb", []) if p.get(k) is not None and p.get("task") in CAP]
    return (rnd(float(np.mean(v))), len(v)) if v else (None, 0)   # :123–125
```

**The Δlogp reported in the paper is the cap2mol subset only** — even though
`attention_attribution.py` runs over all 12 task variants (`DEFAULT_TASKS`, `:361–366`).
Verified against `data/raw/region_Chem-R.json`: cap2mol-only `d_wrong_cot` n = 924,
mean −0.00065 (→ −0.0007 in `attention_perturbation.csv`), `d_wrong_input` n = 840, mean
−0.19538 (→ −0.1954, paper "−0.20"). ChemDFM-R: −0.00020 / −0.12654 (paper "−0.13").
The full-volume `perturb` list for Chem-R is 2182 examples across all tasks.

**Restated vs derived.** The split is defined by whether `syn_in` was found, i.e. whether the
chosen group also appears in the input text (`attention_attribution.py:199–206`). Sheet
`R2_restated_derived` holds the result: restated mean `d_wrong_cot` = −0.0064 (Chem-R),
−0.0010 (Faithful), −0.0092 (ChemDFM-R), −0.0000 (ether-0); derived = −0.0013 / −0.0004 /
−0.0012 / −0.0008. These bracket the paper's quoted ranges exactly. **No script in the release
produces this sheet.**

### 4b — behavioural flip-to-wrong

**Entry point.** `eval/cot_drift.py:265–425` `main()`; condition construction
`build_conditions()` `:202–262`; flip extraction `eval/build_source_data.py:53–59` `_flip()`.

**Algorithm.**
1. For every recorded response (**no ER or correctness filter** — docstring `:5–8`), rebuild
   the prefix `[chat prompt | think_open + CoT + think_close | inter-tag text | answer_open]`
   (`assemble()`, `:220–222`) and greedily regenerate the answer.
2. Conditions (`:234–261`): `base`, `wrong_cot`, `all_wrong_cot`, `drop_cot`, `swap_cot`
   (another example's real CoT for the same task, `:246–247`), `mask_draft`, `corrupt_draft`,
   `syn_cot`, `wrong_input`.
3. Sampling: `SamplingParams(temperature=0.0, max_tokens=320, stop=[mk[3]])` (`:293`) — greedy,
   stops at the model's native answer-close token.
4. Native markup per model: `MARKUP = {"ether-0": ("<|think_start|>", "<|think_end|>",
   "<|answer_start|>", "<|answer_end|>")}` (`:53`), standard tags otherwise (`:54`). The
   perturbation and the regeneration therefore stay in-distribution for ether-0.
5. Correctness of a regenerated answer: `correct()` (`:109–127`) — `s2_success` for S²,
   canonical-SMILES equality for cap2mol/retro, token-Jaccard ≥ 0.5 for mol2cap.
6. Per example the code stores `base_correct` and, per condition, `<cond>_dperf =
   cond_correct − base_correct` and `<cond>_drift` (any answer change) (`:377–392`).

**The paper metric is not `summary.drift_rate`.** `eval/build_source_data.py:53–59`:

```python
def _flip(sub, c):
    # PAPER metric: among originally-correct, fraction that became wrong (dperf==-1).
    present = [e for e in sub if f"{c}_dperf" in e]
    if not present: return float("nan")
    return sum(1 for e in present if e[f"{c}_dperf"] == -1) / len(present)
```
applied to `sub = [e for e in pe if grp(e["task"]) == g and e.get("base_correct") == 1]`
(`:68`). `summary.drift_rate` (`cot_drift.py:394–396`) is *any* answer change over *all*
examples — a different and larger quantity; the contrast is printed by
`eval/verify_paper_metric.py:37–40`.

Task groups: `TRANSLATE = {"cap2mol","mol2cap","retrosynthesis"}`,
everything else is `s2` (`build_source_data.py:50`, `:63`).

Fig 4b plots the `translate` group only (`make_nmi_figures.py:467`) for conditions
`syn_cot / all_wrong_cot / swap_cot` (`:463–464`), with 95 % Wald intervals computed in the
figure code:
```python
errs.append(1.96 * np.sqrt(p * (1 - p) / nn) * 100)     # :473
```
Shipped values (sheet `R2_drift`, translate): Chem-R `all_wrong_cot` **6.95 %** (paper "7 %"),
`wrong_input` **41.30 %** (paper "≈40 %"), `syn_cot` 2.27 %, `swap_cot` 32.75 %, n = 4950.
The "1–11 % / 16–78 %" range in `results.tex:141` spans both task groups and all models and is
the table in `data/RESULTS.md:71–78`; the McNemar tests behind `p<10^{-15}`
(`RESULTS.md:80–82`) are **not** in any released script — see §13.

### 4c — matched-token attention

**Entry point.** `eval/attention_attribution.py:292–358` `matched_attention()`.

**Algorithm.** Find a functional-group synonym that occurs **both** inside the input caption
span `[d0,d1)` and inside the `<think>` span (`:319–328`), locate its token indices on each
side (`:329–332`), run one forward pass with `output_attentions=True` (`:335–336`), and for
each layer take the head-mean, restrict rows to the answer tokens, and take the **per-token
mean** attention to each occurrence:

```python
for A in out.attentions:
    sub = A[0].mean(0).index_select(0, rows)               # answer rows, head-mean   :342
    ain.append(sub.index_select(1, iidx).mean().item())    # per-token attn to input FG  :343
    acot.append(sub.index_select(1, cidx).mean().item())   # per-token attn to CoT FG    :344
```

The two occurrences are the identical string, so the ratio isolates location from content and
from span length.

**Layer.** The stored `attn_in_fg` / `attn_cot_fg` are **per-layer lists**. The published ratio
is read at the **middle layer only**, `l* = L//2`:
`export_stats.py:113–115` `_mid()`, applied at `:127–128`; the same convention is printed by
`attention_attribution.py:581–584`. This is not cosmetic — reproduced from
`data/raw/region_Chem-R.json`:

| model | layers | mid-layer ratio | **all-layer-average ratio** |
|---|---|---|---|
| Chem-R | 32 | **20.06** | 10.72 |
| Chem-R-Faithful | 32 | **20.69** | 9.80 |
| ChemDFM-R | 48 | **4.34** | 4.59 |
| ether-0 | 40 | **0.99** | 1.42 |

Averaging over layers halves the Chem-R number and will not reproduce the paper.

**Subset.** cap2mol only (`CAP = {"cap2mol"}`, `export_stats.py:112`, applied at `:126`).
n_matched (cap2mol) = 265 / 233 / 255 / 283 for Chem-R / Faithful / ChemDFM-R / ether-0.

Fig 4c (`make_nmi_figures.py:484–503`) converts the ratio to a 100 %-split bar:
`pin = ratio/(ratio+1)*100` (`:492`).

### Parameters
| parameter | value | source |
|---|---|---|
| model precision / attention impl | bfloat16, `attn_implementation="eager"` | `attention_attribution.py:469–471` |
| sequence-length cap | 3500 tokens (attention & logprob) | `:133`, `:157`, `:306` |
| `--n_attn` (attention & matched caps) | 300 per regime | `:462` |
| `--n_perturb` | 10⁹ = full volume | `:463` |
| RNG seed | `random.seed(0)` | `:65` (and `cot_drift.py:41`) |
| drift decoding | greedy, `temperature=0.0`, `max_tokens=320` | `cot_drift.py:293` |
| drift engine | vLLM, bf16, `gpu_memory_utilization=0.9`, `max_model_len=4096` | `cot_drift.py:290–291` |
| drift tasks | 7 by default (`DEFAULT_TASKS`) | `cot_drift.py:59–61` |
| attention/attr tasks | 12 (`DEFAULT_TASKS`) | `attention_attribution.py:361–366` |
| Δlogp / matched subset | `cap2mol` only, middle layer | `export_stats.py:112–115` |
| Wald z | 1.96 | `make_nmi_figures.py:473` |

### Inputs
`se_results/<model>/<task>/output.json` (raw traces, `attention_attribution.py:76`,
`cot_drift.py:299`), `data/results/<model>/<task>/*hallucination_details.jsonl` (ER labels,
`:77` / `:305`), `se_results/.../completions.json` (S² metadata, `cot_drift.py:314–317`), and
HF weights from the `HF` map (`attention_attribution.py:43–50`).

### Output
`data/raw/region_<model>.json` (keys `attn`, `perturb`, `matched`, `heatmap`, `region_attr`,
`per_task`, `n_clean`, `n_fabr`) → `data/attention_perturbation.csv` (Fig 4a,c);
`data/raw/drift_<model>.json` (key `per_example`) → sheets `R2_drift`, `R2_drift_by_task`
(Fig 4b).

### Reproduce
```bash
# GPU required (one 8B model in bf16 with eager attention; ~40 GB for the 14B/24B models).
python eval/attention_attribution.py --model Chem-R                 # perturb + matched + attn
python eval/cot_drift.py --model Chem-R                             # behavioural flips (vLLM)

# CPU aggregation:
MOLREHALLU_REGEN=1 python eval/export_stats.py                      # rewrites attention_perturbation.csv
MOLREHALLU_REGEN=1 python eval/build_source_data.py                 # rebuilds R2_* sheets
python eval/verify_paper_metric.py                                  # re-derives R2 flips from data/raw/
cd figures && python -c "import make_nmi_figures as m; m.fig4_mechanism()"
```

---

## 5. Fig 5 — token-type saliency and region attention

### Claim
`results.tex:191–197`

> "For Chem-R and Chem-R-Faithful, the most salient token type per token is the SMILES fragments
> (enrichment $2.5\times$ and $3.2\times$ above the average trace token), while functional-group
> words are depleted ($0.71$–$0.77\times$; Fig. 5a,b). Within-trace attention shows the same
> preference … ether-0 … SMILES tokens are enriched $1.6\times$ … functional-group words are
> depleted to $0.6\times$, and within-trace attention to SMILES is about twice that to
> functional-group words … ChemDFM-R … SMILES fragments account for little trace saliency
> ($1.0\%$ of the total mass; $1.2\times$ per-token enrichment), and within-trace attention
> favours functional-group terms over SMILES."

### 5b — gradient × input saliency

**Entry point.** `eval/attr_probe.py:30–114` `main()`; token categoriser `:15–27`
`categorize()`; aggregation `eval/pull_fullvol.py:23–34` and CSV emit `:73–81`.

**Algorithm.** Per example:
```python
emb  = model.get_input_embeddings()(ids_t).detach().requires_grad_(True)   # attr_probe.py:86
out  = model(inputs_embeds=emb, use_cache=False)                            # :87
logp = torch.log_softmax(out.logits[0].float(), -1)                         # :88
score = logp[pos].gather(1, tgt[:, None]).sum()                             # :91  (pos = ai-1)
model.zero_grad(); score.backward()                                         # :92
sal  = (emb.grad[0].float() * emb[0].float()).sum(-1).abs().detach().cpu()  # :93
tot  = sum(sal[i].item() for i in ci) or 1.0                                # :94  (ci = trace tokens)
... mass[s][c] += sal[i].item() / tot                                       # :99
```
So `s_j = |e_j · ∇_{e_j} log P(A|x,T)|`, normalised **per document** over the trace tokens, and
the target `A` is the model's own recorded answer tokens (right or wrong). Model parameters are
frozen (`:40–41`), gradients flow only into the input embeddings.

Token categories (`categorize`, `:15–27`), checked in this order: `SMILES_frag` (token inside
an RDKit-parseable substring of ≥ 6 characters, from `AA._smiles_tok_in`, `attention_attribution.py:400–407`),
`FG_word` (token inside any functional-group synonym, `AA._fg_tok_in`, `:388–397`), `space`,
`position_digit` (contains a digit), `other_word` (≥ 3 letters), `punct`.

**Enrichment**, the plotted quantity, is computed in `pull_fullvol.py:80–81`:
```python
enrichment = sh[c] / (tc[c] / tot)    # saliency share ÷ global token fraction
```
Note the honest caveat the authors record at `data/RESULTS.md:178–180`: the share is a
per-document average while the token fraction is global (per-document token counts were not
stored), so enrichment is an approximation.

Stratification is `all / er0 / erpos` (`attr_probe.py:49`, `:95`), full volume, **no**
correctness filter (`:45`).

Shipped values (`data/token_examples/r5_gradient.csv`, stratum `all`):

| model | n | SMILES share | SMILES enrich | FG_word enrich | position_digit enrich |
|---|---|---|---|---|---|
| Chem-R | 16 103 | 13.36 % | **2.476** | 0.775 | 1.600 |
| Chem-R-Faithful | 16 100 | 17.79 % | **3.215** | 0.715 | 1.655 |
| ChemDFM-R | 16 064 | **1.01 %** | 1.237 | 0.859 | 1.370 |
| ether-0 | 1 500 | 18.07 % | 1.599 | 0.587 | 0.720 |

All four numbers quoted in the paper (2.5×, 3.2×, 1.6×, 1.2×, 0.71–0.77×, 0.6×, 1.0 % mass)
reproduce. Note ether-0's n is **1500**, matching `methods.tex:50`.

### 5c — within-trace region attention

**Entry point.** `eval/attention_attribution.py:410–455` `region_attr()`, run through the
`--regions_only` branch of `main()` (`:492–515`).

**Algorithm.** One forward pass with `output_attentions=True`; take the **middle layer**, mean
over heads, restrict rows to answer tokens, mean over those rows:
```python
L   = len(out.attentions)                                                        # :430
seq = out.attentions[L // 2][0].mean(0).index_select(0, rows).mean(0)            # :431
```
Then, separately for the input-caption span `[d0, d0+len(desc))` and the trace span
`[t0+7, t1)` (`:448–449`), report the **per-token mean** of `seq` over: all tokens of the
region, its FG-word tokens, and its SMILES-fragment tokens (`region()`, `:434–446`).
`--regions_only` runs over **all** examples with no correctness or ER filter
(`:495`, comment "FULL volume") and checkpoints every 1000 (`:500–502`).

Shipped values (`data/token_examples/r5_region.csv`, stratum `all`, region `trace`, ×10⁻³):
Chem-R SMILES 0.754 / FG 0.132; Chem-R-Faithful 1.247 / 0.140; ChemDFM-R 0.848 / **0.967**
(the reversal the paper reports); ether-0 0.859 / 0.401 (≈2×). Input-region per-token attention
is 3.9–4.6 ×10⁻³ for the three Llama/Qwen-family models against 0.13–0.37 ×10⁻³ in the trace —
the ~11–45× input:trace asymmetry noted in `RESULTS.md:159–161`.

### 5a — token strips

`eval/token_examples.py:30–91` `analyze()` computes **both** gradient saliency (`:54–60`) and
mid-layer answer→token attention (`:63–67`) over the whole sequence for a handful of curated
examples, normalises each over input+trace only (answer excluded as self-referential,
`:74–77`), and dumps per-token records. Example selection is deterministic — first valid
example per `(task, ER-mode)` in `SPECS = [("cap2mol","er0"), ("cap2mol","erpos"),
("retrosynthesis","er0"), ("mol2cap","er0")]` (`:17`), with a token-count window of
`300 ≤ len(ids) ≤ 1500` (`:34`).

`make_nmi_figures.py:510–549` `_token_strip()` renders sheet `R5_token_examples`
(9685 rows), slicing an explicit fraction of the trace (`region=(0.70,1.0)` for
Chem-R-Faithful, `(0.55,1.0)` for ChemDFM-R, `:566–572`) or else auto-selecting the
`win = 120`-token window maximising convolved `SMILES_frag` saliency (`:523–527`).
`sal_norm` is what is coloured (`:517`), i.e. **gradient saliency**, not attention.

`eval/make_token_heatmap.py` renders an alternative attention-based heatmap
(`fig4c.pdf`) from the same JSON; it is **not** one of the eight figures produced by
`make_nmi_figures.main()`.

### Parameters
| parameter | value | source |
|---|---|---|
| saliency formula | `abs((grad·emb).sum(-1))`, per-doc L1-normalised over trace tokens | `attr_probe.py:93–94` |
| target | model's own answer tokens, summed log-prob | `attr_probe.py:89–91` |
| params frozen | yes | `attr_probe.py:40–41` |
| length cap (attr_probe) | 3000 tokens | `attr_probe.py:72` |
| length cap (region_attr) | 3500 tokens | `attention_attribution.py:419` |
| SMILES-fragment definition | RDKit-parseable substring, ≥ 6 chars | `attention_attribution.py:370`, `:400–407` |
| attention layer | `L // 2` | `attention_attribution.py:431`, `token_examples.py:66` |
| `--n_attn` (attr_probe) | 10⁹ = full volume | `attr_probe.py:34` |
| shuffle seed | 0 | `attr_probe.py:46`; `attention_attribution.py:65` |
| ether-0 subsample | 1500 (n reached, not a flag) | `data/raw/gradattr_ether-0.json` |
| token-strip window | 120 tokens | `make_nmi_figures.py:510` |

### Inputs
Same as §4 (`se_results/*/output.json` + details JSONL + HF weights).

### Output
`data/raw/gradattr_<model>.json` (aggregate only: `n`, `n_by_stratum`, `trace_saliency_frac`,
`token_counts`) and `data/raw/region_<model>.json:region_attr` → `eval/pull_fullvol.py` →
`data/token_examples/r5_gradient.csv`, `r5_region.csv` → sheets `R5_grad_enrichment`,
`R5_region_attention` → `figures/fig_scratchpad.pdf`.

### Reproduce
```bash
# GPU:
python eval/attr_probe.py --model Chem-R
python eval/attention_attribution.py --model Chem-R --regions_only
python eval/token_examples.py --model Chem-R          # per-token dump for panel a

# CPU:
python eval/pull_fullvol.py                            # r5_gradient.csv + r5_region.csv + fullvol.txt
cd figures && python -c "import make_nmi_figures as m; m.fig5_scratchpad()"
```

---

## 6. Fig 6 — the drafted-SMILES channel

### Claim
`results.tex:165–171`

> "We tested this structural component directly by perturbing only the SMILES written in the
> trace, leaving the functional-group prose intact and regenerating the answer on
> originally-correct responses that contain such a draft … Replacing the draft with a
> valid-but-wrong structure flips more answers than corrupting the functional-group claims:
> $11.4\%$ versus $5.7\%$ for Chem-R ($2\times$) and $31.9\%$ versus $1.8\%$ for
> Chem-R-Faithful ($18\times$) … the complete SMILES strings appear in $14$–$22\%$ of Chem-R and
> Chem-R-Faithful traces … most draft-writing traces contain only partial structure ($77\%$ …
> $65\%$) … corrupting the draft still flips $11.5\%$ and $34.6\%$ … ChemDFM-R … drafts a SMILES
> in only $8\%$ of correct traces, and corrupting that draft flips $7.3\%$."

### Entry point
- Draft detection and perturbation: `eval/cot_drift.py:137–199` (`find_draft_smiles`,
  `mask_draft_smiles`, `_corrupt_one`, `corrupt_draft_smiles`), wired in at `:251–256`.
- Aggregation: `eval/pull_draft.py:18–45`, and the workbook path
  `eval/build_source_data.py:80–104` `draft_frames()`.
- Panels: `figures/make_nmi_figures.py:620–711` `fig6_draft()`.

### Algorithm

**Draft detection** (`cot_drift.py:137–150`). A candidate is any
`[A-Za-z0-9@+\-\[\]()=#/\\%.]{6,}` run (`_SMILES_RE`, `:49`) that (a) is not already seen,
(b) contains at least one of `0-9 [ ] ( ) = #` — the "must look SMILES-y, not a plain word"
filter (`:145`), and (c) parses under RDKit with **≥ 4 atoms** (`:147–148`).

**Mask condition** (`:153–158`): every detected fragment replaced by the literal `[...]`.

**Corrupt condition** (`:161–199`): for each fragment, mutate up to **2** atoms' element with
the swap table
```python
swap = {6: 7, 7: 8, 8: 7, 16: 8, 9: 17, 17: 9, 35: 17, 15: 7}     # cot_drift.py:168
```
(C→N, N→O, O→N, S→O, F→Cl, Cl→F, Br→Cl, P→N), up to **8** attempts (`:170`), accepting the
first sanitisable result whose canonical SMILES differs from the original (`:184–186`). If no
fragment could be mutated, the condition is not emitted (`ok` flag, `:254–256`).

**Subset.** The paper's "same draft-writing subset" is
`base_correct == 1 and n_draft > 0` — `build_source_data.py:91–92`, mirrored in
`pull_draft.py:33–34`. `n_draft` is recorded per example at `cot_drift.py:262`/`:339`.
`coverage_pct_of_correct = len(draft)/len(bc)` (`build_source_data.py:93`) is Fig 6b.
Flip-to-wrong uses the same `_flip()` as §4b (`dperf == -1`).

Shipped values (sheet `R2_draft_perturbation`, `task_group == "all"` — which is what Fig 6
plots, `make_nmi_figures.py:622`):

| model | n (correct with draft) | coverage | all_wrong (FG name) | mask_draft | **corrupt_draft** | swap | drop |
|---|---|---|---|---|---|---|---|
| Chem-R | 4602 | 71.8 % | 5.71 | 9.89 | **11.36** | 33.72 | 22.36 |
| Chem-R-Faithful | 4298 | 64.7 % | 1.84 | 8.61 | **31.95** | 57.91 | 18.31 |
| ChemDFM-R | 430 | **8.2 %** | 6.28 | 6.98 | **7.26** | 19.77 | 31.16 |

**Fig 6c** cross-plots `corrupt_draft_flip_pct` against the SMILES `enrichment` from
sheet `R5_grad_enrichment` (`make_nmi_figures.py:657–677`) with a degree-1 `np.polyfit` line
(`:666`) — three points, descriptive.

**Fig 6d** is the `draft_copy` column of `R5_region_attention` — a **different** quantity from
`n_draft`: it is the fraction of examples whose canonical answer SMILES equals the canonical
form of some SMILES substring found in the trace, computed at
`attention_attribution.py:452–453`:
```python
ans_c = _canon_smiles(_answer_smiles(ex["answer"]))
copy  = bool(ans_c and any(_canon_smiles(s) == ans_c for s in rt.pop("_found")))
```
Values 0.1409 (Chem-R), 0.2206 (Chem-R-Faithful), 0.0011 (ChemDFM-R) → the paper's "14–22 %"
and "approximately zero".

**Fig 6e** reads sheet `R2b_flip_by_draftcopy` (`make_nmi_figures.py:698`), columns
`early_flip%` and `partial_corrupt_flip%`. Shipped: Chem-R n = 4577 (early 1031, partial 3546 →
**77.5 %** partial), overall corrupt-flip 11.4 %, early 10.8 %, partial **11.5 %**;
Chem-R-Faithful n = 4276 (early 1489, partial 2787 → **65.2 %**), overall 31.9 %, early 26.9 %,
partial **34.6 %**. Every number in `results.tex:169` reproduces. **There is no script in the
release that produces this sheet** — it requires joining `drift_<m>.json:per_example` with the
raw trace text to decide "early" per example, and the raw traces are not shipped. (Its n = 4577
also differs from `R2_draft_perturbation`'s 4602, so the join dropped 25 examples.)

### Parameters
| parameter | value | source |
|---|---|---|
| minimum fragment length | 6 chars | `cot_drift.py:49` |
| minimum fragment size | 4 heavy atoms | `cot_drift.py:148` |
| structural-character filter | must contain one of `0-9 [ ] ( ) = #` | `cot_drift.py:145` |
| atoms mutated per fragment | ≤ 2 | `cot_drift.py:175` |
| mutation attempts | 8 | `cot_drift.py:170` |
| element swap table | `{6:7, 7:8, 8:7, 16:8, 9:17, 17:9, 35:17, 15:7}` | `cot_drift.py:168` |
| mask placeholder | `"[...]"` | `cot_drift.py:153` |
| seed | 0 | `cot_drift.py:41` |
| models with draft results | Chem-R, Chem-R-Faithful, ChemDFM-R (+ether-0 allowed) | `build_source_data.py:88`; `pull_draft.py:10` |

### Inputs / Output
Inputs as §4b. Output `data/raw/drift_<m>.json` → `draft_result.txt` (`pull_draft.py:48`) and
sheet `R2_draft_perturbation` → `figures/fig_draft.pdf`.

### Reproduce
```bash
python eval/cot_drift.py --model Chem-R      # GPU (vLLM); emits mask_draft/corrupt_draft
python eval/pull_draft.py                     # CPU; writes draft_result.txt
cd figures && python -c "import make_nmi_figures as m; m.fig6_draft()"   # CPU
```

---

## 7. Conditional entropy (R3)

### Claim
`results.tex:163–165`

> "Although corrupting its functional-group names has little effect (Fig. 4), replacing the
> whole trace with another molecule's trace raises answer entropy by $0.05$–$0.32$; corrupting
> only the functional-group names changes it by approximately $0.03$ for the three Llama-family
> models."

`methods.tex:41–43`

> "for each example we drew eight answer samples (temperature $0.8$, top-$p$ $0.95$) under three
> trace conditions, empty, real and functional-group-corrupted, and, for the main models, a
> fourth condition that swaps in another molecule's whole trace … We report information gains
> relative to the real-trace condition."

### Entry point
`eval/cot_condsent.py:31–147` `main()`; entropy `:25–28`. Reuses `cot_drift`'s
`build_conditions`, `canon`, `markup_for` (imported at `:22`).

### Algorithm
1. Stage four prefixes per example: `base` (real CoT), `drop_cot` (empty), `all_wrong_cot`
   (every specific FG corrupted), and `swap_cot` when a donor is available (`:88`). The donor
   is another example's real inner CoT from the **same task**, retried up to 5 times to avoid
   picking the example's own (`:78–82`).
2. Sample: `SamplingParams(temperature=0.8, top_p=0.95, n=8, max_tokens=320, stop=[mk[3]])`
   (`:55`).
3. Cluster and score:
```python
def entropy(texts, task):
    c = Counter(CD.canon(task, t) for t in texts)
    n = sum(c.values())
    return -sum((k / n) * math.log(k / n) for k in c.values()) if n else None   # :25–28
```
`math.log` is **natural log**, so the units are nats (as `RESULTS.md:120` and the
`cot_condsent` docstring say). `CD.canon` (`cot_drift.py:92–100`) canonicalises the first
whitespace token of the answer through RDKit, or normalises whitespace/case for mol2cap.
4. Information gains, per example (`:113`, `:118`):
   `ig_presence = H_noCoT − H_realCoT`, `ig_content = H_corrCoT − H_realCoT`,
   `ig_swap = H_swapCoT − H_realCoT`.
5. Two aggregations are stored: `per_task` (`:130`) and a pooled `er_split`
   over all examples / ER=0 / ER>0 (`:132–135`).

**The paper plots the mean-over-tasks aggregation, not the pooled one.**
`eval/build_source_data.py:112–116`:
```python
def mot(k):
    vs = [v[k] for v in pt.values() if v.get(k) is not None]
    return round(sum(vs) / len(vs), 4) if vs else None
summ.append({"model": lab, "aggregation": "mean_over_tasks (PLOTTED in R3)", "n_tasks": len(pt), ...})
```
The pooled rows are written too, labelled `pooled_all_examples` / `pooled_ER=0` /
`pooled_ER>0` (`:118–122`). The difference is large: Chem-R `ig_presence` is **0.2617**
mean-over-tasks vs **0.1904** pooled; Chem-R-Faithful **0.4722** vs **0.2778**.

Shipped `mean_over_tasks` values (sheet `R3_condentropy`):

| model | ig_presence | ig_content | ig_swap |
|---|---|---|---|
| base-a (pre-SFT) | 0.2221 | 0.0346 | — |
| SFT | 0.3865 | 0.0293 | 0.2194 |
| Chem-R | 0.2617 | 0.0387 | 0.2104 |
| +process | 0.4120 | 0.0303 | — |
| Chem-R-Faithful | 0.4722 | 0.0371 | 0.3179 |
| ChemDFM-R | 0.1208 | 0.0433 | 0.0516 |
| ether-0 | 0.6376 | 0.0048 | — |

`ig_swap` spans **0.0516 → 0.3179** = the paper's "0.05–0.32"; `ig_content` for the three
Llama-family models (Chem-R 0.0387, Chem-R-Faithful 0.0371, SFT 0.0293) ≈ 0.03. The
`ig_swap` column is populated only where a swap donor existed and the model was run with it
(4 of 7 models — `n_tasks = 7`, `base-a` only 3).

`eval/cot_info_gain.py` is a **separate, two-condition** probe (free generation vs suppressed
reasoning) with `n_samples=8`, `sp_free` at `max_tokens=1600` and `sp_direct` at 320
(`:99–102`). Its output `data/raw/infogain_<m>.json` is not shipped and no paper number
depends on it.

### Parameters
| parameter | value | source |
|---|---|---|
| samples per condition | 8 | `cot_condsent.py:35`, `:55` |
| temperature / top-p | 0.8 / 0.95 | `cot_condsent.py:55` |
| max answer tokens | 320 | `cot_condsent.py:55` |
| entropy base | natural log (nats) | `cot_condsent.py:28` |
| clustering | exact canonical-SMILES equality (no similarity threshold) | `cot_drift.py:92–100` |
| tasks | 7 (`cot_drift.DEFAULT_TASKS`) | `cot_condsent.py:34` → `cot_drift.py:59–61` |
| donor retries | 5 | `cot_condsent.py:79–82` |
| plotted aggregation | mean over tasks | `build_source_data.py:112–116` |

### Inputs / Output
`se_results/<m>/<task>/output.json` + details JSONL (ER labels, `:68–72`) → `data/raw/condsent_<m>.json`
(`per_task`, `er_split`, `per_example`) → sheets `R3_condentropy` (19 rows), `R3_condentropy_task`
(45 rows). No figure panel; the numbers appear in `results.tex:163–165` prose.

### Reproduce
```bash
python eval/cot_condsent.py --model Chem-R        # GPU (vLLM), 4 prefixes x 8 samples per example
python eval/verify_paper_metric.py                # CPU; prints pooled vs mean-over-tasks side by side
```

---

## 8. The origin ladder

### Claim
`results.tex:88`

> "A comparison along one Chem-R lineage further suggests that hallucination predates
> chemistry-specific fine-tuning: the per-claim rate is highest in the original backbone model,
> is reduced by supervised fine-tuning and answer-only reinforcement learning, and falls to
> $2$–$3\%$ only once the reward grounds the trace's claims, even as task performance saturates
> far earlier (Fig. 2e)."

`methods.tex:79–82`: four points on one lineage, "each scored on the full task suite … we
measured the per-claim fabrication rate, ER, clean-trace fraction, task performance, and
trace-level hedging and abstention rates identified by regular expressions."

### Entry point
`eval/stage_ladder_metrics.py:103–141` `mechanism_stats()` and `:144–174` `main()`.
Panels: `figures/make_nmi_figures.py:220–245` (Fig 2e) and `:717–761` `ed_fig1_ladder()`.

### Algorithm
For each rung, read every response's `details.ER` and accumulate (`:113–129`):

```python
claimed_tot += len(er.get("claimed_fgs", []))
fab_tot     += len(er.get("fabricated_fgs", []))
...
perclaim_fab_rate = fab_tot / claimed_tot          # :136
claims_per_resp   = claimed_tot / n_resp           # :135
hedge_rate        = hedge / n_resp                 # :137
abstain_rate      = abstain / n_resp               # :138
fab_position      = mean(positions)                # :139
```

- `hedge` / `abstain` are regex hits over the `<think>` text: `_HEDGE` (`:57–60`:
  `may|might|maybe|perhaps|possibly|probabl|likely|appears?|seems?|suggest|could|presumably|
  tentativ|not sure|not certain|unclear|i believe|think|assume|guess`) and `_ABSTAIN`
  (`:61–65`: `cannot determin | unable to determin | not enough information | i don't know |
  hard to say | impossible to determin | without more information`).
- `fab_position` (`:88–100`) is the normalised character position of the earliest fabricated-FG
  mention in the `<think>` text, trying `fg`, `fg.replace("_"," ")`, `fg.replace("_","")`.
- `perf`, `ER`, `pct_er0` come from `eval/metrics.py:143` `family_stats(model, GEN_TASKS)`
  (`:155–158`), i.e. the official metrics, sample-weighted.

**Two things to get right about this metric.**

1. `perclaim_fab_rate` here **includes generic functional groups** — `claimed_fgs` and
   `fabricated_fgs` are used raw, with no `GENERIC_FG_NAMES` filter (`:117–118`). The Fig 2c
   per-claim fabrication rate (`100 − cp`, `eval/metrics.py:124–125`) **excludes** them. The
   two quantities carry the same name in the paper (Fig 2c and Fig 2e both say "per-claim
   fabrication") but are computed differently: for Chem-R over the same 12 tasks they are
   **19.34 %** (ladder, pooled Σfab/Σclaimed, generic included) and **22.56 %** (Fig 2c,
   unweighted mean of per-task precision, generic excluded).
2. **The ladder covers 12 task variants** — the three translation tasks plus the nine
   S²-Bench subtasks, `n_resp = 16107` = 3300 + 3300 + 5007 + 9 × 500 for every rung. The
   pre-SFT base model had never been evaluated on S² (its original job hit a 12 h wall-time);
   it was backfilled on 2026-08-01, after which the ladder was recomputed on the full suite.
   `GEN_TASKS` in `eval/stage_ladder_metrics.py` lists all 12, so the released script
   reproduces the shipped counts and rates.

Shipped `data/stage_ladder.csv`:

| model (internal) | stage | perf | ER | %ER=0 | n_resp | claims/resp | **per-claim fab** | hedge | abstain | fab_position |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct-base | base-a (pre-SFT) | 4.53 | 16.88 | 44.9 | 16107 | 3.48 | **34.65 %** | 48.1 % | 1.06 % | 0.371 |
| Chem-R-SFT | SFT | 42.32 | 11.61 | 38.3 | 16107 | 6.01 | **20.57 %** | 77.6 % | 0.03 % | 0.305 |
| Chem-R-v8 | +process | 51.72 | 2.43 | 81.7 | 16107 | 6.51 | **3.72 %** | 77.2 % | 0 | 0.335 |
| Chem-R-v8-coupled | +coupled | 51.45 | 1.73 | 88.0 | 16107 | 6.34 | **2.57 %** | 78.5 % | 0 | 0.308 |
| Chem-R | off-the-shelf GRPO | 50.09 | 10.60 | 40.6 | 16107 | 6.05 | **19.34 %** | 76.8 % | 0.01 % | 0.316 |

Fig 2e / ED Fig 1 plot the four rungs `["base-a (pre-SFT)", "SFT", "off-the-shelf GRPO",
"+coupled"]` (`make_nmi_figures.py:224`, `:725`) — 35 → 21 → 19 → 3 %. The `+process` rung is
deliberately omitted with a reason given in the code comment at `:719–724`.

`Chem-R-v8-coupled` is the internal codename of **Chem-R-Faithful** and `Chem-R-v8` of
`+process`; the mapping table lives at `data/raw/README.md`.

### Parameters
| parameter | value | source |
|---|---|---|
| ladder rungs | 5 declared (`LADDER`) | `stage_ladder_metrics.py:48–54` |
| task list in released code | 3 | `stage_ladder_metrics.py:55` |
| task list that produced the CSV | 12 | inferred from `n_resp = 16107` |
| hedge / abstain regexes | see above | `:57–65` |
| generic FGs in the per-claim rate | **included** | `:117–118` |

### Inputs
`data/results/<model>/<task>/*hallucination_details.jsonl` (claims) joined by id with
`se_results/<model>/<task>/output.json` (`<think>` text, `_load_text`, `:74–85`).

### Output
`data/stage_ladder.csv` → sheet `R1_stage_ladder` → `figures/fig2_widespread.pdf` panel e and
`figures/ed_fig1_ladder.pdf`.

### Reproduce
```bash
MOLREHALLU_REGEN=1 python eval/stage_ladder_metrics.py      # CPU; refuses without the guard (:37–42)
cd figures && python -c "import make_nmi_figures as m; m.ed_fig1_ladder()"
```
The guard exists because the script rewrites `data/stage_ladder.csv` **at module scope** and
silently drops rungs it cannot find.

---

## 9. The mitigation / verification-grounded reward

### Claim
`results.tex:215–216`

> "continuing Chem-R training with the same verifier as an online process reward reduces mean ER
> by $73$–$95\%$ across the four task families without a measurable loss on the reported task
> metrics (caption-to-molecule ER $6.7\to1.8$, retrosynthesis $15.1\to0.8$) … Chem-R-Faithful
> also shows greater SMILES-saliency enrichment ($2.5\times\to3.2\times$) and a higher
> draft-copy rate ($0.14\to0.22$)."

`methods.tex:66–77` gives the reward form, the weights $(0.1,0.4,0.4,0.2)$, the ER gate, and
the grounded term with cap $C=5$.

### Entry point
`reward/chem_merged_v8_ours.py:108–164` `compute_score()`, with `_grounded_term()` `:98–105`,
`_er_count()` `:85–95`, `_accuracy_signal()` `:76–82`.
Evaluation-side comparison: `eval/export_stats.py:85–97` → `data/mitigation.csv`.

### Algorithm

```python
W_FMT, W_ACC, W_ANTI, W_GROUND = 0.1, 0.4, 0.4, 0.2      # :40
GROUNDED_CAP = 5                                          # :41
COUPLED = os.environ.get("COUPLED", "0") == "1"           # :42
...
diag     = diagnose_one(task, {...}, verbose=True)        # :137–140  (the SAME detector)
halluc   = diag["overall_hallucination_score"]            # :141
er_score = diag["hallucination_scores"]["ER_factual_fabrication"]   # :142
acc      = _accuracy_signal(task, diag, metadata, pred, inp)        # :144
grounded = _grounded_term(diag)                                     # :145
fmt      = 1.0 if _FORMAT_PAT.search(ans.strip()) else 0.0          # :149
anti     = 1.0 - halluc / 100.0                                     # :150
acc_paid = acc if (not COUPLED or er_score == 0) else 0.0           # :153
overall  = (W_FMT*fmt + W_ACC*acc_paid + W_ANTI*anti + W_GROUND*grounded
            + LEN_BONUS*length)                                     # :156–157
```

**The gate is exactly `acc_paid = acc if (not COUPLED or er_score == 0) else 0.0`** (`:153`).
`COUPLED` is an environment variable, default `"0"` — the released file therefore computes the
**ungated** `+process` reward unless `COUPLED=1` is set. The gate fires on `er_score == 0`,
which by §1f includes fabricated *generic* groups and fabricated molecular-class claims (the
comment at `:151–152` says so).

**Format**: `_FORMAT_PAT = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)`
(`:34`) — a strict, standard-markup-only check; ether-0-style markup would score 0.

**Grounded term** (`:98–105`):
```python
n_ver, n_fab = _er_count(diag)
if n_ver == 0: return 0.0
precision = n_ver / (n_ver + n_fab)
count     = min(n_ver, GROUNDED_CAP) / GROUNDED_CAP
return 2.0 * count * precision                     # in [0, 2]
```
matching `methods.tex:77`. **Generic groups are excluded from both `n_ver` and `n_fab`**
(`_er_count`, `:93–94`):
```python
ver_specific = {fg for fg in verified   if fg not in GENERIC_FG_NAMES}
fabricated   = {fg for fg in fabricated if fg not in GENERIC_FG_NAMES}
```
So naming "a ring" earns nothing and costs nothing in the grounded term, while it *does* move
ER (and hence the gate and the anti-hallucination term). De-duplication is inherited from
the detector, whose `extract_chemical_entities` returns a set (docstring `:7–8`); `_er_count`
additionally wraps both sides in `set(...)`.

**Accuracy signal** (`:76–82`): `s2_success` for S², `caption_jaccard` for mol2cap, strict
`exact_match` for cap2mol/retrosynthesis — no Tanimoto fallback (comment `:81`).

**Failure mode**: any exception in diagnosis collapses to
`halluc, er_score, n_fab, acc, grounded = 100.0, 100.0, 99, 0.0, 0.0` (`:146–147`) → reward
`0.1·fmt`.

**Length bonus** (`:49–59`, `:155`) is a separate RL-from-base experiment; `LEN_BONUS`
defaults to `0` so `+process` and `+coupled` are unaffected.

### The GRPO run that produced Chem-R-Faithful

Not in the release; reconstructed from the working repo for the record.

- Launcher `MolLM_R_Hallucination_v1/jobs/training/easyr1_merged_v8_coupled.sh` — PBS
  `walltime=48:00:00`, `1 node × 4 GPUs × 300 GB` (`:5–6`); sets **`export COUPLED=1`**
  (`:20`) and nothing else relevant (`LEN_BONUS`, `LEN_TARGET`, `DRAFT_BONUS` unset → 0 / 1500
  / 0). Runs `python3 -m verl.trainer.main config=examples/config_merged_v8_coupled.yaml`
  (`:27`).
- Config `MolLM_R_Hallucination_v1/EasyR1-main/examples/config_merged_v8_coupled.yaml`:

| item | value | line |
|---|---|---|
| algorithm | GRPO (`adv_estimator: grpo`) | 18 |
| base checkpoint | `weidawang/Chem-R-8B` (local HF snapshot) | 33 |
| train / val data | `EasyR1-main/data/merged_4task/{train,test}.parquet` | 2–3 |
| max prompt / response length | 1024 / **2048** | 7–8 |
| rollout batch size | **256** | 9 |
| rollouts per prompt (`n`) | **5** | 51 |
| rollout temperature / top-p | 1.0 / 0.99 | 52–53 |
| learning rate | **1.0e-6** (AdamW, wd 1e-2, warmup 0.0) | 38–41 |
| KL | `use_kl_loss: true`, `kl_penalty: low_var_kl`, **`kl_coef: 1.0e-2`** | 20–22 |
| actor global batch | 128 (micro 4 update / 16 experience) | 26–28 |
| epochs / total steps | 3 epochs → **936 optimizer steps** (80 000 ÷ 256 = 312/epoch) | 76–77 |
| reward | `examples/reward_function/chem_merged_v8_ours.py:compute_score`, `reward_type: batch` | 72–73 |
| seed / shuffle | 1 / true | 13–14 |
| checkpoint dir | `/scratch/.../ckp/easyr1_merged_v8_coupled` (`global_step_936`) | 89 |

- Training data: `train.parquet` = **80 000 rows**, columns `problem / answer / task`, balanced
  20 000 each for cap2mol, mol2cap, retrosynthesis and the nine S² subtasks pooled
  (2170–2288 per subtask); `test.parquet` = 800. Built by
  `EasyR1-main/examples/make_merged_parquet.py:20–25` with `random_state=42` per source
  (`:46–47`) and a global shuffle at `:53–54`.
- The `+process` arm (`config_merged_v8.yaml`) is **byte-identical except for the experiment
  name and checkpoint path**; the only substantive difference is `COUPLED=0` in its launcher.
  That makes the ER gate the single controlled variable between `+process` and
  Chem-R-Faithful.
- Two arms exist in the working repo that the paper does not report: an
  RL-from-base "zero" run (`config_zero_{coupled,process}.yaml`, base
  `Llama-3.1-8B-Instruct`, `max_response_length: 10000`, `LEN_BONUS=0.1`,
  `jobs/training/train_zero.sh:35–36`) — no checkpoint on disk — and a draft-bonus ablation
  (`DRAFT_BONUS=0.15`, `easyr1_merged_v8_coupled_draft.sh:21`).
- The SFT rung has **no training config in either repo**: it is an externally supplied
  pre-GRPO checkpoint evaluated from `/scratch/t0937992/models/llama-3.1-8b-stage2`
  (`jobs/eval/eval_base_ladder.sh:44–47`).

**Evaluation of the mitigation** is `eval/export_stats.py:85–97`: for each family it emits
`(perf, ER, pct_er0, decoupling_gap, overall, cp, gc)` for the three arms
`MIT = [("Chem-R","baseline"), ("+process","process"), ("Chem-R-Faithful","coupled")]` (`:86`).
Shipped `data/mitigation.csv` gives cap2mol ER 6.663 → 1.812 (−72.8 %), mol2cap 7.198 → 1.708
(−76.3 %), retrosynthesis 15.102 → 0.814 (−94.6 %), s2 10.957 → 2.704 (−75.4 %) — the paper's
"73–95 %", with performance 41.03→42.36, 40.31→41.48, 50.03→49.93, 63.96→67.11.

### Parameters
| parameter | value | source |
|---|---|---|
| reward weights | 0.1 / 0.4 / 0.4 / 0.2 | `chem_merged_v8_ours.py:40` |
| grounded cap `C` | 5 | `:41` |
| gate | `COUPLED=1` env var, fires on `er_score == 0` | `:42`, `:153` |
| generic FGs in grounded term | excluded from numerator **and** denominator | `:93–94` |
| length bonus | `LEN_BONUS=0`, `LEN_TARGET=1500` chars | `:49–50` |
| format regex | `<think>…</think>\s*<answer>…</answer>` | `:34` |
| exception fallback | reward 0.1·fmt | `:146–147` |

### Inputs / Output
Called by the RL trainer with `(predicts, ground_truths, tasks, prompts)`; returns per-sample
dicts `{overall, format, accuracy, accuracy_raw, anti_hallucination, grounded, length}`
(`:158–163`). `accuracy` is the **paid** value and `accuracy_raw` the ungated one, so a
training log can separate the gate's effect.

### Reproduce
The reward function itself is CPU-only and testable with no weights:
```bash
COUPLED=1 python -c "
from reward.chem_merged_v8_ours import compute_score
print(compute_score(
  ['<think>The target has a carboxylic acid and a pyrrolidine ring.</think><answer>CC(=O)O</answer>'],
  ['CC(=O)O'], tasks=['cap2mol'], prompts=['Description: a molecule with a carboxylic acid']))"
```
The GRPO run itself is not reproducible from this release (§11).

---

## 10. Human validation

### Claims
`results.tex` does not report the human study; it appears as Extended Data Fig. 2 and in
`limitations.tex:4`:

> "We manually observed $97.3\%$ pooled extraction precision among recovered claims, but the
> claim-stratified audit does not estimate recall because it did not exhaustively annotate
> claims missed by the extractor."

ED Fig 2 caption (rendered by `make_nmi_figures.py:783`): "Automatic detector matches an expert
chemist ($n = 400$ prompts)".

### 10a — claim-level extraction precision (the 97.3 %)

**Entry point.** `human_eval/build_claim_set.py:107–174` `main()` (set construction) and
`human_eval/score_claims.py:35–82` `main()` (scoring).

**Algorithm.** `build_claim_set` walks 6 models × 3 tasks
(`MODELS` `:49–50` = Chem-R, ChemDFM-R, +process, Chem-R-Faithful, DeepSeek-R1, ether-0;
`TASKS` `:51` = cap2mol, mol2cap, retrosynthesis), shuffles ids under `random.seed(0)` (`:41`,
`:116`), and collects at most `MAX_FAB_PER_MT = 60` fabricated and `MAX_VER_PER_MT = 15`
verified claims per (model, task) (`:55–56`), each with a ±280-character context snippet
(`:80`, `:91`) and a rendered 320×240 SVG of the grounding molecule(s) (`:73`, `:129–139`).
It then samples `TARGET_PER_LABEL = 150` (`:52`) per label, shuffles, and splits the verdict
into a separate key (`:162–170`). The annotator sees only `{caption, cid, context, group,
images, task}` and answers `claims / no_claim / unsure` (`:255–257`).

`score_claims.py` tabulates against the key; the headline is
`prec = TP / (TP + FP)` with `TP = fabricated & claims`, `FP = fabricated & no_claim`
(`:54–55`), and the pooled figure is `(fab_claims + ver_claims) / 300` (`:69–71`).

**Current data.** `human_eval/claims_key.json` has exactly **300** entries, 150 `fabricated`
and 150 `verified`, across all 6 models. `human_eval/claim_annotations_RL.json` has one
annotator (`RL`) and 497 labels, of which **300** match the current key; the other 197 use the
pre-rename vocabulary (`present`/`absent`) and internal model names and are counted as
`unmatched`. The confusion table on the 300: fabricated/claims 145, fabricated/no_claim 5,
verified/claims 147, verified/no_claim 3 → fabrication precision **0.967**, verified-side
agreement **0.980**, pooled **292/300 = 97.3 %** — the limitations number.

### 10b — detector-vs-chemist arena (ED Fig 2)

**Entry point.** `MolLM_R_Hallucination_v1/make_latex_tables.py:728–770` `_human_agreement()`.
This function is **not in the release**; `eval/export_stats.py:148–156` imports it inside a bare
`try/except Exception: pass`, so in the release repo that block silently no-ops and
`data/human_eval_agreement.json` is a frozen artefact.

**Algorithm.** Read `human_eval/annotations_round1*.json` (else `annotations.json`) and
`human_eval/samples.json` (`:736–738`). Recompute each panel's aggregate with the same weights
`{IR 0.15, IO 0.25, ER 0.25, EO 0.35}` (`:743–748`) — note `build_annotation_set.panel()`
(`human_eval/build_annotation_set.py:106–108`) stores only the four dimensions, never an
`overall` key. Drop every panel whose label contains `"ours"` to reconstruct the round-1 option
set (`:755–757`). Then

```python
det_ov = min(panels, key=overall)["model"]
det_er = min(panels, key=lambda p: p["scores"]["ER_factual_fabrication"])["model"]   # :759–760
chance += 1.0 / len(panels)                                                          # :762
p0 = ao / n ;  pe = chance / n ;  kappa = (p0 - pe) / (1 - pe)                        # :767–769
```

i.e. Cohen's κ with the expected agreement taken as the mean random-pick probability over each
prompt's option set, not from a marginal confusion matrix.

**Shipped result** (`data/human_eval_agreement.json`):
```json
{"n": 400, "ag_ov": 54.75, "ag_er": 47.5,
 "k_ov": 0.41297..., "k_er": 0.31892..., "rand": 22.9166...}
```
`pe = (300·¼ + 100·⅙)/400 = 0.229167` (300 prompts with 4 surviving panels, 100 with 6).
`(0.5475 − 0.229167)/(1 − 0.229167) = 0.41297`. `make_nmi_figures.py:767–785` `ed_fig2_arena()`
reads this JSON directly.

**`human_eval/compute_kappa.py` computes no kappa.** Despite the name it prints raw agreement
percentages and a chance baseline only (`:48–54`); worse, `:25` does
`v = m["scores"]["overall"]`, a key `build_annotation_set.panel()` never writes, so it raises
`KeyError` on the shipped `samples.json` schema. `human_eval/arena.py:68–73` avoids this by
recomputing the aggregate from the weights, and additionally fits a Plackett–Luce model
(`luce_strengths`, `:27–50`, 500 MM iterations) mapped to Elo as `1000 + 400·log10(strength)`
(`:104–106`).

### Parameters
| parameter | value | source |
|---|---|---|
| annotation prompts | 100 per task × 4 tasks = 400 | `build_annotation_set.py:36–39` |
| annotation tasks | cap2mol, mol2cap, retrosynthesis, s2_MolCustom_FunctionalGroup | `build_annotation_set.py:36–38` |
| core / extra models | 4 core + 2 "ours" (additive, kept out of the shared intersection) | `build_annotation_set.py:27–35` |
| claim-set target | 150 per label, caps 60 fab / 15 ver per (model,task) | `build_claim_set.py:52`, `:55–56` |
| claim-set seed | 0 | `build_claim_set.py:41` |
| context window | ±280 chars | `build_claim_set.py:80` |
| Luce iterations | 500 | `arena.py:27` |
| κ chance model | mean of `1/len(panels)` per prompt | `make_latex_tables.py:762`, `:767` |

### Inputs / Output
Claim track: `se_results`/`data/results` → `human_eval/claims_key.json` +
`annotate_claims.html` → `claim_annotations_<who>.json` → `human_eval/claim_reliability.json`.
Arena track: `human_eval/samples.json` + `annotations*.json` → `data/human_eval_agreement.json`
→ `figures/ed_fig2_arena.pdf`.

### Reproduce
```bash
# CPU throughout.
python human_eval/build_claim_set.py                                   # needs the full tree
python human_eval/score_claims.py human_eval/claim_annotations_*.json  # -> claim_reliability.json
cd figures && python -c "import make_nmi_figures as m; m.ed_fig2_arena()"
```
`build_claim_set.py` and `build_annotation_set.py` cannot run in the release repo: there is no
`se_results/` and `data/results/` contains no `output.json`, so their `load()` returns
`(None, None)` for every (model, task). `samples.json` and `annotations*.json` are gitignored
(`.gitignore:8–11`) and exist only in the working repo, so `arena.py`, `compute_kappa.py` and
`_human_agreement()` cannot be re-run here either.

---

## 11. What is NOT reproducible from the release alone

1. **The generation tree `se_results/` is absent.** Every mechanism probe
   (`attention_attribution.py:76`, `attr_probe.py` via `AA.load_examples`, `cot_drift.py:299`,
   `cot_condsent.py:60`, `cot_info_gain.py:106`, `stage_ladder_metrics.py:76`,
   `token_examples.py`, both `human_eval` builders) reads
   `se_results/<model>/<task>/output.json`. None of them can start. Producing it needs
   `run_multitask_se.py` **plus** `data_loaders.py` and `semantic_entropy.py`, which the
   release deliberately omits (`README.md:14–16`), plus the ChEBI-20 / USPTO-50k / S²-TOMG-Bench
   corpora, which are not redistributed.

2. **The diagnosed tree `data/results/` is a one-directory sample.** Only
   `data/results/Chem-R/cap2mol/Chem-R_cap2mol_hallucination_details.jsonl` ships. That is
   enough for Fig 3a,b and nothing else. `eval/export_stats.py`, `eval/stage_ladder_metrics.py`
   and `eval/build_source_data.py` all raise `SystemExit` without `MOLREHALLU_REGEN=1`
   precisely because running them against this subset would overwrite the submitted CSVs with
   one model's worth of numbers.

3. **Three model checkpoints are released separately.** `HF` maps
   `+process`, `Chem-R-Faithful` and `Chem-R-SFT` to the literal strings
   `"<released separately: …>"` (`attention_attribution.py:47–48`, `:50`; identical maps in
   `cot_drift.py:66–70` and `cot_info_gain.py:43–46`). Any GPU probe on those arms fails at
   `from_pretrained`. Only `Chem-R` (`weidawang/Chem-R-8B`), `ChemDFM-R`, `ether-0` and
   `Llama-3.1-8B-Instruct` resolve.

4. **The EasyR1/verl training stack is not here.** `reward/chem_merged_v8_ours.py` is the reward
   plug-in only; there is no trainer, no GRPO config, no 80k training file. The reward's own
   docstring and `README.md:23–24` say so. The whole `EasyR1-main/` tree (verl trainer, all
   `examples/config_*.yaml`, `examples/make_merged_parquet.py`,
   `data/merged_4task/{train,test}.parquet`) and all 68 PBS scripts under `jobs/` are
   working-repo only. §9 records the configuration.

5. **Semantic entropy has no implementation in the release.** `eval/export_stats.py:44–51`
   only *reads* `se_summary.json`. `MolLM_R_Hallucination_v1/{semantic_entropy.py,
   data_loaders.py}` are both absent, and the released `run_multitask_se.py:27–32` imports
   them — so the shipped generation driver cannot run as shipped. §2 records the clustering
   rules and sampling parameters; the dataset files (`data/chebi-20/`, `data/uspto50k/`,
   `data/s2-bench/`, `data/moleculenet/`) are not redistributed either.

6. **Six workbook sheets have no producer in the release**, acknowledged at
   `data/SOURCE_DATA.md:49–53`: `R5_token_examples`, `R2_restated_derived`,
   `R2b_flip_by_draftcopy`, `Fig3_decoupling_perresponse`, `R2_paired_flips`,
   `Fig1_hallucination_by_model`. A repo-wide grep for those sheet names matches only
   `figures/make_nmi_figures.py`, i.e. the consumer. Of these, `R2b_flip_by_draftcopy` backs
   Fig 6e and `R2_restated_derived` backs a quoted range in `results.tex:143`, so two published
   quantities are terminal data.

7. **`region_*.json` ships for 4 models but `attention_perturbation.csv` has 6 rows.**
   `data/raw/` contains `region_{Chem-R, Chem-R-Faithful, ChemDFM-R, ether-0}.json`; the CSV
   also has `+process` and `DeepSeek-R1`. Re-running `eval/export_stats.py` on the release
   would drop those two rows.

8. **`make_latex_tables.py` is not in the release**, so `eval/export_stats.py:148–156` is dead
   code here and `data/human_eval_agreement.json` cannot be regenerated (§10b).

9. **McNemar tests and bootstrap CIs.** `results.tex:141` cites `p<10^{-15}` and
   `methods.tex:39` "paired McNemar tests"; `data/RESULTS.md:69`, `:80–82` cite 2000-resample
   bootstrap CIs. No released script computes either. Sheet `R2_paired_flips` (3927 rows,
   columns `model, task, uid, flip_trace, flip_input`) holds the paired data the test would
   consume, but not the test.

What **is** end-to-end verifiable on CPU with no weights and no data: the detector
(`diagnose_multitask.diagnose_one`), the reward (`reward.chem_merged_v8_ours.compute_score`),
`eval/pull_fullvol.py`, `eval/pull_draft.py`, `eval/verify_paper_metric.py`,
`human_eval/score_claims.py`, and all eight figures from `data/source_data.xlsx`.

---

## 12. Where the released code differs from what produced the numbers

Ordered by how much a reader running the code would be misled.

1. **`eval/stage_ladder_metrics.py` task list — RESOLVED 2026-08-02.** The script briefly
   declared `GEN_TASKS = ["cap2mol", "mol2cap", "retrosynthesis"]` (11 607 responses) while the
   shipped `data/stage_ladder.csv` already carried `n_resp = 16107` — the same three tasks plus
   the nine S² subtasks — because the CSV was regenerated from a patched copy before the repo
   script was updated. `GEN_TASKS` now lists all 12 variants, and the stale labels in
   `data/RESULTS.md:12`, `:50` and `eval/build_source_data.py:149` were corrected to 16,107.
   Script, data and figures now agree; the rates Fig 2e and ED Fig 1 plot are
   34.65 / 20.57 / 19.34 / 2.57 %.

2. **`reward/chem_merged_v8_ours.py` is a trimmed copy of the file that trained the model.**
   The working-repo original
   (`MolLM_R_Hallucination_v1/EasyR1-main/examples/reward_function/chem_merged_v8_ours.py`,
   210 lines) additionally defines `DRAFT_BONUS` (`:69`), `_SMILES_RE` and `_draft_coverage()`
   (`:62–104`), computes `draft = _draft_coverage(...)` for cap2mol/retrosynthesis (`:199–201`),
   adds `+ DRAFT_BONUS * draft` to the total (`:203`) and returns a `draft_cov` key (`:208`).
   The released 164-line copy strips all of it, and changes `_MAIN`'s default from the absolute
   project path to `"."` (`:23`). Because `DRAFT_BONUS` defaults to `0` and the
   Chem-R-Faithful launcher does not set it
   (`jobs/training/easyr1_merged_v8_coupled.sh:20`), **the two files compute identical rewards
   for the published run** — but the released file cannot reproduce the separate draft-bonus
   ablation, and its aggregate no longer matches the term list in the working-repo original.
   Everything else — `W_FMT/W_ACC/W_ANTI/W_GROUND`, `GROUNDED_CAP`, `COUPLED`, `LEN_BONUS`,
   `LEN_TARGET`, the gate at `:153` — is unchanged.

3. **`eval/build_source_data.py` produces 13 sheets; the shipped workbook has 18.** The five
   extra sheets (§11.6) are written by nothing in either repo. The working-repo copy
   (`MolLM_R_Hallucination_v1/eval/build_source_data.py`) also produces only 13 — the release
   diff is confined to paths, the `MOLREHALLU_REGEN` guard, `LABEL` (internal→display names)
   and `EXCLUDE_MODELS`.

4. **`eval/build_source_data.py:49` `PE_CONDS` omits `wrong_input`**, with a comment at `:164`
   saying it "is not in per_example, so omitted here". The shipped sheet `R2_drift`
   nevertheless contains 13 `wrong_input` rows with the label "FG-name corrupted in the input
   (positive control)" — a string that appears nowhere in `COND_DESC` (`:32–36`). Fig 4b does
   not consume them, but `results.tex:140`'s "approximately 40 %" does.

5. **`ed_fig1_ladder`'s annotation and the ladder y-limits were hard-coded and stale; both were
   fixed on 2026-08-02 while this document was being written.** The 2026-08-01 file had
   `axa.annotate("base fabricates most\n(at 0.3% accuracy)", ...)` and
   `set_ylim(0, 34)` / `set_ylim(0, 60)`, whereas the recomputed 12-task ladder gives the base
   model `perf = 4.53` (point 1). The current file interpolates the value
   (`figures/make_nmi_figures.py:741`, `"…(at %.1f%% accuracy)" % perf[0]`) and derives every
   ladder y-limit from the data through a new `_headroom()` helper (`:788–791`, used at `:234`,
   `:239`, `:740`, `:750`, `:755`). `data/RESULTS.md` was updated to match in the same pass. All other `make_nmi_figures.py` line citations in
   this document were re-verified against the 2026-08-02 file and are unchanged; only the
   region after line 785 shifted.

6. **Fig 2b's `r = -0.54` is hard-coded** (`make_nmi_figures.py:184`), not computed.
   Recomputation over the five plotted points gives −0.533 / −0.535 depending on which
   aggregate is used. (Fig 3a's Pearson, by contrast, *is* computed live at `:347`.)

7. **`data/RESULTS.md` is dated 2026-07-04 and is stale in places** relative to the shipped
   data: the ladder (point 1), and `RESULTS.md:104–106`'s draft table (`coupled` 65 %/4298,
   `ChemDFM` 8 %/430) which does match `R2_draft_perturbation`, but `RESULTS.md:73` quotes
   Chem-R translate `all_wrong` as 7.4 % where the sheet says 6.95 %.

8. **Detector-vs-Methods disagreements** (code is authoritative; Methods is incomplete or
   wrong):
   - `methods.tex:20` states the aggregate as a fixed linear form. The code **renormalizes**
     over the surviving dimensions when IO/EO are skipped on an invalid or missing SMILES
     (`diagnose_hallucination.py:867–896`; `diagnose_multitask.py:73–81`). An invalid molecule
     therefore contributes 0 to IO/EO and the response is scored as `0.375·IR + 0.625·ER`.
   - `methods.tex:28` calls κ "a fixed per-task scale" without giving values. It is **75** for
     cap2mol (`diagnose_hallucination.py:716`, `:744`) and **60** for every other task
     (`diagnose_multitask.py:160, 221, 307, 399, 654`). A given fabrication fraction scores
     25 % lower on retrosynthesis than on cap2mol.
   - `methods.tex:17` says a claim is fabrication only when absent from "the relevant input
     evidence, the prediction **and the reference**". Only cap2mol consults the reference
     (`gt_fgs`, `diagnose_hallucination.py:700`, `:705`). Retrosynthesis grounds on
     product ∪ *predicted* reactants (`diagnose_multitask.py:293–299`), S² on source ∪
     instruction ∪ prediction (`:640–646`), mol2cap and classification on the input molecule
     alone (`:144–145`, `:211–212`). The reference is never consulted on those tasks.
   - `methods.tex:11` describes "rule-based derivation context" as part of the pipeline. It is
     applied **only on cap2mol** (`exclude_derivation=True` at
     `diagnose_hallucination.py:626`, `:688`); the five other diagnosers pass the default
     `False` (`diagnose_multitask.py:147, 213, 300, 392, 647`).
   - `methods.tex:20` defines "per-claim fabrication rate" once. Two different implementations
     carry that name: generic-**excluded**, per-task-then-averaged (`eval/metrics.py:124–125`,
     Fig 2c) and generic-**included**, pooled (`eval/stage_ladder_metrics.py:117–118`,
     Fig 2e). For Chem-R over the same 12 tasks: 22.56 % vs 19.34 %.
   - `methods.tex:11` says the extractor "identifies transformation statements"; the code
     detects derivation clauses only to *suppress* claims (`_DERIV_RE`,
     `diagnose_hallucination.py:402–407`). No transformation is ever scored as a claim in its
     own right.
   - `methods.tex:20` says "grounded claims is the number of distinct verified **specific**
     claims" — matched by `eval/metrics.py:97–99`, and by the reward's `_er_count`
     (`reward/chem_merged_v8_ours.py:93–94`). Consistent; noted because it is the one place
     the generic-exclusion is stated.
   - `methods.tex:36` says semantic entropy uses "caption-level equivalence for text
     generation". The implementation clusters the **`<think>` reasoning text**, not the
     caption (`MolLM_R_Hallucination_v1/semantic_entropy.py:213`), at threshold 0.80 with a
     whitespace-token-Jaccard similarity (the driver supplies no embedding model). The result
     is saturated: mol2cap SE = ln(10) = 2.3026 for four of the five surveyed models. That one
     task is a constant inside every model's twelve-task mean.
   - `methods.tex:5` says S²-Bench "uses 500 prompts (mini size) per subtask for tractability".
     The selection is a head slice of the CSV in file order with no shuffling and no seed
     (`MolLM_R_Hallucination_v1/data_loaders.py:359`), i.e. prompts 1–500 of 5000.

9. **`eval/export_stats.py:32–34` defines a `CLS` classification family** (bace/bbbp/hiv/
   tox21/clintox) and writes those rows into `stats_per_model_task.csv`. No paper figure uses
   them — `make_nmi_figures.py:137`, `:256` filter to `GEN12`. `methods.tex:5` does not mention
   them.

10. **`human_eval/compute_kappa.py` cannot run** against the shipped `samples.json` schema
   (`KeyError: 'overall'` at `:25`) and computes no κ despite its name. The paper's κ comes
   from `make_latex_tables._human_agreement()`, which is not in the release.

11. **Model naming.** `data/stage_ladder.csv` uses internal codenames
    (`Chem-R-v8`, `Chem-R-v8-coupled`) while `eval/stage_ladder_metrics.py:48–54` uses release
    names (`+process`, `Chem-R-Faithful`); the figure code keys on the `stage` column, so this
    is harmless for plotting but breaks any join on `model`. `data/raw/README.md` is the
    authoritative mapping.

---

## 13. Unresolved — claims whose implementation I could not locate

- **`results.tex:98`, mol2cap / retrosynthesis ER–correctness correlations (−0.04, −0.05).**
  No script computes a per-task Pearson between ER and the answer score. `eval/metrics.py`
  exposes `perf_er0`/`perf_erpos` but not a correlation. Sheet `Fig3_decoupling_perresponse`
  contains the necessary per-response `(model, task, ER, exact_match)` rows for all three
  translation tasks (11 607), so the numbers are checkable, but the code that produced them is
  not in either repo.

- **`results.tex:141`, `p<10^{-15}` and the "paired gaps of 15–66 percentage points".**
  `methods.tex:39` promises paired McNemar tests. Nothing in the release runs one. Sheet
  `R2_paired_flips` holds the paired indicators; `data/RESULTS.md:80–82` reports the result.

- **`data/RESULTS.md:69`, "95 % bootstrap CI, 2000 resamples".** No bootstrap anywhere in the
  release. Fig 4b uses a Wald interval computed in the plotting code
  (`make_nmi_figures.py:473`) instead, and the figure caption says so
  (`results.tex:155`).

- **Fig 6e's early-vs-partial split (sheet `R2b_flip_by_draftcopy`).** The "early answer"
  predicate must be a per-example draft-copy test (canonical answer SMILES present in the
  trace), but `drift_<m>.json:per_example` stores only `n_draft`, and `region_<m>.json`'s
  `draft_copy` is not keyed by example id. No script performs the join. Its n (4577) differs
  from `R2_draft_perturbation`'s (4602), confirming a separate computation.

- **`methods.tex:59`, "In a layerwise sensitivity check, the input-versus-trace asymmetry has
  the same direction at every layer except the first and peaks in the middle of the network."**
  The per-layer lists needed for this check are stored in `region_<m>.json:matched` and
  `:attn`, but no released script performs or reports the layerwise sweep.

- **`results.tex:196–198`, ChemDFM-R's "scaffold, positional, and nomenclature cues".** The
  token categoriser (`eval/attr_probe.py:15–27`) has no such category; `other_word` is a
  catch-all. The authors flag this themselves at `data/RESULTS.md:183–187` and
  `limitations.tex:6`. There is no code that isolates these cues.

- **The `+process` arm's `region_*.json` and `condsent_*` swap runs.** `attention_perturbation.csv`
  has a `+process` row (n_perturb 1258) but `data/raw/` has no `region_process.json`; sheet
  `R3_condentropy` shows `ig_swap = NaN` for `+process`, `base-a` and `ether-0` because the
  swap condition was only run for three models (`cot_condsent.py:88` emits `swap_cot` only when
  a donor exists, and the runs differ). This is a coverage gap, not a missing implementation.

Two items that were open until the working repo was inspected, now resolved and recorded above
rather than here: the semantic-entropy clustering rules and sampling parameters
(`methods.tex:36` → §2, `MolLM_R_Hallucination_v1/semantic_entropy.py:159–262`, `:418–444`),
and the GRPO training configuration (`methods.tex:66–77` → §9,
`MolLM_R_Hallucination_v1/EasyR1-main/examples/config_merged_v8_coupled.yaml`). Neither is part
of the public release.
