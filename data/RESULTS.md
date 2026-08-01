# Results — summary + detail (verified 2026-07-04)

Every number below is cross-checked against the raw output files. For each block:
**N (data volume)** and **how the metric is computed**. Regenerate via `eval/*` scripts (see EXPERIMENTS.md).

---

## Data volumes (N)
| Block | N | notes |
|---|---|---|
| Diagnosis test sets (per model) | cap2mol 3300, mol2cap 3300, retro 5007, each s2 subtask 500 (×9) | same sets for all models |
| Stage ladder (per-claim rates) | n_resp 16,107 over all 12 task variants; denominators = total claimed FGs (44k–78k) | **answer-only = cap2mol only, n_resp 3300** (see caveat) |
| Drift (per model) | ~13,500 total; originally-correct: translate ~4–5k, s2 ~1.1–1.5k | base-a only 14 correct (perf≈0 → uninformative) |
| Draft-SMILES perturbation (R2b) | subset = originally-correct WITH a drafted SMILES: Chem-R 4602, coupled 4298, ChemDFM 430 | ChemDFM 8% coverage = negative control |
| Cond-entropy (3-cond) | ~13,500 examples × **8 samples** × 3 prefixes/model | metric-free |
| Info-gain (2-cond) | ~10,800 examples × 8 samples | |
| Attention/perturbation (Δlogp) | n_perturb FULL: Chem-R 2182, ChemDFM 3300, v8 5047, coupled 5292, ether-0 815, DeepSeek 83 | DeepSeek 83 = unusable |
| **Token-type saliency + region-attn (R5)** | **FULL, ER-stratified**: grad Chem-R 16,103 / coupled 16,100 / ChemDFM 16,064; region 16,100 / 16,098 / 16,061 (of 16,107; rest = long/fmt skips) | descriptive probe → no correctness/ER filter (re-run 2026-07-03, replaces earlier n=300/400 sample) |

## Metric definitions (how computed)
- **ER / fabrication**: a claimed functional group is *fabricated* iff, by exact **RDKit SMARTS** matching, it is
  absent from BOTH the input (molecule + caption-named groups) AND the output (pred + GT). ER score ∈ [0,100];
  **per-claim fab-rate = Σ fabricated / Σ claimed** FGs (category-deduped, generic FGs excluded). **%ER=0** =
  fraction of responses with ER score 0.
- **perf (task accuracy)**: cap2mol / retro = canonical-SMILES **exact match**; mol2cap = caption token similarity;
  s2 = official **S2-TOMG success** (`s2_success`). [caveat: s2 edit/opt success does NOT yet weight by
  similarity-to-source — see below.]
- **Decoupling**: perf computed separately on ER=0 vs ER>0 traces.
- **Drift** (metric-dependent, causal): regenerate the answer (greedy) from a perturbed prefix; **flip-to-wrong**
  = among originally-correct responses, fraction that become incorrect; Δperf = mean(cond_correct − base_correct).
  Conditions: syn_cot (ctrl), all_wrong_cot (corrupt all FG = CONTENT), drop_cot (empty = PRESENCE), wrong_input.
- **Cond-entropy** (metric-FREE): per example, 8 samples from {empty / real / corrupted CoT}; answer entropy =
  Shannon over canonical-SMILES clusters. **ig_presence = H(noCoT) − H(realCoT)**; **ig_content = H(corrupt) − H(realCoT)**.
- **Info-gain (2-cond)**: H_free (free generation) − vs H_noCoT; info_gain = H_noCoT − H_free.
- **Perturbation Δlogp**: change in teacher-forced log-prob of the ORIGINAL answer under each perturbation.
- **Gradient×input saliency (R5, causal-importance)**: |∂ logP(answer)/∂ emb · emb| summed over answer tokens,
  attributed to each TRACE token, categorized (SMILES-frag / FG-word / position-digit / other-word / punct).
  Per doc the saliency is normalized to sum 1 over trace tokens; **share** = mean per-doc share on a type;
  **enrichment = share / token-fraction** (>1 ⇒ that token TYPE carries more answer-sensitivity *per token* than
  the average trace token — this is the fair metric; absolute share is confounded by token count). Model params
  frozen (grad only w.r.t. input embeddings). Answer = the model's OWN generated answer (right or wrong).
- **Region attention (R5)**: per-token mean attention from answer tokens to all / FG-word / SMILES-frag tokens,
  in the INPUT region vs the reasoning-TRACE region (mid-layer, eager). **draft-copy** = fraction where the final
  answer SMILES appears (canonically) somewhere in the trace. All ER-stratified (all / ER=0 / ER>0).
- **Ladder mechanism**: hedge% / abstain% = regex over the `<think>` text; fab_position = normalized char
  position of the first fabricated-FG mention; claims/resp = mean distinct claimed FGs.

---

## R1. Stage ladder (origin of hallucination) — all 12 task variants, n_resp 16,107 (answer-only: cap2mol 3300)
| stage | perf | ER | %ER=0 | claims/resp | per-claim fab | hedge% | abstain% |
|---|---|---|---|---|---|---|---|
| base-a (pre-SFT) | 0.3 | 13.09 | 48 | 3.85 | 28.9% | 51.9% | 1.5% |
| SFT | 34.5 | 10.05 | 40 | 6.58 | 18.9% | 79.5% | 0.0% |
| answer-only GRPO* | 43.9 | 6.27 | 68 | 4.76 | 9.1% | 31.1% | 0.0% |
| +process | 46.0 | 2.15 | 81 | 6.74 | 3.4% | 80.2% | 0.0% |
| +coupled | 45.4 | 1.35 | 88 | 6.57 | 2.1% | 82.2% | 0.0% |
| off-the-shelf Chem-R | 44.7 | 10.46 | 38 | 6.58 | 19.4% | 78.3% | 0.0% |

*answer-only = cap2mol only (not comparable across all 3 tasks). **Take-away:** fabrication is highest in the
base (confabulation-under-ignorance, perf 0.3%); SFT reduces it (28.9→18.9%) while eliminating abstention
(1.5→0%); standard training plateaus at 9–19%; only verification-grounded reward → 2–3%.

## R2. Drift — FG-CLAIM perturbation (flip-to-wrong among originally-correct)
SCOPE: we perturb **functional-group claims** (not "CoT content" — the drafted SMILES / reasoning logic are
untouched; whole-CoT content is the **swap_cot** test, now DONE — column below). Conditions: `syn`=synonym (control),
`all_wrong`=corrupt every FG-name claim in the CoT, `swap`=replace the CoT with another molecule's REAL CoT (whole
content), `drop`=empty CoT (presence, OOD-confounded), `wrong_input`=corrupt the FG in the INPUT. % = flip-to-wrong
among originally-correct, [95% bootstrap CI, 2000 resamples]; swap = full-volume point estimate (CI recomputable from `raw/`).

| model / group | syn (ctrl) | all_wrong (FG claim in CoT) | swap (other-mol whole CoT) | drop (presence, OOD) | wrong_input (FG in input) |
|---|---|---|---|---|---|
| Chem-R / translate | 2.1 [1.7,2.5] | 7.4 [6.6,8.1] | **32.7** | 25.9 [24.7,27.1] | **39.8 [36,43.5]** |
| Chem-R / s2 | 1.5 | 4.3 [3.4,5.4] | **24.5** | 10.2 | **41.6 [37.6,45.4]** |
| coupled / translate | 1.0 | 3.5 [3.0,4.0] | **48.5** | 20.6 | **42.1 [38.3,46]** |
| coupled / s2 | 0.1 | 0.9 [0.5,1.4] | **40.4** | 9.0 | **15.7 [12.5,19.4]** |
| ChemDFM / translate | 2.6 | 6.7 [6.0,7.5] | **15.1** | 21.0 | **40.9 [37.7,44]** |
| ChemDFM / s2 | 3.2 | 10.7 [8.9,12.5] | **21.4** | 27.8 | **79.5 [76.4,82.4]** |

**Paired contrasts (McNemar exact + paired-bootstrap Δ CI):**
- **wrong_input − all_wrong (FG in input vs same FG in CoT): Δ = +15 to +68 pp, all SIG (p < 1e-15).**
- all_wrong − syn (FG claim in CoT vs control): Δ = +1 to +7 pp, small but SIG (p 1e-4 to 1e-47).

**Restated vs derived FG claims (Δlogp, teacher-forced):** corrupting a CoT FG claim barely moves the answer
whether the FG is **restated** from the input (also in input; Δlogp −0.001 to −0.009) or **derived** by the
model (CoT-only; −0.0002 to −0.0013) — both ≈0 vs input's −0.23.

**Take-away (careful, scoped to FG claims):** the answer's dependence on a functional-group claim is far higher
when the FG is specified in the **input** than when it appears in the **reasoning trace** (Δ +15–68 pp, p<1e-15);
within the trace, **both restated and derived FG claims are causally inert** (flip +1–7 pp over control; Δlogp ≈0)
— NOT merely redundant restatement. But the **whole CoT is NOT irrelevant**: swapping in another molecule's entire
CoT flips the answer **33–48% (Chem-R/coupled)** — the reasoning's **structural draft is load-bearing** even though
its FG-name *claims* are not (two-channel; the draft channel is characterized in R5). Training does not increase
FG-claim dependence (coupled lowest); ChemDFM uses the draft channel least (swap 15–21%). base-a uninformative (perf≈0).

## R2b. Draft-SMILES perturbation — DIRECT test of the structural-draft channel (FULL)
Perturb ONLY the SMILES the model drafts in its trace (FG prose left intact), regenerate, measure flip-to-wrong
among originally-correct examples that actually draft a SMILES. `mask`=draft→`[...]` (presence); `corrupt`=draft→
valid-but-structurally-wrong SMILES (content/direction). Same subset for the FG-name (`all_wrong`) and whole-CoT
(`swap`) comparisons. ChemDFM = negative control.

| model | draft coverage (of correct) | all_wrong (FG name) | mask (draft) | corrupt (draft) | swap (whole CoT) |
|---|---|---|---|---|---|
| Chem-R | 72% (4602/6406) | 5.7 | 9.9 | 11.4 | 33.7 |
| coupled | 65% (4298/6647) | 1.8 | 8.6 | **31.9** | 57.9 |
| ChemDFM (neg ctrl) | **8%** (430/5250) | 6.3 | 7.0 | 7.3 | 19.8 |

(translate/s2 split in `source_data.xlsx` → sheet `R2_draft_perturbation`.)

**Take-away:** corrupting the drafted SMILES flips the answer far more than corrupting the FG-name claims —
Chem-R **11.4 vs 5.7** (2×), coupled **31.9 vs 1.8** (**18×**) — the DIRECT causal counterpart to R5 (previously
only *inferred* from gradient/attention): the **structural draft is load-bearing; the FG prose is not**.
`corrupt > mask` (the model FOLLOWS the wrong structure, not merely misses a removed one), strongest on coupled,
which also has the highest R5 SMILES-enrichment (3.2×) and draft-copy (0.22) — cross-validates R5. **Negative
control:** ChemDFM barely drafts SMILES (only **8%** of its correct traces) and shows NO draft-specific dependence
(corrupt 7.3 ≈ FG 6.3 ≈ mask 7.0) → model-specific, no structural-draft channel. All effects remain secondary to
the input (cf. R2 wrong_input 40–79%).

## R3. Cond-entropy (metric-free) — mean over tasks
ig_presence = H(noCoT)−H(realCoT); ig_content = H(corrupt FG-names)−H(realCoT); ig_swap = H(other-mol whole CoT)−H(realCoT).
swap only run on the 3 main models (—  = not run). Values are the mean-over-tasks aggregation (the plotted one).
| model | ig_presence | ig_content | ig_swap |
|---|---|---|---|
| base-a | 0.222 | 0.035 | — |
| SFT | 0.383 | 0.026 | — |
| Chem-R | 0.264 | 0.043 | 0.210 |
| Faithful | 0.412 | 0.030 | — |
| coupled | 0.470 | 0.034 | 0.318 |
| ChemDFM | 0.120 | 0.038 | 0.052 |
| ether-0 | 0.638 | 0.005 | — |

**Take-away:** ig_content ≈ 0 (corrupting the **FG-name claims** carries ~no answer-information), yet **ig_swap > 0**
(0.05–0.32 — swapping the **whole** CoT for another molecule's DOES raise answer entropy) and ig_presence > 0 — i.e.
FG-prose is decorative while the **structural-draft content + thinking-space are load-bearing**. Metric-free,
independently confirms the R2/R5 two-channel picture. ChemDFM's ig_swap is lowest (0.05), matching its weak draft channel.

## R4. Mitigation (per family) — from mitigation.csv
ER ↓ 73–95% on all 4 families, perf held/up, %ER=0 ↑, claim precision & verified-FG count ↑. (base→coupled:
cap2mol ER 6.66→1.81, mol2cap 7.20→1.71, retro 15.1→0.81, s2 10.96→2.70.)

## R5. Token-type causal importance in the trace — model-specific CoT channel (FULL, n≈16,100/model)
Descriptive probe of **which token TYPE in the reasoning the answer actually keys on**, per model. FULL volume
(no correctness/ER filter), ER-stratified. gradient enrichment >1 = above-average per-token answer-sensitivity.

**(A) Gradient×input saliency — per-token enrichment (share%/enrich×), `all` stratum:**
| model | SMILES_frag | FG_word | position_digit | other_word | punct |
|---|---|---|---|---|---|
| Chem-R | 13.4% / **2.48×** | 4.7% / 0.77× | 3.2% / 1.60× | 41.3% / 0.77× | 36.9% / 1.13× |
| coupled | 17.8% / **3.21×** | 4.6% / **0.71×** | 3.1% / 1.66× | 39.3% / 0.73× | 34.8% / 1.09× |
| ChemDFM | **1.0%** / 1.24× | 5.3% / 0.86× | 0.6% / 1.37× | 53.2% / 0.81× | 39.5% / 1.49× |

**ER-stratified (SMILES_frag, share%/enrich):** Chem-R ER=0 8.8%/2.23× → ER>0 **16.4%/2.59×** (n 6541/9562);
coupled ER=0 18.0%/3.20× ≈ ER>0 15.9%/3.35× (n 14176/1924); ChemDFM ~1.0%/~1.2× in all strata.
→ *when the trace contains a fabricated FG (ER>0), the answer leans MORE on the SMILES draft, not on the FG words.*

**(B) Region attention + draft-copy, `all` stratum** (INPUT ≫ TRACE per-token; within-trace SMILES vs FG):
| model | INPUT all | TRACE all | INPUT/TRACE | TRACE SMILES | TRACE FG | draft-copy |
|---|---|---|---|---|---|---|
| Chem-R | 0.0045 | 0.0001 | ~45× | **0.0008** | 0.0001 | 0.141 |
| coupled | 0.0039 | 0.0002 | ~20× | **0.0012** | 0.0001 | **0.221** |
| ChemDFM | 0.0046 | 0.0004 | ~11× | 0.0008 | **0.0010** | **0.001** |

**Take-away (scoped carefully):**
1. **Answer attends to INPUT ≫ TRACE** (per-token ~11–45×) in every model — consistent with input-grounding (R2).
2. **Chem-R / coupled — structural-draft channel is load-bearing, FG-prose is not:** within the trace, SMILES
   fragments are the single most causally-important token type (**2.5–3.2× enriched**) while **FG words are
   DEPLETED (0.71–0.80×)**; within-trace attention also goes to SMILES (0.0008–0.0012) not FG (0.0001);
   the model literally drafts the answer structure 14–22% of the time.
3. **The draft reward SHARPENS the load-bearing channel:** coupled vs Chem-R — SMILES enrichment 2.48→**3.21×**,
   draft-copy 0.141→**0.221**, FG stays depleted (0.77→0.71×). The reward concentrates computation on SMILES
   without touching the FG-prose channel. (The reward credits *proper substructures only*, not full-answer pre-emit.)
4. **ChemDFM — model-specific: NO SMILES-draft channel.** SMILES is essentially absent (1.0% mass, ~1.2× enrich,
   draft-copy **0.001**); within-trace attention on FG words ≥ SMILES (0.0010 vs 0.0008), the opposite of Chem-R.
5. **Cross-model:** position-digit is enriched per-token (1.4–1.7×) in all three — ring-closure/position indices
   carry causal weight universally, though they are a small token fraction.

**HONEST CAVEATS (R5):**
- enrichment = (per-doc-averaged saliency share) ÷ (global token fraction) — an approximation (per-doc token
  counts not stored); direction/magnitude robust: matches the earlier n=300 sample (Chem-R ER=0 SMILES 7.5→8.8%,
  FG 4.4→4.5%, punct 39.2→39.7%).
- other_word/punct dominate ABSOLUTE saliency mass only because they are ~90% of tokens — a count effect, not
  importance. Per-token **enrichment** is the fair metric and reverses this (other_word 0.7–0.8× = below average).
- This probe confirms Chem-R externalizes **SMILES structural drafts** and that **ChemDFM lacks that channel**. It
  does NOT independently localize a positive "positional / scaffold / nomenclature" channel for ChemDFM: position-
  digit is enriched but rare (0.6% mass); nomenclature is buried in the `other_word` catch-all and is not
  separable here. That clause of the thesis rests on **region-attention (ChemDFM TRACE FG ≥ SMILES) + R3 condsent**,
  not on gradient. Scope the mechanism claim to **unsupported FG claims**, not "verbal claims" in general.

---

## Cross-check caveats (honest)
1. **s2 success ignores similarity-to-source** for edit/opt — successes could in principle be dissimilar molecules
   (probe showed successes are actually ~0.7 Tanimoto to source, so effect is bounded; a sim-weighted re-run is
   the fallback). Affects R1/R2/R4 s2 numbers slightly, consistently across models.
2. **base-a**: perf≈0 → its drift/perf rows are uninformative (only 14 correct); it contributes to R1 (fabrication)
   and R3 (entropy) only.
3. **DeepSeek** perturbation N=83, mol2cap/others near-0 valid → excluded from mechanism claims.
5. **drop_cot** has an OOD confound (empty `<think>` is off-distribution) → cleanest content signal is all_wrong_cot.
6. **R5 gradient/attention** are DESCRIPTIVE (correlational) probes: they show *what the answer keys on*, not a
   counterfactual. The causal claim is carried by R2 (drift/swap) and R3 (metric-free entropy); R5 corroborates the
   mechanism at token resolution. ChemDFM's positive channel is under-specified (see R5 caveat). enrichment is an
   approximation (per-doc token counts not stored).
