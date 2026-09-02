# Chemist validation of the detector's claim extraction

This directory holds the annotation records behind one number in the paper: the
**97.3% claim-extraction precision** quoted in the Limitations. Everything needed to
recompute it from the raw labels is here.

```bash
python human_eval/score_claims.py
# FABRICATED-flagged: claims=145  no_claim=5   precision = 0.967
# VERIFIED-flagged:   claims=147  no_claim=3   agreement = 0.980
# overall extraction precision = 0.973
```

## What was being checked, and what was not

The detector decides two things about a functional-group claim, and they are not equally
fallible. Whether the group is **structurally present** in a molecule is decided by RDKit
SMARTS, which is deterministic and is not what this audit questions. The error-prone step
is **extraction**: does the model's reasoning actually assert that the *answer* molecule
contains group X, as opposed to negating it, raising it hypothetically, or discussing a
different molecule? That is what a chemist checked.

So the audit measures extraction precision, not detection accuracy, and **recall is not
estimable from it** — the sample contains only claims the detector already extracted, so
claims it silently missed cannot appear.

## The files

| file | what it is |
|---|---|
| `claims_key.json` | The sampled claims and the detector's verdict for each: 300 entries, deliberately balanced at 150 `fabricated` and 150 `verified`. Keys are `model\|task\|sample_id\|verdict\|functional_group`. |
| `claim_annotations_RL.json` | The chemist's labels, blind to the verdict. `annotator` records who labelled (initials). |
| `score_claims.py` | Joins the two and prints the table. Runs from anywhere; writes `claim_reliability.json` beside itself. |

The annotator answered one question per claim: does the model really assert that the
answer molecule has this group? `claims` = yes, extraction correct. `no_claim` = no, the
detector read an assertion that is not there. `unsure` = dropped.

## One name you will not find elsewhere in this repository

The 300 claims are spread over six model variants as they were named when the annotation
was done: `Chem-R`, `Chem-R-Faithful`, `ChemDFM-R`, `DeepSeek-R1`, `ether-0`, and
`+process` (43 claims). `+process` is a reward variant trained against an earlier version
of the detector. It is not released, is not reported in the paper, and is not a
contribution of this work; it appears here only because the published 97.3% is computed
over all 300 sampled claims, so dropping its rows would stop the number reproducing.

`claim_annotations_RL.json` is the export filtered to exactly these 300. An earlier
annotation round asked a different question -- whether a group is structurally present
rather than whether the model asserts it -- and is not part of any number in the paper, so
it is not shipped.

## Not included

The blind forced-choice arena, whose agreement figures are the `Human_eval_agreement`
sheet of `data/source_data.xlsx` (n=400, overall agreement 54.75%, Cohen's kappa 0.41).
Its 20 MB sample pool and the scoring script were written against different versions of
each other and no longer run together, so shipping them would add a broken artefact
rather than a reproducible one. That sheet stays a reported result; the 97.3% here does
not have to be taken on trust.
