# Chemist validation of the detector's claim extraction

This directory holds the annotation records behind one number in the paper: the
**96.9% claim-extraction precision** quoted in the Limitations. Everything needed to
recompute it from the raw labels is here.

```bash
python human_eval/score_claims.py
# FABRICATED-flagged: claims=126  no_claim=5   precision = 0.962
# VERIFIED-flagged:   claims=123  no_claim=3   agreement = 0.976
# overall extraction precision = 0.969
```


## The files

| file | what it is |
|---|---|
| `claims_key.json` | The sampled claims and the detector's verdict for each: 257 entries from the five model variants reported in the paper (131 `fabricated` and 126 `verified`). Keys are `model\|task\|sample_id\|verdict\|functional_group`. |
| `claim_annotations_RL.json` | The chemist's labels, blind to the verdict. `annotator` records who labelled (initials). |
| `score_claims.py` | Joins the two and prints the table. Runs from anywhere; writes `claim_reliability.json` beside itself. |

The annotator answered one question per claim: does the model really assert that the
answer molecule has this group? `claims` = yes, extraction correct. `no_claim` = no, the
detector read an assertion that is not there. `unsure` = dropped.

`claim_annotations_RL.json` is the export filtered to exactly these 257 claims. An earlier
annotation round asked a different question -- whether a group is structurally present
rather than whether the model asserts it -- and is not part of any number in the paper, so
it is not shipped.
