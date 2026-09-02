"""Score the detector's claim-EXTRACTION against human judgment of the model's TEXT.

Structural presence is verified reliably by RDKit/SMARTS; the error-prone step the
detector can get wrong is CLAIM EXTRACTION: does the model's reasoning actually assert
that the answer molecule contains functional group X? (vs. negation, a hypothetical,
or a mention of a different molecule.)

For every sampled claim the detector EXTRACTED such an assertion and classified it:
  fabricated -> asserted-present AND SMARTS says the group is ABSENT
  verified   -> asserted-present AND SMARTS says the group is PRESENT
The human reads the model's wording (BLIND to the verdict) and labels:
  claims    -> the model really asserts the answer molecule has X   (extraction correct)
  no_claim  -> the model does NOT assert it                          (extraction false positive)
  unsure    -> dropped

The paper reports pooled claim-extraction precision across both detector verdicts. We
also print fabrication-specific precision: of the claims flagged fabricated, the fraction
where the model truly makes the claim (absence is trusted from SMARTS, so a confirmed
claim == a real fabrication):
    precision = TP / (TP + FP),  TP = fabricated & claims,  FP = fabricated & no_claim
Recall is NOT estimable here: this set only contains claims the detector already
extracted, so it cannot reveal assertions the detector missed.

Usage:
  python human_eval/score_claims.py human_eval/claim_annotations_*.json
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = ("claims", "no_claim", "unsure")


def main(paths):
    key = json.load(open(os.path.join(ROOT, "claims_key.json")))   # cid -> fabricated / verified
    cnt = {"fabricated": {"claims": 0, "no_claim": 0, "unsure": 0},
           "verified":   {"claims": 0, "no_claim": 0, "unsure": 0}}
    missing = 0
    per_ann = {}
    for p in paths:
        blob = json.load(open(p))
        who = blob.get("annotator", os.path.basename(p))
        pa = per_ann.setdefault(who, {"fabricated": [0, 0, 0], "verified": [0, 0, 0]})
        for cid, human in blob["annotations"].items():
            det = key.get(cid)
            if det not in ("fabricated", "verified") or human not in LABELS:
                missing += 1
                continue
            cnt[det][human] += 1
            pa[det][LABELS.index(human)] += 1

    fab, ver = cnt["fabricated"], cnt["verified"]
    tp, fp = fab["claims"], fab["no_claim"]           # fabricated: claim confirmed / extraction FP
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    scored = tp + fp
    total = scored + ver["claims"] + ver["no_claim"]
    print(f"claims scored: {total}  (unsure dropped: {fab['unsure'] + ver['unsure']}, unmatched: {missing})\n")
    if scored == 0:
        print("no fabricated-flagged claims scored yet."); return
    print("Detector CLAIM-EXTRACTION vs chemist reading the model's text")
    print("(does the model truly assert the answer molecule has the group?)\n")
    print(f"  FABRICATED-flagged: claims={tp}  no_claim={fp}")
    print(f"  fabrication precision = {prec:.3f}   "
          f"(flagged-fabricated where the model really makes the claim; absence trusted from SMARTS)")
    vt, vf = ver["claims"], ver["no_claim"]
    if vt + vf:
        print(f"  VERIFIED-flagged:   claims={vt}  no_claim={vf}  extraction-agree = {vt / (vt + vf):.3f}")
    at, af = tp + vt, fp + vf
    pooled = at / (at + af) if at + af else float("nan")
    if at + af:
        print(f"  overall extraction precision = {pooled:.3f}  (both verdicts pooled)")
    print("  (recall is not estimable from this set -- it contains only extracted claims.)")
    if len(per_ann) > 1:
        print("\nper annotator [fabricated (claims,no_claim,unsure) | verified (...)]:")
        for who, c in per_ann.items():
            print(f"  {who:16s} fab={tuple(c['fabricated'])}  ver={tuple(c['verified'])}")

    out = os.path.join(ROOT, "claim_reliability.json")
    json.dump({"pooled_extraction_precision": pooled,
               "fabrication_precision": prec, "n_fabricated_scored": scored,
               "fabricated": fab, "verified": ver, "unmatched": missing},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    args = sys.argv[1:] or glob.glob(os.path.join(ROOT, "claim_annotations_*.json"))
    if not args:
        print("no annotation files found (human_eval/claim_annotations_*.json)")
    else:
        main(args)
