#!/usr/bin/env python3
"""Stage-ladder mechanism metrics: WHERE does reasoning hallucination come from?

For each model on the training ladder (base-a -> SFT -> answer-only -> process ->
coupled) compute, from the SAME frozen detector's output, quantities that
separate "SFT made it fabricate" from "SFT just made it verbose", and that test
the specific hypothesis that SFT installs obligatory fixed-slot generation:

  perf, ER, %ER=0                 -- via eval/metrics.py (official metrics)
  claims_per_resp                 -- mean # distinct claimed FGs / response
  perclaim_fab_rate               -- sum(fabricated) / sum(claimed)  [claim-weighted]
                                     the verbosity-invariant fabrication RATE
  hedge_rate                      -- frac of traces that hedge ("may/likely/appears/not sure")
  abstain_rate                    -- frac of traces that abstain ("cannot determine/unknown")
  fab_position                    -- mean normalized position (0=start,1=end) of the
                                     first fabricated-FG mention inside <think>
                                     (clustering late/at a fixed slot supports the
                                      "obligatory template slot" mechanism)

Reads results/<model>/<task>/*hallucination_details.jsonl (FG-level) joined by id
with se_results/<model>/<task>/output.json (raw trace text). Writes a CSV to the
paper data folder and prints a table.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

BASE = os.environ.get("MOLLM_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "eval"))
import metrics as MX  # noqa: E402

# ladder order; skip silently if a model has no results yet
LADDER = [
    ("Llama-3.1-8B-Instruct-base", "base-a (pre-SFT)"),
    ("Chem-R-SFT",                 "SFT"),
    ("+process",                  "+process"),
    ("Chem-R-Faithful",          "+coupled"),
    ("Chem-R",                     "off-the-shelf GRPO"),
]
GEN_TASKS = ["cap2mol", "mol2cap", "retrosynthesis"]

_HEDGE = re.compile(
    r"\b(may|might|maybe|perhaps|possibly|possible|probabl|likely|appears?|"
    r"seems?|suggest|could|presumably|tentativ|i(?:'m| am)? not (?:sure|certain)|"
    r"not (?:sure|certain|clear)|unclear|i (?:believe|think|assume|guess))\b", re.I)
_ABSTAIN = re.compile(
    r"\b(cannot (?:be )?determin|can(?:'|no)t (?:be )?determin|unable to determin|"
    r"not enough information|insufficient information|i (?:don'?t|do not) know|"
    r"hard to (?:say|tell)|difficult to determin|impossible to determin|"
    r"without more (?:information|data))\b", re.I)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def _think_text(answer):
    m = _THINK.search(answer or "")
    return (m.group(1) if m else (answer or "")).strip()


def _load_text(model, task):
    """id -> think-segment text, from output.json."""
    fs = glob.glob(f"{BASE}/se_results/{model}/{task}/output.json")
    if not fs:
        return {}
    o = json.load(open(fs[0]))
    recs = o if isinstance(o, list) else o.get("results", o.get("samples", []))
    out = {}
    for r in recs:
        if isinstance(r, dict) and "id" in r:
            out[str(r["id"])] = _think_text(r.get("answer", ""))
    return out


def _fab_position(think, fabricated_fgs):
    """Normalized char position (0..1) of the earliest fabricated-FG mention."""
    if not think or not fabricated_fgs:
        return None
    L = len(think)
    tl = think.lower()
    best = None
    for fg in fabricated_fgs:
        for cand in {fg.lower(), fg.replace("_", " ").lower(), fg.replace("_", "").lower()}:
            i = tl.find(cand)
            if i >= 0:
                best = i if best is None else min(best, i)
    return None if best is None else best / max(L, 1)


def mechanism_stats(model):
    txt_cache = {t: _load_text(model, t) for t in GEN_TASKS}
    n_resp = 0
    claimed_tot = fab_tot = 0
    hedge = abstain = 0
    positions = []
    for task in GEN_TASKS:
        fs = glob.glob(f"{BASE}/data/results/{model}/{task}/*hallucination_details.jsonl")
        if not fs:
            continue
        texts = txt_cache[task]
        for line in open(fs[0]):
            d = json.loads(line)
            er = d.get("details", {}).get("ER", {})
            claimed = er.get("claimed_fgs", []) or []
            fabricated = er.get("fabricated_fgs", []) or []
            n_resp += 1
            claimed_tot += len(claimed)
            fab_tot += len(fabricated)
            think = texts.get(str(d.get("id")), "")
            if think:
                if _HEDGE.search(think):
                    hedge += 1
                if _ABSTAIN.search(think):
                    abstain += 1
                p = _fab_position(think, fabricated)
                if p is not None:
                    positions.append(p)
    if n_resp == 0:
        return None
    return {
        "n_resp": n_resp,
        "claims_per_resp": claimed_tot / n_resp,
        "perclaim_fab_rate": (fab_tot / claimed_tot) if claimed_tot else 0.0,
        "hedge_rate": hedge / n_resp,
        "abstain_rate": abstain / n_resp,
        "fab_position": (sum(positions) / len(positions)) if positions else float("nan"),
        "n_fab_positioned": len(positions),
    }


def main():
    rows = []
    hdr = (f"{'stage':22s} {'perf':>6s} {'ER':>6s} {'%ER0':>5s} "
           f"{'claims/r':>8s} {'fab_rate':>8s} {'hedge%':>7s} {'abstain%':>8s} {'fab_pos':>7s}")
    print(hdr); print("-" * len(hdr))
    for model, label in LADDER:
        mech = mechanism_stats(model)
        if mech is None:
            print(f"{label:22s}  <no results yet>")
            continue
        # official perf/ER over the 3 generative families (comparable across stages)
        agg = MX.family_stats(model, GEN_TASKS) if hasattr(MX, "family_stats") else None
        perf = agg["perf"] if agg else float("nan")
        er = agg["ER"] if agg else float("nan")
        er0 = agg["pct_er0"] if agg else float("nan")
        print(f"{label:22s} {perf:6.1f} {er:6.2f} {er0:5.0f} "
              f"{mech['claims_per_resp']:8.2f} {mech['perclaim_fab_rate']*100:7.1f}% "
              f"{mech['hedge_rate']*100:6.1f}% {mech['abstain_rate']*100:7.1f}% "
              f"{mech['fab_position']:7.2f}")
        rows.append({"model": model, "stage": label, "perf": perf, "ER": er,
                     "pct_er0": er0, **mech})

    if rows:
        import csv
        outdir = os.path.join(BASE, "data")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "stage_ladder.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
