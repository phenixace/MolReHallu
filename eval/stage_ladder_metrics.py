#!/usr/bin/env python3
"""Stage-ladder mechanism metrics: WHERE does reasoning hallucination come from?

For each model on the training ladder (base-a -> SFT -> answer-only -> process ->
coupled) compute, from the SAME frozen detector's output, quantities that
separate "SFT made it fabricate" from "SFT just made it verbose", and that test
the specific hypothesis that SFT installs obligatory fixed-slot generation:

  perf, ER, %ER=0                 -- via eval/metrics.py (official metrics)
  claims_per_resp                 -- mean # distinct claimed FGs / response
  perclaim_fab_rate               -- fabricated / (verified + fabricated) over SPECIFIC groups
                                     only (the six generic names are excluded from both
                                     numerator and denominator), computed per task and then
                                     averaged unweighted over tasks. This is the same quantity
                                     eval/metrics.py reports as claim precision `cp`, so the
                                     ladder's Chem-R rung equals the surveyed Chem-R value.
                                     Generic groups ("ring", "aromatic_ring", ...) verify almost
                                     always, so counting them would dilute the rate in favour of
                                     verbose models -- the opposite of the verbosity-invariance
                                     this metric is for. eval/metrics.py and the training reward
                                     (reward/chem_merged_v8_ours.py::_er_count) both exclude them.
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

# Guard: this script REWRITES data/stage_ladder.csv in place from the full diagnosed results
# tree, which the public release does not ship. Run against the released subset it silently
# drops the ladder rungs it cannot find and degrades the surviving row. It also writes at
# module scope, so importing it is enough to do the damage.
if os.environ.get("MOLREHALLU_REGEN") != "1":
    raise SystemExit(
        "stage_ladder_metrics.py rewrites data/stage_ladder.csv from the full diagnosed "
        "results tree, which is not part of the public release. Refusing to run so the "
        "shipped ladder data is not overwritten. Set MOLREHALLU_REGEN=1 to override."
    )

sys.path.insert(0, os.path.join(BASE, "eval"))
sys.path.insert(0, BASE)
import metrics as MX  # noqa: E402
from diagnose_hallucination import GENERIC_FG_NAMES as GENERIC  # noqa: E402
import io_utils as IO  # noqa: E402  (gz-aware, se_results/ -> data/responses/)


def _alias(label):
    """Directory names for a rung: the release display name and its internal codename.
    A diagnosed tree may be laid out under either, and silently skipping a rung whose
    directory happens to use the other convention is how a ladder loses a stage."""
    alts = [label]
    readme = os.path.join(BASE, "data", "raw", "README.md")
    if os.path.exists(readme):
        for line in open(readme):
            c = [x.strip().strip("`") for x in line.strip().strip("|").split("|")]
            if len(c) < 3 or c[0].startswith(("-", ":", "display")):
                continue
            if label in (c[0], c[1], c[2]):        # filename token / display label / codename
                alts += [x for x in (c[1], c[2], c[0]) if x and x not in alts]
    return alts


def _find(model, task):
    for name in _alias(model):
        fs = IO.find(f"{BASE}/data/results/{name}/{task}/*hallucination_details.jsonl")
        if fs:
            return name, fs
    return model, []

# ladder order; skip silently if a model has no results yet
LADDER = [
    ("Llama-3.1-8B-Instruct-base", "base-a (pre-SFT)"),
    ("Chem-R-SFT",                 "SFT"),
    ("Chem-R-Faithful",          "+coupled"),
    ("Chem-R",                     "off-the-shelf GRPO"),
]
# All 12 task variants the paper reports. The pre-SFT base rung was backfilled on the nine
# S2 subtasks, so every rung is now scored on the same full suite (n_resp = 16,107).
GEN_TASKS = ["cap2mol", "mol2cap", "retrosynthesis",
             "s2_MolCustom_AtomNum", "s2_MolCustom_BondNum", "s2_MolCustom_FunctionalGroup",
             "s2_MolEdit_AddComponent", "s2_MolEdit_DelComponent", "s2_MolEdit_SubComponent",
             "s2_MolOpt_LogP", "s2_MolOpt_MR", "s2_MolOpt_QED"]

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
    fs = []
    for name in _alias(model):
        fs = IO.find(f"{BASE}/se_results/{name}/{task}/output.json")
        if fs:
            break
    if not fs:
        return {}
    o = IO.load_json(fs[0])
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
    claimed_tot = 0
    hedge = abstain = 0
    positions = []
    per_task_rate = []          # specific-claim fabrication rate, one entry per task
    for task in GEN_TASKS:
        _, fs = _find(model, task)
        if not fs:
            continue
        texts = txt_cache[task]
        ver_spec = fab_spec = 0
        for line in IO.open_text(fs[0]):
            d = json.loads(line)
            er = d.get("details", {}).get("ER", {})
            claimed = er.get("claimed_fgs", []) or []
            fabricated = er.get("fabricated_fgs", []) or []
            verified = er.get("verified_fgs")
            if verified is None:
                verified = [x for x in claimed if x not in set(fabricated)]
            n_resp += 1
            claimed_tot += len(claimed)
            ver_spec += len([x for x in verified if x not in GENERIC])
            fab_spec += len([x for x in fabricated if x not in GENERIC])
            think = texts.get(str(d.get("id")), "")
            if think:
                if _HEDGE.search(think):
                    hedge += 1
                if _ABSTAIN.search(think):
                    abstain += 1
                p = _fab_position(think, fabricated)
                if p is not None:
                    positions.append(p)
        if ver_spec + fab_spec:
            per_task_rate.append(fab_spec / (ver_spec + fab_spec))
    if n_resp == 0:
        return None
    return {
        "n_resp": n_resp,
        "claims_per_resp": claimed_tot / n_resp,
        "perclaim_fab_rate": (sum(per_task_rate) / len(per_task_rate)) if per_task_rate else 0.0,
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
        # official perf/ER over all 12 task variants (comparable across stages)
        # metrics.py globs the model directory by name, so hand it the alias that actually
        # has data -- otherwise perf/ER come back NaN for any rung whose directory uses the
        # other naming convention, while the mechanism stats above succeed.
        resolved = next((n for n in _alias(model)
                         if IO.find(f"{BASE}/data/results/{n}/{GEN_TASKS[0]}/*hallucination_details.jsonl")),
                        model)
        agg = MX.family_stats(resolved, GEN_TASKS) if hasattr(MX, "family_stats") else None
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
