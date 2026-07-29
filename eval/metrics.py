"""Single source of truth for per-(model,task) evaluation metrics.

Correct, official metrics only:
  cap2mol / retrosynthesis -> exact match
  mol2cap                  -> BLEU-4 (ChEBI-20 convention)
  s2_*                     -> official S2-TOMG success (NOT validity)
Hallucination read from the re-diagnosed results/ details (input-grounding ER).
Also exposes the ER=0 vs ER>0 performance split (the decoupling analysis) and
GC / CP / information density. Imported by make_latex_tables.py and the eval/
analysis scripts so the LaTeX tables, markdown tables, and figures never diverge.

Needs the se_vllm env (rdkit, nltk, s2_success).
"""
import glob
import json
import os
import re
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE,):
    if p not in sys.path:
        sys.path.insert(0, p)

from diagnose_hallucination import GENERIC_FG_NAMES as GENERIC  # noqa: E402
from s2_success import s2_success  # noqa: E402
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # noqa: E402
from nltk.tokenize import wordpunct_tokenize  # noqa: E402

_SM = SmoothingFunction().method1
_AR = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

S2 = ["s2_MolCustom_AtomNum", "s2_MolCustom_BondNum", "s2_MolCustom_FunctionalGroup",
      "s2_MolEdit_AddComponent", "s2_MolEdit_DelComponent", "s2_MolEdit_SubComponent",
      "s2_MolOpt_LogP", "s2_MolOpt_MR", "s2_MolOpt_QED"]


def _items(path):
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get("results", d)


def details(model, task):
    f = glob.glob(f"{BASE}/data/results/{model}/{task}/*hallucination_details.jsonl")
    if not f:
        return {}
    return {str(json.loads(l)["id"]): json.loads(l) for l in open(f[0])}


def outputs(model, task):
    """{id: {answer, gt, question, metadata}} with S2 metadata merged from completions."""
    of = (glob.glob(f"{BASE}/se_results/{model}/{task}/output.json")
          or glob.glob(f"{BASE}/data/results/{model}/{task}/output.json"))
    if not of:
        return {}
    items = _items(of[0])
    mm = {}
    cf = glob.glob(f"{BASE}/se_results/{model}/{task}/completions.json")
    if cf:
        mm = {str(s.get("id")): (s.get("metadata") or {}) for s in _items(cf[0])}
    res = {}
    for s in items:
        s = dict(s)
        i = str(s.get("id"))
        if not s.get("metadata"):
            s["metadata"] = mm.get(i, {})
        res[i] = s
    return res


def perf_one(task, d, o):
    """Per-example performance with the task's OFFICIAL metric, in [0,1]."""
    if task in ("cap2mol", "retrosynthesis"):
        return 1.0 if d.get("exact_match") else 0.0
    if task == "mol2cap":
        gt = ((o or {}).get("gt") or "").strip()
        pr = d.get("pred_caption") or ""
        if not gt or not pr:
            return 0.0
        return sentence_bleu([wordpunct_tokenize(gt)], wordpunct_tokenize(pr),
                             weights=(0.25,) * 4, smoothing_function=_SM)
    if task.startswith("s2_"):
        # Use the diagnoser's robustly-extracted prediction so performance and
        # hallucination score the SAME molecule (models differ in answer markup:
        # <answer> vs <|answer_start|> etc.); fall back to an <answer> match.
        pred = d.get("pred_smiles")
        if not pred:
            m = _AR.search((o or {}).get("answer", "") or "")
            pred = m.group(1).strip() if m else ""
        return s2_success(task, pred, (o or {}).get("metadata", {}),
                          (o or {}).get("question") or (o or {}).get("input", ""))
    return 0.0


def _gc_one(d):
    v = d.get("details", {}).get("ER", {}).get("verified_fgs", []) or []
    return len({x for x in v if x not in GENERIC})


def task_stats(model, task):
    """Full per-(model,task) stats, or None if not diagnosed."""
    D = details(model, task)
    if not D:
        return None
    O = outputs(model, task)
    perf, er, ov, eo, ir, io, L, gc = [], [], [], [], [], [], [], []
    V = F = valid = 0
    tan = []
    for i, d in D.items():
        o = O.get(i, {})
        perf.append(perf_one(task, d, o))
        hs = d["hallucination_scores"]
        ov.append(d.get("overall_hallucination_score", 0.0))
        er.append(hs["ER_factual_fabrication"]); eo.append(hs["EO_phantom_structure"])
        ir.append(hs["IR_self_contradiction"]); io.append(hs["IO_structural_invalidity"])
        L.append(d.get("reasoning_length", 0)); gc.append(_gc_one(d))
        if d.get("pred_valid"):
            valid += 1
        if isinstance(d.get("tanimoto"), (int, float)):
            tan.append(d["tanimoto"])
        e = d.get("details", {}).get("ER", {})
        V += len([x for x in e.get("verified_fgs", []) if x not in GENERIC])
        F += len([x for x in e.get("fabricated_fgs", []) if x not in GENERIC])
    perf = np.array(perf, float); erA = np.array(er, float); c = erA == 0
    return dict(
        n=len(D),
        perf=100 * perf.mean(),
        perf_er0=100 * perf[c].mean() if c.any() else None,
        perf_erpos=100 * perf[~c].mean() if (~c).any() else None,
        pct_er0=100 * c.mean(),
        overall=float(np.mean(ov)), ER=float(np.mean(er)), EO=float(np.mean(eo)),
        IR=float(np.mean(ir)), IO=float(np.mean(io)),
        gc=float(np.mean(gc)), cp=100 * V / (V + F) if (V + F) else None,
        length=float(np.mean(L)),
        info=1000 * sum(gc) / max(sum(L), 1),
        validity=100 * valid / len(D),
        tanimoto=float(np.mean(tan)) if tan else None,
    )


def family_stats(model, tasks):
    """Sample-weighted aggregate over a task family (e.g. the 9 S2 subtasks)."""
    rs = [(t, task_stats(model, t)) for t in tasks]
    rs = [(t, s) for t, s in rs if s]
    if not rs:
        return None
    N = sum(s["n"] for _, s in rs)
    def w(key):
        vals = [(s[key], s["n"]) for _, s in rs if s.get(key) is not None]
        tot = sum(n for _, n in vals)
        return sum(v * n for v, n in vals) / tot if tot else None
    return dict(n=N, perf=w("perf"), perf_er0=w("perf_er0"), perf_erpos=w("perf_erpos"),
                pct_er0=w("pct_er0"), overall=w("overall"), ER=w("ER"), EO=w("EO"),
                IR=w("IR"), IO=w("IO"), gc=w("gc"), cp=w("cp"), length=w("length"),
                info=w("info"))


# task family -> display label and member tasks
FAMILIES = [("cap2mol", ["cap2mol"]), ("mol2cap", ["mol2cap"]),
            ("retrosynthesis", ["retrosynthesis"]), ("s2", S2)]


if __name__ == "__main__":
    # smoke test
    for m in ["Chem-R"]:
        for lbl, ts in FAMILIES:
            s = family_stats(m, ts)
            if s:
                print(f"{m:18s} {lbl:6s} n={s['n']:5d} perf={s['perf']:5.1f} "
                      f"er0={s['perf_er0']:5.1f} erpos={s['perf_erpos']:5.1f} "
                      f"%er0={s['pct_er0']:4.0f} ER={s['ER']:5.2f} GC={s['gc']:.2f}")
