"""
Build the human-annotation set: SHARED-prompt design.

For each task we pick PROMPTS_PER_TASK prompts that all four models were
evaluated on, and group the four models' reasoning traces under the same prompt
so the annotator rates them side by side (controls for prompt difficulty and
spans the hallucination spectrum). Each model panel carries the auto 2x2 scores
and red-highlight spans for fabricated functional groups.

Run from the repo root:
  python human_eval/build_annotation_set.py
"""
import glob
import json
import os
import re
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# reuse the hardened detector's FG keyword DB so the red/green highlight spans
# mirror exactly what the detector flagged (V4 word-boundary logic).
from diagnose_hallucination import FUNCTIONAL_GROUP_DB
random.seed(0)

MODELS = [("Chem-R", "Chem-R (8B, chem)"),
          ("ChemDFM-R", "ChemDFM-R (14B, chem)"),
          ("DeepSeek-R1", "DeepSeek-Distill (8B, general)"),
          ("ether-0", "ether-0 (24B, general)")]
# Additive panels: shown alongside the core models but EXCLUDED from the
# shared-prompt intersection, so adding them does not change which prompts (uids)
# are selected and any prior annotations on the core set stay aligned.
EXTRA = [("+process", "Chem-R-process (ablation, ours)"),
         ("Chem-R-Faithful", "Chem-R-Faithful (ours)")]
TASKS = [("cap2mol", "cap2mol"), ("mol2cap", "mol2cap"),
         ("retrosynthesis", "retrosynthesis"),
         ("s2_MolCustom_FunctionalGroup", "s2 functional-group")]
PROMPTS_PER_TASK = 100        # shared prompts per task -> x4 model ratings

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# fg canonical name -> highlight keywords, pulled straight from the detector DB
# (the same keyword lists that drove the fabricated/verified verdict).
SYN = {fg: kws for fg, (_smarts, kws) in FUNCTIONAL_GROUP_DB.items()}


def _kw_iter(kw, low):
    # whole-word match (optionally plural), mirroring detector _kw_present:
    # avoids "nitro" highlighting inside "nitrogen", "ether" inside "together".
    pat = re.compile(r"(?<![a-z])" + re.escape(kw.lower()) + r"(?:e?s)?(?![a-z])")
    return pat.finditer(low)


def spans_for(text, fgs, kind="red"):
    low = text.lower(); out = []
    for fg in fgs or []:
        if kind == "red":
            reason = f"claims '{fg.replace('_',' ')}', NOT present in the molecule (possible fabrication)"
        else:
            reason = f"claims '{fg.replace('_',' ')}', verified present in the molecule"
        for syn in SYN.get(fg, [fg.replace("_", " ")]):
            for m in _kw_iter(syn, low):
                out.append([m.start(), m.end(), reason])
    out.sort(); merged = []
    for s, e, r in out:
        if merged and s < merged[-1][1]:
            continue
        merged.append([s, e, r])
    return merged


def load(model, taskdir):
    # output.json may live under se_results/ (our runs) or results/ (collaborator runs)
    o = (glob.glob(f"{ROOT}/se_results/{model}/{taskdir}/output.json")
         or glob.glob(f"{ROOT}/results/{model}/{taskdir}/output.json"))
    d = glob.glob(f"{ROOT}/results/{model}/{taskdir}/*hallucination_details.jsonl")
    if not o or not d:
        return None, None
    O = {str(x["id"]): x for x in json.load(open(o[0]))}
    D = {str(json.loads(l)["id"]): json.loads(l) for l in open(d[0])}
    return O, D


_DELIMS = (("<|think_start|>", "<think>"), ("<|think_end|>", "</think>"),
           ("<|answer_start|>", "<answer>"), ("<|answer_end|>", "</answer>"))


def _norm(t):
    for a, b in _DELIMS:
        if a in t:
            t = t.replace(a, b)
    return t


def panel(o, d):
    ans = _norm(o["answer"])   # normalize ether-0's <|think_start|> etc.
    think = (THINK_RE.search(ans).group(1).strip() if THINK_RE.search(ans) else ans).strip()
    answer = (ANS_RE.search(ans).group(1).strip() if ANS_RE.search(ans) else "")
    hs = d["hallucination_scores"]
    er = d.get("details", {}).get("ER", {})
    fab = er.get("fabricated_fgs", [])
    ver = er.get("verified_fgs", [])
    return {"reasoning": think, "answer": answer,
            "scores": {k: round(hs[k], 1) for k in
                       ["IR_self_contradiction", "IO_structural_invalidity",
                        "ER_factual_fabrication", "EO_phantom_structure"]},
            "exact_match": bool(d.get("exact_match")),
            "highlights": spans_for(think, fab, "red"),
            "highlights_green": spans_for(think, ver, "green")}


def main():
    items = []
    for taskdir, tlabel in TASKS:
        # only models that actually have data for this task (some are cap2mol-only)
        avail = [(m, ml) for m, ml in MODELS if load(m, taskdir)[0] is not None]
        loaded = {m: load(m, taskdir) for m, _ in avail}
        if len(avail) < 2:
            print(f"[skip] {taskdir}: <2 models with data"); continue
        # intersection over CORE models only -> uids stable across EXTRA additions
        common = set.intersection(*[set(O) & set(D) for O, D in loaded.values()])
        common = sorted(common)
        random.shuffle(common)
        # additive panels (loaded separately, never enter `common`)
        extra_avail = [(m, ml) for m, ml in EXTRA if load(m, taskdir)[0] is not None]
        extra_loaded = {m: load(m, taskdir) for m, _ in extra_avail}
        print(f"  {taskdir}: {len(avail)} core + {len(extra_avail)} extra models, "
              f"{len(common)} shared prompts")
        for i in common[:PROMPTS_PER_TASK]:
            o0 = loaded[avail[0][0]][0][i]
            panels = []
            for m, mlabel in avail:
                O, D = loaded[m]
                panels.append({"model": mlabel, **panel(O[i], D[i])})
            for m, mlabel in extra_avail:
                O, D = extra_loaded[m]
                if i in O and i in D:
                    panels.append({"model": mlabel, **panel(O[i], D[i])})
            items.append({"uid": f"{taskdir}|{i}", "task": tlabel, "id": i,
                          "question": o0["question"], "gt": o0["gt"], "models": panels})
    random.shuffle(items)
    out = os.path.join(os.path.dirname(__file__), "samples.json")
    json.dump(items, open(out, "w"), ensure_ascii=False, indent=1)
    ratings = sum(len(it["models"]) for it in items)
    print(f"wrote {out}: {len(items)} shared prompts, {ratings} model panels, "
          f"{round(os.path.getsize(out)/1024)} KB")


if __name__ == "__main__":
    main()
