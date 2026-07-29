"""Summarize the draft-SMILES perturbation experiment (mask_draft / corrupt_draft) vs the
FG-name (all_wrong_cot) and whole-CoT (swap_cot) perturbations, on the SAME subset:
originally-correct examples whose trace actually drafts a SMILES (base_correct==1 & n_draft>0).
flip-to-wrong = among that subset, fraction that turn wrong (<cond>_dperf==-1).
Output: draft_result.txt
"""
import json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AO = os.path.join(BASE, "data", "raw")
MODELS = ["Chem-R", "Chem-R-Faithful", "ChemDFM-R"]
CONDS = ["all_wrong_cot", "mask_draft", "corrupt_draft", "swap_cot", "drop_cot"]
TRANSLATE = {"cap2mol", "mol2cap", "retrosynthesis"}
grp = lambda t: "translate" if t in TRANSLATE else "s2"

out = []
def p(*a): out.append(" ".join(str(x) for x in a))

def flip(sub, c):
    present = [e for e in sub if f"{c}_dperf" in e]
    if not present:
        return float("nan"), 0
    return sum(1 for e in present if e[f"{c}_dperf"] == -1) / len(present), len(present)

p("=" * 92)
p("DRAFT-SMILES perturbation — flip-to-wrong among originally-correct WITH a drafted SMILES")
p("  (mask=remove draft / corrupt=wrong structure) vs all_wrong=FG-name / swap=whole CoT / drop=empty")
p("=" * 92)
for m in MODELS:
    f = f"{AO}/drift_{m}.json"
    if not os.path.exists(f):
        p(f"\n### {m}: NO FILE"); continue
    pe = json.load(open(f))["per_example"]
    bc = [e for e in pe if e.get("base_correct") == 1]
    draft = [e for e in bc if e.get("n_draft", 0) > 0]
    cov = len(draft) / len(bc) if bc else float("nan")
    p(f"\n### {m}  | base_correct={len(bc)}  draft-bearing={len(draft)} ({cov*100:.0f}% coverage)")
    for g in ["all", "translate", "s2"]:
        sub = draft if g == "all" else [e for e in draft if grp(e["task"]) == g]
        if not sub:
            continue
        cells = []
        for c in CONDS:
            r, n = flip(sub, c)
            cells.append(f"{c.replace('_cot','').replace('_draft','D'):>10s}:{r*100:4.1f}%" if r == r else f"{c:>10s}:  NA")
        p(f"  [{g:9s} n={len(sub):5d}]  " + "  ".join(cells))

txt = "\n".join(out)
open("draft_result.txt", "w").write(txt)
print(txt)
