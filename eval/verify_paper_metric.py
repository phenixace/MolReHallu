"""Verify: recompute the PAPER-metric (flip-to-wrong among originally-correct, by task-group)
from the CURRENT full-volume drift_<m>.json per_example, and the mean-over-tasks ig for condsent.
Compares against what RESULTS.md R2/R3 report, to settle whether the earlier tables used a
different metric/aggregation (vs a smaller run). Output: verify_metric.txt
"""
import json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AO = os.path.join(BASE, "data", "raw")
CONDS = ["syn_cot", "wrong_cot", "all_wrong_cot", "drop_cot", "swap_cot"]
TRANSLATE = {"cap2mol", "mol2cap", "retrosynthesis"}


def grp(task):
    return "translate" if task in TRANSLATE else "s2"


out = []
def p(*a): out.append(" ".join(str(x) for x in a))

for m in ["Chem-R", "Chem-R-Faithful", "ChemDFM-R"]:
    f = f"{AO}/drift_{m}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); pe = d["per_example"]
    p(f"\n### {m}  (n_examples={d.get('n_examples')}, base_acc={d.get('base_accuracy'):.3f})")
    # paper metric: among base_correct==1, fraction with <cond>_dperf == -1, by task-group
    for g in ["translate", "s2"]:
        sub = [e for e in pe if grp(e["task"]) == g and e.get("base_correct") == 1]
        row = [f"  {g:9s} n_orig_correct={len(sub):5d} |"]
        for c in CONDS:
            if not sub:
                row.append(f"{c.split('_')[0]}:NA")
                continue
            flip = sum(1 for e in sub if e.get(f"{c}_dperf") == -1) / len(sub)
            row.append(f"{c.replace('_cot','')}:{flip*100:.1f}%")
        p(" ".join(row))
    # also show what summary.drift_rate (any-change over ALL) says, for contrast
    sm = d.get("summary", {})
    p("  [contrast] summary.drift_rate (ANY change / ALL examples): " +
      " ".join(f"{c.replace('_cot','')}:{sm[c]['drift_rate']*100:.1f}%(n{sm[c]['n']})" for c in CONDS if c in sm))

# condsent: pooled vs mean-over-tasks
p("\n=== CONDSENT: pooled (er_split all) vs mean-over-tasks ===")
for m in ["Chem-R", "Chem-R-Faithful", "ChemDFM-R"]:
    f = f"{AO}/condsent_{m}.json"
    if not os.path.exists(f):
        continue
    c = json.load(open(f))
    a = c["er_split"]["all"]
    pt = c["per_task"]
    def mot(k):
        vs = [v[k] for v in pt.values() if v.get(k) is not None]
        return sum(vs) / len(vs) if vs else float("nan")
    p(f"  {m:20s} pooled ig_presence={a['ig_presence']:.3f} ig_content={a['ig_content']:.3f} ig_swap={a.get('ig_swap',float('nan')):.3f}"
      f" | mean-over-tasks ig_presence={mot('ig_presence'):.3f} ig_content={mot('ig_content'):.3f} ig_swap={mot('ig_swap'):.3f}")

txt = "\n".join(out)
open("verify_metric.txt", "w").write(txt)
print(txt)
