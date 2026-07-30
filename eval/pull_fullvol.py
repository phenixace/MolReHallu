"""Detailed full-volume stats for the two descriptive mechanism probes (gradient x input
saliency + region attention), model x ER-stratum. Emits a compact table + a markdown block.
grad enrichment_c = saliency_share_c / token_frac_c  (>1 => that token TYPE carries more
answer-sensitivity per token than the average trace token = fair causal-importance metric).
Run: python eval/pull_fullvol.py  (writes fullvol.txt and .md)
"""
import json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AO = os.path.join(BASE, "data", "raw")
MODELS = ["Chem-R", "Chem-R-Faithful", "ChemDFM-R", "ether-0"]
ORDER = ["SMILES_frag", "FG_word", "position_digit", "other_word", "punct", "space"]
STRATA = ["all", "er0", "erpos"]

out = []
def p(*a): out.append(" ".join(str(x) for x in a))

# ---------- (A) gradient x input saliency ----------
p("=" * 78)
p("(A) GRADIENT x INPUT saliency of ANSWER w.r.t. TRACE tokens  (full volume)")
p("    share = mean per-doc saliency share on that token type (sums to 1 across types)")
p("    enrich = share / token_frac  (>1 = more answer-sensitivity PER TOKEN than avg)")
p("=" * 78)
grad = {}
for m in MODELS:
    d = json.load(open(f"{AO}/gradattr_{m}.json"))
    grad[m] = d
    p(f"\n### {m}   n={d['n']}  (skipped long={d.get('skipped_long')} fmt={d.get('skipped_fmt')})")
    p(f"    n_by_stratum: {d['n_by_stratum']}")
    for s in STRATA:
        tc = d["token_counts"][s]; tot = sum(tc.values()) or 1
        sh = d["trace_saliency_frac"][s]
        p(f"  [{s:5s}] n={d['n_by_stratum'][s]:6d}  " +
          "  ".join(f"{c.split('_')[0][:4]}:{sh[c]*100:4.1f}%/{(sh[c]/(tc[c]/tot) if tc[c] else 0):4.2f}x"
                    for c in ORDER))

# ---------- (B) region attention ----------
p("\n" + "=" * 78)
p("(B) REGION ATTENTION: answer -> attention mass, INPUT vs TRACE region  (full volume)")
p("    per-token mean attention to all / FG-word / SMILES-frag tokens in each region")
p("=" * 78)
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")
region = {}
for m in MODELS:
    d = json.load(open(f"{AO}/region_{m}.json"))
    reg = d.get("region_attr", [])
    region[m] = reg
    p(f"\n### {m}   region_attr n={len(reg)}")
    for s in STRATA:
        if s == "all":
            sub = reg
        elif s == "er0":
            sub = [r for r in reg if r.get("er") == 0]
        else:
            sub = [r for r in reg if r.get("er", 0) > 0]
        if not sub:
            p(f"  [{s:5s}] (none)"); continue
        def rm(region_key, k):
            return mean([r[region_key][k] for r in sub if r.get(region_key) and r[region_key].get(k) is not None])
        dc = mean([1.0 if r.get("draft_copy") else 0.0 for r in sub])
        p(f"  [{s:5s}] n={len(sub):6d} | INPUT all={rm('input','attn_all'):.4f} FG={rm('input','attn_fg'):.4f} SM={rm('input','attn_smiles'):.4f}"
          f" | TRACE all={rm('trace','attn_all'):.4f} FG={rm('trace','attn_fg'):.4f} SM={rm('trace','attn_smiles'):.4f} | draft-copy={dc:.3f}")

txt = "\n".join(out)
open("fullvol.txt", "w").write(txt)
print(txt)

# ---------- machine-readable CSVs into the paper figure bundle ----------
import csv
PB = os.path.join(BASE, "data", "token_examples")
os.makedirs(PB, exist_ok=True)
with open(os.path.join(PB, "r5_gradient.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "stratum", "n", "token_type", "saliency_share", "enrichment", "token_count"])
    for m in MODELS:
        d = grad[m]
        for s in STRATA:
            tc = d["token_counts"][s]; tot = sum(tc.values()) or 1; sh = d["trace_saliency_frac"][s]
            for c in ORDER:
                w.writerow([m, s, d["n_by_stratum"][s], c, round(sh[c], 6),
                            round(sh[c] / (tc[c] / tot), 4) if tc[c] else 0.0, tc[c]])
with open(os.path.join(PB, "r5_region.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "stratum", "n", "region", "attn_all", "attn_fg", "attn_smiles", "draft_copy"])
    for m in MODELS:
        reg = region[m]
        for s in STRATA:
            sub = reg if s == "all" else [r for r in reg if (r.get("er") == 0) == (s == "er0")]
            if not sub:
                continue
            def rm(rk, k):
                return round(mean([r[rk][k] for r in sub if r.get(rk) and r[rk].get(k) is not None]), 6)
            dc = round(mean([1.0 if r.get("draft_copy") else 0.0 for r in sub]), 4)
            for rk in ("input", "trace"):
                w.writerow([m, s, len(sub), rk, rm(rk, "attn_all"), rm(rk, "attn_fg"), rm(rk, "attn_smiles"), dc])
print(f"wrote {PB}/r5_gradient.csv + r5_region.csv")
