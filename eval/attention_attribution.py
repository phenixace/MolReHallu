"""Does the final answer depend on the CoT or on the input?

Mechanistic side of the decoupling result (R3). Without process-level supervision,
a chemical reasoner's answer can collapse onto the input and treat the CoT as
post-hoc rationalization. Two analyses on cap2mol, one model at a time (Chem-R,
ChemDFM-R):

(1) ATTENTION ATTRIBUTION. Teacher-force [prompt | <think>CoT</think> | <answer>A</answer>]
    and measure, per layer, the share of the answer tokens' attention that lands on
    the input (caption) span vs the CoT span. Split by trace cleanliness (ER=0 vs ER>0).

(2) PERTURBATION INTERVENTION (ER=0, correct answers only). The diagnoser found the
    named functional groups to be VERIFIED (present). We corrupt one such mention and
    measure the change in the teacher-forced log-prob of the ORIGINAL correct answer:
      - wrong_cot     : replace ONE verified FG in the CoT with a different (wrong) FG
      - all_wrong_cot : corrupt EVERY specific-FG mention in the CoT (clean-vs-hallucinated
                        replay on the same input -- fully hallucinated reasoning)
      - drop_cot      : blank the entire CoT, keep only the answer tag (mask/truncate)
      - syn_cot       : replace one FG with another SYNONYM of it (negative control, no-op)
      - wrong_input   : instead corrupt the same FG in the INPUT caption (positive control)
    If the answer is decoupled from the CoT, wrong_cot/all_wrong_cot/drop_cot ~ 0 while
    wrong_input << 0. A CoT-faithful (e.g. process-trained) model shows the opposite.

Output: a JSON per model under data/raw/. Plot separately (make_figures).
Run on GPU (se_vllm env). Usage: python eval/attention_attribution.py --model Chem-R
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import diagnose_hallucination as DH  # noqa: E402
import prompts as P  # noqa: E402
import io_utils as IO  # noqa: E402  (gz-aware, se_results/ -> data/responses/)

HF = {"Chem-R": "weidawang/Chem-R-8B",
      "ChemDFM-R": "OpenDFM/ChemDFM-R",
      "ether-0": "futurehouse/ether0",
      "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
      "+process": "<released separately: Chem-R process-reward variant>",
      "Chem-R-Faithful": "<released separately: Chem-R-Faithful weights>",
      "Llama-3.1-8B-Instruct-base": "meta-llama/Llama-3.1-8B-Instruct",
      "Chem-R-SFT": "<released separately: Chem-R SFT checkpoint>"}
# normalize alternative think/answer markup (ether-0) to <think>/<answer>
_DELIMS = [("<|think_start|>", "<think>"), ("<|think_end|>", "</think>"),
           ("<|answer_start|>", "<answer>"), ("<|answer_end|>", "</answer>")]


def _norm_markup(t):
    for a, b in _DELIMS:
        if a in t:
            t = t.replace(a, b)
    return t
GEN = set(DH.GENERIC_FG_NAMES)
# specific FGs with at least one synonym, for perturbation
SPECIFIC = {k: v[1] for k, v in DH.FUNCTIONAL_GROUP_DB.items()
            if k not in GEN and v[1]}
random.seed(0)


def _is_correct(task, d):
    """Task-appropriate 'the model got it right' flag."""
    if task == "mol2cap":                       # caption: exact match is meaningless
        return (d.get("caption_jaccard") or 0.0) >= 0.5
    return bool(d.get("exact_match"))           # cap2mol / retro / s2: exact/official


def load_examples(model, task="cap2mol"):
    of = IO.find(f"{BASE}/se_results/{model}/{task}/output.json")
    df = IO.find(f"{BASE}/data/results/{model}/{task}/*hallucination_details.jsonl")
    if not of or not df:
        return []
    items = IO.load_json(of[0])
    items = items if isinstance(items, list) else items.get("results", [])
    det = {str(json.loads(l)["id"]): json.loads(l) for l in open(df[0])}
    out = []
    for s in items:
        i = str(s.get("id"))
        d = det.get(i)
        if not d or not s.get("answer"):
            continue
        er = d["hallucination_scores"]["ER_factual_fabrication"]
        vfgs = [x for x in d.get("details", {}).get("ER", {}).get("verified_fgs", [])
                if x not in GEN]
        out.append({"id": i, "task": task, "desc": s.get("question", ""),
                    "answer": _norm_markup(s["answer"]),
                    "er": er, "exact": _is_correct(task, d), "vfgs": vfgs})
    return out


def build_text(tok, desc, answer_full, task="cap2mol"):
    msgs = P.build_messages(task, desc)
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = msgs[-1]["content"] + "\n"
    return prompt, prompt + answer_full


def char_to_tok(offsets, lo, hi):
    return [k for k, (a, b) in enumerate(offsets) if b > a and a >= lo and b <= hi]


def spans(tok, full_text, desc, prompt_len):
    """Token spans. The prompt instruction itself contains literal '<think>'/'<answer>'
    markup, so the CoT/answer markers must be located in the GENERATED part only
    (offset >= prompt_len). The input caption is located inside the prompt."""
    enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    ids = enc["input_ids"]
    d0 = full_text.find(desc) if desc else -1
    di = char_to_tok(offs, d0, d0 + len(desc)) if d0 >= 0 else []
    t0 = full_text.find("<think>", prompt_len)
    t1 = full_text.find("</think>", prompt_len)
    ci = char_to_tok(offs, t0 + 7, t1) if (t0 >= 0 and t1 > t0) else []
    a0 = full_text.find("<answer>", prompt_len)
    a1 = full_text.find("</answer>", prompt_len)
    ai = char_to_tok(offs, a0 + 8, a1) if (a0 >= 0 and a1 > a0) else []
    return ids, di, ci, ai


@torch.no_grad()
def attention_profile(model, tok, ex, device):
    prompt, full = build_text(tok, ex["desc"], ex["answer"], ex["task"])
    ids, di, ci, ai = spans(tok, full, ex["desc"], len(prompt))
    if not di or not ci or not ai or len(ids) > 3500:
        return None
    inp = torch.tensor([ids], device=device)
    out = model(input_ids=inp, output_attentions=True, use_cache=False)
    rows = torch.tensor(ai, device=device)
    ic = torch.tensor(di, device=device); cc = torch.tensor(ci, device=device)
    n_in, n_cot = max(len(di), 1), max(len(ci), 1)
    raw, pertok = [], []
    for A in out.attentions:                      # [1, heads, seq, seq]
        a = A[0].mean(0)                           # [seq, seq], head-mean
        sub = a.index_select(0, rows)              # answer rows
        mi = sub.index_select(1, ic).sum().item()
        mc = sub.index_select(1, cc).sum().item()
        raw.append(mc / (mi + mc + 1e-9))          # raw CoT mass share (confounded by span length)
        pin, pcot = mi / n_in, mc / n_cot          # per-token attention weight
        pertok.append(pcot / (pin + pcot + 1e-9))  # per-token CoT share (length-normalized)
    del out
    return {"raw": raw, "pertok": pertok, "n_in": n_in, "n_cot": n_cot}


@torch.no_grad()
def answer_logprob(model, tok, prefix, answer_text, device):
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    full = tok(prefix + answer_text, add_special_tokens=False)["input_ids"]
    if len(full) <= len(pre) or len(full) > 3500:
        return None
    inp = torch.tensor([full], device=device)
    logits = model(input_ids=inp, use_cache=False).logits[0]   # [seq, vocab]
    lp = torch.log_softmax(logits.float(), -1)
    tgt = torch.tensor(full[len(pre):], device=device)
    pos = lp[len(pre) - 1: len(full) - 1]                      # predict answer tokens
    return pos.gather(1, tgt[:, None]).squeeze(1).mean().item()


def find_syn(text, syns):
    low = text.lower()
    for s in sorted(syns, key=len, reverse=True):
        if s.lower() in low:
            return s
    return None


def replace_ci(text, old, new):
    return re.sub(re.escape(old), new, text, count=1, flags=re.I)


def perturb(model, tok, ex, device):
    """Return dict of Δlogprob (condition - orig) or None if not applicable."""
    answer_seg = ex["answer"]
    a0 = answer_seg.rfind("<answer>")
    if a0 < 0:
        return None
    prefix_orig = answer_seg[:a0 + len("<answer>")]
    answer_txt = answer_seg[a0 + len("<answer>"):]            # SMILES + </answer>
    full_prompt, _ = build_text(tok, ex["desc"], "", ex["task"])
    cot = prefix_orig
    # Select a specific FG that appears in the CoT (required for CoT corruption, which
    # works on EVERY task). Prefer one that ALSO appears in the natural-language input
    # (enables the input-corruption positive control) but do NOT require it -- so
    # SMILES-input tasks (mol2cap / retro / most s2) still yield the CoT evidence.
    chosen = syn_cot = syn_in = None
    cot_only = None
    for fg in SPECIFIC:
        sc = find_syn(cot, SPECIFIC[fg])
        if not sc:
            continue
        si = find_syn(ex["desc"], SPECIFIC[fg])
        if si:
            chosen, syn_cot, syn_in = fg, sc, si
            break
        if cot_only is None:
            cot_only = (fg, sc)
    if not chosen and cot_only:
        chosen, syn_cot = cot_only            # CoT-only: no input control this example
    if not chosen:
        return None
    wrong_fg = random.choice([f for f in SPECIFIC if f != chosen])
    wrong_syn = SPECIFIC[wrong_fg][0]
    same_alts = [x for x in SPECIFIC[chosen] if x.lower() != syn_cot.lower()]
    base = full_prompt + prefix_orig
    lp0 = answer_logprob(model, tok, base, answer_txt, device)
    if lp0 is None:
        return None
    res = {"fg": chosen, "wrong_fg": wrong_fg, "lp_orig": lp0}
    # corrupt the entity in the CoT (replace with a wrong group)
    res["d_wrong_cot"] = _delta(model, tok, full_prompt + replace_ci(prefix_orig, syn_cot, wrong_syn),
                                answer_txt, device, lp0)
    # synonym of the same entity in the CoT (negative control, no-op meaning)
    if same_alts:
        res["d_syn_cot"] = _delta(model, tok, full_prompt + replace_ci(prefix_orig, syn_cot, same_alts[0]),
                                  answer_txt, device, lp0)
    # corrupt the same entity in the INPUT instead (positive control) -- only when the
    # group is actually named in the input text (cap2mol / FG-naming s2).
    if syn_in is not None:
        bad_prompt, _ = build_text(tok, replace_ci(ex["desc"], syn_in, wrong_syn), "", ex["task"])
        res["d_wrong_input"] = _delta(model, tok, bad_prompt + prefix_orig, answer_txt, device, lp0)
    # (blunt intervention A) DROP the whole CoT: keep the answer tag, empty the <think>.
    # If the answer is decoupled from the reasoning, removing the entire CoT barely
    # moves the answer log-prob.
    res["d_drop_cot"] = _delta(model, tok, full_prompt + _empty_think(prefix_orig),
                               answer_txt, device, lp0)
    # (blunt intervention B) HALLUCINATE the whole CoT: corrupt EVERY specific-FG
    # mention in the CoT to a wrong group (clean-vs-hallucinated replay on the same
    # input). A CoT-faithful model should lose confidence in the correct answer;
    # a decoupled model is indifferent.
    res["d_all_wrong_cot"] = _delta(model, tok, full_prompt + _corrupt_all_cot(prefix_orig),
                                    answer_txt, device, lp0)
    res["n_cot_fgs"] = _count_cot_fgs(prefix_orig)
    return res


_THINK_INNER = re.compile(r"(<think>).*?(</think>)", re.S | re.I)


def _empty_think(prefix):
    """Blank the CoT content, keeping <think></think> and the answer tag."""
    if _THINK_INNER.search(prefix):
        return _THINK_INNER.sub(r"\1\2", prefix, count=1)
    # no explicit tags: drop everything before <answer>
    a = prefix.rfind("<answer>")
    return "<think></think>" + (prefix[a:] if a >= 0 else prefix)


def _present_cot_fgs(prefix):
    """(fg, synonym) pairs for every specific FG mentioned in the CoT part."""
    a = prefix.rfind("<answer>")
    cot = prefix[:a] if a >= 0 else prefix
    seen = []
    for fg in SPECIFIC:
        s = find_syn(cot, SPECIFIC[fg])
        if s:
            seen.append((fg, s))
    return seen


def _count_cot_fgs(prefix):
    return len(_present_cot_fgs(prefix))


def _corrupt_all_cot(prefix):
    """Replace every specific-FG mention in the CoT with a different (wrong) group."""
    present = _present_cot_fgs(prefix)
    if not present:
        return prefix
    present_fgs = {fg for fg, _ in present}
    out = prefix
    for fg, syn in present:
        alts = [f for f in SPECIFIC if f not in present_fgs]
        if not alts:
            continue
        out = replace_ci(out, syn, SPECIFIC[random.choice(alts)][0])
    return out


def _delta(model, tok, prefix, answer_txt, device, lp0):
    lp = answer_logprob(model, tok, prefix, answer_txt, device)
    return None if lp is None else lp - lp0


@torch.no_grad()
def matched_attention(model, tok, ex, device, want_map=False):
    """Matched-token control: a functional-group word that appears in BOTH the input
    caption AND the CoT. Compare the answer tokens' per-token attention to the input
    occurrence vs the CoT occurrence of the SAME word (identical content, only the
    location differs -> removes the span-length confound). Optionally dump the
    answer->sequence attention for a token-level heatmap. Only fires when the input
    is natural language that names the group (cap2mol / some s2); auto-skips SMILES
    inputs (mol2cap / retro), returning None."""
    prompt, full = build_text(tok, ex["desc"], ex["answer"], ex["task"])
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    ids = enc["input_ids"]
    plen = len(prompt)
    if len(ids) > 3500:
        return None
    t0, t1 = full.find("<think>", plen), full.find("</think>", plen)
    a0, a1 = full.find("<answer>", plen), full.find("</answer>", plen)
    d0 = full.find(ex["desc"]) if ex["desc"] else -1
    if min(t0, t1, a0, a1, d0) < 0 or t1 <= t0 or a1 <= a0:
        return None
    d1 = d0 + len(ex["desc"])
    ai = char_to_tok(offs, a0 + 8, a1)
    if not ai:
        return None
    low = full.lower()
    chosen = syn = None
    for fg in SPECIFIC:
        for s in sorted(SPECIFIC[fg], key=len, reverse=True):
            sl = s.lower()
            if 0 <= low.find(sl, d0, d1) and 0 <= low.find(sl, t0 + 7, t1):
                chosen, syn = fg, s
                break
        if chosen:
            break
    if not chosen:
        return None
    pi = low.find(syn.lower(), d0, d1)
    cj = low.find(syn.lower(), t0 + 7, t1)
    in_fg = char_to_tok(offs, pi, pi + len(syn))
    cot_fg = char_to_tok(offs, cj, cj + len(syn))
    if not in_fg or not cot_fg:
        return None
    out = model(input_ids=torch.tensor([ids], device=device),
                output_attentions=True, use_cache=False)
    rows = torch.tensor(ai, device=device)
    iidx = torch.tensor(in_fg, device=device)
    cidx = torch.tensor(cot_fg, device=device)
    ain, acot = [], []
    for A in out.attentions:
        sub = A[0].mean(0).index_select(0, rows)          # answer rows, head-mean
        ain.append(sub.index_select(1, iidx).mean().item())   # per-token attn to input FG
        acot.append(sub.index_select(1, cidx).mean().item())  # per-token attn to CoT FG
    res = {"fg": chosen, "syn": syn, "attn_in_fg": ain, "attn_cot_fg": acot}
    if want_map:
        L = len(out.attentions)
        seqvec = out.attentions[L // 2][0].mean(0).index_select(0, rows).mean(0)  # mid layer
        res["map"] = {
            "attn": seqvec.float().cpu().tolist(),
            "tokens": [tok.decode([t]) for t in ids],
            "in_fg": in_fg, "cot_fg": cot_fg,
            "input_span": char_to_tok(offs, d0, d1),
            "cot_span": char_to_tok(offs, t0 + 7, t1),
            "ans_span": ai,
        }
    del out
    return res


DEFAULT_TASKS = [
    "cap2mol", "mol2cap", "retrosynthesis",
    "s2_MolCustom_AtomNum", "s2_MolCustom_BondNum", "s2_MolCustom_FunctionalGroup",
    "s2_MolEdit_AddComponent", "s2_MolEdit_DelComponent", "s2_MolEdit_SubComponent",
    "s2_MolOpt_LogP", "s2_MolOpt_MR", "s2_MolOpt_QED",
]

# all FG-name synonyms (incl. generic) for the "answer attends to FG prose?" statistic
_FG_ALL = {k: v[1] for k, v in DH.FUNCTIONAL_GROUP_DB.items() if v[1]}
_SMILES_RE = re.compile(r"[A-Za-z0-9@+\-\[\]()=#/\\%.]{6,}")


def _canon_smiles(s):
    try:
        from rdkit import Chem as _C
        m = _C.MolFromSmiles(s)
        return _C.MolToSmiles(m) if m else None
    except Exception:
        return None


def _answer_smiles(ans):
    m = re.search(r"<answer>(.*?)</answer>", ans, re.S)
    a = (m.group(1) if m else ans).strip()
    return a.split()[0] if a else a


def _fg_tok_in(offs, low, lo, hi):
    """token indices of FG-name synonym occurrences within char span [lo,hi)."""
    s_tok = set()
    for syns in _FG_ALL.values():
        for s in syns:
            st = low.find(s.lower(), lo, hi)
            while st >= 0:
                s_tok |= set(char_to_tok(offs, st, st + len(s)))
                st = low.find(s.lower(), st + 1, hi)
    return s_tok


def _smiles_tok_in(offs, full, lo, hi):
    """token indices of RDKit-parseable SMILES substrings within [lo,hi) + the strings."""
    s_tok = set(); found = []
    for m in _SMILES_RE.finditer(full[lo:hi]):
        s = m.group(0)
        if len(s) >= 6 and _canon_smiles(s):
            found.append(s); s_tok |= set(char_to_tok(offs, lo + m.start(), lo + m.end()))
    return s_tok, found


@torch.no_grad()
def region_attr(model, tok, ex, device):
    """SYMMETRIC salience: per-token answer→attention to FG-name words vs SMILES fragments,
    in the INPUT region AND in the reasoning-TRACE region. Shows what the answer keys on in
    each. + draft_copy: is the final answer SMILES already drafted in the trace?"""
    prompt, full = build_text(tok, ex["desc"], ex["answer"], ex["task"])
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]; ids = enc["input_ids"]; plen = len(prompt)
    if len(ids) > 3500:
        return None
    d0 = full.find(ex["desc"]) if ex["desc"] else -1
    t0, t1 = full.find("<think>", plen), full.find("</think>", plen)
    a0, a1 = full.find("<answer>", plen), full.find("</answer>", plen)
    if d0 < 0 or min(t0, t1, a0, a1) < 0 or t1 <= t0 or a1 <= a0:
        return None
    ai = char_to_tok(offs, a0 + 8, a1)
    if not ai:
        return None
    low = full.lower()
    out = model(input_ids=torch.tensor([ids], device=device), output_attentions=True, use_cache=False)
    L = len(out.attentions); rows = torch.tensor(ai, device=device)
    seq = out.attentions[L // 2][0].mean(0).index_select(0, rows).mean(0).float().cpu()
    del out

    def region(lo, hi):
        allt = set(char_to_tok(offs, lo, hi))
        if not allt:
            return None
        fg = _fg_tok_in(offs, low, lo, hi) & allt
        sm, found = _smiles_tok_in(offs, full, lo, hi)
        sm &= allt

        def pt(idxs):
            idxs = [i for i in idxs if i < len(seq)]
            return (sum(seq[i].item() for i in idxs) / len(idxs)) if idxs else None
        return {"attn_all": pt(allt), "attn_fg": pt(fg), "attn_smiles": pt(sm),
                "n_all": len(allt), "n_fg": len(fg), "n_smiles": len(sm), "_found": found}

    ri = region(d0, d0 + len(ex["desc"]))          # INPUT
    rt = region(t0 + 7, t1)                          # reasoning TRACE
    if ri is None or rt is None:
        return None
    ans_c = _canon_smiles(_answer_smiles(ex["answer"]))
    copy = bool(ans_c and any(_canon_smiles(s) == ans_c for s in rt.pop("_found")))
    ri.pop("_found", None)
    return {"input": ri, "trace": rt, "draft_copy": copy, "task": ex["task"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF))
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)  # ALL generative + s2
    ap.add_argument("--n_attn", type=int, default=300)          # per regime (heavy: output_attentions)
    ap.add_argument("--n_perturb", type=int, default=10**9)     # default: FULL volume (all clean-correct)
    ap.add_argument("--regions_only", action="store_true")      # only add region_attr to existing json (fast)
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(HF[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        HF[args.model], torch_dtype=torch.bfloat16, device_map=dev,
        attn_implementation="eager")
    model.eval()

    # load every task, pool the examples (each tagged with its task)
    all_ex, per_task = [], {}
    for task in args.tasks:
        exs = load_examples(args.model, task)
        c = sum(1 for e in exs if e["er"] == 0 and e["exact"])
        f = sum(1 for e in exs if e["er"] > 0 and e["exact"])
        per_task[task] = {"total": len(exs), "clean": c, "fabr": f}
        all_ex.extend(exs)
        if exs:
            print(f"  {task}: {len(exs)} total, {c} clean-correct, {f} fabr-correct", flush=True)
    random.shuffle(all_ex)                          # mix tasks so caps sample across them
    clean = [e for e in all_ex if e["er"] == 0 and e["exact"]]
    fabr = [e for e in all_ex if e["er"] > 0 and e["exact"]]
    print(f"{args.model}: ALL tasks -> {len(clean)} clean-correct, {len(fabr)} fabricating-correct",
          flush=True)

    od = os.path.join(BASE, "data", "raw")
    fn = os.path.join(od, f"region_{args.model}.json")
    if args.regions_only:
        out = json.load(open(fn)) if os.path.exists(fn) else {"model": args.model}
        reg = []
        for e in all_ex:                                 # FULL volume: no correctness/ER filter, no cap
            r = region_attr(model, tok, e, dev)
            if r:
                r["er"] = e["er"]; r["exact"] = bool(e["exact"])   # tag for ER stratification in the pull
                reg.append(r)
            if len(reg) % 1000 == 0 and len(reg):
                print(f"  region-attr: {len(reg)} done", flush=True)
                out["region_attr"] = reg; json.dump(out, open(fn, "w"))   # checkpoint: survive walltime kill
        out["region_attr"] = reg
        json.dump(out, open(fn, "w"))
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else float("nan")
        def rm(region, key):
            return mean([r[region][key] for r in reg if r[region].get(key) is not None])
        print(f"  region-attr: {len(reg)} examples", flush=True)
        print(f"    INPUT: all={rm('input','attn_all'):.4f} FG={rm('input','attn_fg'):.4f} SMILES={rm('input','attn_smiles'):.4f}", flush=True)
        print(f"    TRACE: all={rm('trace','attn_all'):.4f} FG={rm('trace','attn_fg'):.4f} SMILES={rm('trace','attn_smiles'):.4f}", flush=True)
        print(f"    draft-copy rate: {mean([1.0 if r['draft_copy'] else 0.0 for r in reg]):.2f}", flush=True)
        print(f"updated {fn}", flush=True)
        return

    # (1) attention (capped; multi-task shuffled)
    attn = {"ER0": [], "ERpos": []}
    for tagname, pool in [("ER0", clean), ("ERpos", fabr)]:
        for e in pool[:args.n_attn]:
            fr = attention_profile(model, tok, e, dev)
            if fr:
                fr["task"] = e["task"]
                attn[tagname].append(fr)
        print(f"  attn {tagname}: {len(attn[tagname])} profiles", flush=True)

    # (2) perturbation on ALL clean-correct across tasks (CoT corruption everywhere;
    #     input-corruption control only where the group is named in the input text)
    pert = []
    for e in clean:
        if len(pert) >= args.n_perturb:
            break
        r = perturb(model, tok, e, dev)
        if r:
            r["task"] = e["task"]
            pert.append(r)
    n_inp = sum(1 for p in pert if p.get("d_wrong_input") is not None)
    print(f"  perturbation: {len(pert)} examples ({n_inp} with input-control)", flush=True)

    # (3) matched-token attention + one heatmap example (only fires for NL-input tasks)
    matched, heatmap = [], None
    for e in clean:
        if len(matched) >= args.n_attn:
            break
        r = matched_attention(model, tok, e, dev, want_map=(heatmap is None))
        if r:
            if heatmap is None and "map" in r:
                heatmap = r.pop("map")
            else:
                r.pop("map", None)
            r["task"] = e["task"]
            matched.append(r)
    print(f"  matched-token: {len(matched)} examples (heatmap={'yes' if heatmap else 'no'})", flush=True)

    # (4) symmetric region salience: what the answer attends to in INPUT vs TRACE
    reg = []
    for e in clean:
        if len(reg) >= args.n_attn:
            break
        r = region_attr(model, tok, e, dev)
        if r:
            reg.append(r)
    print(f"  region-attr: {len(reg)} examples", flush=True)

    out = {"model": args.model, "attn": attn, "perturb": pert,
           "matched": matched, "heatmap": heatmap, "region_attr": reg, "per_task": per_task,
           "n_clean": len(clean), "n_fabr": len(fabr)}
    od = os.path.join(BASE, "data", "raw")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"region_{args.model}.json")
    json.dump(out, open(fn, "w"))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")
    for cond in ("d_wrong_cot", "d_all_wrong_cot", "d_drop_cot", "d_syn_cot", "d_wrong_input"):
        print(f"  Δlogprob {cond:16s}: {mean([p.get(cond) for p in pert]):+.3f}", flush=True)
    if matched:
        # read at the MIDDLE layer l*=floor(L/2), matching region_attr and the paper
        # (see export_stats.py / methods.tex). attn_in_fg/attn_cot_fg are per-layer lists.
        def _mid(m, k): return m[k][len(m[k]) // 2]
        ain = mean([_mid(m, "attn_in_fg") for m in matched])
        acot = mean([_mid(m, "attn_cot_fg") for m in matched])
        print(f"  matched-token attn (mid layer): input-FG={ain:.4f}  CoT-FG={acot:.4f}  ratio={ain/max(acot,1e-9):.2f}x", flush=True)
    if reg:
        def rm(region, key):
            return mean([r[region][key] for r in reg if r[region].get(key) is not None])
        print("  region salience (per-token answer->attention):", flush=True)
        print(f"    INPUT: all={rm('input','attn_all'):.4f} FG={rm('input','attn_fg'):.4f} SMILES={rm('input','attn_smiles'):.4f}", flush=True)
        print(f"    TRACE: all={rm('trace','attn_all'):.4f} FG={rm('trace','attn_fg'):.4f} SMILES={rm('trace','attn_smiles'):.4f}", flush=True)
        print(f"    draft-copy rate (answer SMILES drafted in trace): {mean([1.0 if r['draft_copy'] else 0.0 for r in reg]):.2f}", flush=True)
    print(f"wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
