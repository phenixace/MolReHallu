"""CoT-drift test: does corrupting the reasoning change the GENERATED answer?

Stronger than the teacher-forced logprob probe (attention_attribution.py) and it
removes that probe's two weaknesses:
  * NO ER-label / correctness filtering -- runs on the FULL set, so it can't be an
    artifact of "we conditioned on correct answers, which are the easy ones".
  * measures ACTUAL output drift by re-generating the final answer from the edited
    reasoning, not the log-prob of the pre-recorded answer.

For each recorded trace we rebuild the prefix [chat-prompt | <think>CoT</think> ...
<answer>] and greedily generate the answer, once per condition:
  base          : unedited CoT                       (reference regen)
  wrong_cot     : one specific FG in the CoT -> a wrong FG
  all_wrong_cot : every specific FG in the CoT -> wrong (fully corrupted reasoning)
  drop_cot      : CoT emptied (mask/truncate)
  syn_cot       : one FG -> a synonym of itself       (negative control: no drift)
  wrong_input   : same FG corrupted in the INPUT text (only where input names it;
                  positive control: SHOULD drift)
DRIFT = generated answer differs from `base` (canonical SMILES for molecule tasks;
normalized text for mol2cap). If reasoning is decoupled from the answer, the CoT
conditions drift ~ as little as syn_cot; if re-coupled, they drift much more.

Usage: python eval/cot_drift.py --model Chem-R [--tasks ...] [--max_per_task N]
Output: data/raw/drift_<model>.json
"""
import argparse
import glob
import json
import os
import random
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import diagnose_hallucination as DH  # noqa: E402
import prompts as P  # noqa: E402
import io_utils as IO  # noqa: E402  (gz-aware, se_results/ -> data/responses/)
from s2_success import s2_success  # noqa: E402  official S2-TOMG success (constraint satisfaction)
from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")
random.seed(0)

GEN = set(DH.GENERIC_FG_NAMES)
SPECIFIC = {k: v[1] for k, v in DH.FUNCTIONAL_GROUP_DB.items() if k not in GEN and v[1]}
_DELIMS = [("<|think_start|>", "<think>"), ("<|think_end|>", "</think>"),
           ("<|answer_start|>", "<answer>"), ("<|answer_end|>", "</answer>")]
_THINK = re.compile(r"(<think>)(.*?)(</think>)", re.S | re.I)
_ANS = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_SMILES_RE = re.compile(r"[A-Za-z0-9@+\-\[\]()=#/\\%.]{6,}")  # candidate SMILES tokens in the CoT
# Native think/answer markup per model, so we perturb + regenerate IN-DISTRIBUTION
# (ether-0 uses <|think_start|> etc; feeding it normalized <think> is OOD and breaks
# the input-control). (think_open, think_close, answer_open, answer_close).
MARKUP = {"ether-0": ("<|think_start|>", "<|think_end|>", "<|answer_start|>", "<|answer_end|>")}
_STD_MK = ("<think>", "</think>", "<answer>", "</answer>")


def markup_for(model):
    return MARKUP.get(model, _STD_MK)
DEFAULT_TASKS = ["cap2mol", "mol2cap", "retrosynthesis",
                 "s2_MolCustom_FunctionalGroup", "s2_MolEdit_AddComponent",
                 "s2_MolEdit_DelComponent", "s2_MolEdit_SubComponent"]
HF = {"Chem-R": "weidawang/Chem-R-8B",
      "ChemDFM-R": "OpenDFM/ChemDFM-R",
      "ether-0": "futurehouse/ether0",
      "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
      "Chem-R-Faithful": "phenixace/Chem-R-Faithful",
      # stage-ladder: does the model rely on its CoT BEFORE our GRPO?
      "Llama-3.1-8B-Instruct-base": "meta-llama/Llama-3.1-8B-Instruct",
      "Chem-R-SFT": "slayertear/llama-3.1-8b-stage2"}


def _norm(t):
    for a, b in _DELIMS:
        if a in t:
            t = t.replace(a, b)
    return t


def find_syn(text, syns):
    low = text.lower()
    for s in sorted(syns, key=len, reverse=True):
        if s.lower() in low:
            return s
    return None


def replace_ci(text, old, new):
    return re.sub(re.escape(old), new, text, count=1, flags=re.I)


def canon(task, ans):
    """Canonical form for drift comparison."""
    m = _ANS.search(ans)
    a = (m.group(1) if m else ans).strip()
    if task == "mol2cap":                      # free text: normalize whitespace/case
        return re.sub(r"\s+", " ", a.lower()).strip()
    a = a.split()[0] if a else a               # SMILES: first token
    mol = Chem.MolFromSmiles(a)
    return Chem.MolToSmiles(mol) if mol else a.strip()


def _pred_smiles(gen_ans):
    m = _ANS.search(gen_ans)
    a = (m.group(1) if m else gen_ans).strip()
    return a.split()[0] if a else a


def correct(task, gen_ans, gt, meta=None, instruction=""):
    """Task correctness of a generated answer (for Δperformance / absolute perf).
    s2_*   : OFFICIAL S2-TOMG success (constraint satisfaction via s2_success) -- the
             same metric eval/metrics.py uses; exact match is WRONG for S2.
    cap2mol/retro : canonical-SMILES exact match (the task's real metric).
    mol2cap: token Jaccard >= 0.5."""
    if task.startswith("s2_"):
        return float(s2_success(task, _pred_smiles(gen_ans), meta or {}, instruction))
    if not gt:
        return None
    if task == "mol2cap":
        ga = set(canon(task, gen_ans).split())
        gb = set(re.sub(r"\s+", " ", str(gt).lower()).split())
        if not ga or not gb:
            return 0
        return int(len(ga & gb) / len(ga | gb) >= 0.5)
    gt_c = Chem.MolFromSmiles(str(gt).split()[0]) if str(gt).strip() else None
    gt_c = Chem.MolToSmiles(gt_c) if gt_c else str(gt).strip()
    return int(canon(task, gen_ans) == gt_c)


def extract_cot(trace, mk=_STD_MK):
    """Inner CoT text (between think_open/close) of a trace, or None. For donor pools."""
    ts, te = mk[0], mk[1]
    tm = re.search(re.escape(ts) + r"(.*?)" + re.escape(te), trace, re.S)
    return tm.group(1) if tm else None


def find_draft_smiles(cot):
    """Distinct SMILES-like fragments the model DRAFTS in its reasoning (valid, ≥4 atoms,
    and structural-looking — contains a bond/branch/ring char, to avoid matching English words)."""
    seen, out = set(), []
    for frag in _SMILES_RE.findall(cot):
        if frag in seen:
            continue
        seen.add(frag)
        if not re.search(r"[0-9\[\]()=#]", frag):      # must look SMILES-y, not a plain word
            continue
        m = Chem.MolFromSmiles(frag)
        if m and m.GetNumAtoms() >= 4:
            out.append(frag)
    return out


def mask_draft_smiles(cot, frags, placeholder="[...]"):
    """Remove the drafted structure but keep the surrounding prose (tests draft PRESENCE)."""
    out = cot
    for f in frags:
        out = out.replace(f, placeholder)
    return out


def _corrupt_one(frag):
    """A valid but structurally-WRONG SMILES: mutate up to 2 atoms' element (hetero-swap /
    aromatic C→N etc.). Returns None if no valid distinct mutation found."""
    m = Chem.MolFromSmiles(frag)
    if not m or m.GetNumAtoms() < 2:
        return None
    base = Chem.MolToSmiles(m)
    swap = {6: 7, 7: 8, 8: 7, 16: 8, 9: 17, 17: 9, 35: 17, 15: 7}
    n = m.GetNumAtoms()
    for _ in range(8):
        rw = Chem.RWMol(m)
        idxs = list(range(n)); random.shuffle(idxs)
        changed = 0
        for i in idxs:
            if changed >= 2:
                break
            a = rw.GetAtomWithIdx(i); t = swap.get(a.GetAtomicNum())
            if not t:
                continue
            a.SetAtomicNum(t); changed += 1
        if not changed:
            continue
        try:
            m2 = rw.GetMol(); Chem.SanitizeMol(m2); s = Chem.MolToSmiles(m2)
            if s and s != base and Chem.MolFromSmiles(s):
                return s
        except Exception:
            continue
    return None


def corrupt_draft_smiles(cot, frags):
    """Replace each drafted SMILES with a valid-but-wrong structure (tests draft CONTENT/direction)."""
    out, any_ = cot, False
    for f in frags:
        c = _corrupt_one(f)
        if c:
            out = out.replace(f, c); any_ = True
    return out, any_


def build_conditions(task, question, trace, mk=_STD_MK, donor_cot=None):
    """Return {cond: prefix_string} to generate from, or None if no FG in the CoT.
    Uses the model's NATIVE markup mk=(think_open, think_close, ans_open, ans_close)
    to parse the recorded trace and to assemble the regeneration prefix, so ether-0
    (and any custom-token model) is perturbed + regenerated in its own format.
    donor_cot: another example's REAL CoT (same task) → adds a `swap_cot` condition
    (real, fluent, same-distribution reasoning but about a DIFFERENT molecule) — the
    clean control that isolates CoT CONTENT from mere presence of reasoning-shaped text."""
    ts, te, as_, ae = mk
    tm = re.search(re.escape(ts) + r"(.*?)" + re.escape(te), trace, re.S)
    if not tm:
        return None
    cot = tm.group(1)
    a0 = trace.find(as_)
    if a0 < 0:
        return None
    pre_tail = trace[tm.end():a0]              # text between think_close and ans_open

    def assemble(cot_text, q=question):
        c = P.build_messages(task, q)
        return (c, ts + cot_text + te + pre_tail + as_)

    # pick a specific FG present in the CoT
    present = [(fg, find_syn(cot, SPECIFIC[fg])) for fg in SPECIFIC]
    present = [(fg, s) for fg, s in present if s]
    if not present:
        return None
    chosen, syn_cot = present[0]
    wrong_fg = random.choice([f for f in SPECIFIC if f != chosen])
    wrong_syn = SPECIFIC[wrong_fg][0]
    same_alts = [x for x in SPECIFIC[chosen] if x.lower() != syn_cot.lower()]

    conds = {}
    conds["base"] = assemble(cot)
    conds["wrong_cot"] = assemble(replace_ci(cot, syn_cot, wrong_syn))
    # corrupt every present FG
    all_cot = cot
    present_fgs = {fg for fg, _ in present}
    for fg, s in present:
        alts = [f for f in SPECIFIC if f not in present_fgs]
        if alts:
            all_cot = replace_ci(all_cot, s, SPECIFIC[random.choice(alts)][0])
    conds["all_wrong_cot"] = assemble(all_cot)
    conds["drop_cot"] = assemble("")
    if donor_cot is not None:
        conds["swap_cot"] = assemble(donor_cot)   # real CoT from another molecule (same task)
    # DRAFT-SMILES perturbations (only when the CoT actually drafts a structure):
    #   mask_draft   = SMILES -> [...]  (presence of the structural draft)
    #   corrupt_draft= SMILES -> valid-but-wrong structure (content/direction of the draft)
    frags = find_draft_smiles(cot)
    if frags:
        conds["mask_draft"] = assemble(mask_draft_smiles(cot, frags))
        corr, ok = corrupt_draft_smiles(cot, frags)
        if ok:
            conds["corrupt_draft"] = assemble(corr)
    if same_alts:
        conds["syn_cot"] = assemble(replace_ci(cot, syn_cot, same_alts[0]))
    # input control: only if the group is named in the input text
    if find_syn(question, SPECIFIC[chosen]):
        conds["wrong_input"] = assemble(cot, q=replace_ci(question, find_syn(question, SPECIFIC[chosen]), wrong_syn))
    return {"chosen": chosen, "conds": conds, "n_draft": len(frags)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF))
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--max_per_task", type=int, default=10**9)   # default FULL
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[args.model], trust_remote_code=True)
    # Chem-R / merged models ship WITHOUT a chat_template; borrow the base's (same
    # logic run_multitask_se.py used to generate the traces, so prompts match).
    if not getattr(tok, "chat_template", None):
        eos = getattr(tok, "eos_token", "") or ""
        bases = (["meta-llama/Llama-3.1-8B-Instruct"] if "eot_id" in eos
                 else ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"])
        for b in bases:
            try:
                bt = AutoTokenizer.from_pretrained(b, trust_remote_code=True)
                if bt.chat_template:
                    tok.chat_template = bt.chat_template
                    print(f"borrowed chat_template from {b}", flush=True)
                    break
            except Exception as e:
                print(f"fallback {b} failed: {e}", flush=True)
    llm = LLM(model=HF[args.model], dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096, trust_remote_code=True)
    mk = markup_for(args.model)                    # native think/answer tokens
    sp = SamplingParams(temperature=0.0, max_tokens=320, stop=[mk[3]])  # stop at answer-close
    print(f"markup for {args.model}: {mk}", flush=True)

    # collect (uid, cond, prompt_str, task) across all examples
    jobs, meta = [], {}
    for task in args.tasks:
        of = IO.find(f"{BASE}/se_results/{args.model}/{task}/output.json")
        if not of:
            continue
        items = IO.load_json(of[0])
        items = items if isinstance(items, list) else items.get("results", [])
        # ER label per id (full set, NO filtering -- just for the ER-stratified stats)
        df = IO.find(f"{BASE}/data/results/{args.model}/{task}/*hallucination_details.jsonl")
        er_of = {}
        if df:
            for l in IO.open_text(df[0]):
                d = json.loads(l)
                er_of[str(d["id"])] = d["hallucination_scores"]["ER_factual_fabrication"]
        # S2 constraint metadata (source_molecule/added_group/target/... ) for s2_success
        s2meta = {}
        if task.startswith("s2_"):
            cf = IO.find(f"{BASE}/se_results/{args.model}/{task}/completions.json")
            if cf:
                for r in IO.load_json(cf[0]):
                    s2meta[str(r["id"])] = r.get("metadata", {})
        # donor pool: all real inner-CoTs for this task (for the swap_cot control)
        cot_pool = [c for c in (extract_cot(s.get("answer", ""), mk) for s in items) if c and c.strip()]
        random.shuffle(items)
        n = 0
        for s in items:
            if n >= args.max_per_task:
                break
            trace = s.get("answer", "")        # RAW trace, parsed with the model's native markup
            q = s.get("question", "")
            own = extract_cot(trace, mk)
            donor = None
            for _ in range(5):
                cand = random.choice(cot_pool) if cot_pool else None
                if cand and cand != own:
                    donor = cand; break
            bc = build_conditions(task, q, trace, mk, donor_cot=donor)
            if not bc:
                continue
            uid = f"{task}|{s.get('id')}"
            meta[uid] = {"task": task, "gt": s.get("gt", ""),
                         "er": er_of.get(str(s.get("id"))),
                         "n_draft": bc.get("n_draft", 0),
                         "s2meta": s2meta.get(str(s.get("id")), {}), "instr": q}
            for cond, (chat, tail) in bc["conds"].items():
                prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + tail
                jobs.append((uid, cond, prompt))
            n += 1
        print(f"  {task}: {n} examples staged", flush=True)

    print(f"{args.model}: generating {len(jobs)} completions ...", flush=True)
    outs = llm.generate([p for _, _, p in jobs], sp)
    gen = {}
    for (uid, cond, _), o in zip(jobs, outs):
        gen.setdefault(uid, {})[cond] = o.outputs[0].text

    # For each condition: DRIFT (answer changed vs base) and Δperformance (correct
    # after perturbation minus correct at base), overall and stratified by the trace's
    # original ER. No ER filtering anywhere -> the ER split is descriptive, not a gate.
    from collections import defaultdict
    def er_bin(er):
        if er is None:
            return "na"
        return "ER=0" if er == 0 else ("ER_lo" if er <= 25 else "ER_hi")
    drift = defaultdict(lambda: [0, 0])                         # cond -> [drift, n]
    drift_task = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    dperf = defaultdict(lambda: [0, 0])                        # cond -> [sum Δcorrect, n_scored]
    dperf_erbin = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # cond -> erbin -> [sumΔ, n]
    # ABSOLUTE performance per (task, cond): [sum base_correct, sum cond_correct, n]
    acc_tc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    base_acc = [0, 0]
    per_example = []
    for uid, cg in gen.items():
        if "base" not in cg:
            continue
        m = meta[uid]; task = m["task"]; gt = m["gt"]; er = m["er"]
        cb = canon(task, cg["base"])
        cbase = correct(task, cg["base"], gt, m.get("s2meta"), m.get("instr", ""))
        if cbase is not None:
            base_acc[0] += cbase; base_acc[1] += 1
        row = {"uid": uid, "task": task, "er": er, "base_correct": cbase, "n_draft": m.get("n_draft", 0)}
        for cond, txt in cg.items():
            if cond == "base":
                continue
            d = int(canon(task, txt) != cb)
            drift[cond][0] += d; drift[cond][1] += 1
            drift_task[cond][task][0] += d; drift_task[cond][task][1] += 1
            row[cond + "_drift"] = d
            cc = correct(task, txt, gt, m.get("s2meta"), m.get("instr", ""))
            if cbase is not None and cc is not None:
                delta = cc - cbase
                dperf[cond][0] += delta; dperf[cond][1] += 1
                dperf_erbin[cond][er_bin(er)][0] += delta; dperf_erbin[cond][er_bin(er)][1] += 1
                a = acc_tc[task][cond]; a[0] += cbase; a[1] += cc; a[2] += 1
                row[cond + "_dperf"] = delta
        per_example.append(row)

    summary = {c: {"drift_rate": v[0] / v[1] if v[1] else None, "n": v[1],
                   "d_perf": (dperf[c][0] / dperf[c][1] if dperf[c][1] else None)}
               for c, v in drift.items()}
    bytask = {c: {t: (v[0] / v[1] if v[1] else None, v[1]) for t, v in tv.items()}
              for c, tv in drift_task.items()}
    perf_vs_er = {c: {b: {"d_perf": (v[0] / v[1] if v[1] else None), "n": v[1]}
                      for b, v in bins.items()}
                  for c, bins in dperf_erbin.items()}
    # absolute base vs perturbed success rate, per (task, cond)
    abs_perf = {t: {c: {"base": a[0] / a[2], "cond": a[1] / a[2], "n": a[2]}
                    for c, a in cd.items() if a[2]}
                for t, cd in acc_tc.items()}
    out = {"model": args.model, "summary": summary, "by_task": bytask,
           "perf_vs_er": perf_vs_er, "abs_perf": abs_perf,
           "base_accuracy": (base_acc[0] / base_acc[1] if base_acc[1] else None),
           "n_examples": len(per_example), "per_example": per_example}
    od = os.path.join(BASE, "data", "raw")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"drift_{args.model}.json")
    json.dump(out, open(fn, "w"))
    print(f"\n{args.model}: base accuracy {out['base_accuracy']}")
    print(f"{'cond':14s} {'drift%':>7s} {'Δperf':>7s} | Δperf by ER bin")
    for c in ("syn_cot", "wrong_cot", "all_wrong_cot", "mask_draft", "corrupt_draft", "drop_cot", "swap_cot", "wrong_input"):
        if c not in summary:
            continue
        s = summary[c]
        bins = perf_vs_er.get(c, {})
        bstr = "  ".join(f"{b}:{bins[b]['d_perf']:+.3f}(n{bins[b]['n']})"
                         for b in ("ER=0", "ER_lo", "ER_hi") if b in bins)
        dp = s['d_perf']
        print(f"  {c:12s} {s['drift_rate']*100:6.1f}% {dp:+7.3f} | {bstr}")
    print(f"wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
