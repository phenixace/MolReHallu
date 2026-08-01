"""Conditional answer-entropy: does the CoT reduce uncertainty about the answer?
The DIRECT, metric-free test (no success-metric dependence). For each example, sample
N answers from THREE prefixes and take the answer entropy (cluster by canonical SMILES):
  H_noCoT  : prompt + <think></think> + answer-open   (no reasoning)
  H_realCoT: prompt + the model's REAL CoT + answer-open
  H_corrCoT: prompt + the CoT with every FG corrupted + answer-open
  info_gain_presence = H_noCoT  - H_realCoT   (>0 => HAVING a CoT lowers answer entropy)
  info_gain_content  = H_corrCoT - H_realCoT  (>0 => the CoT's CONTENT specifically does)
Reuses cot_drift's markup-aware build_conditions / canon. Output data/raw/condsent_<model>.json
Usage: python eval/cot_condsent.py --model Chem-R [--n_samples 8] [--max_per_task N]
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "eval"))
import cot_drift as CD  # reuse HF, markup_for, build_conditions, canon, _norm, prompts  # noqa: E402


def entropy(texts, task):
    c = Counter(CD.canon(task, t) for t in texts)
    n = sum(c.values())
    return -sum((k / n) * math.log(k / n) for k in c.values()) if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CD.HF))
    ap.add_argument("--tasks", nargs="+", default=CD.DEFAULT_TASKS)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_per_task", type=int, default=10**9)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import glob
    tok = AutoTokenizer.from_pretrained(CD.HF[args.model], trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        eos = getattr(tok, "eos_token", "") or ""
        for b in (["meta-llama/Llama-3.1-8B-Instruct"] if "eot_id" in eos else ["Qwen/Qwen2.5-7B-Instruct"]):
            try:
                bt = AutoTokenizer.from_pretrained(b, trust_remote_code=True)
                if bt.chat_template:
                    tok.chat_template = bt.chat_template; break
            except Exception:
                pass
    llm = LLM(model=CD.HF[args.model], dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096, trust_remote_code=True)
    mk = CD.markup_for(args.model)
    sp = SamplingParams(temperature=0.8, top_p=0.95, n=args.n_samples, max_tokens=320, stop=[mk[3]])

    # stage prefixes for 3 conditions per example
    jobs, meta = [], {}   # jobs: (uid, cond, prompt); meta[uid]=task
    for task in args.tasks:
        of = IO.find(f"{BASE}/se_results/{args.model}/{task}/output.json")
        if not of:
            continue
        items = IO.load_json(of[0])
        items = items if isinstance(items, list) else items.get("results", [])
        cot_pool = [c for c in (CD.extract_cot(s.get("answer", ""), mk) for s in items) if c and c.strip()]
        # ER label per id (to stratify swap analysis by clean ER=0 vs fabricating ER>0)
        import glob as _g
        df = IO.find(f"{BASE}/data/results/{args.model}/{task}/*hallucination_details.jsonl")
        er_of = {}
        if df:
            for l in IO.open_text(df[0]):
                d = json.loads(l); er_of[str(d["id"])] = d["hallucination_scores"]["ER_factual_fabrication"]
        n = 0
        for s in items:
            if n >= args.max_per_task:
                break
            own = CD.extract_cot(s.get("answer", ""), mk)
            donor = None
            for _ in range(5):
                cand = random.choice(cot_pool) if cot_pool else None
                if cand and cand != own:
                    donor = cand; break
            bc = CD.build_conditions(task, s.get("question", ""), s.get("answer", ""), mk, donor_cot=donor)
            if not bc or "all_wrong_cot" not in bc["conds"]:
                continue
            uid = f"{task}|{s.get('id')}"
            meta[uid] = {"task": task, "er": er_of.get(str(s.get("id")))}
            conds = ["base", "drop_cot", "all_wrong_cot"] + (["swap_cot"] if "swap_cot" in bc["conds"] else [])
            for cond in conds:
                chat, tail = bc["conds"][cond]
                p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + tail
                jobs.append((uid, cond, p))
            n += 1
        print(f"  {task}: {n} examples staged", flush=True)

    print(f"{args.model}: {len(jobs)} prefixes x n={args.n_samples} ...", flush=True)
    outs = llm.generate([p for _, _, p in jobs], sp)
    gen = defaultdict(dict)
    for (uid, cond, _), o in zip(jobs, outs):
        gen[uid][cond] = [x.text for x in o.outputs]

    rows = []
    for uid, cg in gen.items():
        if not all(k in cg for k in ("base", "drop_cot", "all_wrong_cot")):
            continue
        task = meta[uid]["task"]; er = meta[uid]["er"]
        h_real = entropy(cg["base"], task)
        h_no = entropy(cg["drop_cot"], task)
        h_corr = entropy(cg["all_wrong_cot"], task)
        if None in (h_real, h_no, h_corr):
            continue
        row = {"task": task, "uid": uid, "er": er, "H_realCoT": h_real, "H_noCoT": h_no,
               "H_corrCoT": h_corr, "ig_presence": h_no - h_real, "ig_content": h_corr - h_real}
        if "swap_cot" in cg:                              # real CoT from another molecule
            h_swap = entropy(cg["swap_cot"], task)
            if h_swap is not None:
                row["H_swapCoT"] = h_swap
                row["ig_swap"] = h_swap - h_real          # >0 => THIS molecule's CoT content matters
        rows.append(row)

    def agg(rs):
        d = {"n": len(rs)}
        for k in ("H_realCoT", "H_noCoT", "H_corrCoT", "ig_presence", "ig_content"):
            d[k] = sum(r[k] for r in rs) / len(rs)
        sw = [r["ig_swap"] for r in rs if "ig_swap" in r]
        d["ig_swap"] = (sum(sw) / len(sw)) if sw else None
        d["n_swap"] = len(sw)
        return d
    tasks = sorted(set(r["task"] for r in rows))
    per_task = {t: agg([r for r in rows if r["task"] == t]) for t in tasks}
    # ER-stratified (the sharp test: does swapping matter even on CLEAN ER=0 traces?)
    clean = [r for r in rows if (r["er"] or 0) == 0]
    fabr = [r for r in rows if (r["er"] or 0) > 0]
    er_split = {"ER=0": agg(clean) if clean else None, "ER>0": agg(fabr) if fabr else None,
                "all": agg(rows) if rows else None}
    out = {"model": args.model, "n_samples": args.n_samples, "per_task": per_task,
           "er_split": er_split, "per_example": rows}
    fn = os.path.join(BASE, "data", "raw", f"condsent_{args.model}.json")
    json.dump(out, open(fn, "w"))
    print(f"\n{args.model}: ig_presence / ig_content / ig_swap  (nats; >0 => that aspect of CoT lowers answer entropy)")
    print(f"{'subset':10s} {'n':>5s} {'igPRES':>7s} {'igCONT':>7s} {'igSWAP':>7s}")
    for lbl in ("all", "ER=0", "ER>0"):
        v = er_split.get(lbl)
        if v:
            sw = f"{v['ig_swap']:+7.3f}" if v['ig_swap'] is not None else "   -   "
            print(f"{lbl:10s} {v['n']:>5d} {v['ig_presence']:+7.3f} {v['ig_content']:+7.3f} {sw} (n_swap={v['n_swap']})")
    print(f"wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
