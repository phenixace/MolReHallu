"""Information-gain of the CoT: does reasoning reduce the model's uncertainty
about the answer? (Motivation analysis for why hallucination is accuracy-invisible.)

For each example we sample N answers TWO ways and take the semantic entropy of the
answer set (molecule tasks: cluster by canonical SMILES):
  H_free  : the model reasons freely (natural <think>..</think><answer>..)
  H_noCoT : reasoning suppressed (prompt + <think></think> + <answer>, direct answer)
  info_gain = H_noCoT - H_free   >=0 means the reasoning collapsed answer uncertainty.

Thesis: fabrication (ER) concentrates where info_gain ~ 0 -- the CoT is not doing
work, so it is free to confabulate and accuracy cannot see it. On tasks where the
CoT carries real information gain (s2 constraint satisfaction) it is load-bearing.

Output: data/raw/infogain_<model>.json  (per-task H_free/H_noCoT/info_gain + ER)
Usage: python eval/cot_info_gain.py --model Chem-R [--n_samples 8] [--max_per_task 100]
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import prompts as P  # noqa: E402
import io_utils as IO  # noqa: E402  (gz-aware, se_results/ -> data/responses/)
from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

_ANS = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_DELIMS = [("<|think_start|>", "<think>"), ("<|think_end|>", "</think>"),
           ("<|answer_start|>", "<answer>"), ("<|answer_end|>", "</answer>")]
# SMILES-answer tasks: canonical-SMILES clustering gives crisp semantic entropy.
DEFAULT_TASKS = ["cap2mol", "retrosynthesis", "s2_MolCustom_FunctionalGroup",
                 "s2_MolCustom_AtomNum", "s2_MolCustom_BondNum",
                 "s2_MolEdit_AddComponent", "s2_MolEdit_SubComponent"]
HF = {"Chem-R": "weidawang/Chem-R-8B",
      "ChemDFM-R": "OpenDFM/ChemDFM-R",
      "ether-0": "futurehouse/ether0",
      "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
      "Chem-R-Faithful": "phenixace/Chem-R-Faithful",
      "Llama-3.1-8B-Instruct-base": "meta-llama/Llama-3.1-8B-Instruct",
      "Chem-R-SFT": "slayertear/llama-3.1-8b-stage2"}


def canon(ans):
    # The answer-CLOSE tag is consumed by the vLLM stop token, so free-generation text
    # looks like "<think>..</think><answer>SMILES" (no closing tag). Pull the text AFTER
    # the last answer-OPEN marker; if none (direct-answer regime -> bare SMILES), use the
    # whole text. (Old code fell back to the full trace and took "<think>" as token 0,
    # collapsing every sample to the same value -> spurious H_free=0.)
    for om in ("<answer>", "<|answer_start|>"):
        i = ans.rfind(om)
        if i >= 0:
            ans = ans[i + len(om):]
            break
    a = ans.strip()
    a = a.split()[0] if a else a
    mol = Chem.MolFromSmiles(a) if a else None
    return Chem.MolToSmiles(mol) if mol else a.strip()


def entropy(answers):
    """Shannon entropy (nats) over canonical-SMILES clusters of the sampled answers."""
    c = Counter(canon(a) for a in answers)
    n = sum(c.values())
    if n == 0:
        return None
    return -sum((k / n) * math.log(k / n) for k in c.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF))
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_per_task", type=int, default=10**9)   # default FULL
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[args.model], trust_remote_code=True)
    if not getattr(tok, "chat_template", None):     # Chem-R/merged: borrow base template
        eos = getattr(tok, "eos_token", "") or ""
        bases = (["meta-llama/Llama-3.1-8B-Instruct"] if "eot_id" in eos
                 else ["Qwen/Qwen2.5-7B-Instruct"])
        for b in bases:
            try:
                bt = AutoTokenizer.from_pretrained(b, trust_remote_code=True)
                if bt.chat_template:
                    tok.chat_template = bt.chat_template; break
            except Exception:
                pass
    llm = LLM(model=HF[args.model], dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096, trust_remote_code=True)
    sp_free = SamplingParams(temperature=0.8, top_p=0.95, n=args.n_samples,
                             max_tokens=1600, stop=["</answer>"])
    sp_direct = SamplingParams(temperature=0.8, top_p=0.95, n=args.n_samples,
                               max_tokens=320, stop=["</answer>"])

    rows = []
    for task in args.tasks:
        of = IO.find(f"{BASE}/se_results/{args.model}/{task}/output.json")
        if not of:
            continue
        items = IO.load_json(of[0])
        items = items if isinstance(items, list) else items.get("results", [])
        df = IO.find(f"{BASE}/data/results/{args.model}/{task}/*hallucination_details.jsonl")
        er_of = {}
        if df:
            for l in IO.open_text(df[0]):
                d = json.loads(l); er_of[str(d["id"])] = d["hallucination_scores"]["ER_factual_fabrication"]
        items = items[:args.max_per_task]
        base = [tok.apply_chat_template(P.build_messages(task, s.get("question", "")),
                                        tokenize=False, add_generation_prompt=True) for s in items]
        free_p = base
        direct_p = [b + "<think></think>\n<answer>" for b in base]
        o_free = llm.generate(free_p, sp_free)
        o_dir = llm.generate(direct_p, sp_direct)
        for s, of_, od_ in zip(items, o_free, o_dir):
            hf = entropy([x.text for x in of_.outputs])       # free: has <answer>SMILES
            hd = entropy([x.text for x in od_.outputs])       # direct: bare SMILES
            if hf is None or hd is None:
                continue
            rows.append({"task": task, "id": str(s.get("id")), "H_free": hf,
                         "H_noCoT": hd, "info_gain": hd - hf, "er": er_of.get(str(s.get("id")))})
        print(f"  {task}: {len([r for r in rows if r['task']==task])} examples", flush=True)

    # aggregate per task
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    per_task = {}
    for t in set(r["task"] for r in rows):
        rt = [r for r in rows if r["task"] == t]
        per_task[t] = {"n": len(rt), "H_free": mean([r["H_free"] for r in rt]),
                       "H_noCoT": mean([r["H_noCoT"] for r in rt]),
                       "info_gain": mean([r["info_gain"] for r in rt]),
                       "ER": mean([r["er"] for r in rt])}
    out = {"model": args.model, "n_samples": args.n_samples,
           "per_task": per_task, "per_example": rows}
    od = os.path.join(BASE, "data", "raw")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"infogain_{args.model}.json")
    json.dump(out, open(fn, "w"))
    print(f"\n{args.model}: info_gain = H_noCoT - H_free (nats), and ER, per task")
    print(f"{'task':30s} {'H_free':>7s} {'H_noCoT':>8s} {'gain':>6s} {'ER':>6s}")
    for t, v in sorted(per_task.items(), key=lambda kv: -(kv[1]['info_gain'] or 0)):
        print(f"{t:30s} {v['H_free']:7.3f} {v['H_noCoT']:8.3f} {v['info_gain']:+6.3f} "
              f"{(v['ER'] or 0):6.2f}")
    print(f"wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
