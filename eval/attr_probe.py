"""Gradient×input saliency (causal-importance) of the ANSWER w.r.t. TRACE tokens, categorized
(SMILES-fragment / FG-word / position-digit / other-word / punct). Verifies the "molecular
scratchpad" narrative: which token TYPE in the reasoning the answer actually keys on, per model.
grad×input is more principled than attention ("what the answer is sensitive to"). One fwd+bwd,
no output_attentions. Output: eval/attn_out/gradattr_<model>.json.
Usage: python eval/attr_probe.py --model Chem-R [--n_attn 200]
"""
import argparse, json, os, re, sys
import torch
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "eval"))
import attention_attribution as AA  # reuse HF, load_examples, build_text, spans, char_to_tok, FG/SMILES span helpers


def categorize(idx, tok_str, fg_tok, sm_tok):
    if idx in sm_tok:
        return "SMILES_frag"
    if idx in fg_tok:
        return "FG_word"
    t = tok_str.strip()
    if t == "":
        return "space"
    if re.fullmatch(r"[0-9]+", t) or re.search(r"\d", t):
        return "position_digit"
    if re.search(r"[A-Za-z]{3,}", t):
        return "other_word"
    return "punct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(AA.HF))
    ap.add_argument("--tasks", nargs="+", default=AA.DEFAULT_TASKS)
    ap.add_argument("--n_attn", type=int, default=10**9)   # default FULL volume
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(AA.HF[args.model])
    model = AutoModelForCausalLM.from_pretrained(AA.HF[args.model], torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)              # grad only wrt input embeddings -> saves memory/time
    dev = "cuda"
    all_ex = []
    for t in args.tasks:
        all_ex += list(AA.load_examples(args.model, t))    # FULL volume: no correctness/ER filter
    import random as _r; _r.seed(0); _r.shuffle(all_ex)

    from collections import defaultdict
    STRATA = ["all", "er0", "erpos"]                       # ER-stratified, full volume
    mass = {s: defaultdict(float) for s in STRATA}
    cnt = {s: defaultdict(int) for s in STRATA}
    ndoc = {s: 0 for s in STRATA}
    n = 0; skipped_long = 0; skipped_fmt = 0
    order = ["SMILES_frag", "FG_word", "position_digit", "other_word", "punct", "space"]
    fn = os.path.join(BASE, "eval", "attn_out", f"gradattr_{args.model}.json")

    def frac(s):
        return {c: (mass[s][c] / ndoc[s] if ndoc[s] else 0.0) for c in order}

    def save():
        json.dump({"model": args.model, "n": n, "skipped_long": skipped_long, "skipped_fmt": skipped_fmt,
                   "n_by_stratum": dict(ndoc),
                   "trace_saliency_frac": {s: frac(s) for s in STRATA},
                   "token_counts": {s: {c: cnt[s][c] for c in order} for s in STRATA}}, open(fn, "w"))

    for e in all_ex:
        if n >= args.n_attn:
            break
        prompt, full = AA.build_text(tok, e["desc"], e["answer"], e["task"])
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        offs, ids = enc["offset_mapping"], enc["input_ids"]
        if len(ids) > 3000:
            skipped_long += 1; continue
        plen = len(prompt)
        t0, t1 = full.find("<think>", plen), full.find("</think>", plen)
        a0, a1 = full.find("<answer>", plen), full.find("</answer>", plen)
        if min(t0, t1, a0, a1) < 0 or t1 <= t0 or a1 <= a0:
            skipped_fmt += 1; continue
        ci = AA.char_to_tok(offs, t0 + 7, t1); ai = AA.char_to_tok(offs, a0 + 8, a1)
        if not ci or not ai:
            skipped_fmt += 1; continue
        low = full.lower()
        fg_tok = AA._fg_tok_in(offs, low, t0 + 7, t1)
        sm_tok, _ = AA._smiles_tok_in(offs, full, t0 + 7, t1)
        ids_t = torch.tensor([ids], device=dev)
        emb = model.get_input_embeddings()(ids_t).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, use_cache=False)
        logp = torch.log_softmax(out.logits[0].float(), -1)
        tgt = torch.tensor([ids[i] for i in ai], device=dev)
        pos = torch.tensor([i - 1 for i in ai], device=dev)
        score = logp[pos].gather(1, tgt[:, None]).sum()
        model.zero_grad(); score.backward()
        sal = (emb.grad[0].float() * emb[0].float()).sum(-1).abs().detach().cpu()
        tot = sum(sal[i].item() for i in ci) or 1.0
        buckets = ["all", "er0" if e["er"] == 0 else "erpos"]
        for i in ci:
            c = categorize(i, tok.decode([ids[i]]), fg_tok, sm_tok)
            for s in buckets:
                mass[s][c] += sal[i].item() / tot
                cnt[s][c] += 1
        for s in buckets:
            ndoc[s] += 1
        n += 1
        if n % 500 == 0:
            print(f"  ...{n} done (skip long={skipped_long} fmt={skipped_fmt})", flush=True)
            save()                                   # checkpoint: survive walltime kill
        del emb, out, sal
    save()
    for s in STRATA:
        fr = frac(s)
        print(f"{args.model} [{s}] (n={ndoc[s]}) — TRACE grad-saliency mass by token type:")
        for c in order:
            print(f"   {c:15s} {fr[c]*100:5.1f}%  (tokens {cnt[s][c]})")
    print(f"wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
