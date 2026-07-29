"""Per-token example dump for the R5 mechanism figure: for a few curated examples per model,
compute per-token gradient x input saliency AND answer->token attention over the WHOLE sequence
(INPUT + TRACE), categorize each token, and export JSON + CSV + a self-contained HTML heatmap.
Deterministic selection (first valid per (task, ER-mode)) — not cherry-picked; swap examples by
editing the SPECS list or the exported data. Output: paper/.../data/token_examples/.
Run: python eval/token_examples.py --model Chem-R   (append per model)
"""
import argparse, json, os, sys, html
import torch
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "eval"))
import attention_attribution as AA
import attr_probe as AP   # categorize()

OUT = os.path.join(BASE, "data", "token_examples")
# (task, ER-mode): comparable set across models — SMILES-draft / fabricated-FG / retro / describe
SPECS = [("cap2mol", "er0"), ("cap2mol", "erpos"), ("retrosynthesis", "er0"), ("mol2cap", "er0")]
CAT_ORDER = ["SMILES_frag", "FG_word", "position_digit", "other_word", "punct", "space"]


def pick(model, task, mode):
    pool = AA.load_examples(model, task)
    if mode == "er0":
        c = [e for e in pool if e["er"] == 0 and e["exact"]]
    else:
        c = [e for e in pool if e["er"] > 0 and e["exact"]] or [e for e in pool if e["er"] > 0]
    return c


def analyze(model_name, tok, model, e, dev):
    prompt, full = AA.build_text(tok, e["desc"], e["answer"], e["task"])
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    offs, ids = enc["offset_mapping"], enc["input_ids"]
    if not (300 <= len(ids) <= 1500):
        return None
    plen = len(prompt)
    d0 = full.find(e["desc"]) if e["desc"] else -1
    t0, t1 = full.find("<think>", plen), full.find("</think>", plen)
    a0, a1 = full.find("<answer>", plen), full.find("</answer>", plen)
    if d0 < 0 or min(t0, t1, a0, a1) < 0 or t1 <= t0 or a1 <= a0:
        return None
    ai = AA.char_to_tok(offs, a0 + 8, a1)
    if not ai:
        return None
    in_tok = set(AA.char_to_tok(offs, d0, d0 + len(e["desc"])))
    tr_tok = set(AA.char_to_tok(offs, t0 + 7, t1))
    an_tok = set(ai)
    low = full.lower()
    fg_all = AA._fg_tok_in(offs, low, 0, len(full))
    sm_all, _ = AA._smiles_tok_in(offs, full, 0, len(full))

    ids_t = torch.tensor([ids], device=dev)
    # (1) gradient x input saliency over the whole sequence
    emb = model.get_input_embeddings()(ids_t).detach().requires_grad_(True)
    out = model(inputs_embeds=emb, use_cache=False)
    logp = torch.log_softmax(out.logits[0].float(), -1)
    tgt = torch.tensor([ids[i] for i in ai], device=dev)
    pos = torch.tensor([i - 1 for i in ai], device=dev)
    model.zero_grad(); (logp[pos].gather(1, tgt[:, None]).sum()).backward()
    sal = (emb.grad[0].float() * emb[0].float()).sum(-1).abs().detach().cpu().tolist()
    del emb, out
    # (2) answer -> token attention (mid layer, mean over heads + answer rows)
    with torch.no_grad():
        o2 = model(input_ids=ids_t, output_attentions=True, use_cache=False)
        L = len(o2.attentions); rows = torch.tensor(ai, device=dev)
        attn = o2.attentions[L // 2][0].mean(0).index_select(0, rows).mean(0).float().cpu().tolist()
        del o2

    def region_of(i):
        if i in an_tok: return "answer"
        if i in tr_tok: return "trace"
        if i in in_tok: return "input"
        return "other"
    # normalize sal/attn over INPUT+TRACE only (answer excluded — self-referential)
    color_idx = [i for i in range(len(ids)) if region_of(i) in ("input", "trace")]
    smax = max((sal[i] for i in color_idx), default=1.0) or 1.0
    amax = max((attn[i] for i in color_idx), default=1.0) or 1.0
    toks = []
    for i in range(len(ids)):
        r = region_of(i)
        txt = full[offs[i][0]:offs[i][1]]
        c = AP.categorize(i, txt, fg_all, sm_all)
        toks.append({"i": i, "text": txt, "region": r, "cat": c,
                     "sal": round(sal[i], 6), "attn": round(attn[i], 6),
                     "sal_norm": round(sal[i] / smax, 4) if r in ("input", "trace") else 0.0,
                     "attn_norm": round(attn[i] / amax, 4) if r in ("input", "trace") else 0.0})
    ga = _answer = AA._answer_smiles(e["answer"])
    return {"model": model_name, "task": e["task"], "id": str(e.get("id", "")),
            "er": e["er"], "exact": bool(e["exact"]),
            "answer_pred": _answer, "answer_gt": str(e.get("gt", "")),
            "n_tokens": len(ids), "tokens": toks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(AA.HF))
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(AA.HF[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        AA.HF[args.model], torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    picked = []
    for task, mode in SPECS:
        got = None
        for e in pick(args.model, task, mode):
            r = analyze(args.model, tok, model, e, dev)
            if r:
                r["spec"] = f"{task}/{mode}"; got = r; break
        if got:
            picked.append(got)
            print(f"  {args.model} {task}/{mode}: id={got['id']} er={got['er']} n={got['n_tokens']}", flush=True)
        else:
            print(f"  {args.model} {task}/{mode}: NONE", flush=True)
    # merge into the shared JSON (one file, all models)
    jf = os.path.join(OUT, "token_examples.json")
    allex = json.load(open(jf)) if os.path.exists(jf) else []
    allex = [x for x in allex if x["model"] != args.model] + picked
    json.dump(allex, open(jf, "w"), indent=1)
    print(f"wrote {jf} ({len(allex)} examples total)", flush=True)


if __name__ == "__main__":
    main()
