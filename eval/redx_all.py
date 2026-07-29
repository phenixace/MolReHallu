"""Full re-diagnosis after the input-grounding ER fix (2026-06-26).

A claimed functional group is fabrication ONLY if it is absent from BOTH the
input (input molecule + input caption/instruction text) AND the output. This
fixes false positives on S2 edit/substitute tasks (the group being removed is
named in the source molecule / instruction) and on cap2mol (groups named in the
caption). Also populates `verified_fgs` for every task so GC is computable.

Merges S2 instance metadata from completions.json (output.json has none).
Run: python eval/redx_all.py [max_workers]
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MAX_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def _items(path):
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get("results", d)


def _meta_map(model, task):
    f = f"{ROOT}/se_results/{model}/{task}/completions.json"
    if not os.path.exists(f):
        return {}
    return {str(s.get("id")): (s.get("metadata") or {}) for s in _items(f)}


def run_pair(args):
    outpath, model, task = args
    from diagnose_multitask import diagnose_one
    try:
        items = _items(outpath)
    except Exception as e:
        return model, task, 0, f"load: {e}"
    mm = _meta_map(model, task) if task.startswith("s2_") else {}
    results = []
    for s in items:
        ans = s.get("answer")
        if not ans:
            continue
        sid = str(s.get("id"))
        meta = s.get("metadata") or mm.get(sid, {}) or {}
        try:
            diag = diagnose_one(
                task,
                {"answer": ans, "question": s.get("question") or s.get("input", ""),
                 "gt": s.get("gt", ""), "metadata": meta},
                verbose=True,
            )
        except Exception as e:
            return model, task, 0, f"diag {sid}: {e}"
        diag["id"] = s.get("id")
        diag["task"] = task
        diag["model"] = model
        results.append(diag)
    od = f"{ROOT}/data/results/{model}/{task}"
    os.makedirs(od, exist_ok=True)
    safe = model.replace("/", "_")
    with open(f"{od}/{safe}_{task}_hallucination_details.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # aggregate summary (schema consumed by make_latex_tables.H/dim)
    from collections import defaultdict
    acc = defaultdict(list)
    tan = []
    exact = valid = 0
    for r in results:
        for k, v in r["hallucination_scores"].items():
            acc[k].append(v)
        acc["overall"].append(r["overall_hallucination_score"])
        if r.get("exact_match"):
            exact += 1
        if r.get("pred_valid"):
            valid += 1
        for tk in ("tanimoto", "reactant_tanimoto", "product_tanimoto"):
            if isinstance(r.get(tk), (int, float)):
                tan.append(r[tk]); break
    n = max(len(results), 1)
    def stat(v):
        mu = sum(v) / len(v)
        return {"mean": round(mu, 2),
                "std": round((sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5, 2),
                "min": round(min(v), 2), "max": round(max(v), 2)}
    summary = {
        "model": model, "task": task, "evaluated_samples": len(results),
        "validity_rate": round(valid / n * 100, 2),
        "exact_match_rate": round(exact / n * 100, 2),
        "hallucination_scores": {k: stat(v) for k, v in acc.items() if v},
    }
    if tan:
        summary["avg_tanimoto"] = round(sum(tan) / len(tan), 4)
    with open(f"{od}/{safe}_{task}_hallucination_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return model, task, len(results), ""


def main():
    outs = sorted(set(glob.glob(f"{ROOT}/se_results/*/*/output.json") +
                      glob.glob(f"{ROOT}/data/results/*/*/output.json")))
    jobs, seen = [], set()
    for o in outs:
        parts = o.split("/")
        task, model = parts[-2], parts[-3]
        if (model, task) in seen:
            continue
        seen.add((model, task))
        jobs.append((o, model, task))
    print(f"re-diagnosing {len(jobs)} (model,task) pairs, {MAX_WORKERS} workers", flush=True)
    ok = fail = 0
    fails = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_pair, j): j for j in jobs}
        for fut in as_completed(futs):
            model, task, n, err = fut.result()
            if err:
                fail += 1
                fails.append((model, task, err))
            else:
                ok += 1
            if (ok + fail) % 20 == 0 or err:
                print(f"  [{ok+fail}/{len(jobs)}] {model}/{task} n={n} {err}", flush=True)
    print(f"\nREDX DONE: {ok} ok, {fail} failed", flush=True)
    for model, task, err in fails:
        print(f"  FAIL {model}/{task}: {err}", flush=True)


if __name__ == "__main__":
    main()
