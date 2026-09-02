"""Re-run the hallucination detector over the released responses.

This is the script that proves the released detector reproduces the released numbers.
The detector kernel is ``diagnose_hallucination.py`` + ``diagnose_multitask.py``; this
harness only feeds it and collects what comes back.

The scoring rule it applies (the input-grounding fix of 2026-06-26): a claimed
functional group counts as fabrication ONLY if it is absent from BOTH the input (input
molecule + input caption/instruction text) AND the output. That is what keeps S2
edit/substitute tasks honest -- the group being removed is named in the source molecule
or the instruction -- and cap2mol likewise, where groups are named in the caption. It
also populates ``verified_fgs`` on every task so grounding coverage is computable.

Two modes, neither of which touches the shipped records by default:

  python eval/redx_all.py --verify [workers]
      Re-diagnose every released (model, task) pair and compare each record against the
      shipped one field by field. Writes nothing. This is the reproducibility check.

  python eval/redx_all.py --out DIR [workers]
      Re-diagnose and write fresh records under DIR, in the shipped layout.

Either mode takes ``--models a,b`` to restrict the run to some models. Workers default to
12; each holds a whole task's records in memory, so lower it on a shared login node.

Running it bare refuses, because the historical default overwrote data/results/ in
place with uncompressed files that would then shadow the shipped .jsonl.gz ones.
Set MOLREHALLU_REGEN=1 to get that old in-place behaviour deliberately.

S2 instance metadata lives in a metadata.json sidecar in the release (the working repo
carried it inside completions.json, which is 985x larger); both are read.
"""
import glob
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import io_utils as IO  # noqa: E402

# Fields compared in --verify. These are every field the paper's numbers are computed
# from; the rest of a record (verbose per-check detail) is diagnostic only.
COMPARED = ("hallucination_scores", "overall_hallucination_score",
            "pred_smiles", "pred_valid", "exact_match")


def _items(path):
    d = IO.load_json(path)
    return d if isinstance(d, list) else d.get("results", d)


def _meta_map(model, task):
    """S2 instance metadata, from completions.json or the released metadata.json sidecar."""
    cf = IO.find(f"{ROOT}/se_results/{model}/{task}/completions.json")
    if cf:
        return {str(s.get("id")): (s.get("metadata") or {}) for s in _items(cf[0])}
    sc = IO.find(f"{ROOT}/se_results/{model}/{task}/metadata.json")
    if sc:
        return {str(k): (v or {}) for k, v in IO.load_json(sc[0]).items()}
    return {}


def _shipped(model, task):
    """The released diagnosis records for one pair, keyed by id."""
    safe = model.replace("/", "_")
    hits = IO.find(f"{ROOT}/data/results/{model}/{task}/{safe}_{task}_hallucination_details.jsonl")
    if not hits:
        return None
    return {str(r.get("id")): r for r in IO.iter_jsonl(hits[0])}


def diagnose_pair(outpath, model, task):
    """Re-diagnose one (model, task) pair. Returns the list of fresh records."""
    from diagnose_multitask import diagnose_one
    mm = _meta_map(model, task) if task.startswith("s2_") else {}
    results = []
    for s in _items(outpath):
        ans = s.get("answer")
        if not ans:
            continue
        sid = str(s.get("id"))
        meta = s.get("metadata") or mm.get(sid, {}) or {}
        diag = diagnose_one(
            task,
            {"answer": ans, "question": s.get("question") or s.get("input", ""),
             "gt": s.get("gt", ""), "metadata": meta},
            verbose=True,
        )
        diag["id"] = s.get("id")
        diag["task"] = task
        diag["model"] = model
        results.append(diag)
    return results


def _summary(model, task, results):
    """Aggregate summary alongside the per-record file, same schema as the shipped one."""
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

    out = {"model": model, "task": task, "evaluated_samples": len(results),
           "validity_rate": round(valid / n * 100, 2),
           "exact_match_rate": round(exact / n * 100, 2),
           "hallucination_scores": {k: stat(v) for k, v in acc.items() if v}}
    if tan:
        out["avg_tanimoto"] = round(sum(tan) / len(tan), 4)
    return out


def run_pair(args):
    outpath, model, task, outdir = args
    try:
        results = diagnose_pair(outpath, model, task)
    except Exception as e:
        return model, task, 0, 0, 0, f"diagnose: {e}"

    if outdir is None:                                    # --verify: compare, write nothing
        ship = _shipped(model, task)
        if ship is None:
            return model, task, len(results), 0, 0, "no shipped records to compare against"
        same = diff = 0
        first = ""
        for r in results:
            s = ship.get(str(r.get("id")))
            if s is None:
                diff += 1
                first = first or f"id {r.get('id')} not in shipped records"
                continue
            bad = [f for f in COMPARED if r.get(f) != s.get(f)]
            if bad:
                diff += 1
                f0 = bad[0]
                first = first or f"id {r.get('id')} {f0}: fresh={r.get(f0)!r} shipped={s.get(f0)!r}"
            else:
                same += 1
        missing = len(ship) - (same + diff)
        if missing:
            first = first or f"{missing} shipped records had no fresh counterpart"
            diff += missing
        return model, task, len(results), same, diff, first

    od = os.path.join(outdir, model, task)
    os.makedirs(od, exist_ok=True)
    safe = model.replace("/", "_")
    with open(f"{od}/{safe}_{task}_hallucination_details.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{od}/{safe}_{task}_hallucination_summary.json", "w") as f:
        json.dump(_summary(model, task, results), f, ensure_ascii=False, indent=2)
    return model, task, len(results), 0, 0, ""


def main():
    argv = sys.argv[1:]
    verify = "--verify" in argv
    outdir = None
    if "--out" in argv:
        outdir = os.path.abspath(argv[argv.index("--out") + 1])
    workers = next((int(a) for a in argv if a.isdigit()), 12)
    only = set(argv[argv.index("--models") + 1].split(",")) if "--models" in argv else None

    if not verify and outdir is None:
        if os.environ.get("MOLREHALLU_REGEN") == "1":
            outdir = os.path.join(ROOT, "data", "results")
        else:
            raise SystemExit(
                "redx_all.py needs a mode.\n"
                "  --verify        re-diagnose and compare against the shipped records "
                "(writes nothing)\n"
                "  --out DIR       re-diagnose and write fresh records under DIR\n"
                "Refusing to default to data/results/: that writes uncompressed .jsonl "
                "beside the shipped .jsonl.gz, which io_utils then prefers, silently "
                "shadowing the released data. Set MOLREHALLU_REGEN=1 to do it anyway.")

    # io_utils resolves se_results/ -> data/responses/ and .json -> .json.gz, so this
    # finds the released tree as well as a working one.
    outs = sorted(set(IO.find(f"{ROOT}/se_results/*/*/output.json") +
                      glob.glob(f"{ROOT}/data/results/*/*/output.json")))
    jobs, seen = [], set()
    for o in outs:
        parts = o.split("/")
        task, model = parts[-2], parts[-3].replace(".json", "")
        if o.endswith(".gz"):
            task = os.path.basename(os.path.dirname(o))
            model = os.path.basename(os.path.dirname(os.path.dirname(o)))
        if only is not None and model not in only:
            continue
        if (model, task) in seen:
            continue
        seen.add((model, task))
        jobs.append((o, model, task, outdir))
    if not jobs:
        raise SystemExit(
            f"found no responses under {ROOT}/data/responses/ or {ROOT}/se_results/ -- "
            "nothing to re-diagnose. (Refusing to report success on an empty run.)")

    mode = "verifying" if verify else f"re-diagnosing -> {outdir}"
    print(f"{mode}: {len(jobs)} (model,task) pairs, {workers} workers", flush=True)
    ok = fail = 0
    tot_same = tot_diff = 0
    fails, mismatched = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_pair, j): j for j in jobs}
        for fut in as_completed(futs):
            model, task, n, same, diff, err = fut.result()
            tot_same += same
            tot_diff += diff
            if err and not diff:
                fail += 1
                fails.append((model, task, err))
            else:
                ok += 1
            if diff:
                mismatched.append((model, task, diff, err))
            if (ok + fail) % 20 == 0 or err:
                print(f"  [{ok+fail}/{len(jobs)}] {model}/{task} n={n} {err}", flush=True)

    print(f"\nREDX DONE: {ok} ok, {fail} failed", flush=True)
    for model, task, err in fails:
        print(f"  FAIL {model}/{task}: {err}", flush=True)
    if verify:
        tot = tot_same + tot_diff
        pct = 100.0 * tot_same / tot if tot else 0.0
        print(f"VERIFY: {tot_same}/{tot} records identical on {', '.join(COMPARED)} "
              f"({pct:.4f}%)", flush=True)
        for model, task, diff, first in mismatched:
            print(f"  MISMATCH {model}/{task}: {diff} records -- {first}", flush=True)
        # A run that compared nothing is a failure, not a pass.
        if tot == 0 or fail:
            raise SystemExit("verification did not compare a complete set of records")
        raise SystemExit(0 if tot_diff == 0 else 1)


if __name__ == "__main__":
    main()
