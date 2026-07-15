"""
Arena interpretation of the forced-choice human annotations.

The annotator picks, per prompt, the LEAST-hallucinatory model from the anonymized
set shown (4 models for most tasks, 6 for cap2mol). That is a top-1-from-N choice,
so the natural model is Luce / Plackett-Luce (the multi-way generalization of
Bradley-Terry that handles variable set sizes). We report:

  * raw votes: wins / appearances and the naive win-rate,
  * Luce arena strengths (MM algorithm, Hunter 2004), normalized and on an
    Elo-like log scale anchored so the mean model = 1000,
  * per-task win-rates,
  * agreement with the automatic detector (the key T7 reliability check):
    does the human's least-hallucinatory pick match the detector's argmin?

Usage: python human_eval/arena.py [annotations.json]
"""
import json
import os
import sys
import collections
import math

ROOT = os.path.dirname(os.path.abspath(__file__))


def luce_strengths(app_sets, wins, iters=500):
    """MM algorithm for the Luce top-1 choice model.
    app_sets: list of model-label sets (one per answered prompt).
    wins: dict model -> number of times chosen.
    Returns dict model -> strength (normalized to mean 1)."""
    models = sorted(wins)
    p = {m: 1.0 for m in models}
    for _ in range(iters):
        # denominator term: for each prompt set S, 1 / sum_{m in S} p_m,
        # accumulated per model over the prompts it appears in.
        denom = {m: 0.0 for m in models}
        for S in app_sets:
            tot = sum(p[m] for m in S)
            if tot <= 0:
                continue
            inv = 1.0 / tot
            for m in S:
                denom[m] += inv
        newp = {m: (wins[m] / denom[m] if denom[m] > 0 else 0.0) for m in models}
        mean = sum(newp.values()) / len(newp)
        if mean > 0:
            newp = {m: v / mean for m, v in newp.items()}
        p = newp
    return p


def main(path):
    blob = json.load(open(path))
    ann = blob["annotations"]
    S = {s["uid"]: s for s in json.load(open(os.path.join(ROOT, "samples.json")))}

    wins = collections.Counter()
    appear = collections.Counter()
    appp_sets = []
    # per task vote tally
    task_app = collections.defaultdict(collections.Counter)
    task_win = collections.defaultdict(collections.Counter)
    # detector agreement
    n = ag_ov = ag_er = 0
    rand = 0.0

    def score(m, key):
        s = m["scores"]
        if key == "overall":
            return (0.15 * s["IR_self_contradiction"] + 0.25 * s["IO_structural_invalidity"]
                    + 0.25 * s["ER_factual_fabrication"] + 0.35 * s["EO_phantom_structure"])
        return s[key]

    def argmin(item, key):
        best, bv = None, 1e9
        for m in item["models"]:
            v = score(m, key)
            if v < bv:
                bv, best = v, m["model"]
        return best

    for uid, a in ann.items():
        if uid not in S or not a.get("chosen"):
            continue
        item = S[uid]
        labels = [m["model"] for m in item["models"]]
        chosen = a["chosen"]
        if chosen not in labels:
            continue
        appp_sets.append(set(labels))
        for m in labels:
            appear[m] += 1
            task_app[item["task"]][m] += 1
        wins[chosen] += 1
        task_win[item["task"]][chosen] += 1
        # detector agreement
        n += 1
        rand += 1.0 / len(labels)
        ag_ov += int(chosen == argmin(item, "overall"))
        ag_er += int(chosen == argmin(item, "ER_factual_fabrication"))

    strengths = luce_strengths(appp_sets, dict(wins))
    # Elo-like: 1000 + 400*log10(strength)   (mean strength = 1 -> 1000)
    elo = {m: 1000 + 400 * math.log10(v) if v > 0 else float("-inf")
           for m, v in strengths.items()}

    print(f"annotator: {blob.get('annotator')}   prompts answered: {n}\n")
    order = sorted(appear, key=lambda m: -strengths.get(m, 0))
    print(f"{'model':32}{'wins':>6}{'appeared':>10}{'win%':>8}{'Luce':>8}{'arena-Elo':>11}")
    for m in order:
        w, ap = wins[m], appear[m]
        wr = 100 * w / ap if ap else 0
        print(f"{m:32}{w:>6}{ap:>10}{wr:>7.1f}%{strengths[m]:>8.2f}{elo[m]:>11.0f}")

    print(f"\n--- detector agreement (human least-halluc pick vs detector argmin) ---")
    print(f"vs min OVERALL: {ag_ov}/{n} = {100*ag_ov/n:.1f}%")
    print(f"vs min ER:      {ag_er}/{n} = {100*ag_er/n:.1f}%")
    print(f"random baseline:{100*rand/n:.1f}%")

    print(f"\n--- per-task win-rate (wins/appearances) ---")
    for t in sorted(task_app):
        print(f"  {t}")
        for m in sorted(task_app[t], key=lambda x: -task_win[t][x]):
            ap = task_app[t][m]; w = task_win[t][m]
            print(f"     {m:32} {w:>4}/{ap:<4} = {100*w/ap:>5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "annotations.json"))
