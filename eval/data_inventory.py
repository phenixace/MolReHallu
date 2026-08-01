#!/usr/bin/env python3
"""Generate the data-source inventory: which model was measured on which dataset, with what
volume, in every experiment reported in the paper.

Nothing here is hardcoded. Models, tasks and counts are discovered from the shipped files:
  data/stats_per_model_task.csv     -> diagnosis coverage + performance
  data/raw/{drift,condsent,gradattr,region}_<model>.json  -> mechanism-experiment coverage
  data/attention_perturbation.csv   -> causal-perturbation / matched-token coverage
  data/raw/README.md                -> filename token <-> sheet label mapping (parsed, not assumed)
  eval/attention_attribution.py     -> model -> HF weights (imported, not copied)

The only declared metadata is task -> dataset, which is a naming fact rather than a measurement.
Provenance: data_loaders.py PATHS/S2_FILES in the analysis repo --
  cap2mol, mol2cap      <- ChEBI-20            (chebi-20/test.txt)
  retrosynthesis        <- USPTO-50k           (uspto50k/USPTO_50K.csv)
  s2_<Family>_<Sub>     <- S2-TOMG-Bench       (s2-bench/<Family>_<Sub>.csv)
  bace/bbbp/hiv/tox21/clintox <- MoleculeNet   (not used in the paper)

Run: python eval/data_inventory.py
Writes data/DATA_INVENTORY.md and data/data_inventory.csv.
"""
import csv
import glob
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
RAW = os.path.join(D, "raw")

# task -> (dataset, dataset file) ; see provenance note above
def dataset_of(task):
    if task in ("cap2mol", "mol2cap"):
        return "ChEBI-20", "chebi-20/test.txt"
    if task == "retrosynthesis":
        return "USPTO-50k", "uspto50k/USPTO_50K.csv"
    if task.startswith("s2_"):
        return "S2-TOMG-Bench", "s2-bench/%s.csv" % task[3:]
    return "MoleculeNet", "moleculenet/%s/%s.csv" % (task, task)


def task_family(task):
    if task.startswith("s2_"):
        return "S2 instruction-following"
    if task in ("cap2mol", "mol2cap", "retrosynthesis"):
        return "translation"
    return "classification"


def raw_name_maps():
    """Parse data/raw/README.md's 3-column mapping table instead of assuming the scheme.
    Returns (filename token -> sheet label, sheet label -> internal codename)."""
    p = os.path.join(RAW, "README.md")
    tok2lab, lab2int = {}, {}
    if not os.path.exists(p):
        return tok2lab, lab2int
    for line in open(p):
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] and not cells[0].startswith(("-", ":", "display")):
            tok2lab[cells[0]] = cells[1]
            lab2int[cells[1]] = cells[2]
    return tok2lab, lab2int


def hf_weights():
    """model -> HF repo (or the release placeholder), imported from the probe's own map."""
    sys.path.insert(0, os.path.join(BASE, "eval"))
    sys.path.insert(0, BASE)
    try:
        import attention_attribution as AA
        return dict(AA.HF)
    except Exception:
        return {}


def paper_tasks():
    """The task set the paper reports, taken from the figure code's own GEN12 list."""
    sys.path.insert(0, os.path.join(BASE, "figures"))
    try:
        import make_nmi_figures as M
        return [t for t, _ in M.GEN12]
    except Exception:
        return []


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(x, nd=2):
    return "" if x is None else ("%.*f" % (nd, x))


def main():
    tok2lab, lab2int = raw_name_maps()
    hf_raw = hf_weights()
    IN_PAPER = paper_tasks()

    int2lab = {v: k for k, v in lab2int.items()}

    def canon(name):
        """Resolve any of {release label, internal codename} to the release label."""
        return int2lab.get(name, name)

    def weights(label):
        """HF entry for a release label, falling back to its internal codename."""
        return hf_raw.get(label) or hf_raw.get(lab2int.get(label, ""), "")

    diag = read_csv(os.path.join(D, "stats_per_model_task.csv"))

    # ---------- 0. origin-ladder rungs (they are not in stats_per_model_task.csv) ---------
    ladder = []
    lp = os.path.join(D, "stage_ladder.csv")
    if os.path.exists(lp):
        ladder = read_csv(lp)
    # derive which tasks the ladder's n_resp corresponds to, instead of asserting it
    by_task_n = {}
    for r in diag:
        n = num(r["n"])
        if n:
            by_task_n.setdefault(r["task"], set()).add(int(n))
    ladder_basis = []
    if ladder:
        target = num(ladder[0]["n_resp"])
        tr = [t for t in by_task_n if task_family(t) == "translation"]
        if target and abs(sum(max(by_task_n[t]) for t in tr) - target) < 1:
            ladder_basis = sorted(tr)

    # ---------- 1. dataset / task inventory (volume per task, from the diagnosis table) ----
    tasks = {}
    for r in diag:
        t = r["task"]
        n = num(r["n"])
        e = tasks.setdefault(t, {"n": set(), "models": set()})
        if n:
            e["n"].add(int(n))
        e["models"].add(r["model"])

    # ---------- 2. mechanism-experiment coverage, discovered from data/raw ----------------
    EXP = {
        "drift": ("CoT / draft perturbation", "per_example"),
        "condsent": ("conditional entropy", "per_example"),
        "gradattr": ("gradient x input saliency", None),
        "region": ("region attention + Dlogp + matched-token", "region_attr"),
    }
    mech = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        stem = os.path.basename(f)[:-len(".json")]
        kind, _, token = stem.partition("_")
        if kind not in EXP:
            continue
        d = json.load(open(f))
        label = tok2lab.get(token, token)
        rec = mech.setdefault((label, kind), {})
        rec["file"] = os.path.basename(f)
        rec["internal"] = d.get("model")
        # per_task is keyed by task; drift's by_task is keyed by CONDITION then task, so for
        # anything carrying per_example records take the task set from the records themselves.
        if isinstance(d.get("per_example"), list) and d["per_example"]:
            rec["tasks"] = sorted({e.get("task") for e in d["per_example"] if e.get("task")})
        else:
            rec["tasks"] = sorted((d.get("per_task") or {}).keys())
        if kind == "gradattr":
            rec["n"] = d.get("n")
            rec["strata"] = d.get("n_by_stratum")
        elif kind == "region":
            rec["n"] = len(d.get("region_attr") or [])
            rec["n_matched"] = len(d.get("matched") or [])
            rec["n_perturb"] = len(d.get("perturb") or [])
        else:
            rec["n"] = len(d.get("per_example") or [])
            if kind == "drift":
                rec["conditions"] = sorted((d.get("summary") or {}).keys())
                rec["base_acc"] = d.get("base_accuracy")
            if kind == "condsent":
                rec["n_samples"] = d.get("n_samples")

    # ---------- 3. write CSV (one row per model x task x experiment) ----------------------
    metric_cols = [c for c in diag[0] if c not in ("model", "task")]
    rows = []
    for r in diag:
        ds, dsf = dataset_of(r["task"])
        row = {
            "experiment": "diagnosis", "model": r["model"], "task": r["task"],
            "task_family": task_family(r["task"]), "dataset": ds, "dataset_file": dsf,
            "reported_in_paper": r["task"] in IN_PAPER, "weights": weights(r["model"]),
        }
        row.update({c: r[c] for c in metric_cols})
        rows.append(row)
    for r in ladder:
        for t in (ladder_basis or ["(3 translation tasks)"]):
            ds, dsf = dataset_of(t) if t in by_task_n else ("(multiple)", "")
            rows.append({
                "experiment": "origin ladder (%s)" % r["stage"], "model": canon(r["model"]), "task": t,
                "task_family": "translation", "dataset": ds, "dataset_file": dsf,
                "reported_in_paper": True, "weights": weights(canon(r["model"])),
                "n": max(by_task_n[t]) if t in by_task_n else r["n_resp"],
                "perf": r["perf"], "ER": r["ER"], "pct_er0": r["pct_er0"],
                "claims_per_resp": r["claims_per_resp"],
                "perclaim_fab_rate": r["perclaim_fab_rate"],
                "hedge_rate": r["hedge_rate"], "abstain_rate": r["abstain_rate"],
                "fab_position": r["fab_position"],
            })
    for (label, kind), rec in sorted(mech.items()):
        for t in (rec["tasks"] or ["(pooled)"]):
            ds, dsf = dataset_of(t) if t != "(pooled)" else ("(multiple)", "")
            rows.append({
                "experiment": EXP[kind][0], "model": label, "task": t,
                "task_family": task_family(t) if t != "(pooled)" else "",
                "dataset": ds, "dataset_file": dsf,
                "reported_in_paper": t in IN_PAPER if t != "(pooled)" else True,
                "n": rec.get("n"), "weights": weights(label),
                "n_conditions": len(rec.get("conditions") or []) or "",
                "base_accuracy": rec.get("base_acc", ""),
                "samples_per_response": rec.get("n_samples", ""),
                "n_matched_token": rec.get("n_matched", ""),
                "n_dlogp": rec.get("n_perturb", ""),
            })
    fixed = ["experiment", "model", "task", "task_family", "dataset", "dataset_file",
             "reported_in_paper", "weights"]
    cols = fixed + [c for c in dict.fromkeys(k for r in rows for k in r) if c not in fixed]
    with open(os.path.join(D, "data_inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)

    # ---------- 4. write markdown ---------------------------------------------------------
    L = []
    A = L.append
    A("# Data-source inventory\n")
    A("Generated by `eval/data_inventory.py` from the shipped files; no value is hardcoded.\n")

    A("\n## 1. Models\n")
    models = sorted({r["model"] for r in diag} | {m for m, _ in mech} |
                    {canon(r["model"]) for r in ladder})
    A("| model | internal codename | weights | diagnosed on | in origin ladder | mechanism experiments |")
    A("|---|---|---|---|---|---|")
    unmapped = []
    for m in models:
        dt = sorted({r["task"] for r in diag if r["model"] == m})
        dp = [t for t in dt if t in IN_PAPER]
        ex = sorted({EXP[k][0] for (mm, k) in mech if mm == m})
        w = weights(m)
        if not w:
            unmapped.append(m)
        A("| %s | %s | %s | %s | %s | %s |" % (
            m, "`%s`" % lab2int[m] if m in lab2int else "-", w or "**unmapped**",
            "%d tasks (%d reported)" % (len(dt), len(dp)) if dt else "-",
            next(("yes (%s)" % r["stage"] for r in ladder if canon(r["model"]) == m), "-"),
            ", ".join(ex) if ex else "-"))
    if unmapped:
        A("\n> No `HF` entry in `eval/attention_attribution.py` for: %s. "
          "Add one so every released model's weights are named in the code." %
          ", ".join("`%s`" % u for u in unmapped))

    A("\n## 2. Datasets and volume\n")
    A("`reported` marks the task set the paper analyses, taken from `GEN12` in "
      "`figures/make_nmi_figures.py`.\n")
    A("| dataset | source file | task | task family | n per model | models | reported |")
    A("|---|---|---|---|---|---|---|")
    for t in sorted(tasks, key=lambda x: (task_family(x), x)):
        ds, dsf = dataset_of(t)
        ns = sorted(tasks[t]["n"])
        A("| %s | `%s` | %s | %s | %s | %d | %s |" % (
            ds, dsf, t, task_family(t),
            str(ns[0]) if len(ns) == 1 else "%d-%d" % (ns[0], ns[-1]),
            len(tasks[t]["models"]), "yes" if t in IN_PAPER else "**no**"))
    A("\n> The MoleculeNet classification tasks were diagnosed but are **not** reported in the "
      "paper; they are listed here only because the shipped statistics table contains them.")

    if ladder:
        A("\n## 2b. Origin ladder\n")
        basis = ", ".join("`%s`" % t for t in ladder_basis) if ladder_basis else "unresolved"
        A("Every rung is scored on the same task basis: %s "
          "(derived here by checking that `n_resp` equals the sum of those tasks' `n`).\n" % basis)
        A("| stage | model | internal codename | weights | n responses | performance | ER | %ER=0 | claims/resp | per-claim fab. |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for r in ladder:
            A("| %s | %s | `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
                r["stage"], canon(r["model"]), lab2int.get(canon(r["model"]), r["model"]),
                weights(canon(r["model"])) or "-",
                r["n_resp"], fmt(num(r["perf"])), fmt(num(r["ER"])), fmt(num(r["pct_er0"])),
                fmt(num(r["claims_per_resp"])), fmt(num(r["perclaim_fab_rate"]), 4)))

    A("\n## 3. Diagnosis: performance and hallucination per model x task\n")
    A("| model | task | dataset | n | performance | ER | overall | %ER=0 | claim prec. | sem. entropy |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(diag, key=lambda r: (r["model"], task_family(r["task"]), r["task"])):
        ds, _ = dataset_of(r["task"])
        A("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["model"], r["task"], ds, r["n"], fmt(num(r["perf"])), fmt(num(r["ER"])),
            fmt(num(r["overall"])), fmt(num(r["pct_er0"])), fmt(num(r.get("cp"))),
            fmt(num(r.get("semantic_entropy")), 4)))

    A("\n## 3b. Complete metric matrix (every column of stats_per_model_task.csv)\n")
    A("| model | task | " + " | ".join(metric_cols) + " |")
    A("|" + "---|" * (len(metric_cols) + 2))
    for r in sorted(diag, key=lambda r: (r["model"], task_family(r["task"]), r["task"])):
        A("| %s | %s | %s |" % (r["model"], r["task"],
                                " | ".join((r[c] or "") for c in metric_cols)))

    A("\n## 4. Mechanism experiments: coverage and volume\n")
    A("| experiment | model | responses | tasks | extra |")
    A("|---|---|---|---|---|")
    for (label, kind), rec in sorted(mech.items(), key=lambda kv: (EXP[kv[0][1]][0], kv[0][0])):
        extra = []
        if kind == "drift":
            extra.append("%d conditions" % len(rec.get("conditions") or []))
            if rec.get("base_acc") is not None:
                extra.append("base acc %.3f" % rec["base_acc"])
        if kind == "condsent" and rec.get("n_samples"):
            extra.append("%s samples/response" % rec["n_samples"])
        if kind == "region":
            extra.append("matched-token n=%d" % rec.get("n_matched", 0))
            extra.append("Dlogp n=%d" % rec.get("n_perturb", 0))
        if kind == "gradattr" and rec.get("strata"):
            extra.append("strata " + json.dumps(rec["strata"]))
        A("| %s | %s | %s | %d | %s |" % (EXP[kind][0], label, rec.get("n"),
                                          len(rec["tasks"]), "; ".join(extra)))

    A("\n### Per-experiment task coverage\n")
    for (label, kind), rec in sorted(mech.items(), key=lambda kv: (EXP[kv[0][1]][0], kv[0][0])):
        A("- **%s / %s** (`%s`, internal `%s`): %s" % (
            EXP[kind][0], label, rec["file"], rec["internal"],
            ", ".join(rec["tasks"]) if rec["tasks"] else "(pooled, no per-task breakdown)"))

    ap = os.path.join(D, "attention_perturbation.csv")
    if os.path.exists(ap):
        A("\n## 5. Causal perturbation and matched-token attention (cap2mol, middle layer)\n")
        r0 = read_csv(ap)
        cols5 = [c for c in r0[0] if c != "model"]
        A("| model | " + " | ".join(cols5) + " |")
        A("|" + "---|" * (len(cols5) + 1))
        for r in r0:
            A("| %s | %s |" % (r["model"], " | ".join(r[c] for c in cols5)))

    A("")
    out = os.path.join(D, "DATA_INVENTORY.md")
    open(out, "w").write("\n".join(L))
    print("wrote %s (%d lines) and data/data_inventory.csv (%d rows)" % (out, len(L), len(rows)))


if __name__ == "__main__":
    main()
