#!/usr/bin/env python3
"""
OpenMolIns (TOMG-shaped, 45k rows, 9 subtasks) -> EasyR1 parquet for s2 GRPO.

OpenMolIns train.csv has only [SubTask, Instruction, molecule]; the numeric
constraints / source molecule live in the Instruction PROSE. We parse them into
the structured metadata that diagnose_s2's EO dimension expects (key names must
match diagnose_multitask: atoms full-name 'carbon', bonds 'single', FGs
'benzene_ring' etc.), then serialise {task, metadata, gt} into the `answer`
column as JSON. The s2 reward (chem_s2_process_ours.py) decodes it and calls
diagnose_one for the full 2x2.

Columns: problem (system + s2 prompt), answer (JSON), task (s2_* diagnose name).
"""
import argparse
import csv
import json
import os
import re
import sys

import pandas as pd

_MAIN = os.environ.get("MOLLM_PROJECT_DIR", ".")
sys.path.insert(0, _MAIN)
from prompts import PROMPTS, SYSTEM_PROMPT  # noqa: E402

SUBTASK_TO_TASK = {
    "AtomNum": "s2_MolCustom_AtomNum", "BondNum": "s2_MolCustom_BondNum",
    "FunctionalGroup": "s2_MolCustom_FunctionalGroup",
    "AddComponent": "s2_MolEdit_AddComponent", "DelComponent": "s2_MolEdit_DelComponent",
    "SubComponent": "s2_MolEdit_SubComponent",
    "LogP": "s2_MolOpt_LogP", "MR": "s2_MolOpt_MR", "QED": "s2_MolOpt_QED",
}

_ATOM_KEYS = {"carbon", "oxygen", "nitrogen", "sulfur", "fluorine", "chlorine",
              "bromine", "iodine", "phosphorus", "boron", "silicon"}
_BOND_KEYS = {"single", "double", "triple", "rotatable", "aromatic"}
# instruction FG phrase -> diagnose_s2 FG key
_FG_MAP = {
    "benzene ring": "benzene_ring", "hydroxyl": "hydroxyl", "anhydride": "anhydride",
    "aldehyde": "aldehyde", "ketone": "ketone", "carboxyl": "carboxyl",
    "ester": "ester", "amide": "amide", "amine": "amine", "nitro": "nitro",
    "halo": "halo", "thioether": "thioether", "nitrile": "nitrile",
}

_MOL_RE = re.compile(r"molecule\s+([^\s.,]+(?:\([^\s]*\))?[^\s.,]*)")
_NUM_WORD = re.compile(r"(\d+)\s+([a-zA-Z][a-zA-Z ]*?)\s+(atoms?|bonds?|group)")


def _extract_source(instruction):
    m = _MOL_RE.search(instruction)
    return m.group(1).rstrip(".,") if m else ""


def parse_metadata(subtask, instruction, molecule):
    """Return diagnose_s2-compatible metadata for one row."""
    if subtask in ("AtomNum", "BondNum", "FunctionalGroup"):
        constraints = {}
        for num, name, kind in _NUM_WORD.findall(instruction):
            name = name.strip().lower()
            if kind.startswith("atom") and name in _ATOM_KEYS:
                constraints[name] = num
            elif kind.startswith("bond") and name in _BOND_KEYS:
                constraints[name] = num
            elif kind == "group":
                key = _FG_MAP.get(name) or _FG_MAP.get(name.replace(" group", ""))
                if key:
                    constraints[key] = num
        return {"constraints": constraints}
    # MolEdit / MolOpt: source molecule embedded in prose
    meta = {"source_molecule": _extract_source(instruction) or molecule}
    if subtask in ("LogP", "MR", "QED"):
        meta["target_property"] = subtask
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/openmolins/train.csv")
    ap.add_argument("--out_dir", default=os.path.join(_MAIN, "EasyR1-main/data/openmolins_s2"))
    ap.add_argument("--val_frac", type=int, default=50, help="hold out 1/val_frac per subtask for val")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    train, val = [], []
    per_sub = {}
    for r in rows:
        sub = r["SubTask"]
        task = SUBTASK_TO_TASK.get(sub)
        if task is None:
            continue
        instr = r["Instruction"]
        meta = parse_metadata(sub, instr, r.get("molecule", ""))
        prompt = SYSTEM_PROMPT + "\n\n" + PROMPTS[task].replace("{input}", instr)
        # gt: source molecule for edit/opt; empty for MolCustom (constraint-only)
        gt = meta.get("source_molecule", "")
        answer = json.dumps({"task": task, "metadata": meta, "gt": gt})
        rec = {"problem": prompt, "answer": answer, "task": task}
        per_sub[sub] = per_sub.get(sub, 0) + 1
        (val if per_sub[sub] % args.val_frac == 0 else train).append(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(train).to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    pd.DataFrame(val).to_parquet(os.path.join(args.out_dir, "test.parquet"), index=False)
    print(f"train={len(train)} val={len(val)} -> {args.out_dir}")
    # sanity: show one parsed metadata per task family
    import collections
    seen = collections.OrderedDict()
    for rec in train:
        a = json.loads(rec["answer"])
        if a["task"] not in seen:
            seen[a["task"]] = a["metadata"]
    for t, m in seen.items():
        print(f"  {t}: {m}")


if __name__ == "__main__":
    main()
