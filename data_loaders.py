#!/usr/bin/env python3
"""
Unified data loaders for molecular reasoning tasks.

Every loader returns a list of dicts with this standard schema:
    {
        "id": str,           # unique sample identifier
        "input": str,        # text fed into the prompt template (via {input})
        "gt": str | float,   # ground truth answer
        "task": str,         # task name
        "metadata": dict,    # task-specific extras (e.g. S2 constraints)
    }

Tasks covered:
    cap2mol / mol2cap         (ChEBI-20)
    bace / bbbp / hiv / tox21 / clintox   (MoleculeNet classification)
    retrosynthesis            (USPTO-50K)
    s2_MolCustom_AtomNum / BondNum / FunctionalGroup
    s2_MolEdit_AddComponent / DelComponent / SubComponent
    s2_MolOpt_LogP / MR / QED
"""

import csv
import json
import os
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ----------------------------------------------------------------------------
# Path map
# ----------------------------------------------------------------------------

PATHS = {
    "chebi20_test":   os.path.join(DATA_DIR, "chebi-20/test.txt"),
    "chebi20_train":  os.path.join(DATA_DIR, "chebi-20/train.txt"),
    "chebi20_val":    os.path.join(DATA_DIR, "chebi-20/validation.txt"),
    "moleculenet":    os.path.join(DATA_DIR, "moleculenet"),
    "uspto50k":       os.path.join(DATA_DIR, "uspto50k/USPTO_50K.csv"),
    "s2bench":        os.path.join(DATA_DIR, "s2-bench"),
}


# ----------------------------------------------------------------------------
# ChEBI-20: cap2mol & mol2cap
# ----------------------------------------------------------------------------

def _read_chebi20(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) >= 3:
                rows.append({"cid": parts[0], "smiles": parts[1], "description": parts[2]})
    return rows


def load_chebi20_cap2mol(split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    path = PATHS[f"chebi20_{split}"]
    rows = _read_chebi20(path)
    data = [
        {
            "id": r["cid"],
            "input": r["description"],
            "gt": r["smiles"],
            "task": "cap2mol",
            "metadata": {},
        }
        for r in rows
    ]
    return data[:max_samples] if max_samples else data


def load_chebi20_mol2cap(split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    path = PATHS[f"chebi20_{split}"]
    rows = _read_chebi20(path)
    data = [
        {
            "id": r["cid"],
            "input": r["smiles"],
            "gt": r["description"],
            "task": "mol2cap",
            "metadata": {},
        }
        for r in rows
    ]
    return data[:max_samples] if max_samples else data


# ----------------------------------------------------------------------------
# MoleculeNet classification
# ----------------------------------------------------------------------------
# Data layout (downloaded from scikit-fingerprints/MoleculeNet_*):
#   data/moleculenet/<name>/<name>.csv               (SMILES + label columns)
#   data/moleculenet/<name>/ogb_splits_<name>.json   (train/valid/test indices)
# For tox21 we aggregate the 12 sub-task columns into a single any-positive label
# (matching the prompt "predict whether it is toxic"). Use load_tox21_per_task
# for the per-endpoint version.
# ----------------------------------------------------------------------------

MOLNET_LABEL_COLS = {
    "bace":    "label",
    "bbbp":    "label",
    "hiv":     "label",
    "tox21":   None,        # multi-label, see _aggregate_tox21
    "clintox": "CT_TOX",
}


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ogb_split(name: str, split: str) -> List[int]:
    path = os.path.join(PATHS["moleculenet"], name, f"ogb_splits_{name}.json")
    with open(path) as f:
        s = json.load(f)
    if split == "val":
        split = "valid"
    return s[split]


def _aggregate_tox21(row: Dict[str, str]) -> Optional[str]:
    tox_cols = [c for c in row.keys() if c not in ("SMILES", "smiles")]
    vals = [row.get(c, "") for c in tox_cols]
    has_label = any(v not in ("", None) for v in vals)
    if not has_label:
        return None
    any_pos = any(v in ("1", "1.0") for v in vals)
    return "Yes" if any_pos else "No"


def _aggregate_clintox(row: Dict[str, str]) -> Optional[str]:
    # ClinTox: CT_TOX = 1 means clinical toxicity (positive). We follow the
    # standard MoleculeNet convention.
    v = row.get("CT_TOX", "")
    if v in ("", None):
        return None
    return "Yes" if v in ("1", "1.0") else "No"


def load_moleculenet(
    task: str,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generic MoleculeNet loader returning Yes/No labels for the task prompt."""
    if task not in MOLNET_LABEL_COLS:
        raise ValueError(f"Unknown MoleculeNet task: {task}")
    csv_path = os.path.join(PATHS["moleculenet"], task, f"{task}.csv")
    rows = _read_csv_rows(csv_path)
    indices = _ogb_split(task, split)

    data: List[Dict[str, Any]] = []
    for idx in indices:
        if idx >= len(rows):
            continue
        row = rows[idx]
        smiles = row.get("SMILES") or row.get("smiles") or ""
        if not smiles:
            continue

        if task == "tox21":
            gt = _aggregate_tox21(row)
        elif task == "clintox":
            gt = _aggregate_clintox(row)
        else:
            col = MOLNET_LABEL_COLS[task]
            v = row.get(col, "")
            gt = "Yes" if v in ("1", "1.0") else ("No" if v in ("0", "0.0") else None)

        if gt is None:
            continue

        data.append({
            "id": f"{task}_{split}_{idx}",
            "input": smiles,
            "gt": gt,
            "task": task,
            "metadata": {"row_index": idx},
        })

    return data[:max_samples] if max_samples else data


# ----------------------------------------------------------------------------
# USPTO-50k retrosynthesis
# ----------------------------------------------------------------------------
# CSV columns: id, class, reactions (reactants>>product)
# We expose retrosynthesis test rows. Standard splits aren't bundled here,
# so by default we use the last 5000 rows as a held-out test (deterministic).
# ----------------------------------------------------------------------------

USPTO50K_DEFAULT_TEST_SIZE = 5007  # standard test split size for USPTO-50K


def load_uspto50k(
    split: str = "test",
    max_samples: Optional[int] = None,
    test_size: int = USPTO50K_DEFAULT_TEST_SIZE,
) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(PATHS["uspto50k"])
    n = len(rows)
    # Deterministic split: last `test_size` rows are test, before that valid,
    # and the rest is train. Matches the conventional retrosynthesis layout.
    test_start = max(0, n - test_size)
    val_start = max(0, test_start - test_size)
    if split == "test":
        subset = rows[test_start:]
    elif split in ("valid", "val"):
        subset = rows[val_start:test_start]
    elif split == "train":
        subset = rows[:val_start]
    else:
        raise ValueError(f"Unknown split: {split}")

    data: List[Dict[str, Any]] = []
    for i, r in enumerate(subset):
        rxn = r.get("reactions", "")
        if ">>" not in rxn:
            continue
        reactants, product = rxn.split(">>", 1)
        data.append({
            "id": f"uspto_{split}_{i}",
            "input": product.strip(),
            "gt": reactants.strip(),
            "task": "retrosynthesis",
            "metadata": {
                "rxn_class": r.get("class", ""),
                "rxn_id": r.get("id", ""),
            },
        })
    return data[:max_samples] if max_samples else data


def load_uspto50k_forward(
    split: str = "test",
    max_samples: Optional[int] = None,
    test_size: int = USPTO50K_DEFAULT_TEST_SIZE,
) -> List[Dict[str, Any]]:
    """Forward reaction prediction from the same USPTO-50K rows: given the
    reactants, predict the product (the reverse direction of retrosynthesis)."""
    rows = _read_csv_rows(PATHS["uspto50k"])
    n = len(rows)
    test_start = max(0, n - test_size)
    val_start = max(0, test_start - test_size)
    if split == "test":
        subset = rows[test_start:]
    elif split in ("valid", "val"):
        subset = rows[val_start:test_start]
    elif split == "train":
        subset = rows[:val_start]
    else:
        raise ValueError(f"Unknown split: {split}")

    data: List[Dict[str, Any]] = []
    for i, r in enumerate(subset):
        rxn = r.get("reactions", "")
        if ">>" not in rxn:
            continue
        reactants, product = rxn.split(">>", 1)
        data.append({
            "id": f"rxnpred_{split}_{i}",
            "input": reactants.strip(),
            "gt": product.strip(),
            "task": "reaction_prediction",
            "metadata": {
                "rxn_class": r.get("class", ""),
                "rxn_id": r.get("id", ""),
            },
        })
    return data[:max_samples] if max_samples else data


# ----------------------------------------------------------------------------
# S2-TOMG-Bench
# ----------------------------------------------------------------------------
# 3 task groups × 3 subtasks each:
#   MolCustom: AtomNum, BondNum, FunctionalGroup (de-novo generation under
#              composition constraints; no GT molecule, only constraint vector)
#   MolEdit:   AddComponent, DelComponent, SubComponent (edit a given molecule)
#   MolOpt:    LogP, MR, QED (modify to increase/decrease a property)
#
# For S2-bench there is no single-correct-answer ground truth -- evaluation is
# constraint-satisfaction style. We expose the constraint metadata so callers
# can verify with the existing S2-TOMG-Bench evaluate.py or our own scorer.
# ----------------------------------------------------------------------------

S2_FILE_MAP = {
    "s2_MolCustom_AtomNum":          "MolCustom_AtomNum.csv",
    "s2_MolCustom_BondNum":          "MolCustom_BondNum.csv",
    "s2_MolCustom_FunctionalGroup":  "MolCustom_FunctionalGroup.csv",
    "s2_MolEdit_AddComponent":       "MolEdit_AddComponent.csv",
    "s2_MolEdit_DelComponent":       "MolEdit_DelComponent.csv",
    "s2_MolEdit_SubComponent":       "MolEdit_SubComponent.csv",
    "s2_MolOpt_LogP":                "MolOpt_LogP.csv",
    "s2_MolOpt_MR":                  "MolOpt_MR.csv",
    "s2_MolOpt_QED":                 "MolOpt_QED.csv",
}


def _s2_extract_metadata(task: str, row: Dict[str, str]) -> Dict[str, Any]:
    """Pull out the numeric constraint / source molecule from an S2 row."""
    meta: Dict[str, Any] = {}
    if task.startswith("s2_MolCustom_"):
        meta["constraints"] = {
            k: v for k, v in row.items()
            if k != "Instruction" and v not in ("", None)
        }
    elif task.startswith("s2_MolEdit_"):
        meta["source_molecule"] = row.get("molecule", "")
        if "added_group" in row:
            meta["added_group"] = row.get("added_group", "")
        if "deleted_group" in row:
            meta["deleted_group"] = row.get("deleted_group", "")
        if "subed_group" in row:
            meta["subed_group"] = row.get("subed_group", "")
    elif task.startswith("s2_MolOpt_"):
        meta["source_molecule"] = row.get("molecule", "")
        prop = task.split("_", 2)[-1]
        meta["target_property"] = prop
        if prop in row:
            try:
                meta["source_value"] = float(row[prop])
            except (TypeError, ValueError):
                meta["source_value"] = row.get(prop, "")
    return meta


def load_s2bench(
    task: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if task not in S2_FILE_MAP:
        raise ValueError(f"Unknown S2-bench task: {task}")
    path = os.path.join(PATHS["s2bench"], S2_FILE_MAP[task])
    rows = _read_csv_rows(path)

    data: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        instr = r.get("Instruction", "").strip()
        if not instr:
            continue
        meta = _s2_extract_metadata(task, r)
        sid = r.get("index") or str(i)
        # gt = source molecule for edit/opt tasks (we expect output != source);
        # for MolCustom there is no GT, only constraints.
        gt = meta.get("source_molecule", "")
        data.append({
            "id": f"{task}_{sid}",
            "input": instr,
            "gt": gt,
            "task": task,
            "metadata": meta,
        })
    return data[:max_samples] if max_samples else data


# ----------------------------------------------------------------------------
# Unified dispatch
# ----------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cap2mol":         {"loader": load_chebi20_cap2mol,   "se_kind": "cap2mol",        "family": "chebi20"},
    "mol2cap":         {"loader": load_chebi20_mol2cap,   "se_kind": "mol2cap",        "family": "chebi20"},
    "bace":            {"loader": lambda **kw: load_moleculenet("bace", **kw),    "se_kind": "classification", "family": "moleculenet"},
    "bbbp":            {"loader": lambda **kw: load_moleculenet("bbbp", **kw),    "se_kind": "classification", "family": "moleculenet"},
    "hiv":             {"loader": lambda **kw: load_moleculenet("hiv", **kw),     "se_kind": "classification", "family": "moleculenet"},
    "tox21":           {"loader": lambda **kw: load_moleculenet("tox21", **kw),   "se_kind": "classification", "family": "moleculenet"},
    "clintox":         {"loader": lambda **kw: load_moleculenet("clintox", **kw), "se_kind": "classification", "family": "moleculenet"},
    "retrosynthesis":  {"loader": load_uspto50k,           "se_kind": "retrosynthesis", "family": "uspto50k"},
    "reaction_prediction": {"loader": load_uspto50k_forward, "se_kind": "cap2mol", "family": "uspto50k"},
}
for _t in S2_FILE_MAP:
    TASK_REGISTRY[_t] = {
        "loader": (lambda t=_t, **kw: load_s2bench(t, **kw)),
        "se_kind": "cap2mol",   # SMILES output; cluster by Tanimoto
        "family":  "s2bench",
    }


def load_task(task: str, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Top-level dispatch. `split` is honored only by ChEBI-20 / MoleculeNet / USPTO-50K."""
    spec = TASK_REGISTRY.get(task)
    if spec is None:
        raise ValueError(f"Unknown task: {task}. Known: {sorted(TASK_REGISTRY)}")
    loader = spec["loader"]
    family = spec["family"]
    kwargs: Dict[str, Any] = {"max_samples": max_samples}
    if family in ("chebi20", "moleculenet", "uspto50k"):
        kwargs["split"] = split
    return loader(**kwargs)


ALL_TASKS: List[str] = list(TASK_REGISTRY.keys())


if __name__ == "__main__":
    for t in ALL_TASKS:
        try:
            d = load_task(t, max_samples=2)
            print(f"{t:35s} -> {len(d)} samples  e.g. id={d[0]['id'] if d else '-'}")
        except Exception as e:
            print(f"{t:35s} -> ERROR {type(e).__name__}: {e}")
