#!/usr/bin/env python3
"""
Build the all-task joint-training parquet by mixing the per-task parquets,
balanced-sampled so the large tasks (retro/s2) don't drown the core ChEBI tasks.

Each task keeps its own prompt + answer format (s2 answer = JSON with metadata,
others = plain gt). The merged reward (chem_merged_ours.py) dispatches by the
`task` column, so no format unification is needed here.
"""
import argparse
import os
import sys

import pandas as pd

_MAIN = os.environ.get("MOLLM_PROJECT_DIR", ".")
BASE = os.path.join(_MAIN, "EasyR1-main", "data")

# (parquet dir, per-task train cap). None cap = take all.
SOURCES = [
    ("chebi20", 20000),          # cap2mol
    ("mol2cap", 20000),
    ("retrosynthesis", 20000),
    ("openmolins_s2", 20000),    # 9 s2 subtasks already mixed inside
    # ("bace", None),            # add once bace eval confirms it helps
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(BASE, "merged_4task"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # cap2mol parquet was written with task="Molecule_Design" (legacy EasyR1
    # naming); the merged reward dispatches via diagnose_one which expects
    # "cap2mol". Normalize here.
    TASK_RENAME = {"Molecule_Design": "cap2mol"}

    train_parts, test_parts = [], []
    for d, cap in SOURCES:
        tr = pd.read_parquet(os.path.join(BASE, d, "train.parquet"))
        te = pd.read_parquet(os.path.join(BASE, d, "test.parquet"))
        tr["task"] = tr["task"].replace(TASK_RENAME)
        te["task"] = te["task"].replace(TASK_RENAME)
        if cap and len(tr) > cap:
            tr = tr.sample(n=cap, random_state=args.seed)
        train_parts.append(tr)
        # cap val per source to keep periodic eval fast
        test_parts.append(te.sample(n=min(len(te), 200), random_state=args.seed))
        print(f"  {d}: train {len(tr)} | val {min(len(te),200)}")

    train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    test.to_parquet(os.path.join(args.out_dir, "test.parquet"), index=False)
    print(f"\nMERGED: train={len(train)} test={len(test)} -> {args.out_dir}")
    print("task distribution (train):")
    print(train["task"].value_counts().to_string())


if __name__ == "__main__":
    main()
