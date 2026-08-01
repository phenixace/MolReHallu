#!/usr/bin/env python3
"""
Convert ChEBI-20 cap2mol splits into EasyR1 (veRL) parquet format.

Output columns (see verl/utils/dataset.py + examples/config.yaml):
  problem : the rendered cap2mol prompt (becomes the user message)
  answer  : ground-truth SMILES (the reward's `ground_truth`)
  task    : "Molecule_Design"  (cap2mol == description -> SMILES)

We render the SAME cap2mol prompt the eval pipeline uses (prompts.PROMPTS),
prefixed with the system line so train- and eval-time prompts match. EasyR1's
dataset only builds a user message, so the system text is folded into `problem`.

Usage:
  python make_chebi_parquet.py \
      --chebi_dir $MOLLM_PROJECT_DIR/data/chebi-20 \
      --out_dir   $MOLLM_PROJECT_DIR/EasyR1-main/data/chebi20 \
      --max_train 5000
"""

import argparse
import os
import sys

import pandas as pd

_MAIN = os.environ.get(
    "MOLLM_PROJECT_DIR", "."
)
sys.path.insert(0, _MAIN)
from prompts import PROMPTS, SYSTEM_PROMPT  # noqa: E402

CAP2MOL = PROMPTS["cap2mol"]


def load_chebi(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t", 2)
            if len(parts) >= 3:
                rows.append({"cid": parts[0], "smiles": parts[1], "desc": parts[2]})
    return rows


def to_parquet(rows, out_path, max_n=None):
    if max_n:
        rows = rows[:max_n]
    records = []
    for r in rows:
        prompt = SYSTEM_PROMPT + "\n\n" + CAP2MOL.replace("{input}", r["desc"])
        records.append(
            {"problem": prompt, "answer": r["smiles"], "task": "Molecule_Design"}
        )
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  wrote {len(df):>6} rows -> {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chebi_dir", default=os.path.join(_MAIN, "data/chebi-20"))
    ap.add_argument("--out_dir", default=os.path.join(_MAIN, "EasyR1-main/data/chebi20"))
    ap.add_argument("--max_train", type=int, default=5000)
    ap.add_argument("--max_val", type=int, default=500)
    args = ap.parse_args()

    train = load_chebi(os.path.join(args.chebi_dir, "train.txt"))
    val = load_chebi(os.path.join(args.chebi_dir, "validation.txt"))
    test = load_chebi(os.path.join(args.chebi_dir, "test.txt"))
    print(f"Loaded: train={len(train)} val={len(val)} test={len(test)}")

    to_parquet(train, os.path.join(args.out_dir, "train.parquet"), args.max_train)
    # Use the test split for periodic validation (val_files); cap for speed.
    to_parquet(test, os.path.join(args.out_dir, "test.parquet"), args.max_val)

    # Sanity print one record
    df = pd.read_parquet(os.path.join(args.out_dir, "train.parquet"))
    print("\nSample record:")
    print("  problem[:160]:", repr(df.iloc[0]["problem"][:160]))
    print("  answer       :", df.iloc[0]["answer"])
    print("  task         :", df.iloc[0]["task"])


if __name__ == "__main__":
    main()
