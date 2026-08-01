#!/usr/bin/env python3
"""
Generate EasyR1 parquet for ANY task, reusing data_loaders + prompts so the
prompt and ground-truth match the eval pipeline exactly.

Columns: problem (system + rendered prompt), answer (gt), task (diagnose name).

Usage:
  python make_task_parquet.py --task mol2cap --max_train 100000 --max_val 500
"""
import argparse
import os
import sys

import pandas as pd

_MAIN = os.environ.get("MOLLM_PROJECT_DIR", ".")
sys.path.insert(0, _MAIN)
from data_loaders import load_task          # noqa: E402
from prompts import PROMPTS, SYSTEM_PROMPT   # noqa: E402


def build(task, split, max_n):
    data = load_task(task, split=split, max_samples=max_n)
    tmpl = PROMPTS[task]
    rows = []
    for d in data:
        prompt = SYSTEM_PROMPT + "\n\n" + tmpl.replace("{input}", str(d["input"]))
        rows.append({"problem": prompt, "answer": str(d["gt"]), "task": task})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_train", type=int, default=100000)
    ap.add_argument("--max_val", type=int, default=500)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(_MAIN, "EasyR1-main/data", args.task)
    os.makedirs(out_dir, exist_ok=True)

    # train split (fall back to test if a task has no train split)
    try:
        tr = build(args.task, "train", args.max_train)
    except Exception as e:
        print(f"[warn] no train split for {args.task} ({e}); using test for both")
        tr = build(args.task, "test", args.max_train)
    te = build(args.task, "test", args.max_val)

    tr.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    te.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    print(f"{args.task}: train={len(tr)} test={len(te)} -> {out_dir}")
    print("  sample problem[:120]:", repr(tr.iloc[0]['problem'][:120]))
    print("  sample answer:", repr(tr.iloc[0]['answer'][:80]))


if __name__ == "__main__":
    main()
