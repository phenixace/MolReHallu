#!/usr/bin/env python3
"""
Multi-task Semantic Entropy + hallucination-ready sampling.

For each (model, task) pair this script:
  1. Generates N completions per sample with the task-specific prompt
  2. Writes `completions.json` for offline SE / dual-entropy analysis
  3. Writes `output.json` (first completion per sample) so the same run can be
     fed straight into diagnose_hallucination.py (no extra inference needed)
  4. Computes semantic-entropy summary + per-sample details

Inputs come from data_loaders.TASK_REGISTRY and prompts.PROMPTS — adding a new
task means editing those two files only.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_loaders import ALL_TASKS, TASK_REGISTRY, load_task
from prompts import SYSTEM_PROMPT, build_messages
from semantic_entropy import (
    evaluate_semantic_entropy,
    sample_completions,
)


def _write_outputs_json(samples: List[Dict[str, Any]], path: str, model_name: str, task: str) -> None:
    """Convert completions[] into the single-answer schema diagnose_hallucination.py expects."""
    out = []
    for s in samples:
        comps = s.get("completions", [])
        out.append({
            "id": s["id"],
            "question": s["input"],
            "gt": s.get("gt", ""),
            "answer": comps[0] if comps else "",
            "model": model_name,
            "task": task,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def run_task(task_name: str, backend, args) -> Dict[str, Any]:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")
    spec = TASK_REGISTRY[task_name]
    se_kind = spec["se_kind"]

    print(f"\n{'='*60}")
    print(f"  Task: {task_name}   SE-kind: {se_kind}")
    print(f"{'='*60}")

    data = load_task(task_name, split="test", max_samples=args.max_samples)
    print(f"  Loaded {len(data)} samples")

    task_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    comp_path = os.path.join(task_dir, "completions.json")
    out_path = os.path.join(task_dir, "output.json")

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(comp_path):
        with open(comp_path) as f:
            for item in json.load(f):
                existing[str(item["id"])] = item
        print(f"  Resuming: {len(existing)} already done")

    all_samples: List[Dict[str, Any]] = list(existing.values())
    written = 0

    for item in tqdm(data, desc=f"Sampling [{task_name}]"):
        sid = str(item["id"])
        if sid in existing:
            continue

        messages = build_messages(task_name, item["input"])
        completions = sample_completions(
            backend, messages,
            n_samples=args.n_samples,
            temperature=args.temperature,
        )

        entry = {
            "id": sid,
            "input": item["input"],
            "gt": item.get("gt", ""),
            "completions": completions,
            "task": task_name,
            "metadata": item.get("metadata", {}),
        }
        all_samples.append(entry)
        existing[sid] = entry
        written += 1

        if written % 10 == 0:
            with open(comp_path, "w") as f:
                json.dump(all_samples, f, ensure_ascii=False, indent=2)

    with open(comp_path, "w") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"  Completions saved: {comp_path}")

    # First-completion output.json for hallucination diagnosis
    _write_outputs_json(all_samples, out_path, args.model_name, task_name)
    print(f"  Diagnosis-ready output saved: {out_path}")

    print(f"  Computing semantic entropy ...")
    results = evaluate_semantic_entropy(all_samples, task=se_kind)

    summary_path = os.path.join(task_dir, "se_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results["summary"], f, ensure_ascii=False, indent=2)

    details_path = os.path.join(task_dir, "se_details.jsonl")
    with open(details_path, "w") as f:
        for entry in results["per_sample"]:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    s = results["summary"]
    se = s.get("semantic_entropy", {})
    print(f"  SE mean={se.get('mean','N/A')} "
          f"median={se.get('median','N/A')} "
          f"n={s.get('n_samples', 0)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task Semantic Entropy + completions for molecular LLMs"
    )
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="unknown")
    parser.add_argument(
        "--tasks", nargs="+", default=["cap2mol"],
        choices=ALL_TASKS + ["all"],
        help="Tasks to run. Use 'all' for every registered task.",
    )
    parser.add_argument("--backend", choices=["hf", "vllm", "api"],
                        default="vllm")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per task (for testing)")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device(s) — overrides CUDA_VISIBLE_DEVICES")

    args = parser.parse_args()

    if "all" in args.tasks:
        args.tasks = list(ALL_TASKS)

    if args.gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"Initializing {args.backend} backend: {args.model_id}")
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        llm = LLM(
            model=args.model_id,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

        # Some fine-tunes (e.g. Chem-R) ship without chat_template. Detect the
        # base architecture from EOS token and patch in a sensible template.
        if not getattr(tokenizer, "chat_template", None):
            print(f"[WARN] {args.model_id} has no chat_template; applying fallback")
            base_fallbacks = []
            eos = getattr(tokenizer, "eos_token", "") or ""
            if "eot_id" in eos:
                base_fallbacks.append("meta-llama/Llama-3.1-8B-Instruct")
            else:
                base_fallbacks.append("Qwen/Qwen2.5-7B-Instruct")
                base_fallbacks.append("mistralai/Mistral-7B-Instruct-v0.2")
            for base in base_fallbacks:
                try:
                    base_tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
                    if base_tok.chat_template:
                        tokenizer.chat_template = base_tok.chat_template
                        print(f"  borrowed chat_template from {base}")
                        break
                except Exception as e:
                    print(f"  fallback {base} failed: {e}")

        class VLLMWrapper:
            def __init__(self, llm, tokenizer, max_tokens):
                self.llm = llm
                self.tokenizer = tokenizer
                self.sampling_params = SamplingParams(max_tokens=max_tokens)
        backend = VLLMWrapper(llm, tokenizer, args.max_new_tokens)
    else:
        raise ValueError(f"Backend {args.backend} not supported in multitask script")

    print(f"Tasks to run: {args.tasks}")
    print(f"Samples per prompt: {args.n_samples}")
    print(f"Max samples per task: {args.max_samples or 'all'}")

    all_results: Dict[str, Any] = {}
    for task_name in args.tasks:
        try:
            results = run_task(task_name, backend, args)
            all_results[task_name] = results["summary"]
        except FileNotFoundError as e:
            print(f"  SKIPPED {task_name}: {e}")
        except Exception as e:
            print(f"  ERROR on {task_name}: {e}")
            import traceback
            traceback.print_exc()

    overview_path = os.path.join(args.output_dir, "overview.json")
    with open(overview_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nOverview saved to {overview_path}")


if __name__ == "__main__":
    main()
