#!/usr/bin/env python3
"""
Semantic Entropy for Molecular LLM Hallucination Detection.

Measures model uncertainty by sampling multiple completions for the same prompt
and computing the entropy of their semantic clustering. High entropy indicates
the model is uncertain and likely hallucinating.

Supports multiple task types:
  - cap2mol: caption → SMILES (cluster by Tanimoto similarity)
  - mol2cap: SMILES → caption (cluster by text embedding similarity)
  - property: SMILES → numeric value (cluster by value proximity)

Reference:
  Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for
  Uncertainty Estimation in Natural Language Generation", ICLR 2023.
"""

import math
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ============================================================================
# Output Extraction Helpers
# ============================================================================

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# ether-0 native tokens (read directly; do NOT string-replace, since ether-0
# also mentions a stray '<answer>' in its reasoning text).
_ETHER_ANS_RE = re.compile(r"<\|answer_start\|>(.*?)(?:<\|answer_end\|>|\Z)", re.DOTALL)
_ETHER_THINK_RE = re.compile(r"<\|think_start\|>(.*?)(?:<\|think_end\|>|\Z)", re.DOTALL)


def _maybe_selfies_to_smiles(smi):
    """Convert SELFIES (e.g. MolReasoner's '[C][C][=O]...') to SMILES so
    downstream RDKit clustering works; pass SMILES through unchanged."""
    try:
        from diagnose_hallucination import _is_selfies, _selfies_to_smiles
        if _is_selfies(smi):
            conv = _selfies_to_smiles(smi)
            if conv:
                return conv
    except Exception:
        pass
    return smi


def extract_smiles(text: str) -> Optional[str]:
    if not text:
        return None
    m = _ETHER_ANS_RE.search(text)
    smi = m.group(1).strip() if m else None
    if smi is None:
        ms = _ANSWER_RE.findall(text)
        smi = ms[-1].strip() if ms else None
    if smi and 0 < len(smi) < 1000:
        return _maybe_selfies_to_smiles(smi)
    return None


def extract_reasoning(text: str) -> Optional[str]:
    if not text:
        return None
    m = _ETHER_THINK_RE.search(text) or _THINK_RE.search(text)
    return m.group(1).strip() if m else None


def _parse_mol(smi: str):
    if not HAS_RDKIT or not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _canonical(smi: str) -> Optional[str]:
    mol = _parse_mol(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


# ============================================================================
# Similarity Functions (per task type)
# ============================================================================

def tanimoto_similarity(smi1: str, smi2: str) -> float:
    """Morgan fingerprint Tanimoto between two SMILES. Returns 0.0 on failure."""
    if not HAS_RDKIT:
        return 0.0
    m1, m2 = _parse_mol(smi1), _parse_mol(smi2)
    if m1 is None or m2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def text_embedding_similarity(
    text1: str, text2: str, model=None
) -> float:
    """Cosine similarity between sentence embeddings.
    If no model is provided, falls back to token-overlap Jaccard."""
    if model is not None:
        emb = model.encode([text1, text2], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def numeric_similarity(val1: float, val2: float, tolerance: float = 0.1) -> float:
    """Similarity for numeric predictions (property tasks)."""
    diff = abs(val1 - val2)
    return max(0.0, 1.0 - diff / max(tolerance, 1e-9))


def classification_similarity(a: str, b: str) -> float:
    """Exact match for classification tasks (Yes/No, class labels)."""
    return 1.0 if a.strip().lower() == b.strip().lower() else 0.0


def reaction_similarity(smi1: str, smi2: str) -> float:
    """Similarity for reaction tasks (product/reactant SMILES).
    Handles multi-component SMILES separated by '.'."""
    parts1 = sorted(s.strip() for s in smi1.split(".") if s.strip())
    parts2 = sorted(s.strip() for s in smi2.split(".") if s.strip())
    can1 = [_canonical(s) or s for s in parts1]
    can2 = [_canonical(s) or s for s in parts2]
    if can1 == can2:
        return 1.0
    if not HAS_RDKIT:
        return 0.0
    sims = []
    for s1 in parts1:
        best = max((tanimoto_similarity(s1, s2) for s2 in parts2), default=0.0)
        sims.append(best)
    return sum(sims) / max(len(sims), 1)


# ============================================================================
# Greedy Semantic Clustering
# ============================================================================

def cluster_by_similarity(
    items: List[Any],
    sim_fn: Callable[[Any, Any], float],
    threshold: float = 0.85,
) -> List[List[int]]:
    """Greedy agglomerative clustering: assign each item to the first cluster
    whose centroid (first element) has similarity >= threshold, else new cluster.
    Returns list of clusters, each a list of indices."""
    clusters: List[List[int]] = []
    representatives: List[int] = []

    for i, item in enumerate(items):
        assigned = False
        for ci, rep_idx in enumerate(representatives):
            if sim_fn(items[rep_idx], item) >= threshold:
                clusters[ci].append(i)
                assigned = True
                break
        if not assigned:
            representatives.append(i)
            clusters.append([i])

    return clusters


# ============================================================================
# Entropy Computation
# ============================================================================

def entropy_from_clusters(clusters: List[List[int]], total: int) -> float:
    """Shannon entropy of the cluster distribution. Higher = more uncertain."""
    if total <= 0:
        return 0.0
    H = 0.0
    for c in clusters:
        p = len(c) / total
        if p > 0:
            H -= p * math.log(p)
    return H


def normalized_entropy(clusters: List[List[int]], total: int) -> float:
    """Entropy normalized to [0, 1]. 0 = all same cluster, 1 = all different."""
    H = entropy_from_clusters(clusters, total)
    max_H = math.log(total) if total > 1 else 1.0
    return H / max_H


# ============================================================================
# Task-Specific Extractors
# ============================================================================

TASK_EXTRACTORS: Dict[str, Callable[[str], Any]] = {
    "cap2mol": lambda text: extract_smiles(text),
    "mol2cap": lambda text: extract_reasoning(text) or text,
    "property": lambda text: _extract_numeric(text),
    "classification": lambda text: _extract_classification(text),
    "reaction": lambda text: extract_smiles(text),
    "retrosynthesis": lambda text: extract_smiles(text),
}


def _extract_numeric(text: str) -> Optional[float]:
    m = _ANSWER_RE.search(text)
    if m:
        try:
            return float(m.group(1).strip())
        except ValueError:
            pass
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return float(nums[-1]) if nums else None


def _extract_classification(text: str) -> Optional[str]:
    m = _ANSWER_RE.search(text)
    if m:
        ans = m.group(1).strip().lower()
        if ans in ("yes", "no"):
            return ans
        if "yes" in ans:
            return "yes"
        if "no" in ans:
            return "no"
        return ans
    return None


TASK_SIM_FNS: Dict[str, Callable] = {
    "cap2mol": tanimoto_similarity,
    "mol2cap": text_embedding_similarity,
    "property": numeric_similarity,
    "classification": classification_similarity,
    "reaction": reaction_similarity,
    "retrosynthesis": reaction_similarity,
}

TASK_THRESHOLDS: Dict[str, float] = {
    "cap2mol": 0.85,
    "mol2cap": 0.80,
    "property": 0.90,
    "classification": 0.99,
    "reaction": 0.85,
    "retrosynthesis": 0.85,
}


# ============================================================================
# Core: Compute Semantic Entropy for One Sample
# ============================================================================

def compute_semantic_entropy(
    completions: List[str],
    task: str = "cap2mol",
    threshold: Optional[float] = None,
    embedding_model=None,
) -> Dict[str, Any]:
    """Compute semantic entropy from N sampled completions for one prompt.

    Args:
        completions: list of N raw model outputs (each with <think>/<answer>)
        task: one of "cap2mol", "mol2cap", "property"
        threshold: similarity threshold for clustering (default per task)
        embedding_model: sentence-transformers model for mol2cap task

    Returns:
        dict with entropy scores and diagnostic info
    """
    extractor = TASK_EXTRACTORS.get(task, TASK_EXTRACTORS["cap2mol"])
    sim_fn = TASK_SIM_FNS.get(task, TASK_SIM_FNS["cap2mol"])
    if threshold is None:
        threshold = TASK_THRESHOLDS.get(task, 0.85)

    extracted = [extractor(c) for c in completions]
    valid_pairs = [(i, v) for i, v in enumerate(extracted) if v is not None]

    result: Dict[str, Any] = {
        "n_samples": len(completions),
        "n_valid": len(valid_pairs),
        "n_invalid": len(completions) - len(valid_pairs),
        "task": task,
        "threshold": threshold,
    }

    if len(valid_pairs) < 2:
        result.update({
            "semantic_entropy": 0.0,
            "normalized_entropy": 0.0,
            "n_clusters": len(valid_pairs),
            "cluster_sizes": [1] * len(valid_pairs),
            "note": "too few valid samples",
        })
        return result

    valid_items = [v for _, v in valid_pairs]

    if task == "mol2cap" and embedding_model is not None:
        _sim = lambda a, b: text_embedding_similarity(a, b, embedding_model)
    else:
        _sim = sim_fn

    clusters = cluster_by_similarity(valid_items, _sim, threshold)
    n = len(valid_items)

    se = entropy_from_clusters(clusters, n)
    nse = normalized_entropy(clusters, n)

    result.update({
        "semantic_entropy": round(se, 4),
        "normalized_entropy": round(nse, 4),
        "n_clusters": len(clusters),
        "cluster_sizes": [len(c) for c in clusters],
    })

    if task == "cap2mol":
        result["unique_canonical"] = len(set(
            _canonical(s) or s for s in valid_items
        ))
        if HAS_RDKIT:
            valid_mols = [s for s in valid_items if _parse_mol(s) is not None]
            result["validity_rate"] = (
                round(len(valid_mols) / len(valid_items), 4)
                if valid_items else 0.0
            )

    return result


# ============================================================================
# Dual-Layer Entropy: Output + Reasoning
# ============================================================================

def compute_dual_entropy(
    completions: List[str],
    task: str = "cap2mol",
    threshold: Optional[float] = None,
    reasoning_threshold: float = 0.75,
    embedding_model=None,
) -> Dict[str, Any]:
    """Compute both output-level and reasoning-level semantic entropy.

    Captures two orthogonal uncertainty signals:
    - Output entropy: are the final answers semantically diverse?
    - Reasoning entropy: are the reasoning paths diverse?

    Low output entropy + high reasoning entropy suggests the model
    arrives at the same answer via inconsistent reasoning (IR signal).
    """
    output_se = compute_semantic_entropy(
        completions, task=task, threshold=threshold,
        embedding_model=embedding_model,
    )

    reasonings = [extract_reasoning(c) or "" for c in completions]
    valid_reasonings = [r for r in reasonings if len(r.strip()) > 20]

    if len(valid_reasonings) < 2:
        reasoning_se = {
            "semantic_entropy": 0.0,
            "normalized_entropy": 0.0,
            "n_clusters": len(valid_reasonings),
            "note": "too few valid reasonings",
        }
    else:
        if embedding_model is not None:
            _sim = lambda a, b: text_embedding_similarity(
                a, b, embedding_model
            )
        else:
            _sim = text_embedding_similarity

        r_clusters = cluster_by_similarity(
            valid_reasonings, _sim, reasoning_threshold
        )
        n = len(valid_reasonings)
        reasoning_se = {
            "semantic_entropy": round(
                entropy_from_clusters(r_clusters, n), 4
            ),
            "normalized_entropy": round(
                normalized_entropy(r_clusters, n), 4
            ),
            "n_clusters": len(r_clusters),
            "cluster_sizes": [len(c) for c in r_clusters],
        }

    return {
        "output": output_se,
        "reasoning": reasoning_se,
        "entropy_gap": round(
            reasoning_se["normalized_entropy"]
            - output_se["normalized_entropy"], 4
        ),
    }


# ============================================================================
# Multi-Sample Generation (wraps existing inference backends)
# ============================================================================

def sample_completions(
    backend,
    messages: List[Dict],
    n_samples: int = 10,
    temperature: float = 0.8,
) -> List[str]:
    """Generate n_samples completions from a backend for the same prompt.

    Works with HFBackend, VLLMBackend, or APIBackend from inference.py.
    Temporarily overrides temperature for diversity.
    """
    completions = []

    if hasattr(backend, "llm"):
        from vllm import SamplingParams
        prompt = backend.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=getattr(backend, "sampling_params", None)
            and backend.sampling_params.max_tokens or 4096,
            n=n_samples,
        )
        outputs = backend.llm.generate([prompt], params)
        completions = [o.text for o in outputs[0].outputs]
    else:
        for _ in range(n_samples):
            completions.append(backend.generate(messages))

    return completions


# ============================================================================
# Batch Evaluation: Compute SE for a Dataset
# ============================================================================

def evaluate_semantic_entropy(
    samples: List[Dict[str, Any]],
    task: str = "cap2mol",
    threshold: Optional[float] = None,
    embedding_model=None,
) -> Dict[str, Any]:
    """Evaluate semantic entropy over a list of samples.

    Each sample dict must have:
      - "completions": List[str] of N sampled outputs
      - "id" or "cid": sample identifier
    Optionally:
      - "gt" or "gt_smiles": ground truth for correlation analysis

    Returns aggregate statistics and per-sample results.
    """
    per_sample = []
    all_se = []
    all_nse = []

    for sample in samples:
        completions = sample.get("completions", [])
        if len(completions) < 2:
            continue

        se_result = compute_dual_entropy(
            completions, task=task, threshold=threshold,
            embedding_model=embedding_model,
        )

        sid = str(sample.get("id", sample.get("cid", "")))
        entry = {"id": sid, **se_result}

        gt = sample.get("gt", sample.get("gt_smiles"))
        if gt and task == "cap2mol":
            smiles_list = [extract_smiles(c) for c in completions]
            valid_smiles = [s for s in smiles_list if s]
            if valid_smiles:
                gt_sims = [tanimoto_similarity(gt, s) for s in valid_smiles]
                entry["gt_max_tanimoto"] = round(max(gt_sims), 4)
                entry["gt_mean_tanimoto"] = round(
                    sum(gt_sims) / len(gt_sims), 4
                )
                exact = any(
                    _canonical(s) == _canonical(gt) for s in valid_smiles
                )
                entry["gt_exact_match_in_samples"] = exact

        per_sample.append(entry)
        all_se.append(se_result["output"]["semantic_entropy"])
        all_nse.append(se_result["output"]["normalized_entropy"])

    summary = _compute_summary(all_se, all_nse, per_sample, task)
    return {"summary": summary, "per_sample": per_sample}


def _compute_summary(
    all_se: List[float],
    all_nse: List[float],
    per_sample: List[Dict],
    task: str,
) -> Dict[str, Any]:
    if not all_se:
        return {"n_samples": 0}

    se_arr = np.array(all_se)
    nse_arr = np.array(all_nse)

    summary: Dict[str, Any] = {
        "n_samples": len(all_se),
        "task": task,
        "semantic_entropy": {
            "mean": round(float(se_arr.mean()), 4),
            "std": round(float(se_arr.std()), 4),
            "median": round(float(np.median(se_arr)), 4),
            "min": round(float(se_arr.min()), 4),
            "max": round(float(se_arr.max()), 4),
        },
        "normalized_entropy": {
            "mean": round(float(nse_arr.mean()), 4),
            "std": round(float(nse_arr.std()), 4),
            "median": round(float(np.median(nse_arr)), 4),
        },
    }

    bins = {"low": 0, "medium": 0, "high": 0}
    for nse in all_nse:
        if nse < 0.3:
            bins["low"] += 1
        elif nse < 0.7:
            bins["medium"] += 1
        else:
            bins["high"] += 1
    n = len(all_nse)
    summary["uncertainty_distribution"] = {
        k: {"count": v, "ratio": round(v / n * 100, 2)}
        for k, v in bins.items()
    }

    if task == "cap2mol":
        gt_sims = [
            s["gt_mean_tanimoto"] for s in per_sample
            if "gt_mean_tanimoto" in s
        ]
        if gt_sims and len(gt_sims) == len(all_nse):
            corr = float(np.corrcoef(all_nse, gt_sims)[0, 1])
            summary["entropy_vs_accuracy_correlation"] = round(corr, 4)

    return summary
