#!/usr/bin/env python3
"""
Task-aware hallucination diagnosis on top of the 2x2 (IR/IO/ER/EO) taxonomy.

This module *extends* diagnose_hallucination.py to non-cap2mol tasks. The four
dimensions are reinterpreted task-by-task so the same schema can be used
across the evaluation matrix.

Task                 IR (reason)             IO (output validity)        ER (extrinsic-reason)        EO (extrinsic-output)
cap2mol              self-contradiction      invalid/inconsistent SMILES factual fab. of FGs/classes  phantom structure (Tanimoto)
mol2cap              self-contradiction      malformed caption           wrong FG/class claims        caption-similarity to GT
classification       self-contradiction      malformed Yes/No            FG/class claims about input  wrong label (0 / 100)
retrosynthesis       self-contradiction      invalid reactant SMILES     fabricated atoms vs product  reactant-set similarity to GT
s2_MolCustom         self-contradiction      invalid SMILES              fabricated FGs/classes       constraint violation magnitude
s2_MolEdit / Opt     self-contradiction      invalid / unchanged SMILES  fabricated FGs/classes       distance from source / wrong direction

Usage:
    python diagnose_multitask.py \
        --task bace \
        --model_output outputs/Chem-R/bace/output.json \
        --model_name Chem-R \
        --output_dir results/Chem-R/bace
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import diagnose_hallucination as DH  # reuse cap2mol scorers & helpers

DIMENSION_WEIGHTS = DH.DIMENSION_WEIGHTS  # IR(0.15) + IO(0.25) + ER(0.25) + EO(0.35)


# ============================================================================
# Helpers shared across task variants
# ============================================================================

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _extract_answer(answer: str) -> str:
    region = DH.extract_answer_region(answer)
    return region if region is not None else (answer or "").strip()


def _extract_yes_no(answer: str) -> Optional[str]:
    region = _extract_answer(answer)
    m = _YESNO_RE.search(region)
    if m:
        return m.group(1).lower()
    return None


def _extract_smiles(answer: str) -> Optional[str]:
    return DH.extract_answer_smiles(answer)


_DIM_FULL_KEY = {
    "IR": "IR_self_contradiction",
    "IO": "IO_structural_invalidity",
    "ER": "ER_factual_fabrication",
    "EO": "EO_phantom_structure",
}


def _aggregate(scores: Dict[str, float], dims_skipped: Dict[str, bool]) -> float:
    """Weighted aggregate over IR/IO/ER/EO with renormalisation when a
    dimension is skipped (e.g. invalid output)."""
    active = {k: w for k, w in DIMENSION_WEIGHTS.items() if not dims_skipped.get(k)}
    total_w = sum(active.values()) or 1.0
    return round(
        sum(scores[_DIM_FULL_KEY[k]] * (w / total_w) for k, w in active.items()),
        2,
    )


def _base_result(reasoning: Optional[str]) -> Dict[str, Any]:
    return {
        "reasoning_length": len(reasoning) if reasoning else 0,
        "hallucination_scores": {
            "IR_self_contradiction": 0.0,
            "IO_structural_invalidity": 0.0,
            "ER_factual_fabrication": 0.0,
            "EO_phantom_structure": 0.0,
        },
    }


# ============================================================================
# cap2mol — thin wrapper around the existing diagnose_single
# ============================================================================

def diagnose_cap2mol(input_text: str, answer: str, gt: str, verbose: bool = False) -> Dict[str, Any]:
    return DH.diagnose_single(input_text, answer, gt, verbose=verbose)


# ============================================================================
# mol2cap
# ============================================================================

def _text_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def diagnose_mol2cap(input_smi: str, answer: str, gt_caption: str, verbose: bool = False) -> Dict[str, Any]:
    reasoning = DH.extract_think(answer) or answer
    pred_caption = _extract_answer(answer)

    result = _base_result(reasoning)
    result["pred_caption"] = pred_caption[:500]
    result["pred_valid"] = bool(pred_caption and len(pred_caption.split()) >= 3)

    # IR — same as cap2mol
    ir, ir_det = DH.score_ir_self_contradiction(reasoning)
    result["hallucination_scores"]["IR_self_contradiction"] = ir

    # IO — caption well-formed (>=10 chars, "molecule is" pattern, etc.)
    io_score = 0.0
    io_det: Dict[str, Any] = {}
    if not pred_caption or len(pred_caption) < 10:
        io_score = 80.0
        io_det["reason"] = "empty/short caption"
    else:
        if "molecule is" not in pred_caption.lower():
            io_score += 25.0
            io_det["missing_phrase"] = "molecule is"
        if len(pred_caption.split()) < 8:
            io_score += 20.0
            io_det["too_few_words"] = True
    result["hallucination_scores"]["IO_structural_invalidity"] = round(io_score, 2)

    # ER — claims in (reasoning + caption) that contradict the input molecule
    input_mol = DH.parse_mol(input_smi)
    input_fgs = DH._get_mol_fgs(input_mol) if input_mol else set()
    claimed_fgs = (
        DH.extract_chemical_entities(reasoning + " " + pred_caption)
          .get("functional_groups", set())
    )
    fabricated = [f for f in claimed_fgs if f not in input_fgs]
    verified = [f for f in claimed_fgs if f in input_fgs]
    er_score = 0.0
    er_det: Dict[str, Any] = {
        "input_fgs": list(input_fgs),
        "claimed_fgs": list(claimed_fgs),
        "verified_fgs": verified,
        "fabricated_fgs": fabricated,
    }
    if claimed_fgs:
        er_score = min(100.0, (len(fabricated) / len(claimed_fgs)) * 60.0)
    result["hallucination_scores"]["ER_factual_fabrication"] = round(er_score, 2)

    # EO — caption similarity to GT caption (jaccard, transformed)
    sim = _text_jaccard(pred_caption, gt_caption)
    result["caption_jaccard"] = round(sim, 4)
    result["exact_match"] = sim >= 0.85
    eo_score = max(0.0, (1.0 - sim) * 100.0)
    result["hallucination_scores"]["EO_phantom_structure"] = round(eo_score, 2)
    if result["exact_match"]:
        result["hallucination_scores"]["EO_phantom_structure"] = 0.0

    skipped = {"IR": False, "IO": False, "ER": not bool(claimed_fgs), "EO": False}
    result["overall_hallucination_score"] = _aggregate(result["hallucination_scores"], skipped)

    if verbose:
        result["details"] = {"IR": ir_det, "IO": io_det, "ER": er_det,
                             "EO": {"jaccard": sim}}
    return result


# ============================================================================
# Classification (bace / bbbp / hiv / tox21 / clintox)
# ============================================================================

def diagnose_classification(input_smi: str, answer: str, gt: str, verbose: bool = False) -> Dict[str, Any]:
    reasoning = DH.extract_think(answer) or answer
    pred = _extract_yes_no(answer)

    result = _base_result(reasoning)
    result["pred_label"] = pred
    result["pred_valid"] = pred in ("yes", "no")
    result["exact_match"] = (pred is not None and gt is not None and pred == gt.lower())

    # IR
    ir, ir_det = DH.score_ir_self_contradiction(reasoning)
    result["hallucination_scores"]["IR_self_contradiction"] = ir

    # IO — Yes/No format compliance
    if pred is None:
        io_score = 70.0
        io_det = {"reason": "no yes/no in <answer>"}
    elif pred in ("yes", "no"):
        io_score = 0.0
        io_det = {"format_ok": True}
    else:
        io_score = 40.0
        io_det = {"reason": "unrecognised label"}
    result["hallucination_scores"]["IO_structural_invalidity"] = io_score

    # ER — FG fabrication relative to input molecule
    input_mol = DH.parse_mol(input_smi)
    input_fgs = DH._get_mol_fgs(input_mol) if input_mol else set()
    claimed = DH.extract_chemical_entities(reasoning).get("functional_groups", set())
    fabricated = [f for f in claimed if f not in input_fgs]
    verified = [f for f in claimed if f in input_fgs]
    er_score = 0.0
    er_det: Dict[str, Any] = {"claimed_fgs": list(claimed),
                              "verified_fgs": verified,
                              "fabricated_fgs": fabricated}
    if claimed:
        er_score = min(100.0, len(fabricated) / max(len(claimed), 1) * 60.0)
    result["hallucination_scores"]["ER_factual_fabrication"] = round(er_score, 2)

    # EO — incorrect label
    if pred is None:
        eo_score = 0.0  # IO already penalised the missing label
        eo_det = {"skipped": True, "reason": "no label"}
    elif result["exact_match"]:
        eo_score = 0.0
        eo_det = {"correct": True}
    else:
        eo_score = 100.0
        eo_det = {"correct": False, "pred": pred, "gt": gt}
    result["hallucination_scores"]["EO_phantom_structure"] = eo_score

    skipped = {"IR": False, "IO": False, "ER": not claimed,
               "EO": pred is None}
    result["overall_hallucination_score"] = _aggregate(result["hallucination_scores"], skipped)

    if verbose:
        result["details"] = {"IR": ir_det, "IO": io_det,
                             "ER": er_det, "EO": eo_det}
    return result


# ============================================================================
# Retrosynthesis
# ============================================================================

def _split_reactants(smi: str) -> List[str]:
    return [p.strip() for p in (smi or "").split(".") if p.strip()]


def _canonical_set(smi: str) -> List[str]:
    parts = _split_reactants(smi)
    can = [DH.canonical(p) for p in parts]
    can = [c for c in can if c]
    return sorted(can)


def diagnose_retrosynthesis(product_smi: str, answer: str, gt_reactants: str,
                            verbose: bool = False) -> Dict[str, Any]:
    reasoning = DH.extract_think(answer) or answer
    pred_block = _extract_answer(answer)
    pred_set = _canonical_set(pred_block)
    gt_set = _canonical_set(gt_reactants)

    result = _base_result(reasoning)
    result["pred_reactants_block"] = pred_block[:400]
    result["pred_reactants_canonical"] = pred_set
    result["gt_reactants_canonical"] = gt_set
    result["exact_match"] = bool(pred_set) and pred_set == gt_set

    # IR
    ir, ir_det = DH.score_ir_self_contradiction(reasoning)
    result["hallucination_scores"]["IR_self_contradiction"] = ir

    # IO — every reactant must parse
    parsed = [DH.parse_mol(p) for p in _split_reactants(pred_block)]
    n_total = len(parsed)
    n_valid = sum(1 for m in parsed if m is not None)
    if n_total == 0:
        io_score = 80.0
        io_det = {"reason": "no reactants"}
    else:
        invalid_ratio = 1.0 - n_valid / n_total
        io_score = invalid_ratio * 80.0
        io_det = {"valid": n_valid, "total": n_total}
    result["hallucination_scores"]["IO_structural_invalidity"] = round(io_score, 2)
    result["pred_valid"] = n_total > 0 and n_valid == n_total

    # ER — reasoning-claimed FGs that are absent in product AND in predicted reactants
    product_mol = DH.parse_mol(product_smi)
    product_fgs = DH._get_mol_fgs(product_mol) if product_mol else set()
    pred_fgs: set = set()
    for m in parsed:
        if m is not None:
            pred_fgs |= DH._get_mol_fgs(m)
    real_fgs = product_fgs | pred_fgs
    claimed = DH.extract_chemical_entities(reasoning).get("functional_groups", set())
    fabricated = [f for f in claimed if f not in real_fgs]
    verified = [f for f in claimed if f in real_fgs]
    er_score = 0.0
    er_det = {"claimed_fgs": list(claimed), "verified_fgs": verified,
              "fabricated_fgs": fabricated}
    if claimed:
        er_score = min(100.0, len(fabricated) / max(len(claimed), 1) * 60.0)
    result["hallucination_scores"]["ER_factual_fabrication"] = round(er_score, 2)

    # EO — set-level reactant similarity (best-Tanimoto matching)
    if not pred_set or not gt_set:
        eo_score = 80.0 if pred_set != gt_set else 0.0
        eo_det = {"reason": "empty set on one side"}
        result["reactant_tanimoto"] = 0.0
    elif result["exact_match"]:
        eo_score = 0.0
        eo_det = {"exact": True}
        result["reactant_tanimoto"] = 1.0
    else:
        sims = []
        for g in gt_set:
            best = 0.0
            for p in pred_set:
                t = DH.tanimoto(g, p) or 0.0
                if t > best:
                    best = t
            sims.append(best)
        mean_sim = sum(sims) / len(sims)
        eo_score = max(0.0, (1.0 - mean_sim) * 100.0)
        eo_det = {"mean_best_tanimoto": round(mean_sim, 4)}
        # Expose as a top-level continuous accuracy signal so the GRPO reward
        # gets a gradient (mirrors cap2mol's tanimoto / mol2cap's caption_jaccard).
        result["reactant_tanimoto"] = round(mean_sim, 4)
    result["hallucination_scores"]["EO_phantom_structure"] = round(eo_score, 2)

    skipped = {"IR": False, "IO": False, "ER": not claimed,
               "EO": not pred_set}
    result["overall_hallucination_score"] = _aggregate(result["hallucination_scores"], skipped)

    if verbose:
        result["details"] = {"IR": ir_det, "IO": io_det,
                             "ER": er_det, "EO": eo_det}
    return result


def diagnose_reaction_prediction(reactants_smi: str, answer: str, gt_product: str,
                                 verbose: bool = False) -> Dict[str, Any]:
    """Forward reaction prediction: given reactants, predict the product.

    Dual of :func:`diagnose_retrosynthesis` -- here the reactants are the given
    context and the product is predicted, so the ``real`` functional groups for
    ER are those present in the reactants OR the predicted product.
    """
    reasoning = DH.extract_think(answer) or answer
    pred_block = _extract_answer(answer)
    pred_set = _canonical_set(pred_block)
    gt_set = _canonical_set(gt_product)

    result = _base_result(reasoning)
    result["pred_product_block"] = pred_block[:400]
    result["pred_product_canonical"] = pred_set
    result["gt_product_canonical"] = gt_set
    result["exact_match"] = bool(pred_set) and pred_set == gt_set

    # IR
    ir, ir_det = DH.score_ir_self_contradiction(reasoning)
    result["hallucination_scores"]["IR_self_contradiction"] = ir

    # IO -- every predicted product must parse
    parsed = [DH.parse_mol(p) for p in _split_reactants(pred_block)]
    n_total = len(parsed)
    n_valid = sum(1 for m in parsed if m is not None)
    if n_total == 0:
        io_score = 80.0
        io_det = {"reason": "no product"}
    else:
        invalid_ratio = 1.0 - n_valid / n_total
        io_score = invalid_ratio * 80.0
        io_det = {"valid": n_valid, "total": n_total}
    result["hallucination_scores"]["IO_structural_invalidity"] = round(io_score, 2)
    result["pred_valid"] = n_total > 0 and n_valid == n_total

    # ER -- reasoning-claimed FGs absent in BOTH reactants and predicted product
    real_fgs: set = set()
    for r in _split_reactants(reactants_smi):
        m = DH.parse_mol(r)
        if m is not None:
            real_fgs |= DH._get_mol_fgs(m)
    for m in parsed:
        if m is not None:
            real_fgs |= DH._get_mol_fgs(m)
    claimed = DH.extract_chemical_entities(reasoning).get("functional_groups", set())
    fabricated = [f for f in claimed if f not in real_fgs]
    verified = [f for f in claimed if f in real_fgs]
    er_score = 0.0
    er_det = {"claimed_fgs": list(claimed), "verified_fgs": verified,
              "fabricated_fgs": fabricated}
    if claimed:
        er_score = min(100.0, len(fabricated) / max(len(claimed), 1) * 60.0)
    result["hallucination_scores"]["ER_factual_fabrication"] = round(er_score, 2)

    # EO -- product similarity (best-Tanimoto matching)
    if not pred_set or not gt_set:
        eo_score = 80.0 if pred_set != gt_set else 0.0
        eo_det = {"reason": "empty set on one side"}
        result["product_tanimoto"] = 0.0
    elif result["exact_match"]:
        eo_score = 0.0
        eo_det = {"exact": True}
        result["product_tanimoto"] = 1.0
    else:
        sims = []
        for g in gt_set:
            best = 0.0
            for p in pred_set:
                t = DH.tanimoto(g, p) or 0.0
                if t > best:
                    best = t
            sims.append(best)
        mean_sim = sum(sims) / len(sims)
        eo_score = max(0.0, (1.0 - mean_sim) * 100.0)
        eo_det = {"mean_best_tanimoto": round(mean_sim, 4)}
        # Top-level continuous accuracy signal for the GRPO reward.
        result["product_tanimoto"] = round(mean_sim, 4)
    result["hallucination_scores"]["EO_phantom_structure"] = round(eo_score, 2)

    skipped = {"IR": False, "IO": False, "ER": not claimed, "EO": not pred_set}
    result["overall_hallucination_score"] = _aggregate(result["hallucination_scores"], skipped)

    if verbose:
        result["details"] = {"IR": ir_det, "IO": io_det,
                             "ER": er_det, "EO": eo_det}
    return result


# ============================================================================
# S2-bench
# ============================================================================

_S2_ATOM_KEYS = ("carbon", "oxygen", "nitrogen", "sulfur", "fluorine",
                 "chlorine", "bromine", "iodine", "phosphorus", "boron",
                 "silicon", "selenium", "tellurium", "arsenic", "antimony",
                 "bismuth", "polonium")

_S2_BOND_KEYS = ("single", "double", "triple", "rotatable", "aromatic")

# Functional-group SMARTS lifted from S2-TOMG-Bench's evaluation utility.
_S2_FG_SMARTS = {
    "benzene_ring": "c1ccccc1",
    "hydroxyl":     "[OX2H]",
    "anhydride":    "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "aldehyde":     "[CX3H1](=O)[#6]",
    "ketone":       "[#6][CX3](=O)[#6]",
    "carboxyl":     "[CX3](=O)[OX2H1]",
    "ester":        "[CX3](=O)[OX2H0][#6]",
    "amide":        "[NX3][CX3](=[OX1])",
    "amine":        "[NX3;H2,H1;!$(NC=O)]",
    "nitro":        "[NX3](=O)=O",
    "halo":         "[F,Cl,Br,I]",
    "thioether":    "[#16X2]([#6])[#6]",
    "nitrile":      "[NX1]#[CX2]",
    "thiol":        "[#16X2H]",
    "sulfide":      "[#16X2]([#6])[#6]",
    "disulfide":    "[#16X2][#16X2]",
    "sulfoxide":    "[#16X3](=[OX1])[#6]",
    "sulfone":      "[#16X4](=[OX1])(=[OX1])[#6]",
    "borane":       "[BX3]",
}


def _count_atoms(mol) -> Dict[str, int]:
    if mol is None:
        return {}
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    sym_map = {"C": "carbon", "O": "oxygen", "N": "nitrogen", "S": "sulfur",
               "F": "fluorine", "Cl": "chlorine", "Br": "bromine", "I": "iodine",
               "P": "phosphorus", "B": "boron", "Si": "silicon",
               "Se": "selenium", "Te": "tellurium", "As": "arsenic",
               "Sb": "antimony", "Bi": "bismuth", "Po": "polonium"}
    counts = {v: 0 for v in sym_map.values()}
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym in sym_map:
            counts[sym_map[sym]] += 1
    return counts


def _count_bonds(mol) -> Dict[str, int]:
    if mol is None:
        return {}
    counts = {k: 0 for k in _S2_BOND_KEYS}
    if DH.HAS_RDKIT:
        from rdkit.Chem import rdMolDescriptors
        for b in mol.GetBonds():
            btype = b.GetBondTypeAsDouble()
            if b.GetIsAromatic():
                counts["aromatic"] += 1
            elif btype == 1.0:
                counts["single"] += 1
            elif btype == 2.0:
                counts["double"] += 1
            elif btype == 3.0:
                counts["triple"] += 1
        counts["rotatable"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return counts


def _count_fgs(mol) -> Dict[str, int]:
    if mol is None or not DH.HAS_RDKIT:
        return {}
    from rdkit import Chem
    out = {}
    for name, sm in _S2_FG_SMARTS.items():
        pat = Chem.MolFromSmarts(sm)
        if pat is None:
            continue
        out[name] = len(mol.GetSubstructMatches(pat))
    return out


def _s2_constraint_eo(task: str, pred_smi: Optional[str], metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """EO score for S2 = magnitude of constraint violation."""
    mol = DH.parse_mol(pred_smi) if pred_smi else None
    if mol is None:
        return 100.0, {"reason": "invalid SMILES"}

    if task.endswith("AtomNum"):
        counts = _count_atoms(mol)
        constraints = metadata.get("constraints", {})
        deviations = []
        for k in _S2_ATOM_KEYS:
            target_str = constraints.get(k, "0")
            try:
                target = int(target_str)
            except (TypeError, ValueError):
                continue
            if target <= 0:
                continue
            got = counts.get(k, 0)
            deviations.append(abs(got - target) / max(target, 1))
        if not deviations:
            return 0.0, {"reason": "no positive atom constraints"}
        mean_dev = sum(deviations) / len(deviations)
        return round(min(100.0, mean_dev * 100.0), 2), {"mean_atom_deviation": round(mean_dev, 3)}

    if task.endswith("BondNum"):
        counts = _count_bonds(mol)
        constraints = metadata.get("constraints", {})
        deviations = []
        for k in _S2_BOND_KEYS:
            target_str = constraints.get(k, "0")
            try:
                target = int(target_str)
            except (TypeError, ValueError):
                continue
            if target <= 0:
                continue
            got = counts.get(k, 0)
            deviations.append(abs(got - target) / max(target, 1))
        if not deviations:
            return 0.0, {"reason": "no positive bond constraints"}
        mean_dev = sum(deviations) / len(deviations)
        return round(min(100.0, mean_dev * 100.0), 2), {"mean_bond_deviation": round(mean_dev, 3)}

    if task.endswith("FunctionalGroup"):
        counts = _count_fgs(mol)
        constraints = metadata.get("constraints", {})
        deviations = []
        for k in _S2_FG_SMARTS:
            target_str = constraints.get(k, "0")
            try:
                target = int(target_str)
            except (TypeError, ValueError):
                continue
            if target <= 0:
                continue
            got = counts.get(k, 0)
            deviations.append(abs(got - target) / max(target, 1))
        if not deviations:
            return 0.0, {"reason": "no positive FG constraints"}
        mean_dev = sum(deviations) / len(deviations)
        return round(min(100.0, mean_dev * 100.0), 2), {"mean_fg_deviation": round(mean_dev, 3)}

    # MolEdit / MolOpt: compare to source molecule. The source should be modified
    # in a specific way; we use Tanimoto to score the "phantom" component
    # (over-modification = low similarity, no-op = exact match to source).
    src = metadata.get("source_molecule", "")
    if not src:
        return 50.0, {"reason": "no source molecule in metadata"}
    src_mol = DH.parse_mol(src)
    if src_mol is None:
        return 50.0, {"reason": "invalid source molecule"}

    src_can = DH.canonical(src)
    pred_can = DH.canonical(pred_smi)
    if src_can == pred_can:
        # No edit performed at all → high "phantom" because the model just
        # echoed the source.
        return 60.0, {"reason": "no edit performed"}

    sim = DH.tanimoto(src, pred_smi) or 0.0
    # For Edit/Opt we expect moderate similarity (most of the scaffold kept,
    # one component swapped). Penalise both too-low and too-high distance.
    if sim < 0.2:
        eo = 80.0
    elif sim > 0.95:
        eo = 40.0
    else:
        eo = max(0.0, (1.0 - sim) * 50.0)
    return round(eo, 2), {"source_pred_tanimoto": round(sim, 4)}


def diagnose_s2(task: str, instruction: str, answer: str,
                metadata: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    reasoning = DH.extract_think(answer) or answer
    pred_smi = _extract_smiles(answer)

    result = _base_result(reasoning)
    result["pred_smiles"] = pred_smi
    result["pred_valid"] = pred_smi is not None and DH.parse_mol(pred_smi) is not None

    # IR
    ir, ir_det = DH.score_ir_self_contradiction(reasoning)
    result["hallucination_scores"]["IR_self_contradiction"] = ir

    # IO — structural validity
    if pred_smi is None:
        io_score, io_det = 80.0, {"reason": "no SMILES"}
    elif DH.parse_mol(pred_smi) is None:
        io_score, io_det = 80.0, {"reason": "invalid SMILES"}
    else:
        io_score, io_det = 0.0, {"valid": True}
    result["hallucination_scores"]["IO_structural_invalidity"] = io_score

    # ER — reasoning factual fabrication. A claimed FG is fabricated ONLY if it
    # is absent from BOTH the input (source molecule + instruction-named groups)
    # AND the predicted molecule. S2 edit/substitute tasks legitimately discuss
    # the group being removed/replaced — that group lives in the source molecule
    # or is named in the instruction, so it must not count as a hallucination.
    pred_mol = DH.parse_mol(pred_smi) if pred_smi else None
    pred_fgs = DH._get_mol_fgs(pred_mol) if pred_mol else set()
    src = metadata.get("source_molecule", "") or ""
    src_mol = DH.parse_mol(src) if src else None
    src_fgs = DH._get_mol_fgs(src_mol) if src_mol else set()
    instr_fgs = DH.extract_chemical_entities(instruction).get("functional_groups", set())
    real_fgs = pred_fgs | src_fgs | instr_fgs
    claimed = DH.extract_chemical_entities(reasoning).get("functional_groups", set())
    fabricated = [f for f in claimed if f not in real_fgs]
    verified = [f for f in claimed if f in real_fgs]
    er_score = 0.0
    er_det = {"claimed_fgs": list(claimed), "verified_fgs": verified,
              "fabricated_fgs": fabricated, "input_fgs": list(src_fgs | instr_fgs)}
    if claimed:
        er_score = min(100.0, len(fabricated) / max(len(claimed), 1) * 60.0)
    result["hallucination_scores"]["ER_factual_fabrication"] = round(er_score, 2)

    # EO — constraint-aware
    eo_score, eo_det = _s2_constraint_eo(task, pred_smi, metadata)
    result["hallucination_scores"]["EO_phantom_structure"] = eo_score
    result["exact_match"] = eo_score == 0.0

    skipped = {"IR": False, "IO": False, "ER": not claimed, "EO": False}
    result["overall_hallucination_score"] = _aggregate(result["hallucination_scores"], skipped)

    if verbose:
        result["details"] = {"IR": ir_det, "IO": io_det,
                             "ER": er_det, "EO": eo_det}
    return result


# ============================================================================
# Top-level dispatch
# ============================================================================

def diagnose_one(task: str, sample: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """sample must have: id, question (= input), gt, answer, plus optional metadata."""
    answer = sample.get("answer", "") or ""
    input_text = sample.get("question") or sample.get("input") or ""
    gt = sample.get("gt", "")
    metadata = sample.get("metadata", {}) or {}

    if task == "cap2mol":
        return diagnose_cap2mol(input_text, answer, gt, verbose=verbose)
    if task == "mol2cap":
        return diagnose_mol2cap(input_text, answer, gt, verbose=verbose)
    if task in ("bace", "bbbp", "hiv", "tox21", "clintox"):
        return diagnose_classification(input_text, answer, gt, verbose=verbose)
    if task == "retrosynthesis":
        return diagnose_retrosynthesis(input_text, answer, gt, verbose=verbose)
    if task == "reaction_prediction":
        return diagnose_reaction_prediction(input_text, answer, gt, verbose=verbose)
    if task.startswith("s2_"):
        return diagnose_s2(task, input_text, answer, metadata, verbose=verbose)
    raise ValueError(f"Unknown task: {task}")


# ============================================================================
# Batch driver
# ============================================================================

def evaluate_file(task: str, model_name: str, model_output_path: str,
                  output_dir: str, verbose: bool = False,
                  max_samples: Optional[int] = None) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"  Diagnosing: {model_name} / {task}")
    print(f"  Input:      {model_output_path}")
    print(f"{'='*70}")

    with open(model_output_path) as f:
        samples = json.load(f)
    if max_samples:
        samples = samples[:max_samples]

    results = []
    score_acc: Dict[str, List[float]] = defaultdict(list)
    tanimoto_acc: List[float] = []
    exact = 0
    valid = 0

    for sample in tqdm(samples, desc=f"  {model_name}/{task}"):
        if not sample.get("answer"):
            continue
        diag = diagnose_one(task, sample, verbose=verbose)
        diag["id"] = sample.get("id")
        diag["task"] = task
        diag["model"] = model_name
        results.append(diag)

        for k, v in diag["hallucination_scores"].items():
            score_acc[k].append(v)
        score_acc["overall"].append(diag["overall_hallucination_score"])
        if diag.get("exact_match"):
            exact += 1
        if diag.get("pred_valid"):
            valid += 1
        if diag.get("tanimoto") is not None:
            tanimoto_acc.append(diag["tanimoto"])

    summary: Dict[str, Any] = {
        "model": model_name,
        "task": task,
        "evaluated_samples": len(results),
        "validity_rate": round(valid / max(len(results), 1) * 100, 2),
        "exact_match_rate": round(exact / max(len(results), 1) * 100, 2),
        "hallucination_scores": {
            k: {
                "mean": round(sum(v) / len(v), 2),
                "std": round((sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5, 2),
                "min": round(min(v), 2),
                "max": round(max(v), 2),
            }
            for k, v in score_acc.items() if v
        },
    }
    if tanimoto_acc:
        summary["avg_tanimoto"] = round(sum(tanimoto_acc) / len(tanimoto_acc), 4)

    # Distribution buckets
    bins = {"low": 0, "moderate": 0, "high": 0}
    for r in results:
        s = r["overall_hallucination_score"]
        if s < 30: bins["low"] += 1
        elif s < 60: bins["moderate"] += 1
        else: bins["high"] += 1
    summary["hallucination_distribution"] = {
        k: {"count": v, "ratio": round(v / max(len(results), 1) * 100, 2)}
        for k, v in bins.items()
    }

    os.makedirs(output_dir, exist_ok=True)
    safe = model_name.replace("/", "_")
    with open(os.path.join(output_dir, f"{safe}_{task}_hallucination_details.jsonl"), "w") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    with open(os.path.join(output_dir, f"{safe}_{task}_hallucination_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  Validity: {summary['validity_rate']}%   "
          f"Exact: {summary['exact_match_rate']}%")
    for k, st in summary["hallucination_scores"].items():
        print(f"  {k:32s} mean={st['mean']:6.2f}  std={st['std']:6.2f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Task-aware hallucination diagnosis (2x2)")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    evaluate_file(args.task, args.model_name, args.model_output,
                  args.output_dir, verbose=args.verbose, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
