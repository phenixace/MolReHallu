#!/usr/bin/env python3
"""
Reasoning Hallucination Diagnosis for Molecular LLMs (v2).

2x2 Taxonomy:
                    Reasoning (CoT)              Output (SMILES)
  Intrinsic    IR: Self-Contradiction        IO: Structural Invalidity
  Extrinsic    ER: Factual Fabrication       EO: Phantom Structure

  IR — Self-contradictions, flip-flopping, excessive correction in the
       reasoning trace.  Purely internal signal; no external reference needed.
  IO — The predicted SMILES is chemically invalid, violates physicochemical
       constraints implied by the description, or contradicts the model's own
       reasoning (correctly identified FGs missing from output, SMILES
       fragments in reasoning absent from prediction).
  ER — The reasoning asserts verifiable chemical facts that are *wrong*
       when checked against the predicted molecule or ground truth.
       Only penalises claims that can be objectively falsified with RDKit.
  EO — The predicted molecule is structurally valid but substantially
       different from the ground truth (Tanimoto, scaffold, FG mismatch).

Usage:
  python diagnose_hallucination.py \\
    --model_output outputs/chem_r/output.json \\
    --model_name Chem-R \\
    --output_dir results/

  python diagnose_hallucination.py --batch_config batch_config.json --output_dir results/
"""

import json
import os
import re
import math
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors, rdMolDescriptors, AllChem, DataStructs
    )
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available. Structural analysis will be limited.")

try:
    import selfies as sf
    HAS_SELFIES = True
except ImportError:
    HAS_SELFIES = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === STUB: sections added via StrReplace below ===
# ============================================================================
# Chemical Knowledge Base
# ============================================================================

# Functional-group / substructure knowledge base.  Each entry maps a canonical
# name to (SMARTS pattern, [reasoning keywords]).  A reasoning claim is VERIFIED
# if the named SMARTS matches the predicted-or-GT molecule, and FABRICATED
# otherwise.  Design rules (audited 2026-06-20):
#   * Generic claims ("aromatic ring", "ring", "halogen", "double bond") match
#     ANY instance, so naming a non-benzene aromatic is not a false positive.
#   * Specific groups use tight SMARTS to avoid spurious verification.
#   * Every SMARTS is validated against positive + negative reference molecules
#     by tests/validate_fg_db.py; do not edit a pattern without re-running it.
FUNCTIONAL_GROUP_DB: Dict[str, Tuple[str, List[str]]] = {
    # ---- generic structural features (permissive: match any instance) -------
    "aromatic_ring":  ("[a]",                              ["aromatic ring", "aromatic", "aryl", "aromatic system", "aromatic moiety", "aromatic core"]),
    "ring":           ("[R]",                              ["ring system", "cyclic", "carbocycle", "ring structure"]),
    "heterocycle":    ("[!#6;R]",                          ["heterocycle", "heterocyclic", "heteroaromatic", "heterocyclic ring", "heteroatom in the ring"]),
    "halogen":        ("[F,Cl,Br,I]",                      ["halogen", "halide", "halo", "halogenated"]),
    "double_bond":    ("[$(*=*),a]",                       ["double bond", "unsaturation", "unsaturated"]),
    "triple_bond":    ("*#*",                              ["triple bond"]),
    # ---- oxygen-containing groups -------------------------------------------
    "hydroxyl":       ("[OX2H]",                           ["hydroxy", "hydroxyl", "alcohol", "-oh", "phenol", "phenolic"]),
    "carboxyl":       ("[CX3](=O)[OX2H1,OX1-]",            ["carboxyl", "carboxylic acid", "cooh", "carboxy"]),
    "carboxylate":    ("[CX3](=O)[O-]",                    ["carboxylate"]),
    "ester":          ("[#6][CX3](=[OX1])[OX2H0][#6]",     ["ester", "acetate", "methyl ester", "ethyl ester", "ester linkage", "ester bond", "lactone"]),
    "ether":          ("[OX2;!a;!$(O[CX3]=[OX1])]([#6])[#6]", ["ether", "methoxy", "ethoxy", "alkoxy"]),
    "ketone":         ("[#6][CX3](=O)[#6]",                ["ketone", "keto"]),
    "aldehyde":       ("[CX3H1](=O)[#6]",                  ["aldehyde", "formyl", "-cho"]),
    "carbonyl":       ("[CX3]=[OX1]",                      ["carbonyl", "acyl", "acetyl", "oxo"]),
    "carbonate":      ("[CX3](=[OX1])([OX2])[OX2]",        ["carbonate", "carbonate ester"]),
    "anhydride":      ("[CX3](=[OX1])[OX2][CX3]=[OX1]",    ["anhydride"]),
    "epoxide":        ("[OX2r3]",                          ["epoxide", "oxirane", "epoxy"]),
    "peroxide":       ("[OX2][OX2]",                       ["peroxide", "peroxy"]),
    "acetal":         ("[CX4]([OX2])[OX2]",                ["acetal", "ketal"]),
    "enol":           ("[OX2H][CX3]=[CX3]",                ["enol"]),
    # ---- nitrogen-containing groups -----------------------------------------
    "amino":          ("[$([NX3;!$(NC=O);!$(N=*);!$([NX3]([OX1])=O)]),$([NX4+;!$(NC=O)])]",
                                                           ["amino", "amine", "nh2", "primary amine", "secondary amine", "tertiary amine", "ammonium"]),
    "amide":          ("[NX3][CX3](=[OX1])",               ["amide", "amido", "carboxamide", "peptide bond", "amide bond", "lactam"]),
    "imine":          ("[CX3]=[NX2]",                      ["imine", "schiff base", "schiff", "azomethine"]),
    "amidine":        ("[NX3][CX3]=[NX2]",                 ["amidine"]),
    "guanidine":      ("[NX3][CX3](=[NX2])[NX3]",          ["guanidine", "guanidinium"]),
    "nitrile":        ("[NX1]#[CX2]",                      ["nitrile", "cyano", "carbonitrile"]),
    "nitro":          ("[$([NX3](=O)=O),$([NX3+](=O)[O-])]", ["nitro", "nitro group", "-no2"]),
    "nitroso":        ("[NX2]=[OX1]",                      ["nitroso"]),
    "n_oxide":        ("[#7+][OX1-]",                      ["n-oxide", "n oxide", "amine oxide", "pyridine n-oxide"]),
    "hydrazine":      ("[NX3;!$(NC=O)][NX3;!$(NC=O)]",     ["hydrazine", "hydrazino"]),
    "hydrazone":      ("[NX3][NX2]=[CX3]",                 ["hydrazone"]),
    "oxime":          ("[CX3]=[NX2][OX2H]",                ["oxime"]),
    "azo":            ("[NX2]=[NX2]",                      ["azo"]),
    "azide":          ("[$([#7]=[#7+]=[#7-]),$([#7-][#7+]#[#7])]", ["azide", "azido"]),
    "isocyanate":     ("[NX2]=[CX2]=[OX1]",                ["isocyanate"]),
    "isothiocyanate": ("[NX2]=[CX2]=[SX1]",                ["isothiocyanate"]),
    "carbamate":      ("[NX3][CX3](=[OX1])[OX2][#6]",      ["carbamate", "urethane"]),
    "urea":           ("[NX3][CX3](=[OX1])[NX3]",          ["urea"]),
    "thiourea":       ("[NX3][CX3](=[SX1])[NX3]",          ["thiourea"]),
    # ---- sulfur / phosphorus / boron / silicon ------------------------------
    "sulfhydryl":     ("[#16X2H]",                         ["thiol", "sulfhydryl", "mercapto", "-sh"]),
    "thioether":      ("[#16X2]([#6])[#6]",                ["thioether", "sulfide"]),
    "disulfide":      ("[SX2][SX2]",                       ["disulfide"]),
    "sulfoxide":      ("[$([SX3]=[OX1])]",                 ["sulfoxide"]),
    "sulfonyl":       ("[SX4](=[OX1])(=[OX1])",            ["sulfonyl", "sulfone"]),
    "sulfonamide":    ("[SX4](=[OX1])(=[OX1])[NX3]",       ["sulfonamide"]),
    "sulfonic_acid":  ("[SX4](=[OX1])(=[OX1])[OX2H,OX1-]", ["sulfonic acid", "sulfonate", "sulfo"]),
    "thioester":      ("[#6][CX3](=[OX1])[SX2]",           ["thioester"]),
    "thioamide":      ("[#6][CX3](=[SX1])[NX3]",           ["thioamide", "thioamido"]),
    "phosphate":      ("[PX4](=[OX1])([OX2,OX1-])([OX2,OX1-])[OX2,OX1-]", ["phosphate", "phospho", "phosphoryl"]),
    "phosphonate":    ("[PX4](=[OX1])([#6])([OX2,OX1-])",  ["phosphonate", "phosphonic"]),
    "phosphine":      ("[PX3;!$(P=O)]",                    ["phosphine", "phosphino"]),
    "boronic_acid":   ("[BX3]([OX2H,OX1-])[OX2H,OX1-]",    ["boronic", "boronate"]),
    "silyl":          ("[Si]",                             ["silyl", "silane", "siloxane", "tms"]),
    # ---- halogens (specific) ------------------------------------------------
    "halide_f":       ("[F;X0,X1]",                            ["fluoro", "fluoride", "fluorine", "fluorinated"]),
    "halide_cl":      ("[Cl;X0,X1]",                           ["chloro", "chloride", "chlorine", "chlorinated"]),
    "halide_br":      ("[Br;X0,X1]",                           ["bromo", "bromide", "bromine", "brominated"]),
    "halide_i":       ("[I;X0,X1]",                            ["iodo", "iodide", "iodine", "iodinated"]),
    # ---- unsaturation (specific carbon-carbon) ------------------------------
    "vinyl":          ("[CX3]=[CX3]",                      ["alkene", "vinyl", "olefin", "ethylene"]),
    "alkyne":         ("[CX2]#[CX2]",                      ["alkyne", "acetylene"]),
    # ---- carbocyclic aromatics ----------------------------------------------
    "phenyl":         ("c1ccccc1",                         ["phenyl", "benzene", "benzene ring", "benzyl", "benzo"]),
    "naphthalene":    ("c1ccc2ccccc2c1",                   ["naphthalene", "naphthyl", "naphtho"]),
    # ---- nitrogen heteroaromatics -------------------------------------------
    "pyridine":       ("c1ccncc1",                         ["pyridine", "pyridyl", "pyridinyl"]),
    "pyrimidine":     ("c1cncnc1",                         ["pyrimidine", "pyrimidinyl"]),
    "pyrazine":       ("c1cnccn1",                         ["pyrazine"]),
    "pyridazine":     ("c1ccnnc1",                         ["pyridazine"]),
    "pyrrole":        ("c1cc[n]c1",                       ["pyrrole", "pyrrolyl"]),
    "pyrazole":       ("c1cc[n]n1",                       ["pyrazole"]),
    "imidazole":      ("c1c[n]cn1",                       ["imidazole", "imidazolyl"]),
    "triazole":       ("[$(c1nc[n]n1),$(c1cn[n]n1),$(c1nn[n]n1)]", ["triazole"]),
    "tetrazole":      ("c1nnn[n]1",                       ["tetrazole"]),
    # ---- oxygen / sulfur heteroaromatics ------------------------------------
    "furan":          ("c1ccoc1",                          ["furan", "furyl", "furanyl", "furo"]),
    "thiophene":      ("c1ccsc1",                          ["thiophene", "thienyl", "thiophenyl", "thieno"]),
    "oxazole":        ("c1ocnc1",                          ["oxazole"]),
    "isoxazole":      ("c1conc1",                          ["isoxazole"]),
    "thiazole":       ("c1scnc1",                          ["thiazole"]),
    "isothiazole":    ("c1ccsn1",                          ["isothiazole"]),
    # ---- fused heteroaromatics ----------------------------------------------
    "indole":         ("c1ccc2[n]ccc2c1",                 ["indole", "indolyl", "indolo"]),
    "benzofuran":     ("c1ccc2occc2c1",                    ["benzofuran"]),
    "benzothiophene": ("c1ccc2sccc2c1",                    ["benzothiophene"]),
    "benzimidazole":  ("c1ccc2[n]cnc2c1",                 ["benzimidazole"]),
    "benzoxazole":    ("c1ccc2ocnc2c1",                    ["benzoxazole"]),
    "benzothiazole":  ("c1ccc2scnc2c1",                    ["benzothiazole"]),
    "quinoline":      ("c1ccc2ncccc2c1",                   ["quinoline", "quinolyl", "quinolinyl"]),
    "isoquinoline":   ("c1ccc2cnccc2c1",                   ["isoquinoline"]),
    "quinazoline":    ("c1ccc2ncncc2c1",                   ["quinazoline"]),
    "purine":         ("c1ncc2nc[n]c2n1",                ["purine"]),
    "carbazole":      ("c1ccc2c(c1)[n]c1ccccc12",        ["carbazole"]),
    "coumarin":       ("O=c1ccc2ccccc2o1",                 ["coumarin"]),
    # ---- saturated heterocycles ---------------------------------------------
    "piperidine":     ("[NX3]1[CX4][CX4][CX4][CX4][CX4]1", ["piperidine", "piperidinyl"]),
    "piperazine":     ("[NX3]1[CX4][CX4][NX3][CX4][CX4]1", ["piperazine"]),
    "morpholine":     ("[OX2]1[CX4][CX4][NX3][CX4][CX4]1", ["morpholine", "morpholino", "morpholinyl"]),
    "pyrrolidine":    ("[NX3]1[CX4][CX4][CX4][CX4]1",      ["pyrrolidine", "pyrrolidinyl"]),
    "tetrahydrofuran":("[OX2]1[CX4][CX4][CX4][CX4]1",      ["tetrahydrofuran", "oxolane"]),
    "tetrahydropyran":("[OX2]1[CX4][CX4][CX4][CX4][CX4]1", ["tetrahydropyran", "oxane"]),
    "aziridine":      ("[NX3]1[CX4][CX4]1",                ["aziridine"]),
    "oxetane":        ("[OX2]1[CX4][CX4][CX4]1",           ["oxetane"]),
    # ---- saturated carbocycles ----------------------------------------------
    "cyclopropane":   ("[CX4]1[CX4][CX4]1",                ["cyclopropane", "cyclopropyl"]),
    "cyclopentane":   ("[CX4]1[CX4][CX4][CX4][CX4]1",      ["cyclopentane", "cyclopentyl"]),
    "cyclohexane":    ("[CX4]1[CX4][CX4][CX4][CX4][CX4]1", ["cyclohexane", "cyclohexyl"]),
}

# Generic features are deliberately easy to satisfy; downstream consumers (e.g.
# the grounded-claims reward) may want to weight them less than specific groups.
GENERIC_FG_NAMES = frozenset({
    "aromatic_ring", "ring", "heterocycle", "halogen", "double_bond", "triple_bond",
})
MOLECULAR_CLASS_HINTS = {
    "steroid":    {"keywords": ["steroid", "pregnane", "androstane", "estrane", "cholest"],
                   "min_rings": 4, "min_carbons": 17},
    "sugar":      {"keywords": ["saccharide", "sugar", "glucose", "galactose", "glycosyl", "glycoside"],
                   "min_oxygens": 3},
    "amino_acid": {"keywords": ["amino acid", "alanine", "glycine", "valine", "leucine"],
                   "requires_N": True, "requires_O": True},
    "nucleotide": {"keywords": ["nucleotide", "nucleoside", "adenine", "guanine", "purine"],
                   "requires_N": True, "requires_P": True},
    "fatty_acid": {"keywords": ["fatty acid", "long-chain", "oleic", "palmitic", "stearic"],
                   "min_carbons": 8},
    "alkaloid":   {"keywords": ["alkaloid"], "requires_N": True, "min_rings": 2},
    "peptide":    {"keywords": ["peptide", "dipeptide", "tripeptide"],
                   "requires_N": True, "requires_O": True},
    "terpene":    {"keywords": ["terpene", "terpenoid", "monoterpene", "sesquiterpene"],
                   "min_carbons": 10},
    "flavonoid":  {"keywords": ["flavonoid", "flavone", "isoflavone", "catechin"],
                   "min_rings": 3},
}

BIO_ROLE_KEYWORDS = [
    "metabolite", "inhibitor", "agonist", "antagonist", "cofactor",
    "vitamin", "hormone", "neurotransmitter", "antioxidant", "antibiotic",
    "drug", "toxin", "allergen", "has a role as", "it has a role",
    "signaling", "marker", "EC ", "plant metabolite", "human metabolite",
    "mouse metabolite",
]


# ============================================================================
# Utility Functions
# ============================================================================

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


_ETHER_ANS_RE = re.compile(r"<\|answer_start\|>(.*?)(?:<\|answer_end\|>|\Z)", re.DOTALL)
_ETHER_THINK_RE = re.compile(r"<\|think_start\|>(.*?)(?:<\|think_end\|>|\Z)", re.DOTALL)
_STD_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STD_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANY_ANS_START = re.compile(r"<\|answer_start\|>|<answer>")


def extract_answer_region(text: Optional[str]) -> Optional[str]:
    """Format-aware answer extraction. ether-0 emits native <|answer_start|>..
    <|answer_end|> tokens AND sometimes mentions a stray '<answer>' in its
    reasoning, so we read ether-0's native tokens first (unambiguous) and only
    fall back to standard <answer>..</answer> (taking the LAST pair, to dodge any
    stray earlier mention)."""
    if not text:
        return None
    m = _ETHER_ANS_RE.search(text)
    if m:
        return m.group(1).strip()
    ms = _STD_ANS_RE.findall(text)
    if ms:
        return ms[-1].strip()
    return None


def extract_think(answer: str) -> Optional[str]:
    if not answer:
        return None
    m = _ETHER_THINK_RE.search(answer)
    if m:
        return m.group(1).strip()
    m = _STD_THINK_RE.search(answer)
    if m:
        return m.group(1).strip()
    a = _ANY_ANS_START.search(answer)   # no think tags: reasoning = before answer
    if a and a.start() > 0:
        return answer[:a.start()].strip()
    return None


_SELFIES_RE = re.compile(r"^\[(?:C|N|O|S|P|F|Cl|Br|I|B|#|=|/|\\|@|@@|Ring|Branch|Expl)")


def _is_selfies(s: str) -> bool:
    return bool(s and s.startswith("[") and _SELFIES_RE.match(s))


def _sanitize_selfies(selfies_str: str) -> str:
    s = selfies_str
    s = s.replace("[C@@H1]", "[C@@]").replace("[C@H1]", "[C@]")
    s = s.replace("[C@@H]", "[C@@]").replace("[C@H]", "[C@]")
    s = s.replace("[\\C@@H]", "[\\C@@]").replace("[\\C@H]", "[\\C@]")
    s = s.replace("[/C@@H]", "[/C@@]").replace("[/C@H]", "[/C@]")
    return s


def _selfies_to_smiles(selfies_str: str) -> Optional[str]:
    if not HAS_SELFIES:
        return None
    sanitized = _sanitize_selfies(selfies_str)
    try:
        smi = sf.decoder(sanitized)
        if smi and HAS_RDKIT and Chem.MolFromSmiles(smi) is not None:
            return smi
        return smi if smi else None
    except Exception:
        return None


def extract_answer_smiles(answer: str) -> Optional[str]:
    smi = extract_answer_region(answer)
    if smi:
        smi = smi.strip().strip("\n").strip()
        if not (0 < len(smi) < 1000):
            return None
        if _is_selfies(smi):
            converted = _selfies_to_smiles(smi)
            if converted is not None:
                return converted
        return smi
    return None


def parse_mol(smi: str):
    if not HAS_RDKIT or not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def canonical(smi: str) -> Optional[str]:
    if not HAS_RDKIT or not smi:
        return None
    mol = parse_mol(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def tanimoto(smi1: str, smi2: str) -> Optional[float]:
    if not HAS_RDKIT:
        return None
    m1, m2 = parse_mol(smi1), parse_mol(smi2)
    if m1 is None or m2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)


def mol_properties(mol) -> Dict:
    if mol is None:
        return {}
    props = {}
    try:
        props["formula"] = rdMolDescriptors.CalcMolFormula(mol)
        props["mw"] = round(Descriptors.MolWt(mol), 2)
        props["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
        props["num_rings"] = rdMolDescriptors.CalcNumRings(mol)
        props["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        props["num_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        props["num_hba"] = rdMolDescriptors.CalcNumHBA(mol)
        props["num_hbd"] = rdMolDescriptors.CalcNumHBD(mol)
        atom_counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
        props["atom_counts"] = dict(atom_counts)
        props["num_carbons"] = atom_counts.get("C", 0)
        props["has_N"] = atom_counts.get("N", 0) > 0
        props["has_O"] = atom_counts.get("O", 0) > 0
        props["has_S"] = atom_counts.get("S", 0) > 0
        props["has_P"] = atom_counts.get("P", 0) > 0
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        props["num_chiral"] = len(chiral)
        ring_info = mol.GetRingInfo()
        props["ring_sizes"] = sorted(len(r) for r in ring_info.AtomRings())
    except Exception:
        pass
    return props


def separate_structural_desc(description: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", description)
    structural = []
    for s in sentences:
        sl = s.lower()
        if not any(kw.lower() in sl for kw in BIO_ROLE_KEYWORDS):
            structural.append(s)
    return " ".join(structural) if structural else description


_KW_CACHE: Dict[str, "re.Pattern"] = {}


def _kw_pattern(kw: str) -> "re.Pattern":
    pat = _KW_CACHE.get(kw)
    if pat is None:
        pat = re.compile(r"(?<![a-z])" + re.escape(kw.lower()) + r"(?:e?s)?(?![a-z])")
        _KW_CACHE[kw] = pat
    return pat


def _kw_present(kw: str, tl: str) -> bool:
    """Whole-word keyword match: not flanked by letters, allowing a plural
    s/es. Prevents spurious substring hits like 'nitro' inside 'nitrogen' or
    'ether' inside 'together'."""
    return _kw_pattern(kw).search(tl) is not None


# A group named inside a derivation / transformation clause refers to a PRECURSOR
# or a reaction, not the molecule itself ("derived from an acid", "replacing the
# carboxylic acid", "condensation of ..."), so it must not be scored as a
# fabrication. With exclude_derivation, a group counts only if it has at least one
# occurrence OUTSIDE such a clause (a genuine claim about the molecule).
_DERIV_RE = re.compile(
    r"derived from|deriv(?:es|ed|ing)|obtained (?:from|by)|condens|replac|"
    r"convert|precursor|starting material|reactant|formed (?:by|from)|"
    r"comes from|originat|tautomer|conjugate (?:acid|base)|reacts? with|"
    r"from (?:the )?(?:acid|amine|alcohol|aldehyde|acyl|carboxylic|parent)|"
    r"of (?:the )?(?:acid|parent)")
_DERIV_WINDOW = 60


def _fg_claimed(keywords: List[str], tl: str, exclude_derivation: bool) -> bool:
    if not exclude_derivation:
        return any(_kw_present(kw, tl) for kw in keywords)
    for kw in keywords:
        for m in _kw_pattern(kw).finditer(tl):
            span = tl[max(0, m.start() - _DERIV_WINDOW): m.end() + _DERIV_WINDOW]
            if not _DERIV_RE.search(span):
                return True
    return False


def extract_chemical_entities(text: str, exclude_derivation: bool = False) -> Dict[str, set]:
    tl = normalize(text)
    entities: Dict[str, set] = defaultdict(set)
    for fg_name, (_, keywords) in FUNCTIONAL_GROUP_DB.items():
        if _fg_claimed(keywords, tl, exclude_derivation):
            entities["functional_groups"].add(fg_name)
    for cls_name, cls_info in MOLECULAR_CLASS_HINTS.items():
        if _fg_claimed(cls_info["keywords"], tl, exclude_derivation):
            entities["mol_classes"].add(cls_name)
    element_pats = {
        "nitrogen": r"\bnitrogen\b|\bnitro\b|\bamino\b|\bamine\b",
        "oxygen":   r"\boxygen\b|\bhydroxy\b|\boxide\b",
        "sulfur":   r"\bsulfur\b|\bthio\b|\bmercapto\b",
        "phosphorus": r"\bphospho\b|\bphosphate\b",
        "fluorine": r"\bfluor\b", "chlorine": r"\bchlor\b",
        "bromine":  r"\bbrom\b",
    }
    for elem, pat in element_pats.items():
        if re.search(pat, tl):
            entities["elements"].add(elem)
    return dict(entities)


_SMARTS_CACHE: Dict[str, Any] = {}


def _compiled_fg_patterns():
    """Compile (once) every FUNCTIONAL_GROUP_DB SMARTS; cache for reuse across
    the tens of thousands of molecules diagnosed per run."""
    if not _SMARTS_CACHE:
        for fg_name, (smarts, _) in FUNCTIONAL_GROUP_DB.items():
            pat = Chem.MolFromSmarts(smarts) if HAS_RDKIT else None
            if pat is not None:
                _SMARTS_CACHE[fg_name] = pat
    return _SMARTS_CACHE


def _get_mol_fgs(mol) -> set:
    """Return the set of functional group names present in a molecule."""
    if mol is None:
        return set()
    found = set()
    for fg_name, pat in _compiled_fg_patterns().items():
        if mol.HasSubstructMatch(pat):
            found.add(fg_name)
    return found


def extract_smiles_fragments(text: str) -> List[str]:
    """Extract SMILES-like substrings from reasoning text."""
    pats = [
        r"(?<!\w)([A-Z][a-z]?(?:[\[\]()=#@/\\+\-\d]|[A-Z][a-z]?){3,})(?!\w)",
        r"(?<!\w)((?:[CNOSPFIBcnos][\[\]()=#@/\\+\-\d]*){4,})(?!\w)",
    ]
    frags = []
    for p in pats:
        for m in re.findall(p, text):
            if len(m) >= 4 and any(c in m for c in "CNOcno"):
                if not re.match(r"^[A-Z][a-z]+$", m):
                    frags.append(m)
    return frags


# ============================================================================
# IR — Intrinsic × Reasoning: Self-Contradiction
# ============================================================================

def score_ir_self_contradiction(reasoning: str) -> Tuple[float, Dict]:
    """
    Detect self-contradictions, excessive self-correction, and flip-flopping
    within the reasoning trace.  Purely internal — no external reference.

    Score: 0 (consistent) to 100 (highly contradictory).
    """
    if not reasoning or len(reasoning.strip()) < 20:
        return 0.0, {"reason": "too short to assess"}

    rl = normalize(reasoning)
    details: Dict[str, Any] = {}
    score = 0.0

    correction_patterns = [
        (r"\bwait\b.*\b(?:wrong|incorrect|mistake|no)\b", 5),
        (r"\bactually\b.*\b(?:should be|is not|isn't)\b", 4),
        (r"\blet me reconsider\b", 4),
        (r"\bi was wrong\b|\bi made a mistake\b|\bthat's incorrect\b", 6),
        (r"\bcorrection\b", 3),
        (r"\bon second thought\b", 3),
        (r"\bno,?\s*(?:wait|actually)\b", 4),
    ]
    total_corrections = 0
    for pat, weight in correction_patterns:
        count = len(re.findall(pat, rl))
        total_corrections += count
        score += count * weight
    details["self_corrections"] = total_corrections

    flip_phrases = re.findall(
        r"\b(?:this (?:is|should be)|the (?:smiles|molecule|structure)"
        r" (?:is|should be))\s+([^\n.]{5,50})",
        rl,
    )
    if len(flip_phrases) > 3:
        score += (len(flip_phrases) - 3) * 5
        details["conclusion_attempts"] = len(flip_phrases)

    seen: set = set()
    repeated = 0
    for s in reasoning.split("."):
        s_norm = normalize(s)
        if len(s_norm) > 20:
            if s_norm in seen:
                repeated += 1
            seen.add(s_norm)
    if repeated > 2:
        score += repeated * 3
    details["repeated_sentences"] = repeated

    if len(reasoning) > 10000:
        score += min(20, (len(reasoning) - 10000) / 1000)
        details["excessive_length"] = len(reasoning)

    return round(min(100.0, score), 2), details
# ============================================================================
# IO — Intrinsic × Output: Structural Invalidity
# ============================================================================

def score_io_structural_invalidity(
    description: str, pred_smi: Optional[str], reasoning: Optional[str] = None
) -> Tuple[float, Dict]:
    """
    Check whether the predicted SMILES is chemically valid and whether it
    satisfies basic physicochemical constraints implied by the description
    and the model's own reasoning trace.

    Two sub-signals:
      (a) Description constraint violation: predicted structure lacks FGs or
          class properties stated in the description.
      (b) Reasoning transmission failure: the reasoning correctly identifies
          FGs from the description, but the predicted SMILES doesn't contain
          them (model "knew" but failed to transmit to output).

    This does NOT compare to ground truth — only checks internal consistency.

    Score: 0 (valid and consistent) to 100 (invalid or grossly inconsistent).
    """
    details: Dict[str, Any] = {}

    if not pred_smi:
        return 0.0, {"valid": False, "skipped": True, "reason": "no prediction"}

    mol = parse_mol(pred_smi)
    if mol is None:
        return 0.0, {"valid": False, "skipped": True, "reason": "invalid SMILES",
                     "pred": pred_smi[:100]}

    details["valid"] = True
    details["skipped"] = False
    score = 0.0

    props = mol_properties(mol)
    details["formula"] = props.get("formula", "")

    desc_ents = extract_chemical_entities(
        separate_structural_desc(description)
    )
    desc_classes = desc_ents.get("mol_classes", set())
    desc_fgs = desc_ents.get("functional_groups", set())

    class_violations = []
    for cls_name in desc_classes:
        info = MOLECULAR_CLASS_HINTS.get(cls_name, {})
        if "min_rings" in info:
            if props.get("num_rings", 0) < info["min_rings"]:
                class_violations.append(
                    f"{cls_name}: need >={info['min_rings']} rings,"
                    f" got {props.get('num_rings', 0)}"
                )
        if info.get("requires_N") and not props.get("has_N"):
            class_violations.append(f"{cls_name}: requires N")
        if info.get("requires_O") and not props.get("has_O"):
            class_violations.append(f"{cls_name}: requires O")
        if info.get("requires_P") and not props.get("has_P"):
            class_violations.append(f"{cls_name}: requires P")
        if "min_carbons" in info:
            if props.get("num_carbons", 0) < info["min_carbons"]:
                class_violations.append(
                    f"{cls_name}: need >={info['min_carbons']} C,"
                    f" got {props.get('num_carbons', 0)}"
                )

    if class_violations:
        score += min(40.0, len(class_violations) * 10)
        details["class_violations"] = class_violations

    pred_fgs = _get_mol_fgs(mol)
    missing_fgs = desc_fgs - pred_fgs
    if missing_fgs and desc_fgs:
        ratio = len(missing_fgs) / len(desc_fgs)
        score += ratio * 30
        details["desc_fgs_missing_in_pred"] = list(missing_fgs)

    # --- Reasoning transmission faithfulness (from H2) ---
    if reasoning and len(reasoning.strip()) > 20:
        trace_ents = extract_chemical_entities(reasoning, exclude_derivation=True)
        trace_fgs = trace_ents.get("functional_groups", set())
        correctly_identified = desc_fgs & trace_fgs
        if correctly_identified:
            unfaithful_fgs = [
                fg for fg in correctly_identified if fg not in pred_fgs
            ]
            if unfaithful_fgs:
                unfaithful_ratio = len(unfaithful_fgs) / len(correctly_identified)
                score += unfaithful_ratio * 25
                details["unfaithful_fgs"] = unfaithful_fgs
                details["unfaithfulness_ratio"] = round(unfaithful_ratio, 3)

        # SMILES fragment consistency: check if sub-structures mentioned
        # in reasoning are present in the final prediction
        smiles_frags = extract_smiles_fragments(reasoning)
        if smiles_frags:
            frag_consistent = 0
            frag_total = 0
            for frag in smiles_frags:
                frag_mol = parse_mol(frag)
                if frag_mol is not None:
                    frag_total += 1
                    try:
                        if mol.HasSubstructMatch(frag_mol):
                            frag_consistent += 1
                    except Exception:
                        pass
            if frag_total > 0:
                frag_ratio = 1.0 - (frag_consistent / frag_total)
                if frag_ratio > 0:
                    score += frag_ratio * 15
                    details["smiles_frag_inconsistency"] = round(frag_ratio, 3)
                    details["smiles_frags_checked"] = frag_total

    return round(min(100.0, score), 2), details
# ============================================================================
# ER — Extrinsic × Reasoning: Factual Fabrication
# ============================================================================

def score_er_factual_fabrication(
    description: str,
    reasoning: str,
    pred_smi: Optional[str],
    gt_smi: str,
) -> Tuple[float, Dict]:
    """
    Check whether the reasoning asserts chemical facts that are *objectively
    wrong* when verified against the predicted molecule or ground truth.

    Key difference from old H1: we only penalise claims that can be falsified
    by RDKit.  If the reasoning mentions a functional group that IS present in
    either the predicted molecule or the GT molecule, that is NOT fabrication
    — the model may be drawing on legitimate chemical knowledge beyond what
    the description explicitly states.

    Score: 0 (no fabrication) to 100 (heavy fabrication).
    """
    _empty = {"claimed_fgs": [], "verified_fgs": [], "fabricated_fgs": []}
    if not reasoning:
        return 0.0, {"reason": "no reasoning", **_empty}

    trace_ents = extract_chemical_entities(reasoning, exclude_derivation=True)
    trace_fgs = trace_ents.get("functional_groups", set())
    trace_classes = trace_ents.get("mol_classes", set())

    if not trace_fgs and not trace_classes:
        return 0.0, {"reason": "no verifiable claims in reasoning", **_empty}
    details: Dict[str, Any] = {}

    pred_mol = parse_mol(pred_smi) if pred_smi else None
    gt_mol = parse_mol(gt_smi) if gt_smi else None

    pred_fgs = _get_mol_fgs(pred_mol)
    gt_fgs = _get_mol_fgs(gt_mol)
    # A claim is NOT fabrication if the functional group is already named in the
    # input description: the model is grounding on the prompt, not inventing it.
    desc_ents = extract_chemical_entities(separate_structural_desc(description))
    desc_fgs = desc_ents.get("functional_groups", set())
    real_fgs = pred_fgs | gt_fgs | desc_fgs

    fabricated_fgs = []
    verified_fgs = []
    for fg in trace_fgs:
        if fg in real_fgs:
            verified_fgs.append(fg)
        else:
            fabricated_fgs.append(fg)

    checks = max(len(trace_fgs), 1)
    penalties = len(fabricated_fgs) * 5.0
    details["claimed_fgs"] = list(trace_fgs)
    details["verified_fgs"] = verified_fgs
    details["fabricated_fgs"] = fabricated_fgs

    desc_classes = desc_ents.get("mol_classes", set())
    wrong_classes = []
    for cls in trace_classes:
        if cls in desc_classes:
            continue
        info = MOLECULAR_CLASS_HINTS.get(cls, {})
        mol_to_check = pred_mol or gt_mol
        if mol_to_check:
            props = mol_properties(mol_to_check)
            plausible = True
            if "min_rings" in info:
                if props.get("num_rings", 0) < info["min_rings"]:
                    plausible = False
            if info.get("requires_N") and not props.get("has_N"):
                plausible = False
            if not plausible:
                wrong_classes.append(cls)

    if wrong_classes:
        penalties += len(wrong_classes) * 8.0
        details["wrong_classes"] = wrong_classes
        checks += len(trace_classes)

    score = min(100.0, (penalties / max(checks, 1)) * 15)
    details["penalties"] = round(penalties, 2)
    details["checks"] = checks
    return round(score, 2), details
# ============================================================================
# EO — Extrinsic × Output: Phantom Structure
# ============================================================================

def score_eo_phantom_structure(
    pred_smi: Optional[str], gt_smi: str
) -> Tuple[float, Dict]:
    """
    The predicted molecule is structurally valid but substantially different
    from the ground truth.

    Score: 0 (correct or very close) to 100 (completely different molecule).
    """
    details: Dict[str, Any] = {}
    if not pred_smi:
        return 0.0, {"skipped": True, "reason": "no prediction"}

    pred_mol = parse_mol(pred_smi)
    if pred_mol is None:
        return 0.0, {"skipped": True, "reason": "invalid SMILES"}

    gt_can = canonical(gt_smi)
    pred_can = canonical(pred_smi)
    if gt_can and pred_can and gt_can == pred_can:
        return 0.0, {"exact_match": True}

    sim = tanimoto(gt_smi, pred_smi)
    details["tanimoto"] = sim

    if sim is not None:
        score = max(0.0, (1.0 - sim) * 100)
    else:
        score = 80.0

    if HAS_RDKIT:
        gt_mol = parse_mol(gt_smi)
        if gt_mol and pred_mol:
            gt_p = mol_properties(gt_mol)
            pred_p = mol_properties(pred_mol)
            if gt_p.get("formula") != pred_p.get("formula"):
                details["formula_mismatch"] = {
                    "gt": gt_p.get("formula"),
                    "pred": pred_p.get("formula"),
                }
            if gt_p.get("num_rings", 0) != pred_p.get("num_rings", 0):
                details["ring_count_mismatch"] = {
                    "gt": gt_p.get("num_rings"),
                    "pred": pred_p.get("num_rings"),
                }
            gt_fgs = _get_mol_fgs(gt_mol)
            pred_fgs = _get_mol_fgs(pred_mol)
            details["missing_fgs"] = list(gt_fgs - pred_fgs)
            details["extra_fgs"] = list(pred_fgs - gt_fgs)

    return round(score, 2), details
# ============================================================================
# Aggregate Hallucination Diagnosis
# ============================================================================

DIMENSION_WEIGHTS = {
    "IR": 0.15,
    "IO": 0.25,
    "ER": 0.25,
    "EO": 0.35,
}


def diagnose_single(
    description: str, answer: str, gt_smi: str, verbose: bool = False
) -> Dict[str, Any]:
    """Run full 2x2 hallucination diagnosis on a single sample."""
    reasoning = extract_think(answer)
    pred_smi = extract_answer_smiles(answer)

    if reasoning is None:
        reasoning = answer
    if pred_smi is None:
        smi_match = re.search(
            r"[A-Z][A-Za-z0-9@+\-\[\]\\/()\=#]{5,}", answer
        )
        if smi_match:
            pred_smi = smi_match.group(0)

    is_exact = False
    if pred_smi and gt_smi:
        pred_can = canonical(pred_smi)
        gt_can = canonical(gt_smi)
        if pred_can and gt_can:
            is_exact = pred_can == gt_can

    result: Dict[str, Any] = {
        "pred_smiles": pred_smi,
        "pred_valid": parse_mol(pred_smi) is not None if pred_smi else False,
        "exact_match": is_exact,
        "tanimoto": tanimoto(gt_smi, pred_smi) if pred_smi else None,
        "reasoning_length": len(reasoning) if reasoning else 0,
    }

    if not pred_smi:
        result["validity_issue"] = "no_prediction"
    elif parse_mol(pred_smi) is None:
        result["validity_issue"] = "invalid_smiles"
    else:
        result["validity_issue"] = None

    ir_score, ir_det = score_ir_self_contradiction(reasoning or "")
    io_score, io_det = score_io_structural_invalidity(description, pred_smi, reasoning=reasoning)
    er_score, er_det = score_er_factual_fabrication(
        description, reasoning or "", pred_smi, gt_smi
    )
    eo_score, eo_det = score_eo_phantom_structure(pred_smi, gt_smi)

    result["hallucination_scores"] = {
        "IR_self_contradiction": ir_score,
        "IO_structural_invalidity": io_score,
        "ER_factual_fabrication": er_score,
        "EO_phantom_structure": eo_score,
    }

    # Dynamic weighting: when SMILES is invalid, IO and EO are skipped (0),
    # so only use IR + ER with renormalized weights
    io_skipped = io_det.get("skipped", False)
    eo_skipped = eo_det.get("skipped", False)

    if io_skipped or eo_skipped:
        active_weights = {"IR": DIMENSION_WEIGHTS["IR"], "ER": DIMENSION_WEIGHTS["ER"]}
        if not io_skipped:
            active_weights["IO"] = DIMENSION_WEIGHTS["IO"]
        if not eo_skipped:
            active_weights["EO"] = DIMENSION_WEIGHTS["EO"]
        total_w = sum(active_weights.values())
        overall = sum(
            result["hallucination_scores"][f"{k}_{v}"] * (w / total_w)
            for (k, w), v in zip(
                active_weights.items(),
                [{"IR": "self_contradiction", "IO": "structural_invalidity",
                  "ER": "factual_fabrication", "EO": "phantom_structure"}[k]
                 for k in active_weights],
            )
        )
    else:
        overall = sum(
            result["hallucination_scores"][f"{k}_{v}"] * w
            for (k, w), v in zip(
                DIMENSION_WEIGHTS.items(),
                ["self_contradiction", "structural_invalidity",
                 "factual_fabrication", "phantom_structure"],
            )
        )
    result["overall_hallucination_score"] = round(overall, 2)

    if verbose:
        result["details"] = {
            "IR": ir_det, "IO": io_det,
            "ER": er_det, "EO": eo_det,
        }

    return result
# ============================================================================
# Data Loading
# ============================================================================

def load_chebi20_test(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            if len(parts) >= 3:
                data.append({
                    "cid": parts[0],
                    "gt_smiles": parts[1],
                    "description": parts[2],
                })
    return data


def load_model_output(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    output_map = {}
    for item in data:
        key = str(item.get("id", item.get("cid", "")))
        output_map[key] = item
    return output_map
# ============================================================================
# Batch Evaluation & Reporting
# ============================================================================

def evaluate_model(
    model_name: str,
    model_output_path: str,
    test_data: List[Dict],
    output_dir: str,
    verbose: bool = False,
    max_samples: Optional[int] = None,
) -> Dict:
    print(f"\n{'='*70}")
    print(f"  Evaluating: {model_name}")
    print(f"  Output: {model_output_path}")
    print(f"{'='*70}")

    model_output = load_model_output(model_output_path)
    print(f"  Loaded {len(model_output)} model outputs")

    results = []
    all_scores: Dict[str, List[float]] = defaultdict(list)
    matched = 0
    samples = test_data[:max_samples] if max_samples else test_data

    for sample in tqdm(samples, desc=f"  Diagnosing {model_name}"):
        cid = sample["cid"]
        gt_smi = sample["gt_smiles"]
        desc = sample["description"]

        out = model_output.get(cid)
        if out is None:
            continue
        matched += 1

        answer = out.get("answer", "")
        if not answer:
            continue

        diag = diagnose_single(desc, answer, gt_smi, verbose=verbose)
        diag["cid"] = cid
        diag["model"] = model_name
        results.append(diag)

        for k, v in diag["hallucination_scores"].items():
            all_scores[k].append(v)
        all_scores["overall"].append(diag["overall_hallucination_score"])

    print(f"  Matched {matched} samples with model outputs")

    summary: Dict[str, Any] = {
        "model": model_name,
        "total_samples": len(samples),
        "matched_samples": matched,
        "evaluated_samples": len(results),
    }

    if results:
        valid_count = sum(1 for r in results if r["pred_valid"])
        exact_count = sum(1 for r in results if r["exact_match"])
        summary["validity_rate"] = round(
            valid_count / len(results) * 100, 2
        )
        summary["exact_match_rate"] = round(
            exact_count / len(results) * 100, 2
        )
        tani_vals = [
            r["tanimoto"] for r in results if r["tanimoto"] is not None
        ]
        if tani_vals:
            summary["avg_tanimoto"] = round(
                sum(tani_vals) / len(tani_vals), 4
            )
        summary["avg_reasoning_length"] = round(
            sum(r["reasoning_length"] for r in results) / len(results), 1
        )

        summary["hallucination_scores"] = {}
        for k, vals in all_scores.items():
            summary["hallucination_scores"][k] = {
                "mean": round(sum(vals) / len(vals), 2),
                "median": round(sorted(vals)[len(vals) // 2], 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
                "std": round(
                    (sum((v - sum(vals) / len(vals)) ** 2
                         for v in vals) / len(vals)) ** 0.5, 2
                ),
            }

        bins = {"low": 0, "moderate": 0, "high": 0}
        for r in results:
            s = r["overall_hallucination_score"]
            if s < 30:
                bins["low"] += 1
            elif s < 60:
                bins["moderate"] += 1
            else:
                bins["high"] += 1
        summary["hallucination_distribution"] = {
            k: {"count": v, "ratio": round(v / len(results) * 100, 2)}
            for k, v in bins.items()
        }

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace(" ", "_")

    details_path = os.path.join(
        output_dir, f"{safe_name}_hallucination_details.jsonl"
    )
    with open(details_path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    summary_path = os.path.join(
        output_dir, f"{safe_name}_hallucination_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _print_summary(model_name, summary)
    return summary


def _print_summary(model_name: str, summary: Dict) -> None:
    print(f"\n  --- Summary for {model_name} ---")
    print(f"  Evaluated: {summary.get('evaluated_samples', 0)} samples")
    if "validity_rate" in summary:
        print(f"  Validity: {summary['validity_rate']}%")
        print(f"  Exact Match: {summary['exact_match_rate']}%")
        if "avg_tanimoto" in summary:
            print(f"  Avg Tanimoto: {summary['avg_tanimoto']}")
        print(f"  Hallucination Scores (mean):")
        for k, v in summary.get("hallucination_scores", {}).items():
            print(f"    {k}: {v['mean']} (std={v['std']})")
        print(f"  Distribution:")
        for k, v in summary.get("hallucination_distribution", {}).items():
            print(f"    {k}: {v['count']} ({v['ratio']}%)")


def generate_comparative_report(
    summaries: List[Dict], output_dir: str
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("COMPARATIVE HALLUCINATION DIAGNOSIS REPORT (v2: 2x2)")
    lines.append(f"Models evaluated: {len(summaries)}")
    lines.append("=" * 80)

    header = f"{'Metric':<35}"
    for s in summaries:
        header += f" {s['model']:<18}"
    lines.append(header)
    lines.append("-" * (35 + 19 * len(summaries)))

    metrics = [
        ("Evaluated Samples", "evaluated_samples"),
        ("Validity Rate (%)", "validity_rate"),
        ("Exact Match Rate (%)", "exact_match_rate"),
        ("Avg Tanimoto Similarity", "avg_tanimoto"),
    ]
    for label, key in metrics:
        row = f"{label:<35}"
        for s in summaries:
            val = s.get(key, "N/A")
            if isinstance(val, float):
                row += f" {val:<18.2f}"
            else:
                row += f" {str(val):<18}"
        lines.append(row)

    lines.append("")
    lines.append("--- Hallucination Scores (mean) ---")
    h_keys = [
        "IR_self_contradiction",
        "IO_structural_invalidity",
        "ER_factual_fabrication",
        "EO_phantom_structure",
        "overall",
    ]
    for hk in h_keys:
        row = f"{hk:<35}"
        for s in summaries:
            hs = s.get("hallucination_scores", {}).get(hk, {})
            val = hs.get("mean", "N/A")
            if isinstance(val, (int, float)):
                row += f" {val:<18.2f}"
            else:
                row += f" {str(val):<18}"
        lines.append(row)

    lines.append("")
    lines.append("--- Distribution ---")
    for level in ["low", "moderate", "high"]:
        row = f"{level:<35}"
        for s in summaries:
            dist = s.get("hallucination_distribution", {}).get(level, {})
            ratio = dist.get("ratio", "N/A")
            if isinstance(ratio, (int, float)):
                row += f" {ratio:>5.1f}%{'':<12}"
            else:
                row += f" {str(ratio):<18}"
        lines.append(row)

    report = "\n".join(lines)
    report_path = os.path.join(
        output_dir, "comparative_hallucination_report.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n{report}")
    return report
# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reasoning Hallucination Diagnosis v2 (2x2 taxonomy)"
    )
    parser.add_argument(
        "--chebi_test", type=str,
        default=os.path.join(BASE_DIR, "data/chebi-20/test.txt"),
    )
    parser.add_argument("--model_output", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="unknown")
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(BASE_DIR, "results"),
    )
    parser.add_argument("--batch_config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading ChEBI-20 test data ...")
    test_data = load_chebi20_test(args.chebi_test)
    print(f"Loaded {len(test_data)} test samples")

    summaries = []

    if args.batch_config:
        with open(args.batch_config, "r") as f:
            configs = json.load(f)
        for cfg in configs:
            s = evaluate_model(
                model_name=cfg["name"],
                model_output_path=cfg["output"],
                test_data=test_data,
                output_dir=args.output_dir,
                verbose=args.verbose,
                max_samples=args.max_samples,
            )
            summaries.append(s)
    elif args.model_output:
        s = evaluate_model(
            model_name=args.model_name,
            model_output_path=args.model_output,
            test_data=test_data,
            output_dir=args.output_dir,
            verbose=args.verbose,
            max_samples=args.max_samples,
        )
        summaries.append(s)
    else:
        print("No model output specified. Use --model_output or --batch_config.")
        return

    if len(summaries) > 1:
        generate_comparative_report(summaries, args.output_dir)

    all_path = os.path.join(args.output_dir, "all_summaries.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nAll summaries saved to: {all_path}")


if __name__ == "__main__":
    main()
