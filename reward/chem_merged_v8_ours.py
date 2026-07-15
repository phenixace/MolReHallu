"""merged_v8 reward: task-correct accuracy + anti-hallucination + grounded(count x precision).

Accuracy is the TRUE per-task metric (not the broken proxies):
  cap2mol / retrosynthesis -> exact match (1/0)
  mol2cap                  -> caption similarity (0..1, token Jaccard)
  s2_*                     -> official S2-TOMG success (1/0), via s2_success
FG categories are already de-duplicated by the detector (extract_chemical_entities
returns a set), so spamming the same group cannot inflate ER or grounded.

overall = 0.1*format + 0.4*accuracy + 0.4*(1 - halluc/100) + 0.2*grounded

Set env COUPLED=1 for the decoupling-aware variant: the accuracy reward is paid
ONLY when the trace is clean (ER == 0). A correct answer reached through a
fabricating trace earns no accuracy reward; the clean/grounded CoT reward is paid
on the trace regardless.
"""
import json
import os
import re
import sys
from typing import Dict, List, Optional, Union

_MAIN = os.environ.get("MOLLM_PROJECT_DIR", ".")
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from diagnose_multitask import diagnose_one  # noqa: E402
from diagnose_hallucination import GENERIC_FG_NAMES  # noqa: E402
from s2_success import s2_success  # noqa: E402

_FORMAT_PAT = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_ANSWER_PAT = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_INPUT_MARKER = {
    "cap2mol": "Description:", "mol2cap": "SMILES:",
    "retrosynthesis": "Product:", "reaction_prediction": "Reactants:",
}
W_FMT, W_ACC, W_ANTI, W_GROUND = 0.1, 0.4, 0.4, 0.2
GROUNDED_CAP = 5
COUPLED = os.environ.get("COUPLED", "0") == "1"
# Thinking-length bonus (RL-from-base / "zero" runs only; 0 => off, so v8/coupled
# are unchanged). Rewards LONGER reasoning inside <think>...</think> so a model
# initialized from the base (no SFT) learns to actually reason. Anti-hacking: the
# bonus SATURATES at LEN_TARGET chars (no benefit to rambling), is paid ONLY on a
# valid <think>/<answer> format, and coexists with the anti-hallucination + grounded
# terms that penalize padding the CoT with fake chemistry.
LEN_BONUS = float(os.environ.get("LEN_BONUS", "0"))     # weight; try ~0.1
LEN_TARGET = float(os.environ.get("LEN_TARGET", "1500"))  # chars of think content to saturate
_THINK_PAT = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _think_len_term(ans: str) -> float:
    """Length of the CoT content, capped and normalized to [0,1]."""
    m = _THINK_PAT.search(ans or "")
    if not m:
        return 0.0
    return min(len(m.group(1).strip()), LEN_TARGET) / LEN_TARGET




def _extract_input(task: str, prompt: str) -> str:
    marker = "Instruction:" if task.startswith("s2_") else _INPUT_MARKER.get(task)
    if marker and marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def _answer_smiles(ans: str) -> str:
    m = _ANSWER_PAT.search(ans)
    return (m.group(1).strip() if m else ans.strip())


def _accuracy_signal(task: str, diag: Dict, metadata: Dict, pred: str, instruction: str) -> float:
    if task.startswith("s2_"):
        return float(s2_success(task, pred, metadata, instruction))
    if task == "mol2cap":
        return float(diag.get("caption_jaccard") or 0.0)
    # cap2mol / retrosynthesis: exact match only (no similarity fallback)
    return 1.0 if diag.get("exact_match") else 0.0


def _er_count(diag: Dict):
    """(# distinct verified-specific FGs, # distinct fabricated FGs). Already de-duped."""
    er = diag.get("details", {}).get("ER", {})
    verified = er.get("verified_fgs")
    if verified is None:
        claimed = er.get("claimed_fgs", []) or []
        fabricated = set(er.get("fabricated_fgs", []) or [])
        verified = [fg for fg in claimed if fg not in fabricated]
    ver_specific = {fg for fg in verified if fg not in GENERIC_FG_NAMES}
    fabricated = {fg for fg in (er.get("fabricated_fgs", []) or []) if fg not in GENERIC_FG_NAMES}
    return len(ver_specific), len(fabricated)


def _grounded_term(diag: Dict) -> float:
    """count x precision, in [0,2]: reward saying MORE verified groups AND being precise."""
    n_ver, n_fab = _er_count(diag)
    if n_ver == 0:
        return 0.0
    precision = n_ver / (n_ver + n_fab)               # in (0,1]
    count = min(n_ver, GROUNDED_CAP) / GROUNDED_CAP    # in (0,1]
    return 2.0 * count * precision


def compute_score(
    predicts: List[str],
    ground_truths: List[Union[str, dict]],
    tasks: Optional[List[str]] = None,
    prompts: Optional[List[str]] = None,
    **kwargs,
) -> List[Dict[str, float]]:
    n = len(predicts)
    tasks = tasks or ["cap2mol"] * n
    prompts = prompts or [""] * n
    scores: List[Dict[str, float]] = []
    for i in range(n):
        task = tasks[i]
        ans = predicts[i] or ""
        gt_raw = ground_truths[i]
        metadata: Dict = {}
        if task.startswith("s2_"):
            try:
                payload = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
                metadata = payload.get("metadata", {})
                gt = payload.get("gt", "")
                task = payload.get("task", task)
            except Exception:
                gt = ""
        else:
            gt = gt_raw if isinstance(gt_raw, str) else str(gt_raw)

        inp = _extract_input(task, prompts[i] or "")
        try:
            diag = diagnose_one(
                task, {"answer": ans, "question": inp, "gt": gt, "metadata": metadata},
                verbose=True,
            )
            halluc = float(diag.get("overall_hallucination_score", 50.0))
            n_ver, n_fab = _er_count(diag)
            acc = _accuracy_signal(task, diag, metadata, _answer_smiles(ans), inp)
            grounded = _grounded_term(diag)
        except Exception:
            halluc, n_fab, acc, grounded = 100.0, 99, 0.0, 0.0

        fmt = 1.0 if _FORMAT_PAT.search(ans.strip()) else 0.0
        anti = 1.0 - halluc / 100.0
        # Decoupling-aware: pay accuracy only when the trace is clean (no fabrication).
        acc_paid = acc if (not COUPLED or n_fab == 0) else 0.0
        # Thinking-length bonus (off unless LEN_BONUS>0); paid only on valid format.
        length = _think_len_term(ans) if fmt else 0.0
        overall = (W_FMT * fmt + W_ACC * acc_paid + W_ANTI * anti + W_GROUND * grounded
                   + LEN_BONUS * length)
        scores.append({
            "overall": float(overall), "format": float(fmt),
            "accuracy": float(acc_paid), "accuracy_raw": float(acc),
            "anti_hallucination": float(anti), "grounded": float(grounded),
            "length": float(length),
        })
    return scores
