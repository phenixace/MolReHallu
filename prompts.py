#!/usr/bin/env python3
"""
Prompt templates for all supported tasks.

Every prompt must use the `{input}` placeholder, which is filled with the
`input` field from the data loader (see data_loaders.py).

The reasoning-format envelope (think/answer tags) is shared so that all tasks
produce uniformly parsable output for hallucination diagnosis and semantic
entropy. The instruction text is task-specific.
"""

from typing import Dict, List

SYSTEM_PROMPT = "You are an expert chemist."

REASONING_FOOTER = (
    "You should put your reasoning in <think> </think> tags. "
    "The final answer MUST BE put in <answer> </answer> tags. "
    "Please strictly follow the format."
)

# ----------------------------------------------------------------------------
# ChEBI-20
# ----------------------------------------------------------------------------

PROMPT_CAP2MOL = (
    "Your task is to solve the given problem step by step. "
    + REASONING_FOOTER + "\n"
    "Now predict the SMILES representation for the following "
    "molecular design requirement:\nDescription: {input}"
)

PROMPT_MOL2CAP = (
    "Your task is to solve the given problem step by step. "
    + REASONING_FOOTER.replace("<answer> </answer>",
                               "<answer> The molecule is ... </answer>") + "\n"
    "Now describe the following molecule based on its SMILES "
    "representation:\nSMILES: {input}"
)

# ----------------------------------------------------------------------------
# MoleculeNet classification — Yes/No prompts
# ----------------------------------------------------------------------------

PROMPT_BACE = (
    "Your task is to solve the given problem step by step. "
    "Given the SMILES string of a molecule, predict whether it can inhibit (Yes) "
    "the BACE1 enzyme or cannot inhibit (No) BACE1. Please answer with only "
    "Yes or No. " + REASONING_FOOTER + "\n"
    "Now predict the BACE1 inhibition potential (Yes or No) for the following "
    "molecule:\nSMILES: {input}"
)

PROMPT_BBBP = (
    "Your task is to solve the given problem step by step. "
    "Given the SMILES string of a molecule, predict whether it penetrates the "
    "blood-brain barrier (Yes) or not (No). Please answer with only Yes or No. "
    + REASONING_FOOTER + "\n"
    "Now predict the BBB penetration (Yes or No) for the following molecule:\n"
    "SMILES: {input}"
)

PROMPT_HIV = (
    "Your task is to solve the given problem step by step. "
    "Given the SMILES string of a molecule, predict its ability to inhibit "
    "HIV replication. Answer Yes or No. " + REASONING_FOOTER + "\n"
    "Now predict the HIV replication inhibition (Yes or No) for the following "
    "molecule:\nSMILES: {input}"
)

PROMPT_TOX21 = (
    "Your task is to solve the given problem step by step. "
    "Given the SMILES string of a molecule, predict whether it is toxic (Yes) "
    "in any of the Tox21 endpoints (NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, "
    "NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53) "
    "or not toxic (No). " + REASONING_FOOTER + "\n"
    "Now predict the toxicity (Yes or No) for the following molecule:\n"
    "SMILES: {input}"
)

PROMPT_CLINTOX = (
    "Your task is to solve the given problem step by step. "
    "Given the SMILES string of a molecule, predict whether it has clinical "
    "trial toxicity (Yes) or not (No). " + REASONING_FOOTER + "\n"
    "Now predict the clinical toxicity (Yes or No) for the following "
    "molecule:\nSMILES: {input}"
)

# ----------------------------------------------------------------------------
# Retrosynthesis
# ----------------------------------------------------------------------------

PROMPT_RETROSYNTHESIS = (
    "Your task is to solve the given problem step by step. "
    + REASONING_FOOTER + "\n"
    "Note: If multiple reactants are predicted, they MUST be separated by a "
    "period `.` instead of commas.\n"
    "Now predict the reactants for the following product:\nProduct: {input}"
)

PROMPT_REACTION_PREDICTION = (
    "Your task is to solve the given problem step by step. "
    + REASONING_FOOTER + "\n"
    "Now predict the product of the following reaction:\nReactants: {input}"
)

# ----------------------------------------------------------------------------
# S2-TOMG-Bench
# ----------------------------------------------------------------------------
# Each instruction already contains the full chemistry constraint (atom count,
# functional group requirement, edit specification, etc.). We wrap it with the
# same reasoning envelope so the same parsers can be used.

PROMPT_S2_MOLCUSTOM = (
    "Your task is to solve the given problem step by step. "
    "Generate a SMILES string that exactly satisfies the composition "
    "constraints in the instruction. " + REASONING_FOOTER + "\n"
    "Instruction: {input}"
)

PROMPT_S2_MOLEDIT = (
    "Your task is to solve the given problem step by step. "
    "Output the SMILES of the edited molecule that exactly satisfies the "
    "edit instruction. " + REASONING_FOOTER + "\n"
    "Instruction: {input}"
)

PROMPT_S2_MOLOPT = (
    "Your task is to solve the given problem step by step. "
    "Output the SMILES of a molecule that satisfies the optimisation "
    "instruction. " + REASONING_FOOTER + "\n"
    "Instruction: {input}"
)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

PROMPTS: Dict[str, str] = {
    "cap2mol":         PROMPT_CAP2MOL,
    "mol2cap":         PROMPT_MOL2CAP,
    "bace":            PROMPT_BACE,
    "bbbp":            PROMPT_BBBP,
    "hiv":             PROMPT_HIV,
    "tox21":           PROMPT_TOX21,
    "clintox":         PROMPT_CLINTOX,
    "retrosynthesis":  PROMPT_RETROSYNTHESIS,
    "reaction_prediction": PROMPT_REACTION_PREDICTION,
    "s2_MolCustom_AtomNum":         PROMPT_S2_MOLCUSTOM,
    "s2_MolCustom_BondNum":         PROMPT_S2_MOLCUSTOM,
    "s2_MolCustom_FunctionalGroup": PROMPT_S2_MOLCUSTOM,
    "s2_MolEdit_AddComponent":      PROMPT_S2_MOLEDIT,
    "s2_MolEdit_DelComponent":      PROMPT_S2_MOLEDIT,
    "s2_MolEdit_SubComponent":      PROMPT_S2_MOLEDIT,
    "s2_MolOpt_LogP":               PROMPT_S2_MOLOPT,
    "s2_MolOpt_MR":                 PROMPT_S2_MOLOPT,
    "s2_MolOpt_QED":                PROMPT_S2_MOLOPT,
}


def build_messages(task: str, input_text: str, system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    if task not in PROMPTS:
        raise ValueError(f"No prompt for task: {task}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": PROMPTS[task].format(input=input_text)},
    ]


if __name__ == "__main__":
    from data_loaders import load_task, ALL_TASKS
    for t in ALL_TASKS:
        try:
            s = load_task(t, max_samples=1)[0]
            msgs = build_messages(t, s["input"])
            print(f"\n=== {t} ===")
            print(msgs[1]["content"][:300].replace("\n", " | "))
        except Exception as e:
            print(f"{t}: ERROR {e}")
