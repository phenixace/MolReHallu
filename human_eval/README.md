# Human validation of the 2×2 diagnoser

Validate the automatic hallucination detector against expert judgment with a
**blind forced-choice** task.

**Design**: 100 prompts per task (cap2mol, mol2cap, retrosynthesis,
S2 functional-group) = 400 prompts. Each prompt is shown with **all models'**
answers side by side. The models are **anonymized and shuffled** per prompt
(labelled Model A/B/C…), so the rater is not biased by model identity or order.
Edit `MODELS`/`TASKS`/`PROMPTS_PER_TASK` in `build_annotation_set.py` to resize.

**Task**: read each model's reasoning and pick the **one whose reasoning
hallucinates the least**. As hints, an automatic checker marks claims it thinks
are **fabricated (red)** or **verified present (green)** — these are suggestions,
not the answer; use your own judgment. Model names and detector scores are hidden.

## Run locally (no install needed)

```bash
cd human_eval
python -m http.server 8000
# open http://localhost:8000 in a browser
```

The page loads `samples.json`. For each prompt: read the anonymized panels, click
**pick as least hallucinatory** on one. Progress auto-saves in the browser; click
**Download** to get `annotations.json` and send it back. Two annotators can each
run independently (put your name in the box) for an inter-rater check.

## Score it

```bash
python human_eval/arena.py human_eval/annotations.json        # arena ranking + agreement
python human_eval/compute_kappa.py human_eval/annotations.json # agreement only
```
`arena.py` reports the **arena leaderboard** (votes, win-rate, Luce/Plackett-Luce
strengths and an Elo-like scale — the right model for top-1-from-N choices with
variable set sizes) plus detector agreement: how often the human's least-halluc
pick matches the detector's argmin (overall and ER) vs the random baseline — the
key reliability number for the paper.

## Annotation rounds

The trained models (`EXTRA` in `build_annotation_set.py`) are shown as **additive
panels**: they appear alongside the core models but are kept out of the shared-prompt
intersection, so the **uids stay identical** across rounds and earlier annotations
remain aligned. Re-annotate and Download a fresh `annotations.json` to score a round
that includes them.

## Regenerate the sample set

```bash
python human_eval/build_annotation_set.py   # rewrites samples.json
```
Source data: reasoning traces in `../se_results/<model>/<task>/output.json` (or
`../results/...` for collaborator runs), auto-diagnoses in
`../results/<model>/<task>/*hallucination_details.jsonl`. Highlights use the
detector's `fabricated_fgs` (red) and `verified_fgs` (green).

## What the shipped annotation files contain

All annotation records ship unmodified, including the identifiers used while the
study was run. Two things about them are worth stating plainly, because neither
is what the released figures use.

**Internal training codenames.** `claims_key.json` keys models as
`Chem-R-v8`, `Chem-R-v8-coupled`, `ChemDFM-R-14B` and
`DeepSeek-R1-Distill-Llama-8B`. These are the internal training names; the
release display names are, respectively, `+process`, `Chem-R-Faithful`,
`ChemDFM-R` and `DeepSeek-R1-Distill` (the same mapping as `data/raw/README.md`).
The records were left as collected rather than renamed after the fact, so that
what is published is exactly what the annotator scored.

**The arena set spans more models than the paper reports.** `samples.json` and
`annotations*.json` come from a blind forced-choice round that also included
`Mol-R1`, `MolReasoner`, `Chem-R-Merged` and an earlier `Chem-R-Faithful`
checkpoint. None of these are part of the released study: the earlier
`Chem-R-Faithful` is a superseded checkpoint that happens to share the name with
the released model, and the other three were dropped before the analyses reported
here. They remain in the raw records because removing panels from a forced-choice
comparison would change what the annotator was actually choosing between. No
figure or number in the paper is computed from them.

**Annotator field.** `annotator` holds the initials of the chemist who scored the
set. The per-claim audit was scored by one annotator; `annotations.json` and
`annotations-2.json` are two passes over the same 400 prompts.

### Reproducing the paper's extraction-precision number

```bash
cd human_eval && python score_claims.py
# claims scored: 300 ...
# overall extraction precision = 0.973
```

This is the $97.3\%$ pooled extraction precision quoted in the Limitations
section. `claims_key.json` and `claim_annotations_RL.json` are a matched pair —
the key indexes exactly the 300 claims that were annotated, so replacing either
one alone will silently drop records and change the number.
