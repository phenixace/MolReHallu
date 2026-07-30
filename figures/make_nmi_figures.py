#!/usr/bin/env python3
"""Generate the NMI plotted main-display figures from Source Data.

``main()`` produces every plotted figure shipped in this folder:
fig1_measures, fig2_widespread, fig3_accuracy_gap, fig4_mechanism,
fig_scratchpad (fig5), fig_draft (fig6), ed_fig1_ladder, ed_fig2_arena.

Only the conceptual-framework schematic (fig1_framework) is drawn by hand by the
authors and is not produced here; it is not shipped in this folder either.

Design:
  * Nature artwork spec via ``nature_figures.py`` (5-7 pt fonts, 89/183 mm
    widths, editable-text vector PDF + 600-dpi PNG).
  * A muted, harmonious palette (one hue per model); constrained layout, short
    titles, panel letters lifted clear of titles, and annotations parked in
    empty regions so nothing overlaps.
  * One argument beat per figure:
      Fig 2  pervasive, not explained by confidence or length
      Fig 3  decoupled from answer correctness
      Fig 4  answer grounded in the input; verbal claims inert
      Fig 5  model-specific molecular scratchpad
      Fig 6  structural drafts are causally load-bearing

Series come from ``data/source_data.xlsx`` (+ ``attention_perturbation.csv`` for
the teacher-forced Delta-logp, + the raw cap2mol details JSONL for the
per-response distribution).  Do not hand-edit numbers here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import nature_figures as nf  # applies the Nature rcParams on import

# --------------------------------------------------------------------------- #
# Paths / data
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent           # .../MolHallu/nmi/figures
PROJ = HERE.parent                            # .../ChemR_Hallucination_NMI
DATA = PROJ / "data"
REPRO = PROJ / "data" / "results"
SHEETS = pd.read_excel(DATA / "source_data.xlsx", sheet_name=None)


def sheet(name: str) -> pd.DataFrame:
    return SHEETS[name].copy()


# --------------------------------------------------------------------------- #
# Palette — muted editorial, one hue per model, colourblind-reasonable
# --------------------------------------------------------------------------- #
# ---- Cool editorial palette (blue / purple / grey / green); each hue = one value + role ----
BLUE   = "#3E6DA0"   # Chem-R            / SMILES channel
GREEN  = "#5B9B79"   # ChemDFM-R         / faithful, correct, clean, performance (good)
GREY   = "#8A8E99"   # ether-0           / functional-group channel (neutral, decorative)
PURPLE = "#8A6BA6"   # R1-Distill-8B     / fabricated, wrong (bad)
BLUELT = "#8FB4D6"   # Chem-R-Faithful   (light-blue tint of the Chem-R lineage)
INDIGO = "#3B4E86"   # Chem-R-Faithful (deep blue-violet, lineage)
GRAYD  = "#4F5866"   # input copy  (dark neutral)
GRAYL  = "#C7CBD2"   # trace copy  (light neutral)
GOOD_LT = "#BBD9C8"  # light-green fill (good)
BAD_LT  = "#D6C7E0"  # light-purple fill (bad)
GRAYREF = "0.55"     # gridlines, reference / identity lines

PAL = {
    "Chem-R": BLUE, "ChemDFM-R": GREEN, "ether-0": GREY, "DeepSeek-R1-Distill": PURPLE,
    "Chem-R-Faithful": INDIGO,
}
# semantic / channel aliases -- reuse palette hues so a colour means one thing throughout
C_SMILES, C_FG, C_GOOD, C_BAD = BLUE, GREY, GREEN, PURPLE
# ONE cool sequential ramp for all continuous data (heatmap AND token saliency); darker = higher
CMAP = LinearSegmentedColormap.from_list(
    "cool_seq", ["#F3F3F1", "#CAD3DA", "#8FA3B8", "#546B8C", "#2C3E5C"])

# raw workbook name -> canonical display name
DISPLAY = {
    "Chem-R": "Chem-R", "ChemDFM-R": "ChemDFM-R", "ether-0": "ether-0",
    "DeepSeek-R1-Distill": "DeepSeek-R1-Distill", "Chem-R-Faithful": "Chem-R-Faithful", "+process": "+process",
}
# compact labels for tight legends / annotations
SHORT = {
    "Chem-R": "Chem-R", "ChemDFM-R": "ChemDFM-R", "ether-0": "ether-0",
    "DeepSeek-R1-Distill": "R1-Distill", "Chem-R-Faithful": "Faithful",
}

# The surveyed set is defined per figure as SURV5 (it includes Chem-R-Faithful); an older
# 4-model SURVEY constant used to live here and is deliberately gone — reusing it silently
# dropped Chem-R-Faithful from Fig. 2.
GEN12 = [
    ("cap2mol", "cap2mol"), ("mol2cap", "mol2cap"), ("retrosynthesis", "retro"),
    ("s2_MolCustom_AtomNum", "MC-Atom"), ("s2_MolCustom_BondNum", "MC-Bond"),
    ("s2_MolCustom_FunctionalGroup", "MC-FG"), ("s2_MolEdit_AddComponent", "ME-Add"),
    ("s2_MolEdit_DelComponent", "ME-Del"), ("s2_MolEdit_SubComponent", "ME-Sub"),
    ("s2_MolOpt_LogP", "MO-LogP"), ("s2_MolOpt_MR", "MO-MR"), ("s2_MolOpt_QED", "MO-QED"),
]


def pc(raw: str) -> str:
    return PAL[DISPLAY.get(raw, raw)]


def plabel(ax, letter: str) -> None:
    """Bold panel letter, lifted above the title so it never collides."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-3, 9),
                textcoords="offset points", fontsize=8, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)


def title(ax, text: str) -> None:
    ax.set_title(text, fontsize=7, fontweight="bold", pad=4)


def save(fig, stem: str) -> None:
    nf.save(fig, HERE / stem)


# --------------------------------------------------------------------------- #
# Fig 2 — pervasive; not confidence or length
# --------------------------------------------------------------------------- #
def fig2_widespread() -> None:
    d = sheet("Diagnosis_model_task")
    SURV5 = ["Chem-R", "Chem-R-Faithful", "ChemDFM-R", "ether-0", "DeepSeek-R1-Distill"]
    # paper analyses the 12 generation+S2 tasks only (no classification/CLS tasks);
    # restrict here so every per-model aggregate (SE, overall, ER, cp, length) is
    # over the same 12-task universe as the panel-a heatmap.
    d = d[d.model.isin(SURV5) & d.task.isin([t for t, _ in GEN12])]

    def agg(m, c):
        return float(d[d.model == m][c].mean())

    fig = plt.figure(figsize=(nf.COL2, nf.COL2 * 0.82), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0])

    # (a) heatmap models x 12 tasks
    axa = fig.add_subplot(gs[0, :])
    plabel(axa, "a")
    tasks = [t for t, _ in GEN12]
    mat = np.array([[float(d[(d.model == m) & (d.task == t)]["ER"].mean()) for t in tasks]
                    for m in SURV5])
    vmax = float(np.nanmax(mat))
    im = axa.imshow(mat, cmap=CMAP, vmin=0, vmax=vmax, aspect="auto")
    axa.set_xticks(range(len(tasks)))
    axa.set_xticklabels([lab for _, lab in GEN12], rotation=40, ha="right", fontsize=5.5)
    axa.set_yticks(range(len(SURV5)))
    axa.set_yticklabels([DISPLAY[m] for m in SURV5])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            axa.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.0,
                     color="white" if v > 0.5 * vmax else "0.15")
    axa.tick_params(length=0)
    for s in axa.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axa, fraction=0.03, pad=0.025)
    cb.set_label("ER fabrication", fontsize=6)
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    title(axa, "Fabrication spans every model and task family")

    # (b) bubble: semantic entropy vs overall, size ~ ER
    axb = fig.add_subplot(gs[1, 0])
    plabel(axb, "b")
    for m in SURV5:
        x, y, er = agg(m, "semantic_entropy"), agg(m, "overall"), agg(m, "ER")
        axb.scatter(x, y, s=er * 7 + 18, color=pc(m), alpha=0.85, edgecolor="white",
                    linewidth=0.6, zorder=3)
        axb.annotate(DISPLAY[m], (x, y), xytext=(5, 4), textcoords="offset points",
                     fontsize=5.2, color=pc(m))
    axb.set_xlabel("semantic entropy (confidence)")
    axb.set_ylabel("overall hallucination")
    axb.set_xlim(0.99, 1.48)
    axb.set_ylim(5, 55)
    axb.text(0.97, 0.95, r"$r = -0.54$", transform=axb.transAxes, ha="right",
             va="top", fontsize=5.6, color=GRAYREF)
    title(axb, r"Confidence $\neq$ faithfulness")

    # (c) claim precision lollipop
    axc = fig.add_subplot(gs[1, 1])
    plabel(axc, "c")
    fab = {m: 100 - agg(m, "cp") for m in SURV5}
    order = sorted(SURV5, key=lambda m: fab[m])
    for i, m in enumerate(order):
        v = fab[m]
        axc.plot([0, v], [i, i], color=pc(m), lw=1.4, zorder=1)
        axc.scatter(v, i, s=26, color=pc(m), zorder=3, edgecolor="white", linewidth=0.5)
        axc.text(v + 2, i, f"{v:.0f}", va="center", fontsize=5.4, color=pc(m))
    axc.set_yticks(range(len(order)))
    axc.set_yticklabels([DISPLAY[m] for m in order], fontsize=5.6)
    axc.set_xlim(0, 62)
    axc.set_xlabel("per-claim fabrication rate (%)")
    title(axc, "Even the most faithful fabricates")

    # (d) length vs ER scatter (log-x)
    axd = fig.add_subplot(gs[2, 0])
    plabel(axd, "d")
    for m in SURV5:
        x, y = agg(m, "length"), agg(m, "ER")
        ha = "right" if m == SURV5[-1] else "left"
        dx = -5 if ha == "right" else 5
        axd.scatter(x, y, s=32, color=pc(m), edgecolor="white", linewidth=0.6, zorder=3)
        axd.annotate(DISPLAY[m], (x, y), xytext=(dx, 4), textcoords="offset points",
                     fontsize=5.2, color=pc(m), ha=ha)
    axd.set_xscale("log")
    axd.set_xlim(7e2, 2.2e4)
    axd.set_xlabel("reasoning length (chars)")
    axd.set_ylabel("ER fabrication")
    title(axd, r"Length $\neq$ faithfulness")

    # (e) training-stage ladder: fabrication falls, performance saturates
    axe = fig.add_subplot(gs[2, 1])
    plabel(axe, "e")
    lad = sheet("R1_stage_ladder").set_index("stage")
    lin = ["base-a (pre-SFT)", "SFT", "off-the-shelf GRPO", "+coupled"]
    labs = ["base", "SFT", "Chem-R", "Faithful"]
    fabv = [float(lad.loc[s, "perclaim_fab_rate"]) * 100 for s in lin]
    perf = [float(lad.loc[s, "perf"]) for s in lin]
    xx = np.arange(len(lin))
    axe.plot(xx, fabv, "-o", color=C_BAD, ms=3.5, lw=1.3)
    for xi, v in zip(xx, fabv):
        axe.text(xi, v + 1.4, f"{v:.0f}", ha="center", fontsize=5.0, color=C_BAD)
    axe.set_ylabel("per-claim fabrication (%)", color=C_BAD)
    axe.tick_params(axis="y", labelcolor=C_BAD)
    axe.set_ylim(0, 34)
    axr = axe.twinx()
    axr.plot(xx, perf, "-s", color=C_GOOD, ms=3, lw=1.3)
    axr.set_ylabel("performance (%)", color=C_GOOD)
    axr.tick_params(axis="y", labelcolor=C_GOOD)
    axr.set_ylim(0, 60)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(C_GOOD)
    axr.spines["top"].set_visible(False)
    axe.set_xticks(xx)
    axe.set_xticklabels(labs, fontsize=5.4)
    title(axe, "Fabrication starts in the base model")
    save(fig, "fig2_widespread")


# --------------------------------------------------------------------------- #
# Fig 1 (d,e) — overall hallucination + IR/IO/ER/EO decomposition per model.
# Generated separately; the authors composite it under the hand-drawn framework
# panels a-c. Data: Diagnosis_model_task (per-model mean over the 12 gen tasks).
# --------------------------------------------------------------------------- #
def fig1_measures() -> None:
    d = sheet("Diagnosis_model_task")
    d = d[d.task.isin([t for t, _ in GEN12])]
    MODELS = ["Chem-R", "Chem-R-Faithful", "ChemDFM-R", "ether-0", "DeepSeek-R1-Distill"]
    W = [("IR", 0.15, "#C7CBD2"), ("IO", 0.25, "#8A8E99"),
         ("ER", 0.25, "#8A6BA6"), ("EO", 0.35, "#3E6DA0")]

    def wmean(m, c):                       # simple mean over the generative tasks
        return float(d[d.model == m][c].mean())

    contrib = {m: [w * wmean(m, k) for k, w, _ in W] for m in MODELS}
    overall = {m: sum(contrib[m]) for m in MODELS}
    perf = {m: wmean(m, "perf") for m in MODELS}
    order = sorted(MODELS, key=lambda m: overall[m])          # ascending by overall
    x = np.arange(len(order))
    xl = [SHORT.get(m, m) for m in order]

    fig = plt.figure(figsize=(nf.COL2, nf.COL2 * 0.38), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

    # (d) overall hallucination vs task performance
    axd = fig.add_subplot(gs[0, 0]); plabel(axd, "d")
    w = 0.38
    b1 = axd.bar(x - w / 2, [overall[m] for m in order], w, color=C_BAD, label="hallucination")
    b2 = axd.bar(x + w / 2, [perf[m] for m in order], w, color=C_GOOD, label="performance")
    axd.bar_label(b1, fmt="%.1f", fontsize=4.6, padding=1)
    axd.bar_label(b2, fmt="%.0f", fontsize=4.6, padding=1)
    axd.set_xticks(x); axd.set_xticklabels(xl, rotation=18, ha="right", fontsize=5.6)
    axd.set_ylabel("score")
    axd.set_ylim(0, max(max(overall.values()), max(perf.values())) * 1.18)
    axd.legend(fontsize=5.0, loc="upper center", frameon=False, borderpad=0.2,
               handlelength=0.9, handletextpad=0.4, ncol=2, columnspacing=1.0)
    title(axd, "Hallucination vs performance")

    # (e) composition: share of each model's overall from the four error types
    axe = fig.add_subplot(gs[0, 1]); plabel(axe, "e")
    bottom = np.zeros(len(order))
    for j, (dim, wt, col) in enumerate(W):
        share = np.array([100 * contrib[m][j] / overall[m] for m in order])
        axe.bar(x, share, 0.64, bottom=bottom, color=col, label=dim,
                edgecolor="white", linewidth=0.4)
        for xi, v, b in zip(x, share, bottom):
            if v >= 6:
                axe.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                         fontsize=4.6, color="white")
        bottom += share
    axe.set_xticks(x); axe.set_xticklabels(xl, rotation=18, ha="right", fontsize=5.6)
    axe.set_ylabel("share of overall (%)")
    axe.set_ylim(0, 100)
    axe.legend(ncol=4, fontsize=4.8, loc="upper center", bbox_to_anchor=(0.5, -0.24),
               frameon=False, columnspacing=0.8, handlelength=0.9, handletextpad=0.4)
    title(axe, "Hallucination composition")
    save(fig, "fig1_measures")


# --------------------------------------------------------------------------- #
# Fig 3 — decoupled from correctness
# --------------------------------------------------------------------------- #
def _chemr_cap2mol():
    p = REPRO / "Chem-R" / "cap2mol" / "Chem-R_cap2mol_hallucination_details.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    er = np.array([r["hallucination_scores"]["ER_factual_fabrication"] for r in recs], float)
    exact = np.array([1 if r.get("exact_match") else 0 for r in recs])
    return er, exact


def fig3_accuracy_gap() -> None:
    er, exact = _chemr_cap2mol()
    corr, wrong = er[exact == 1], er[exact == 0]
    fig = plt.figure(figsize=(nf.COL2, nf.COL2 * 0.56), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    # (a) violin + inner quartile box -- correct vs wrong ER distributions
    axa = fig.add_subplot(gs[0, :2])
    plabel(axa, "a")
    data, cols = [corr, wrong], [C_GOOD, C_BAD]
    parts = axa.violinplot(data, positions=[1, 2], showextrema=False, widths=0.82)
    for pcbody, c in zip(parts["bodies"], cols):
        pcbody.set_facecolor(c); pcbody.set_alpha(0.28)
        pcbody.set_edgecolor(c); pcbody.set_linewidth(1.1)
    bp = axa.boxplot(data, positions=[1, 2], widths=0.12, showfliers=False,
                     patch_artist=True, medianprops=dict(color="white", lw=1.1),
                     whiskerprops=dict(color="0.4", lw=0.7), capprops=dict(color="0.4", lw=0.7),
                     boxprops=dict(lw=0), zorder=5)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.95)
    for i, (v, c) in enumerate(zip(data, cols), start=1):
        axa.annotate(f"mean {v.mean():.1f}", (i, v.mean()), xytext=(11, 0),
                     textcoords="offset points", fontsize=5.6, color=c, va="center",
                     fontweight="bold", zorder=6)
    axa.set_xticks([1, 2]); axa.set_xticklabels(["correct answer", "wrong answer"])
    axa.set_ylabel("ER fabrication")
    axa.set_ylim(-2, 30)
    r = np.corrcoef(er, exact)[0, 1]
    axa.text(0.5, 0.99, f"Pearson$(ER,$ exact$) = {r:.2f}$;  $n = {len(er)}$",
             transform=axa.transAxes, ha="center", va="top", fontsize=5.8,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.4))
    title(axa, "Correct and wrong answers fabricate alike")

    # (b) mosaic -- each cell's AREA is its share of responses (width = fabricating vs
    #     faithful, height within a column = correct vs wrong)
    axb = fig.add_subplot(gs[0, 2])
    plabel(axb, "b")
    n = len(er)
    rf = np.sum((exact == 1) & (er > 0)) / n
    rc = np.sum((exact == 1) & (er == 0)) / n
    wf = np.sum((exact == 0) & (er > 0)) / n
    wc = np.sum((exact == 0) & (er == 0)) / n
    fab, faith = rf + wf, rc + wc
    g = 0.012
    cells = [(0, fab, 1 - rf / fab, rf / fab, C_BAD, rf, "right-yet\nfabricating", "white"),
             (0, fab, 0, wf / fab, BAD_LT, wf, "wrong &\nfabricating", "0.15"),
             (fab, faith, 1 - rc / faith, rc / faith, C_GOOD, rc, "right &\nfaithful", "white"),
             (fab, faith, 0, wc / faith, GOOD_LT, wc, "wrong-yet\nfaithful", "0.15")]
    for x0, w, y0, h, c, pct, lab, tc in cells:
        axb.add_patch(Rectangle((x0 + g / 2, y0 + g / 2), w - g, h - g, facecolor=c,
                                edgecolor="none"))
        axb.text(x0 + w / 2, y0 + h / 2 + 0.055, f"{pct * 100:.0f}%", ha="center",
                 va="center", fontsize=8, fontweight="bold", color=tc)
        axb.text(x0 + w / 2, y0 + h / 2 - 0.11, lab, ha="center", va="center",
                 fontsize=4.7, color=tc)
    axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.set_aspect("equal")
    axb.set_xticks([fab / 2, fab + faith / 2])
    axb.set_xticklabels(["fabricating", "faithful"], fontsize=5.2)
    axb.set_yticks([]); axb.tick_params(length=0)
    for s in axb.spines.values():
        s.set_visible(False)
    title(axb, r"Correct $\neq$ faithful")

    # (c) clean vs fabricating performance scatter
    SURV5 = ["Chem-R", "Chem-R-Faithful", "ChemDFM-R", "ether-0", "DeepSeek-R1-Distill"]
    fam = sheet("Diagnosis_family"); fam = fam[fam.model.isin(SURV5)]
    axc = fig.add_subplot(gs[1, :2])
    plabel(axc, "c")
    mk = {"cap2mol": "o", "mol2cap": "s", "retrosynthesis": "^", "s2": "D"}
    for _, row in fam.iterrows():
        axc.scatter(row["perf_er0"], row["perf_erpos"], s=28, color=pc(row["model"]),
                    marker=mk.get(row["family"], "o"), edgecolor="white", linewidth=0.5,
                    alpha=0.9, zorder=3)
    axc.plot([0, 72], [0, 72], color=GRAYREF, ls=(0, (4, 3)), lw=0.7, zorder=0)
    axc.set_xlim(-3, 72); axc.set_ylim(-3, 72)
    axc.set_xlabel("performance, clean traces (ER = 0)")
    axc.set_ylabel("performance,\nfabricating traces (ER > 0)")
    title(axc, "Fabrication barely moves accuracy")
    tleg = [Line2D([], [], marker=mk[f], ls="", color="0.45", markersize=3.2,
                   label=f.replace("retrosynthesis", "retro")) for f in mk]
    mleg = [Line2D([], [], marker="s", ls="", color=pc(m), markersize=3.2, label=DISPLAY[m])
            for m in SURV5]
    leg1 = axc.legend(handles=tleg, loc="lower right", fontsize=4.8, handletextpad=0.2,
                      labelspacing=0.2, title="task", title_fontsize=4.8, borderpad=0.3)
    axc.add_artist(leg1)
    axc.legend(handles=mleg, loc="upper left", fontsize=4.8, handletextpad=0.2,
               labelspacing=0.2, borderpad=0.3)

    # (d) % clean traces per family
    axd = fig.add_subplot(gs[1, 2])
    plabel(axd, "d")
    fams = ["cap2mol", "mol2cap", "retrosynthesis", "s2"]
    x = np.arange(len(fams)); w = 0.16
    for k, m in enumerate(SURV5):
        vals = [float(fam[(fam.model == m) & (fam.family == f)]["pct_er0"].mean()) for f in fams]
        axd.bar(x + (k - 2) * w, vals, w, color=pc(m))
    axd.set_xticks(x)
    axd.set_xticklabels(["cap2mol", "mol2cap", "retro", "s2"], rotation=25, ha="right", fontsize=5.2)
    axd.set_ylabel("clean traces (%)")
    axd.set_ylim(0, 100)
    title(axd, "Clean traces are scarce on hard tasks")
    save(fig, "fig3_accuracy_gap")


# --------------------------------------------------------------------------- #
# Fig 4 — answer grounded in input; verbal claims inert
# --------------------------------------------------------------------------- #
def fig4_mechanism() -> None:
    ap = pd.read_csv(DATA / "attention_perturbation.csv").set_index("model")
    drift = sheet("R2_drift")
    fig = plt.figure(figsize=(nf.COL2, nf.COL2 * 0.36), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.2, 0.85])

    # (a) teacher-forced Delta log p — lollipop
    axa = fig.add_subplot(gs[0, 0])
    plabel(axa, "a")
    rows = [("Chem-R", "Chem-R"), ("ChemDFM-R", "ChemDFM-R")]
    conds = [("corrupt claim\nin trace", "d_wrong_cot"), ("synonym\ncontrol", "d_syn_cot"),
             ("corrupt fact\nin input", "d_wrong_input")]
    y = np.arange(len(conds))
    for off, (disp, raw) in zip([0.16, -0.16], rows):
        vals = [float(ap.loc[raw, c]) for _, c in conds]
        for v, yy in zip(vals, y + off):
            axa.plot([0, v], [yy, yy], color=PAL[disp], lw=1.0, zorder=1)
        axa.plot(vals, y + off, ls="", marker="o", ms=4, color=PAL[disp], label=disp, zorder=3)
        axa.annotate(f"{vals[2]:.2f}", (vals[2], y[2] + off), xytext=(-3, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=5.0,
                     color=PAL[disp])
    axa.axvline(0, color="0.3", lw=0.6)
    # the two trace/synonym conditions are measured but ~0 -- say so, so they don't
    # read as missing data
    axa.annotate("both $|\\Delta\\log p| < 0.001$\n(the claim is inert)", (0, 0.5),
                 xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                 fontsize=5.0, color="0.4")
    axa.set_yticks(y); axa.set_yticklabels([c for c, _ in conds], fontsize=5.6)
    axa.set_xlabel(r"$\Delta\log p$(correct answer)")
    axa.set_xlim(-0.21, 0.075)
    axa.legend(loc="lower left", fontsize=5.2, handletextpad=0.2, borderpad=0.3)
    title(axa, "Corrupt the claim vs the input")

    # (b) behavioural drift flip-to-wrong (translate) with 95% CI
    axb = fig.add_subplot(gs[0, 1])
    plabel(axb, "b")
    dconds = [("syn_cot", "synonym\n(ctrl)"), ("all_wrong_cot", "FG claims\nin trace"),
              ("swap_cot", "whole trace\nswapped")]
    x = np.arange(len(dconds)); w = 0.36
    for off, (disp, raw) in zip([-w / 2, w / 2], [("Chem-R", "Chem-R"), ("ChemDFM-R", "ChemDFM-R")]):
        sub = drift[(drift.model == raw) & (drift.task_group == "translate")]
        vals, errs = [], []
        for cond, _ in dconds:
            row = sub[sub.condition == cond]
            p = float(row["flip_to_wrong_pct"].iloc[0]) / 100
            nn = float(row["n_orig_correct"].iloc[0])
            vals.append(p * 100); errs.append(1.96 * np.sqrt(p * (1 - p) / nn) * 100)
        axb.bar(x + off, vals, w, yerr=errs, color=PAL[disp], label=disp,
                error_kw=dict(lw=0.6, ecolor="0.3"))
    axb.set_xticks(x); axb.set_xticklabels([l for _, l in dconds], fontsize=5.6)
    axb.set_ylabel("flip-to-wrong among\noriginally-correct (%)")
    axb.legend(loc="upper left", fontsize=5.2, borderpad=0.3)
    title(axb, "Claims inert; whole trace is not")

    # (c) for a functional-group word that appears in BOTH the input and the trace,
    #     what share of the answer's attention lands on each copy?  This reframes the
    #     input/trace ratio (20x, 21x, 4x) as an intuitive 100%-split bar.
    axc = fig.add_subplot(gs[0, 2])
    plabel(axc, "c")
    order = [("Chem-R", "Chem-R"), ("Faithful", "Chem-R-Faithful"),
             ("ChemDFM-R", "ChemDFM-R")]
    y = np.arange(len(order))[::-1]
    C_IN, C_TR = GRAYD, GRAYL              # input copy (dark), trace copy (light)
    for yy, (disp, raw) in zip(y, order):
        ratio = float(ap.loc[raw, "matched_ratio_input_over_cot"])
        pin = ratio / (ratio + 1) * 100
        axc.barh(yy, pin, color=C_IN, edgecolor="white", linewidth=0.5)
        axc.barh(yy, 100 - pin, left=pin, color=C_TR, edgecolor="white", linewidth=0.5)
        axc.text(pin / 2, yy, f"{pin:.0f}%", ha="center", va="center", color="white",
                 fontsize=5.8, fontweight="bold")
        axc.text(102, yy, f"{ratio:.0f}$\\times$", ha="left", va="center", fontsize=5.2,
                 color="0.35")
    axc.set_xlim(0, 117); axc.set_ylim(-0.6, len(order) - 0.4)
    axc.set_yticks(y); axc.set_yticklabels([d for d, _ in order], fontsize=5.6)
    axc.set_xticks([0, 50, 100])
    axc.set_xlabel("attention on the input (dark)\nvs trace (light) copy, %")
    title(axc, "Same word: answer looks at input")
    save(fig, "fig4_mechanism")


# --------------------------------------------------------------------------- #
# Fig 5 — model-specific molecular scratchpad
# --------------------------------------------------------------------------- #
def _token_strip(ax, model_raw, task, ex_id, heading, focus="SMILES_frag", win=120,
                 region=None):
    te = sheet("R5_token_examples")
    sub = te[(te.model == model_raw) & (te.task == task)
             & (te["id"].astype(str) == str(ex_id))
             & (te.region == "trace")].sort_values("token_index")
    sal = sub["sal_norm"].to_numpy(float)
    cats = sub["cat"].to_numpy()
    if region is not None:
        # show a specific slice of the trace (fractions), e.g. the reactant-proposal
        # region where the model drafts the answer -- not the input-restatement block.
        n = len(sal); lo, hi = int(region[0] * n), int(region[1] * n)
        sal = sal[lo:hi]; cats = cats[lo:hi]
    elif len(sal) > win:
        focus_sal = np.where(cats == focus, sal, 0.0)
        base = focus_sal if focus_sal.sum() > 0 else sal
        s0 = int(np.argmax(np.convolve(base, np.ones(win), "valid")))
        sal = sal[s0:s0 + win]; cats = cats[s0:s0 + win]
    v = (sal - sal.min()) / (sal.ptp() + 1e-9)
    ax.imshow(v[None, :], cmap=CMAP, aspect="auto", extent=[0, len(v), 0, 1])
    # contiguous SMILES-fragment runs -> one shaded span below (a written-out SMILES
    # string tokenizes into many adjacent SMILES tokens); FG words -> ticks above
    k = 0
    while k < len(cats):
        if cats[k] == "SMILES_frag":
            j = k
            while j < len(cats) and cats[j] == "SMILES_frag":
                j += 1
            ax.add_patch(Rectangle((k, -0.36), j - k, 0.30, facecolor=C_SMILES,
                                   edgecolor="none", clip_on=False))
            k = j
        else:
            if cats[k] == "FG_word":
                ax.plot([k + .5, k + .5], [1.04, 1.32], color=C_FG, lw=1.0, clip_on=False)
            k += 1
    ax.set_xlim(0, len(v)); ax.set_ylim(-0.42, 1.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(heading, fontsize=5.8, pad=8, loc="left")


def fig5_scratchpad() -> None:
    grad = sheet("R5_grad_enrichment")
    reg = sheet("R5_region_attention")
    models3 = [("Chem-R", "Chem-R"), ("Chem-R-Faithful", "Chem-R-Faithful"),
               ("ChemDFM-R", "ChemDFM-R"), ("ether-0", "ether-0")]
    fig = plt.figure(figsize=(nf.COL2, nf.COL2 * 0.74), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.0])

    # (a) token-saliency strips
    axa = fig.add_subplot(gs[0, :]); axa.set_axis_off()
    plabel(axa, "a")
    axa.set_title("Answer-saliency where each model proposes the reactant answer",
                  fontsize=7, fontweight="bold", loc="left", pad=2)
    s1 = axa.inset_axes([0.0, 0.70, 1.0, 0.19])
    _token_strip(s1, "Chem-R-Faithful", "retrosynthesis", "uspto_test_1",
                 "Chem-R-Faithful  —  drafts the reactants as partial SMILES fragments (each attended)",
                 region=(0.70, 1.0))
    s2 = axa.inset_axes([0.0, 0.39, 1.0, 0.19])
    _token_strip(s2, "ChemDFM-R", "retrosynthesis", "uspto_test_31",
                 "ChemDFM-R  —  proposes them in words and naming; no SMILES",
                 region=(0.55, 1.0))
    s3 = axa.inset_axes([0.0, 0.08, 1.0, 0.19])
    _token_strip(s3, "ether-0", "retrosynthesis", "uspto_test_76",
                 "ether-0  —  also drafts the reactants as SMILES fragments")
    axa.legend(handles=[Rectangle((0, 0), 1, 1, color=C_SMILES),
                        Line2D([], [], color=C_FG, lw=1.4)],
               labels=["SMILES string", "FG token"],
               loc="upper right", fontsize=5.2, ncol=2, handlelength=1.2, borderpad=0.3,
               bbox_to_anchor=(1.0, 1.08))

    # (b) gradient saliency enrichment by token type
    axb = fig.add_subplot(gs[1, 0])
    plabel(axb, "b")
    types = [("SMILES_frag", "SMILES"), ("FG_word", "FG word"), ("position_digit", "position")]
    x = np.arange(len(types)); w = 0.2
    for k, (raw, disp) in enumerate(models3):
        sub = grad[(grad.model == raw) & (grad.stratum == "all")]
        vals = [float(sub[sub.token_type == t]["enrichment"].iloc[0]) for t, _ in types]
        bars = axb.bar(x + (k - 1.5) * w, vals, w, color=PAL[disp], label=SHORT[disp])
        axb.bar_label(bars, fmt="%.1f", fontsize=4.4, padding=1)
    axb.axhline(1, color="0.3", ls=(0, (4, 3)), lw=0.6)
    axb.set_xticks(x); axb.set_xticklabels([l for _, l in types], fontsize=5.6)
    axb.set_ylabel("saliency enrichment\n(>1 = above average)")
    axb.legend(loc="upper right", fontsize=5.0, handlelength=1.0, borderpad=0.3)
    title(axb, "What the answer keys on")

    # (c) within-trace attention: SMILES vs FG
    axc = fig.add_subplot(gs[1, 1])
    plabel(axc, "c")
    x = np.arange(len(models3)); w = 0.36
    smi = [float(reg[(reg.model == r) & (reg.stratum == "all") & (reg.region == "trace")]["attn_smiles"].iloc[0]) * 1e3
           for r, _ in models3]
    fgv = [float(reg[(reg.model == r) & (reg.stratum == "all") & (reg.region == "trace")]["attn_fg"].iloc[0]) * 1e3
           for r, _ in models3]
    axc.bar(x - w / 2, smi, w, color=C_SMILES, label="to SMILES")
    axc.bar(x + w / 2, fgv, w, color=C_FG, label="to FG word")
    axc.set_xticks(x); axc.set_xticklabels([SHORT[d] for _, d in models3], rotation=15,
                                           ha="right", fontsize=5.4)
    axc.set_ylabel(r"within-trace attention ($\times10^{-3}$)")
    axc.legend(loc="upper left", fontsize=5.0, borderpad=0.3)
    title(axc, "Where trace attention goes")

    save(fig, "fig_scratchpad")


# --------------------------------------------------------------------------- #
# Fig 6 — structural drafts are causally load-bearing  (use task_group == "all")
# --------------------------------------------------------------------------- #
def fig6_draft() -> None:
    dp = sheet("R2_draft_perturbation")
    dp = dp[dp.task_group == "all"].set_index("model")
    grad = sheet("R5_grad_enrichment")
    fig = plt.figure(figsize=(nf.COL2, nf.COL2 * 0.62), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    # (a) flip-to-wrong by perturbation, Chem-R vs Chem-R-Faithful
    axa = fig.add_subplot(gs[0, 0:2])
    plabel(axa, "a")
    conds = [("all_wrong_flip_pct", "FG-name\nclaim"), ("mask_draft_flip_pct", "mask\ndraft"),
             ("corrupt_draft_flip_pct", "corrupt\ndraft"), ("swap_flip_pct", "swap\ntrace")]
    x = np.arange(len(conds)); w = 0.38
    for off, (raw, disp) in zip([-w / 2, w / 2],
                                [("Chem-R", "Chem-R"), ("Chem-R-Faithful", "Chem-R-Faithful")]):
        vals = [float(dp.loc[raw, c]) for c, _ in conds]
        axa.bar(x + off, vals, w, color=PAL[disp], label=SHORT[disp])
    axa.set_xticks(x); axa.set_xticklabels([l for _, l in conds], fontsize=5.6)
    axa.set_ylabel("flip-to-wrong among\noriginally-correct (%)")
    axa.legend(loc="upper left", fontsize=5.2, borderpad=0.3)
    title(axa, "Corrupting the draft flips answers")

    # (b) draft coverage — negative control
    axb = fig.add_subplot(gs[0, 2])
    plabel(axb, "b")
    models3 = [("Chem-R", "Chem-R"), ("Chem-R-Faithful", "Chem-R-Faithful"),
               ("ChemDFM-R", "ChemDFM-R")]
    cov = [float(dp.loc[r, "coverage_pct_of_correct"]) for r, _ in models3]
    bars = axb.bar(range(3), cov, 0.6, color=[PAL[d] for _, d in models3])
    axb.bar_label(bars, fmt="%.0f%%", fontsize=5.4, padding=1)
    axb.set_xticks(range(3)); axb.set_xticklabels([SHORT[d] for _, d in models3], rotation=15,
                                                  ha="right", fontsize=5.4)
    axb.set_ylabel("correct traces that\ndraft a SMILES (%)")
    axb.set_ylim(0, 100)
    title(axb, "ChemDFM-R barely drafts")

    # (c) cross-validation: corrupt-draft flip vs SMILES enrichment
    axc = fig.add_subplot(gs[1, 0])
    plabel(axc, "c")
    emap = {"Chem-R": "Chem-R", "Chem-R-Faithful": "Chem-R-Faithful", "ChemDFM-R": "ChemDFM-R"}
    xs, ys, cs, labs = [], [], [], []
    for raw, disp in models3:
        e = float(grad[(grad.model == emap[raw]) & (grad.stratum == "all")
                       & (grad.token_type == "SMILES_frag")]["enrichment"].iloc[0])
        xs.append(e); ys.append(float(dp.loc[raw, "corrupt_draft_flip_pct"]))
        cs.append(PAL[disp]); labs.append(SHORT[disp])
    axc.plot(np.sort(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.sort(xs)), color=GRAYREF,
             lw=0.8, ls=(0, (4, 3)), zorder=0)
    axc.scatter(xs, ys, s=42, color=cs, edgecolor="white", linewidth=0.8, zorder=3)
    for xv, yv, lab, c in zip(xs, ys, labs, cs):
        ha = "right" if lab == "Faithful" else "left"
        dx = -6 if ha == "right" else 6
        axc.annotate(lab, (xv, yv), xytext=(dx, 3), textcoords="offset points",
                     fontsize=5.0, color=c, ha=ha)
    axc.set_xlim(0.9, 3.7)
    axc.set_xlabel("trace SMILES saliency enrichment")
    axc.set_ylabel("corrupt-draft\nflip-to-wrong (%)")
    title(axc, "Two probes agree")

    # (d) early-answer rate: complete answer SMILES already written in the trace
    axdc = fig.add_subplot(gs[1, 1])
    plabel(axdc, "d")
    regd = sheet("R5_region_attention")
    m3r = [("Chem-R", "Chem-R"), ("Chem-R-Faithful", "Chem-R-Faithful"),
           ("ChemDFM-R", "ChemDFM-R")]
    dcv = [float(regd[(regd.model == r) & (regd.stratum == "all") & (regd.region == "trace")]["draft_copy"].iloc[0])
           for r, _ in m3r]
    barsdc = axdc.bar(range(3), dcv, 0.6, color=[PAL[d] for _, d in m3r])
    axdc.bar_label(barsdc, fmt="%.2f", fontsize=5.4, padding=1)
    axdc.set_xticks(range(3)); axdc.set_xticklabels([SHORT[d] for _, d in m3r], rotation=15,
                                                    ha="right", fontsize=5.4)
    axdc.set_ylabel("complete answer SMILES\nin trace (fraction)")
    axdc.set_ylim(0, max(dcv) * 1.3)
    title(axdc, "Answer can be written early")

    # (e) not an early-answer artefact: partial-only vs early-answer corrupt-draft flip
    axd = fig.add_subplot(gs[1, 2])
    plabel(axd, "e")
    rd = sheet("R2b_flip_by_draftcopy").set_index("model")
    m2 = [("Chem-R", "Chem-R"), ("Chem-R-Faithful", "Faithful")]
    xx = np.arange(len(m2)); wd = 0.38
    early = [float(rd.loc[raw, "early_flip%"]) for raw, _ in m2]
    part = [float(rd.loc[raw, "partial_corrupt_flip%"]) for raw, _ in m2]
    b1 = axd.bar(xx - wd / 2, early, wd, color=GRAYL, label="early answer")
    b2 = axd.bar(xx + wd / 2, part, wd, color=C_SMILES, label="partial only")
    axd.bar_label(b1, fmt="%.0f", fontsize=4.6, padding=1)
    axd.bar_label(b2, fmt="%.0f", fontsize=4.6, padding=1)
    axd.set_xticks(xx); axd.set_xticklabels([d for _, d in m2], fontsize=5.4)
    axd.set_ylabel("corrupt-draft\nflip-to-wrong (%)")
    axd.legend(loc="upper left", fontsize=4.8, borderpad=0.3, handlelength=1.0)
    title(axd, "Not just early answers")
    save(fig, "fig_draft")


# --------------------------------------------------------------------------- #
# Extended Data Fig. 1 — where fabrication comes from (training-stage ladder)
# --------------------------------------------------------------------------- #
def ed_fig1_ladder() -> None:
    d = sheet("R1_stage_ladder").set_index("stage")
    # The released Chem-R is a fully-trained answer-only GRPO model scored on the full
    # task suite (n=11607); it is the answer-only point.  The separate acc-only
    # ablation is omitted -- it was scored on cap2mol only (n=3300), so its per-claim
    # rate is not comparable to the other stages.
    lin = ["base-a (pre-SFT)", "SFT", "off-the-shelf GRPO", "+process", "+coupled"]
    labs = ["base\n(pre-SFT)", "SFT", "Chem-R\n(answer-only)", "+process", "+coupled"]
    fab = [float(d.loc[s, "perclaim_fab_rate"]) * 100 for s in lin]
    perf = [float(d.loc[s, "perf"]) for s in lin]
    x = np.arange(len(lin))
    fig = plt.figure(figsize=(nf.COL2, nf.COL2 * 0.36), layout="constrained")
    gs = fig.add_gridspec(1, 2)

    # (a) per-claim fabrication along the lineage
    axa = fig.add_subplot(gs[0, 0]); plabel(axa, "a")
    axa.plot(x, fab, "-o", color=C_BAD, ms=4, lw=1.3)
    for xi, v in zip(x, fab):
        axa.text(xi, v + 1.1, f"{v:.0f}", ha="center", fontsize=5.4, color=C_BAD)
    axa.set_xticks(x); axa.set_xticklabels(labs, fontsize=5.4)
    axa.set_ylabel("per-claim fabrication (%)")
    axa.set_ylim(0, 34)
    axa.annotate("base fabricates most\n(at 0.3% accuracy)", (0.30, 0.84),
                 xycoords="axes fraction", fontsize=5.0, color=GRAYREF, va="top")
    title(axa, "Only a grounded reward cuts fabrication")

    # (b) accuracy saturates early; only grounding cuts fabrication (twin axis)
    axb = fig.add_subplot(gs[0, 1]); plabel(axb, "b")
    axb.plot(x, perf, "-o", color=C_GOOD, ms=4, lw=1.3)
    axb.set_ylabel("task performance (%)", color=C_GOOD)
    axb.tick_params(axis="y", labelcolor=C_GOOD)
    axb.set_ylim(0, 60)
    axr = axb.twinx()
    axr.plot(x, fab, "-s", color=C_BAD, ms=3.5, lw=1.3)
    axr.set_ylabel("per-claim fabrication (%)", color=C_BAD)
    axr.tick_params(axis="y", labelcolor=C_BAD)
    axr.set_ylim(0, 34)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(C_BAD)
    axr.spines["top"].set_visible(False)
    axb.set_xticks(x); axb.set_xticklabels(labs, fontsize=5.4)
    title(axb, "Accuracy saturates; grounding cuts fabrication")
    save(fig, "ed_fig1_ladder")


# --------------------------------------------------------------------------- #
# Extended Data Fig. 2 — detector agrees with an expert chemist (arena)
# --------------------------------------------------------------------------- #
def ed_fig2_arena() -> None:
    j = json.loads((DATA / "human_eval_agreement.json").read_text())
    fig, ax = nf.new_fig(nf.COL1, nf.COL1 * 0.74)
    conds = ["overall-\nhallucination pick", "ER-\nfabrication pick"]
    ag = [j["ag_ov"], j["ag_er"]]
    ks = [j["k_ov"], j["k_er"]]
    x = np.arange(2)
    ax.bar(x, ag, 0.55, color=[C_SMILES, C_FG])
    ax.axhline(j["rand"], color=GRAYREF, ls=(0, (4, 3)), lw=0.8)
    ax.text(1.42, j["rand"] + 1.6, f"random {j['rand']:.0f}%", ha="right", fontsize=5.6,
            color=GRAYREF)
    for xi, a, k in zip(x, ag, ks):
        ax.text(xi, a + 1.6, f"{a:.0f}%\n$\\kappa$ = {k:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=6)
    ax.set_ylabel("chemist agrees with detector's\nlowest-hallucination model (%)")
    ax.set_ylim(0, 70)
    ax.set_title("Automatic detector matches an expert chemist\n($n = 400$ prompts)",
                 fontsize=7, fontweight="bold")
    save(fig, "ed_fig2_arena")


def main() -> None:
    fig1_measures()
    fig2_widespread()
    fig3_accuracy_gap()
    fig4_mechanism()
    fig5_scratchpad()
    fig6_draft()
    ed_fig1_ladder()
    ed_fig2_arena()


if __name__ == "__main__":
    main()
