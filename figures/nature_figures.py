"""
nature_figures.py
Publication-quality figure toolkit for Nature-family journals.

Styling follows Nature's artwork guidelines:
- Sans-serif fonts (Arial/Helvetica), final size 5-7 pt
- Single-column width 89 mm, double-column 183 mm, max height 247 mm
- Thin axis lines (0.5 pt), ticks pointing outward, no top/right spines
- Colourblind-safe categorical palette (Wong, Nature Methods 8, 441 (2011))
- Vector PDF + 600-dpi PNG export, with text kept EDITABLE (fonttype 42)

Requires: matplotlib, numpy.  (No seaborn needed.)

Quick start
-----------
    import nature_figures as nf          # applies the style on import
    fig, ax = nf.new_fig(nf.COL1)         # single-column figure
    ax.plot(...)
    nf.save(fig, "figures/fig1")          # -> fig1.pdf + fig1.png

Run `python nature_figures.py` to regenerate every example into ./figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- Nature physical dimensions (mm -> inch) ---------------------------------
MM = 1 / 25.4
COL1 = 89 * MM     # single column   ~3.50 in
COL15 = 120 * MM   # 1.5 column      ~4.72 in
COL2 = 183 * MM    # double column   ~7.20 in
MAXH = 247 * MM    # maximum figure height

# --- Colourblind-safe categorical palette (Wong 2011) ------------------------
WONG = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
}
# Default cycle: strong, well-separated colours first.
# Order validated (Machado-2009 CVD simulation, white surface): worst
# adjacent-pair dE = 17.2 (safe threshold 12) across protan/deutan/tritan.
# Usage limits from the same validation:
#   - orange / sky blue: ~2.3:1 contrast on white -> fine for bars, markers,
#     fills; avoid for hairline-thin lines without markers
#   - yellow: 1.3:1 -> fills only, never lines or text
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#000000", "#F0E442"]

# Continuous data: perceptually uniform only — never 'jet'/rainbow.
CMAP_SEQ = "viridis"      # sequential (magnitude): one scale, light -> dark
CMAP_SEQ_PRINT = "cividis"  # sequential, optimised for CVD + grayscale print
CMAP_DIV = "RdBu_r"       # diverging (signed data): neutral at the midpoint
GRAY = "0.4"              # reference lines: identity, chance level, zero

# --- Morandi palette (muted alternative style) --------------------------------
# All 10 swatches sit below the OKLCH chroma floor (C 0.012-0.055 < 0.10):
# hue alone cannot carry series identity, so compensations are mandatory —
# a marker per line series, ink edges + value labels on fills, and a legend.
# Contrast on white: slate 5.3:1, slate2 3.5:1 (both fine for lines);
# sage 2.4:1 (lines only with lw >= 1.2 + markers); everything lighter is
# fill/background only. Keep viridis for continuous data even in this style.
MORANDI = {
    "slate":    "#5E6C82",  # darkest — first-choice line colour
    "sage":     "#81B3A9",
    "slate2":   "#7F8A9B",
    "grayblue": "#899FB0",  # CVD dE to slate2 = 7.1 — never use both
    "beige":    "#D6CDBE",
    "palesage": "#B3C6BB",  # CVD dE to beige = 5.2 — never adjacent
    "paleblue": "#B7CBD5",
    "paleaqua": "#C1DDDB",
    "palemint": "#D1DED7",
    "cream":    "#EBE6DE",  # source swatch mislabels the hex; RGB 235,230,222
}
MORANDI_INK = "#3A4454"     # derived darker slate: edges, annotations, brackets
# Bars/area fills, drawn with MORANDI_INK edges. Order validated
# (Machado-2009 CVD simulation): worst adjacent-pair dE = 12.9 (threshold 12).
PALETTE_MORANDI = [MORANDI["slate"], MORANDI["sage"], MORANDI["beige"],
                   MORANDI["slate2"], MORANDI["palesage"]]
# Line plots: at most these three on white, each with its own marker.
MORANDI_LINES = [MORANDI["slate"], MORANDI["sage"], MORANDI["slate2"]]
MORANDI_MARKERS = ["o", "s", "^"]


def set_style(palette=None):
    """Apply Nature-style rcParams globally. Called automatically on import.

    palette: colour cycle to use — defaults to the Wong palette.
    For the muted look: set_style(palette=PALETTE_MORANDI).
    """
    palette = palette or PALETTE
    mpl.rcParams.update({
        # --- fonts ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "mathtext.fontset": "dejavusans",
        # --- keep text editable in vector output (crucial for Nature) ---
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # --- lines, ticks, spines ---
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # --- clean, journal look ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=palette),
        "legend.frameon": False,
        # --- figure / export ---
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
    })


def new_fig(width=COL1, height=None, **kw):
    """A figure sized to a Nature column width. height defaults to 0.72*width."""
    if height is None:
        height = width * 0.72
    height = min(height, MAXH)
    return plt.subplots(figsize=(width, height), **kw)


def save(fig, path, formats=("pdf", "png")):
    """Save as vector PDF + 600-dpi PNG. Text stays selectable in the PDF.

    The PDF gets no CreationDate. Matplotlib stamps one by default, which makes an
    otherwise identical rebuild differ byte for byte -- and since the rendered figures
    are committed, that would dirty the tree on every regeneration and hide a real
    change among six spurious ones. Without it, rebuilding from an unchanged workbook
    reproduces all six files exactly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(path.with_suffix(f".{ext}"), metadata=meta)
    plt.close(fig)
    print("saved:", ", ".join(str(path.with_suffix("." + e)) for e in formats))


def panel_label(ax, letter, dx=-20, dy=2):
    """Bold panel letter (a, b, c ...) at the top-left of a panel.

    Offsets are in points from the axes corner, so the letter lands in the
    same physical spot on every panel — including axes shrunk by
    set_aspect("equal") — unlike axes-fraction coordinates.
    """
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=8, fontweight="bold", ha="left", va="bottom")


def sig_bracket(ax, x1, x2, y, text, h=None, lw=0.5):
    """Significance bracket between x positions x1 and x2 (data coords).

    Nature style: give the exact P value with an italic P, not stars —
    e.g. sig_bracket(ax, 1, 2, 0.95, "$P$ = 0.003").
    """
    if h is None:
        h = 0.015 * float(np.diff(ax.get_ylim())[0])
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=lw, color="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=6)


# =============================================================================
# Example figures (all data are synthetic — replace with yours)
# =============================================================================

def example_grouped_bar(ax=None):
    """Grouped bars with error bars — the classic benchmark comparison."""
    tasks = ["Property", "Reaction", "Retrosyn.", "Naming"]
    models = ["Baseline", "Prev. SOTA", "Ours"]
    data = np.array([[0.62, 0.55, 0.48, 0.71],
                     [0.78, 0.70, 0.66, 0.85],
                     [0.86, 0.81, 0.79, 0.91]])
    err = np.full_like(data, 0.02)

    made = ax is None
    if made:
        _, ax = new_fig(COL1)
    x = np.arange(len(tasks))
    w = 0.26
    for i, m in enumerate(models):
        ax.bar(x + (i - 1) * w, data[i], w, yerr=err[i], label=m,
               capsize=1.5, error_kw={"lw": 0.5}, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=1.0, handlelength=1.0)
    return ax


def example_scaling(ax=None):
    """Log-x line plot with shaded error bands — scaling / ablation curves."""
    x = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])
    lx = np.log10(x / 1e6)
    made = ax is None
    if made:
        _, ax = new_fig(COL1)
    for name, (base, slope) in {"Ours": (0.55, 0.40),
                                "Baseline": (0.50, 0.30)}.items():
        y = base + slope * (lx / lx.max())
        line, = ax.plot(x, y, marker="o", label=name)
        ax.fill_between(x, y - 0.02, y + 0.02, alpha=0.15, lw=0,
                        color=line.get_color())
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")
    return ax


def example_scatter(ax=None):
    """Predicted-vs-measured scatter with identity line and R^2."""
    rng = np.random.default_rng(2)
    n = 250
    true = rng.uniform(-3, 3, n)
    pred = true + rng.normal(0, 0.45, n)
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)
    made = ax is None
    if made:
        _, ax = new_fig(COL1)
    # rasterized=True keeps the PDF small for big point clouds
    # while axes and text stay as editable vector graphics.
    ax.scatter(true, pred, s=5, alpha=0.55, edgecolor="none",
               color=WONG["blue"], rasterized=True)
    lim = [-3.6, 3.6]
    ax.plot(lim, lim, ls="--", lw=0.5, color="0.4", zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.text(0.05, 0.9, f"$R^2$ = {r2:.2f}", transform=ax.transAxes)
    return ax


def example_heatmap(fig=None, ax=None):
    """Heatmap with a slim colourbar — confusion / similarity / perf grid."""
    labels = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(3)
    m = rng.uniform(0, 0.4, (5, 5))
    np.fill_diagonal(m, rng.uniform(0.8, 1.0, 5))
    made = ax is None
    if made:
        fig, ax = new_fig(COL1)
    im = ax.imshow(m, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Score")
    return ax


def example_violin(ax=None):
    """Violin plot with median lines — distribution comparison across groups."""
    rng = np.random.default_rng(4)
    groups = ["Ours", "Baseline", "Random"]
    data = [rng.normal(mu, sd, 200)
            for mu, sd in [(0.82, 0.07), (0.66, 0.11), (0.50, 0.15)]]
    made = ax is None
    if made:
        _, ax = new_fig(COL1)
    parts = ax.violinplot(data, showmeans=False, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(WONG["skyblue"])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_linewidth(0.5)
            parts[key].set_color("black")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    return ax


def example_box_strip(ax=None):
    """Box plot + jittered individual points + exact-P bracket.

    Nature's statistics policy: for small n, show every data point
    (dot/box plots) rather than bar +/- SEM, and report exact P values.
    State n and the test in the figure legend.
    """
    rng = np.random.default_rng(5)
    groups = ["Baseline", "Ours"]
    data = [rng.normal(0.66, 0.05, 12), rng.normal(0.80, 0.04, 12)]
    made = ax is None
    if made:
        _, ax = new_fig(COL1 * 0.62, COL1 * 0.72)
    ax.boxplot(data, widths=0.5, showfliers=False,
               medianprops={"color": "black", "lw": 0.8},
               boxprops={"lw": 0.5}, whiskerprops={"lw": 0.5},
               capprops={"lw": 0.5})
    for i, (d, c) in enumerate(zip(data, [WONG["blue"], WONG["vermilion"]])):
        x = i + 1 + rng.uniform(-0.14, 0.14, d.size)  # jitter
        ax.scatter(x, d, s=6, color=c, alpha=0.85,
                   edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_ylim(0.5, 0.95)
    sig_bracket(ax, 1, 2, 0.90, "$P$ = 3 × 10$^{-4}$")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_ylabel("Accuracy")
    return ax


def example_morandi():
    """Morandi-style two-panel demo — the compensations in action.

    Muted colours carry almost no hue signal, so identity rides on a
    second channel everywhere: one marker per line series (a), ink edges
    + direct value labels on bars (b). Max 3 line series in this style.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.36))
    # a) training curves: line + marker + low-alpha band of the same colour
    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 25)
    series = {"Ours": 0.90, "Baseline": 0.72, "Ablation": 0.55}
    for (name, k), c, m in zip(series.items(), MORANDI_LINES, MORANDI_MARKERS):
        y = k * (1 - np.exp(-x / 3)) + rng.normal(0, 0.006, x.size)
        ax1.plot(x, y, marker=m, markersize=2.5, markevery=3,
                 lw=1.2, color=c, label=name)
        ax1.fill_between(x, y - 0.04, y + 0.04, color=c, alpha=0.15, lw=0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower right")
    # b) one bar per method: fill palette + ink edge + value on the cap
    methods = ["Ours", "SOTA", "Base", "Rand.", "Chance"]
    vals = [0.86, 0.79, 0.71, 0.64, 0.52]
    bars = ax2.bar(methods, vals, 0.62, color=PALETTE_MORANDI,
                   edgecolor=MORANDI_INK, linewidth=0.4)
    ax2.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    for ax, letter in zip((ax1, ax2), "ab"):
        panel_label(ax, letter)
    fig.tight_layout(w_pad=2.5)
    return fig


def example_multipanel():
    """A realistic double-column, 4-panel Nature figure with a,b,c,d labels."""
    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.62))
    example_grouped_bar(axes[0, 0])
    example_scaling(axes[0, 1])
    example_scatter(axes[1, 0])
    example_box_strip(axes[1, 1])
    for ax, letter in zip(axes.flat, "abcd"):
        panel_label(ax, letter)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    return fig


set_style()  # apply on import


if __name__ == "__main__":
    out = Path("figures")
    # Single-panel examples
    for name, func in [("bar", example_grouped_bar),
                       ("scaling", example_scaling),
                       ("scatter", example_scatter),
                       ("violin", example_violin),
                       ("box_strip", example_box_strip)]:
        fig, ax = new_fig(COL1)
        func(ax)
        save(fig, out / f"fig_{name}")
    # Heatmap needs the figure handle for the colourbar
    fig, ax = new_fig(COL1)
    example_heatmap(fig, ax)
    save(fig, out / "fig_heatmap")
    # Multi-panel
    save(example_multipanel(), out / "fig_multipanel")
    # Morandi variant
    save(example_morandi(), out / "fig_morandi")
    print("All figures written to ./figures/")
