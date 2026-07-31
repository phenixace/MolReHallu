"""Figure 4c — answer->trace attention token heatmap (the signature 'molecular scratchpad' visual).
Renders a real reasoning trace with each token tinted by how much the ANSWER attends to it, showing
Chem-R keys on SMILES fragments while ChemDFM-R keys on scaffold/positional/nomenclature words.
Uses ATTENTION (what the abstract cites), not the spiky per-token gradient. Output: figures/fig4c.*
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)
EX = json.load(open(os.path.join(D, "token_examples", "token_examples.json")))
NAVY, TEAL, BRICK, GREEN, MUT, INK, SEC = "#3C5488", "#00A087", "#E64B35", "#008300", "#898781", "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("att", ["#f4f6fa", "#a9bcd8", NAVY])
TYPEC = {"SMILES_frag": TEAL, "FG_word": BRICK, "position_digit": GREEN}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8})
COLS = 96


def pick(model, spec):
    e = next((x for x in EX if x["model"] == model and x.get("spec") == spec), None)
    return [t for t in e["tokens"] if t["region"] == "trace"] if e else None


def dense_window(tr, size=150):
    a = np.array([t["attn_norm"] for t in tr])
    if len(tr) <= size:
        return tr
    best, bi = -1, 0
    for i in range(0, len(tr) - size, 5):
        s = a[i:i + size].sum()
        if s > best:
            best, bi = s, i
    return tr[bi:bi + size]


def render(ax, tr, title):
    a = np.array([t["attn_norm"] for t in tr])
    hi = np.percentile(a, 96) or 1.0
    norm = lambda v: min(1.0, (v / hi)) ** 0.75
    col, row = 0, 0
    for t in tr:
        txt = t["text"].replace("\n", " ") or " "
        w = max(1, len(txt))
        if col + w > COLS:
            col, row = 0, row + 1
        val = norm(t["attn_norm"])
        fc = CMAP(val)
        ec = TYPEC.get(t["cat"], "none")
        ax.add_patch(Rectangle((col, -row), w, 0.92, facecolor=fc, edgecolor=ec,
                               linewidth=1.1 if ec != "none" else 0, zorder=2))
        tcol = "white" if val > 0.55 else INK
        ax.text(col + w / 2, -row + 0.46, txt, ha="center", va="center", family="monospace",
                fontsize=5.2, color=tcol, zorder=3)
        col += w
    ax.set_xlim(0, COLS); ax.set_ylim(-row - 1, 1.2)
    ax.set_title(title, fontsize=8.5, loc="left", pad=4)
    ax.axis("off")
    return row + 1


def main():
    chem = dense_window(pick("Chem-R", "retrosynthesis/er0"))
    # ChemDFM: prefer same task; fall back to cap2mol if retro has no informative trace
    dfm_r = pick("ChemDFM-R", "retrosynthesis/er0")
    dfm = dense_window(dfm_r if dfm_r and len(dfm_r) > 40 else pick("ChemDFM-R", "cap2mol/er0"))
    dfm_spec = "retrosynthesis" if (dfm_r and len(dfm_r) > 40) else "cap2mol"

    fig = plt.figure(figsize=(7.4, 4.6))
    ax1 = fig.add_axes([0.03, 0.55, 0.82, 0.34])
    ax2 = fig.add_axes([0.03, 0.10, 0.82, 0.34])
    render(ax1, chem, "Chem-R  (retrosynthesis trace)  —  answer attends to SMILES fragments")
    render(ax2, dfm, f"ChemDFM-R  ({dfm_spec} trace)  —  answer attends to scaffold / positional / nomenclature words")
    # colorbar
    cax = fig.add_axes([0.88, 0.30, 0.017, 0.4])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("answer→token attention\n(within trace, normalized)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6); cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    # type legend
    for i, (lab, c) in enumerate([("SMILES fragment", TEAL), ("FG word", BRICK), ("position digit", GREEN)]):
        fig.add_artist(plt.Line2D([0.06 + i * 0.20], [0.95], marker="s", ms=7, mfc="none", mec=c, mew=1.6, ls=""))
        fig.text(0.075 + i * 0.20, 0.945, lab, fontsize=6.8, color=SEC, va="center")
    fig.suptitle("Figure 4c  |  Where each model looks in its own reasoning (answer→trace attention)",
                 x=0.03, ha="left", fontsize=10, fontweight="bold", y=0.995)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig4c.{ext}"), bbox_inches="tight", dpi=300)
    print(f"wrote fig4c  (Chem-R {len(chem)} tok, ChemDFM {len(dfm)} tok, dfm task={dfm_spec})", flush=True)


if __name__ == "__main__":
    main()
