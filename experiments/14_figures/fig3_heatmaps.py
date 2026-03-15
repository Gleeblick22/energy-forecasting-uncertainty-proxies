"""
F3 — 2x2 Overconfidence Heatmaps
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig3_heatmaps.pdf
         results/uci/figures/fig3_heatmaps.png
Run from project root:
    cd ~/projects/energy-forecasting-uncertainty-proxies
    python experiments/14_figures/fig3_heatmaps.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# --- colours ---
UCI_BLUE   = "#1565C0"
GEF_GREEN  = "#2E7D32"
DANGER_RED = "#B71C1C"   # slightly lighter red — white text readable
SAFE_COL   = "#C8E6C9"   # clear green
WARN_COL   = "#BBDEFB"   # clear blue
CAUT_COL   = "#FFF9C4"   # clear yellow — distinct from green
DANGER_COL = "#B71C1C"

plt.style.use("seaborn-v0_8-whitegrid")

OUT_DIR = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

data = {
    "uci": {
        "P1\nEnsemble\nVariance": {
            "cells": [[0.4086, 0.1586], [0.3414, 0.0914]],
            "or_overall": 0.091, "or_extreme": 0.011, "binom_p": 1.00000
        },
        "P2\nARIMA\nPI Width": {
            "cells": [[0.1550, 0.2075], [0.5950, 0.0425]],
            "or_overall": 0.042, "or_extreme": 0.053, "binom_p": 0.08570
        },
        "P3\nResidual\nVolatility": {
            "cells": [[0.4065, 0.1565], [0.3435, 0.0935]],
            "or_overall": 0.093, "or_extreme": 0.062, "binom_p": 0.99975
        }
    },
    "gefcom": {
        "P1\nEnsemble\nVariance": {
            "cells": [[0.4466, 0.1966], [0.3034, 0.0534]],
            "or_overall": 0.053, "or_extreme": 0.018, "binom_p": 1.00000
        },
        "P2\nARIMA\nPI Width": {
            "cells": [[0.2839, 0.1824], [0.4661, 0.0676]],
            "or_overall": 0.068, "or_extreme": 0.103, "binom_p": 0.00009
        },
        "P3\nResidual\nVolatility": {
            "cells": [[0.4271, 0.1771], [0.3229, 0.0729]],
            "or_overall": 0.073, "or_extreme": 0.086, "binom_p": 0.08188
        }
    }
}

proxies    = ["P1\nEnsemble\nVariance", "P2\nARIMA\nPI Width", "P3\nResidual\nVolatility"]
cell_names = [["SAFE", "WARNED"], ["CAUTIOUS", "DANGEROUS"]]
cell_bg    = [[SAFE_COL, WARN_COL], [CAUT_COL, DANGER_COL]]
cell_tc    = [["#1B5E20", "#0D47A1"], ["#5D4037", "white"]]

def draw_heatmap(ax, cells, grid_color, or_overall, or_extreme, binom_p):
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(2):
        for c in range(2):
            val  = cells[r][c]
            bg   = cell_bg[r][c]
            tc   = cell_tc[r][c]
            name = cell_names[r][c]
            x    = c
            y    = 1 - r  # flip: row 0 = top

            lw = 3.5 if name == "DANGEROUS" else 1.5
            ec = "#7B0000" if name == "DANGEROUS" else "white"

            rect = mpatches.FancyBboxPatch(
                (x + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.02",
                facecolor=bg, edgecolor=ec,
                linewidth=lw, zorder=2
            )
            ax.add_patch(rect)

            ax.text(x + 0.5, y + 0.65, name,
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color=tc, zorder=3)
            ax.text(x + 0.5, y + 0.30, f"{val*100:.1f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=tc, zorder=3)

    # column headers — horizontal above each column
    for c, lbl in enumerate(["Low Proxy", "High Proxy"]):
        ax.text(c + 0.5, 2.08, lbl,
                ha="center", va="bottom",
                fontsize=8, color="#555555")

    # row headers — horizontal left of each row
    for r, lbl in enumerate(["Low Error", "High Error"]):
        ax.text(-0.06, 1 - r + 0.5, lbl,
                ha="right", va="center",
                fontsize=8, color="#555555")

    # OR stats below
    elev = "(+)" if or_extreme > or_overall else "(-)"
    sig  = "***" if binom_p < 0.001 else ("*" if binom_p < 0.05 else "ns")
    ax.text(1.0, -0.14,
            f"OR_all={or_overall:.3f}  OR_ext={or_extreme:.3f} {elev}  p={binom_p:.5f} {sig}",
            ha="center", va="top", fontsize=7.5,
            color=grid_color, fontweight="bold")

    # border
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 1.96, 1.96,
        boxstyle="round,pad=0.0",
        facecolor="none", edgecolor=grid_color,
        linewidth=1.5, zorder=1
    ))

# --- figure: 3 rows x 2 cols ---
fig, axes = plt.subplots(3, 2, figsize=(9, 10))
fig.subplots_adjust(hspace=0.55, wspace=0.4,
                    left=0.14, right=0.97,
                    top=0.94, bottom=0.10)

col_titles  = ["UCI — Portugal Grid", "GEFCom2014 — New England Grid"]
col_colors  = [UCI_BLUE, GEF_GREEN]
datasets    = ["uci", "gefcom"]
row_labels  = ["P1\nEnsemble\nVariance", "P2\nARIMA\nPI Width", "P3\nResidual\nVolatility"]

for row_idx, proxy in enumerate(proxies):
    for col_idx, (dataset, gc) in enumerate(zip(datasets, col_colors)):
        ax = axes[row_idx][col_idx]
        d  = data[dataset][proxy]
        draw_heatmap(ax, d["cells"], gc,
                     d["or_overall"], d["or_extreme"], d["binom_p"])

        # proxy row label — left of left panel only, no overlap
        if col_idx == 0:
            fig.text(0.01, ax.get_position().y0 +
                     ax.get_position().height / 2,
                     row_labels[row_idx],
                     ha="left", va="center",
                     fontsize=9, fontweight="bold",
                     color="black")

        # column title — top row only
        if row_idx == 0:
            ax.set_title(col_titles[col_idx],
                         fontsize=12, fontweight="bold",
                         color=gc, pad=20)

# colour legend
legend_patches = [
    mpatches.Patch(facecolor=SAFE_COL,   edgecolor="#1B5E20", label="SAFE — low error, low proxy"),
    mpatches.Patch(facecolor=WARN_COL,   edgecolor="#0D47A1", label="WARNED — low error, high proxy"),
    mpatches.Patch(facecolor=CAUT_COL,   edgecolor="#5D4037", label="CAUTIOUS — high error, low proxy"),
    mpatches.Patch(facecolor=DANGER_COL, edgecolor="#7B0000", label="DANGEROUS — high error, high proxy"),
]
fig.legend(handles=legend_patches, loc="lower center",
           ncol=2, fontsize=8.5, frameon=True,
           bbox_to_anchor=(0.55, -0.04))

# --- save ---
pdf_path = os.path.join(OUT_DIR, "fig3_heatmaps.pdf")
png_path = os.path.join(OUT_DIR, "fig3_heatmaps.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
