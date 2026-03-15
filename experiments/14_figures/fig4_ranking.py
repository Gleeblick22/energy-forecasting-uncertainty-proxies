"""
F4 — Proxy Ranking Bar Chart (Winkler Score)
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig4_ranking.pdf
         results/uci/figures/fig4_ranking.png
Run from project root:
    cd ~/projects/energy-forecasting-uncertainty-proxies
    python experiments/14_figures/fig4_ranking.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# --- PDD colours ---
UCI_BLUE  = "#1565C0"
GEF_GREEN = "#2E7D32"

plt.style.use("seaborn-v0_8-whitegrid")

OUT_DIR = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- verified data from results_summary_FINAL.csv ---
methods = ["P1\nEnsemble\nVariance", "P2\nARIMA\nPI Width",
           "P3\nResidual\nVolatility", "Conformal\nBenchmark"]

uci_winkler = [202.6498, 171.4659, 220.0946, 166.1550]
gef_winkler = [ 55.1499,  97.4419,  45.9661,  75.3455]

x = np.arange(len(methods))

def draw_bars(ax, values, title, color):
    bars = ax.bar(x, values, width=0.65, color=color,
                  alpha=0.85, edgecolor="white", linewidth=1.2,
                  zorder=3)

    # value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.012,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

    # conformal reference line — label at left edge above line
    conf_val = values[-1]
    ax.axhline(conf_val, color="black", linewidth=1.2,
               linestyle="--", alpha=0.5, zorder=2)
    ax.text(x[0] - 0.32, conf_val + max(values) * 0.025,
            "Conformal benchmark",
            ha="left", fontsize=8, color="black", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Winkler Score (lower = better)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=color, pad=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, max(values) * 1.20)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(wspace=0.25)

draw_bars(axes[0], uci_winkler, "UCI — Portugal Grid",           UCI_BLUE)
draw_bars(axes[1], gef_winkler, "GEFCom2014 — New England Grid", GEF_GREEN)

# --- save ---
pdf_path = os.path.join(OUT_DIR, "fig4_ranking.pdf")
png_path = os.path.join(OUT_DIR, "fig4_ranking.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
