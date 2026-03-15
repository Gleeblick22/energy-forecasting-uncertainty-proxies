"""
F2 — Proxy vs Error Scatter
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig2_proxy_scatter.pdf
         results/uci/figures/fig2_proxy_scatter.png
Run from project root:
    cd ~/projects/energy-forecasting-uncertainty-proxies
    python experiments/14_figures/fig2_proxy_scatter.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

# --- PDD colours ---
UCI_BLUE  = "#1565C0"
GEF_GREEN = "#2E7D32"
EXT_RED   = "#C00000"
NORM_GREY = "#90A4AE"

plt.style.use("seaborn-v0_8-whitegrid")

# --- paths ---
UCI_PROXY = "results/uci/tables/confidence_proxies_uci.csv"
GEF_PROXY = "results/gefcom/tables/confidence_proxies_gefcom.csv"
OUT_DIR   = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load ---
uci = pd.read_csv(UCI_PROXY, index_col=0)
gef = pd.read_csv(GEF_PROXY, index_col=0)

# --- draw panel ---
def draw_panel(ax, df, title, grid_color, rho_all, rho_ext,
               show_threshold=False, y_clip_q=0.995):

    norm = df[df["is_extreme_demand"] == 0]
    ext  = df[df["is_extreme_demand"] == 1]

    y_clip  = df["lstm_abs_error"].quantile(y_clip_q)
    x_min   = df["ensemble_variance"][df["ensemble_variance"] > 0].min()
    x_clip  = df["ensemble_variance"].quantile(0.99)

    # log scale x — consistent across both panels
    ax.set_xscale("log")

    # normal points
    ax.scatter(norm["ensemble_variance"], norm["lstm_abs_error"],
               c=NORM_GREY, alpha=0.18, s=5, linewidths=0)

    # extreme points
    ax.scatter(ext["ensemble_variance"], ext["lstm_abs_error"],
               c=EXT_RED, alpha=0.55, s=14, linewidths=0)

    # 75th percentile threshold line — UCI only
    if show_threshold:
        err_75 = df["lstm_abs_error"].quantile(0.75)
        ax.axhline(err_75, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.6, zorder=2)
        # label placed above line, anchored left
        ax.text(x_min * 3, err_75 + y_clip * 0.04,
                f"75th pctile ({err_75:.1f} MWh)",
                ha="left", fontsize=8, color="black", alpha=0.75)

    # rho annotation — bottom right
    ax.text(0.97, 0.05,
            f"ρ_all = {rho_all:+.3f}\nρ_extreme = {rho_ext:+.3f}",
            transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=grid_color, linewidth=1.5),
            color=grid_color, fontweight="bold")

    ax.set_xlim(left=x_min * 0.8, right=x_clip * 1.5)
    ax.set_ylim(bottom=0, top=y_clip)

    # legend top left
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=NORM_GREY, markersize=7,
               label=f"Normal (n={len(norm):,})"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=EXT_RED, markersize=7,
               label=f"Extreme (n={len(ext):,})")
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=9, frameon=True, framealpha=0.9)

    ax.set_xlabel("P1 — Ensemble Variance (MWh², log scale)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color=grid_color)
    ax.tick_params(labelsize=10)

# --- figure ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.subplots_adjust(wspace=0.32)

draw_panel(axes[0], uci,
           title="UCI — Portugal Grid",
           grid_color=UCI_BLUE,
           rho_all=0.1893, rho_ext=0.0089,
           show_threshold=False,
           y_clip_q=0.995)

draw_panel(axes[1], gef,
           title="GEFCom2014 — New England Grid",
           grid_color=GEF_GREEN,
           rho_all=0.4396, rho_ext=0.4815,
           show_threshold=False,
           y_clip_q=0.97)

axes[0].set_ylabel("Absolute LSTM Forecast Error (MWh)", fontsize=12)
axes[1].set_ylabel("")

# --- save ---
pdf_path = os.path.join(OUT_DIR, "fig2_proxy_scatter.pdf")
png_path = os.path.join(OUT_DIR, "fig2_proxy_scatter.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
