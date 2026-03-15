"""
F1 — Error by Demand Regime (Boxplot)
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig1_error_regime.pdf
         results/uci/figures/fig1_error_regime.png
Run from project root:
    cd ~/projects/energy-forecasting-uncertainty-proxies
    python experiments/14_figures/fig1_error_regime.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --- PDD colours ---
UCI_BLUE  = "#1565C0"
GEF_GREEN = "#2E7D32"
NORM_GREY = "#B0BEC5"

plt.style.use("seaborn-v0_8-whitegrid")

# --- paths (relative to project root) ---
UCI_PROXY = "results/uci/tables/confidence_proxies_uci.csv"
GEF_PROXY = "results/gefcom/tables/confidence_proxies_gefcom.csv"
OUT_DIR   = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load data ---
uci = pd.read_csv(UCI_PROXY, index_col=0)
gef = pd.read_csv(GEF_PROXY, index_col=0)

uci["regime"] = uci["is_extreme_demand"].map({0: "Normal", 1: "Extreme"})
gef["regime"] = gef["is_extreme_demand"].map({0: "Normal", 1: "Extreme"})

# --- draw panel ---
def draw_panel(ax, df, title, color_ext, ylabel):
    data_norm = df.loc[df["regime"] == "Normal",  "lstm_abs_error"].values
    data_ext  = df.loc[df["regime"] == "Extreme", "lstm_abs_error"].values

    bp = ax.boxplot(
        [data_norm, data_ext],
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2, alpha=0.3, linestyle="none"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2)
    )

    bp["boxes"][0].set_facecolor(NORM_GREY)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(color_ext)
    bp["boxes"][1].set_alpha(0.85)

    for flier, col in zip(bp["fliers"], [NORM_GREY, color_ext]):
        flier.set_markerfacecolor(col)
        flier.set_markeredgecolor(col)

    norm_mean = data_norm.mean()
    ext_mean  = data_ext.mean()
    ax.plot(1, norm_mean, "D", color="white", markeredgecolor="black",
            markersize=7, zorder=5)
    ax.plot(2, ext_mean,  "D", color="white", markeredgecolor="black",
            markersize=7, zorder=5)

    pct = (ext_mean - norm_mean) / norm_mean * 100
    ax.annotate(
        f"+{pct:.1f}% at\nextreme demand",
        xy=(2, ext_mean), xytext=(2.35, ext_mean),
        fontsize=9, color=color_ext, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color_ext, lw=1.2),
        va="center"
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Normal\nDemand", "Extreme\nDemand"], fontsize=10)
    ax.tick_params(axis="x", pad=18)
    # --- title coloured to match dataset PDD colour ---
    ax.set_title(title, fontsize=13, fontweight="bold",
                 pad=10, color=color_ext)
    if ylabel:
        ax.set_ylabel("Absolute Forecast Error (MWh)", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(0.5, 3.0)

    n_norm = len(data_norm)
    n_ext  = len(data_ext)
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    ax.text(1, ymin - ymax * 0.06, f"n={n_norm:,}",
            ha="center", fontsize=9, color="gray")
    ax.text(2, ymin - ymax * 0.06, f"n={n_ext:,}",
            ha="center", fontsize=9, color="gray")

# --- figure ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
fig.subplots_adjust(wspace=0.35, bottom=0.18)

draw_panel(axes[0], uci, "UCI — Portugal Grid",
           color_ext=UCI_BLUE, ylabel=True)
draw_panel(axes[1], gef, "GEFCom2014 — New England Grid",
           color_ext=GEF_GREEN, ylabel=False)

# --- legend ---
mean_patch = plt.Line2D([0], [0], marker="D", color="w",
                         markerfacecolor="white", markeredgecolor="black",
                         markersize=7, label="Mean")
norm_patch = mpatches.Patch(facecolor=NORM_GREY, alpha=0.75, label="Normal demand")
uci_patch  = mpatches.Patch(facecolor=UCI_BLUE,  alpha=0.85, label="Extreme demand (UCI)")
gef_patch  = mpatches.Patch(facecolor=GEF_GREEN, alpha=0.85, label="Extreme demand (GEFCom2014)")

fig.legend(handles=[norm_patch, uci_patch, gef_patch, mean_patch],
           loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

# --- save ---
pdf_path = os.path.join(OUT_DIR, "fig1_error_regime.pdf")
png_path = os.path.join(OUT_DIR, "fig1_error_regime.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
