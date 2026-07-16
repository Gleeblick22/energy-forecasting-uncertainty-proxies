"""
Temperature Ablation — Figure
P1 Spearman rho: With vs Without Temperature Features
GEFCom2014 Cross-Grid Ablation Study

Run from: ~/projects/energy-forecasting-uncertainty-proxies/
Command:  /usr/bin/python3 experiments/20_ablation_temperature/plot_ablation_figure.py
Output:   results/gefcom_ablation/figures/fig_ablation_temperature.pdf
          results/gefcom_ablation/figures/fig_ablation_temperature.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.size"] = 10
plt.style.use("seaborn-v0_8-whitegrid")
import numpy as np
from pathlib import Path

OUT_DIR = Path("results/uci/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

labels    = ["All hours", "Normal hours", "Extreme hours"]
with_temp = [0.4396, 0.4217, 0.4815]
no_temp   = [0.4298, 0.3908, 0.6069]

x     = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

bars1 = ax.bar(x - width/2, with_temp, width,
               label="With temperature (original)",
               color="#1565C0", zorder=3)
bars2 = ax.bar(x + width/2, no_temp, width,
               label="Without temperature (ablation)",
               color="#2E7D32", zorder=3)

ax.axhline(y=0.50, color="#888780", linewidth=1.2,
           linestyle="--", zorder=2,
           label="Cohen large effect (ρ = 0.50)")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.008,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom",
            fontsize=9, color="#1565C0", fontweight="bold")

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.008,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom",
            fontsize=9, color="#2E7D32", fontweight="bold")

ax.annotate("",
    xy=(x[2] + width/2, no_temp[2] - 0.01),
    xytext=(x[2] + width/2, with_temp[2] + 0.035),
    arrowprops=dict(arrowstyle="->",
                    color="#e67e22", lw=1.5))
ax.text(x[2] + width/2 + 0.13,
        (with_temp[2] + no_temp[2])/2,
        "Δρ = +0.125",
        fontsize=9, color="#e67e22", va="center")

ax.set_ylabel("Spearman ρ", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.30, 0.72)
ax.set_yticks([0.30, 0.35, 0.40, 0.45,
               0.50, 0.55, 0.60, 0.65, 0.70])
ax.yaxis.set_tick_params(labelsize=9)
ax.grid(axis="y", color="#e1e0d9",
        linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig13_ablation_temperature.pdf",
            dpi=1200, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig13_ablation_temperature.png",
            dpi=1200, bbox_inches="tight")
plt.close()
print("Saved PDF and PNG to results/gefcom_ablation/figures/")
