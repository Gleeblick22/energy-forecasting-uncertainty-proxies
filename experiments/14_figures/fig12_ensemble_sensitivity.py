"""
Fig 12 — Ensemble Size Sensitivity
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig12_ensemble_sensitivity.pdf
         results/uci/figures/fig12_ensemble_sensitivity.png
Run from project root:
    python experiments/14_figures/fig12_ensemble_sensitivity.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# --- PDD colours — identical to fig10 and fig11 ---
UCI_BLUE   = "#1565C0"
GEF_GREEN  = "#2E7D32"
NORM_GREY  = "#9E9E9E"
DANGER_RED = "#C62828"

plt.style.use("seaborn-v0_8-whitegrid")

# --- paths ---
DATA_PATH = "results/17_ensemble_sensitivity/sensitivity_results.csv"
OUT_DIR   = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load and sort ---
df     = pd.read_csv(DATA_PATH)
uci    = df[df['grid'] == 'uci'].sort_values('n_seeds').reset_index(drop=True)
gefcom = df[df['grid'] == 'gefcom'].sort_values('n_seeds').reset_index(drop=True)

seeds   = uci['n_seeds'].tolist()
uci_rho = uci['rho_extreme'].tolist()
gef_rho = gefcom['rho_extreme'].tolist()

BONFERRONI = 0.0083

# --- figure ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
fig.subplots_adjust(bottom=0.20)

# --- UCI line ---
ax.plot(seeds, uci_rho,
        color=UCI_BLUE, linestyle='--', linewidth=1.4,
        marker='s', markersize=6,
        markerfacecolor='white', markeredgecolor=UCI_BLUE,
        markeredgewidth=1.5,
        label='UCI Portugal (Mediterranean)',
        zorder=3)

# --- GEFCom line ---
ax.plot(seeds, gef_rho,
        color=GEF_GREEN, linestyle='-', linewidth=1.4,
        marker='o', markersize=6,
        markerfacecolor=GEF_GREEN, markeredgecolor=GEF_GREEN,
        label='GEFCom2014 New England (Continental)',
        zorder=3)

# --- Bonferroni threshold ---
ax.axhline(y=BONFERRONI, color='black', linestyle=':', linewidth=1.0,
           alpha=0.7, zorder=1,
           label='Bonferroni threshold (α = 0.0083)')

# --- Zero reference line ---
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8,
           alpha=0.4, zorder=1)

# --- Annotate UCI 10-seed borderline ---
rho_10_uci = uci.loc[uci['n_seeds'] == 10, 'rho_extreme'].values[0]
ax.annotate(
    'borderline at 10 seeds\n(p=0.008, exceeds α=0.0083)',
    xy=(10, rho_10_uci),
    xytext=(14, 0.22),
    fontsize=9, color=UCI_BLUE, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=UCI_BLUE, lw=1.0,
                    connectionstyle='arc3,rad=-0.25'),
    bbox=dict(boxstyle='round,pad=0.3', fc='white',
              ec=UCI_BLUE, lw=0.8, alpha=0.95),
    ha='left'
)

# --- Annotate GEFCom 50-seed strengthening ---
rho_50_gef = gefcom.loc[gefcom['n_seeds'] == 50, 'rho_extreme'].values[0]
ax.annotate(
    f'rho = +{rho_50_gef:.3f} at 50 seeds',
    xy=(50, rho_50_gef),
    xytext=(36, 0.57),
    fontsize=9, color=GEF_GREEN, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=GEF_GREEN, lw=1.0,
                    connectionstyle='arc3,rad=0.20'),
    bbox=dict(boxstyle='round,pad=0.3', fc='white',
              ec=GEF_GREEN, lw=0.8, alpha=0.95),
    ha='left'
)

# --- Annotate UCI non-significant region ---
ax.text(28, -0.03,
        'UCI: non-significant\nat all seed sizes',
        fontsize=9, color=DANGER_RED, style='italic',
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                  ec=DANGER_RED, lw=0.6, alpha=0.90))

# --- Axes formatting ---
ax.set_title('P1 Ensemble Variance Reliability Across Ensemble Sizes',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Ensemble Size (number of seeds)', fontsize=12)
ax.set_ylabel('Spearman rho at Extreme Demand Hours\n(proxy vs LSTM absolute error)', fontsize=12)
ax.set_xticks(seeds)
ax.set_xticklabels([str(s) for s in seeds], fontsize=11)
ax.set_ylim(-0.10, 0.68)
ax.set_xlim(3, 57)
ax.tick_params(axis='y', labelsize=10)

# --- Legend ---
legend_handles = [
    mlines.Line2D([], [],
                  color=UCI_BLUE, linestyle='--', linewidth=1.4,
                  marker='s', markersize=5,
                  markerfacecolor='white', markeredgecolor=UCI_BLUE,
                  label='UCI Portugal (Mediterranean)'),
    mlines.Line2D([], [],
                  color=GEF_GREEN, linestyle='-', linewidth=1.4,
                  marker='o', markersize=5,
                  markerfacecolor=GEF_GREEN,
                  label='GEFCom2014 New England (Continental)'),
    mlines.Line2D([], [],
                  color='black', linestyle=':', linewidth=1.0,
                  label='Bonferroni threshold (alpha = 0.0083)'),
]

fig.legend(
    handles=legend_handles,
    loc='lower center', ncol=3, fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
    frameon=True, edgecolor='#CCCCCC'
)

# --- Save ---
pdf_path = os.path.join(OUT_DIR, "fig12_ensemble_sensitivity.pdf")
png_path = os.path.join(OUT_DIR, "fig12_ensemble_sensitivity.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

# --- Validation ---
print("\nVALIDATION:")
print(f"  UCI    rho values: {uci_rho}")
print(f"  GEFCom rho values: {gef_rho}")
print(f"  Seed sizes:        {seeds}")
print("fig12 COMPLETE.")
