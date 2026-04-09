"""
Degradation Curve — Proxy Reliability Across Demand Percentiles (Extension 1)
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig_degradation_curve.pdf
         results/uci/figures/fig_degradation_curve.png
Run from project root:
    python experiments/14_figures/fig_degradation_curve.py
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

# --- paths ---
DATA_PATH = "results/15_degradation_curve/degradation_results.csv"
OUT_DIR   = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load ---
df = pd.read_csv(DATA_PATH)

BONFERRONI_ALPHA = 0.0083

PROXY_STYLE = {
    'P1': {'color': UCI_BLUE,  'ls': '-',  'lw': 1.8, 'marker': 'o', 'ms': 3.5, 'label': 'P1 Ensemble Variance'},
    'P2': {'color': NORM_GREY, 'ls': '--', 'lw': 1.4, 'marker': 's', 'ms': 3.0, 'label': 'P2 PI Width (ARIMA)'},
    'P3': {'color': GEF_GREEN, 'ls': ':',  'lw': 1.6, 'marker': '^', 'ms': 3.5, 'label': 'P3 Residual Volatility'},
}

UCI_COLLAPSE = {'P1': 81, 'P3': 85}

# --- draw panel ---
def draw_panel(ax, grid, title, grid_color, ylabel):
    gdf = df[df['grid'] == grid]

    for proxy, style in PROXY_STYLE.items():
        pdata = gdf[gdf['proxy'] == proxy].sort_values('pct')
        ax.plot(
            pdata['pct'], pdata['rho'],
            color=style['color'],
            ls=style['ls'],
            lw=style['lw'],
            marker=style['marker'],
            ms=style['ms'],
            label=style['label'],
            zorder=3
        )

    # Zero reference line
    ax.axhline(0, color='black', lw=0.8, ls='-', alpha=0.4, zorder=1)

    # Non-significant shaded band
    ax.axhspan(-0.08, 0.08, alpha=0.10, color=NORM_GREY, zorder=0)
    ax.text(70.4, 0.075, 'Non-significant zone (α=0.0083)',
            fontsize=9, color='gray', va='top')

    # Collapse annotations — UCI only
    if grid == 'uci':
        for proxy, pct in UCI_COLLAPSE.items():
            style = PROXY_STYLE[proxy]
            pdata = gdf[gdf['proxy'] == proxy].sort_values('pct')
            rho_val = pdata.loc[pdata['pct'] == pct, 'rho'].values
            if len(rho_val) > 0:
                ax.axvline(pct, color=style['color'],
                           ls=':', lw=1.2, alpha=0.8, zorder=2)
                ax.annotate(
                    f'{proxy}: collapses\nat {pct}th pct',
                    xy=(pct, rho_val[0]),
                    xytext=(pct + 1.5, rho_val[0] + 0.06),
                    fontsize=9,
                    color=style['color'],
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=style['color'], lw=1.2),
                    va='center'
                )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10, color=grid_color)
    ax.set_xlim(69.5, 98.5)
    ax.set_xticks(range(70, 99, 5))
    ax.set_xlabel('Demand Percentile Threshold', fontsize=12)
    if ylabel:
        ax.set_ylabel('Spearman ρ (proxy vs LSTM MAE)', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

# --- figure ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
fig.subplots_adjust(wspace=0.35, bottom=0.18)

draw_panel(axes[0], 'uci',    'UCI — Portugal Grid',           UCI_BLUE,  ylabel=True)
draw_panel(axes[1], 'gefcom', 'GEFCom2014 — New England Grid', GEF_GREEN, ylabel=False)

# --- legend ---
p1_patch = mpatches.Patch(facecolor=UCI_BLUE,  alpha=0.85, label='P1 Ensemble Variance')
p2_patch = mpatches.Patch(facecolor=NORM_GREY, alpha=0.75, label='P2 PI Width (ARIMA)')
p3_patch = mpatches.Patch(facecolor=GEF_GREEN, alpha=0.85, label='P3 Residual Volatility')

fig.legend(
    handles=[p1_patch, p2_patch, p3_patch],
    loc='lower center', ncol=3, fontsize=9,
    bbox_to_anchor=(0.5, -0.02), frameon=True
)

# --- save ---
pdf_path = os.path.join(OUT_DIR, "fig_degradation_curve.pdf")
png_path = os.path.join(OUT_DIR, "fig_degradation_curve.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
