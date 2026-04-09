"""
Fig A — Proxy Reliability Degradation Curve (Extension 1)
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig_A_degradation_curve.png
         results/uci/figures/fig_A_degradation_curve.pdf
Run from project root:
    python experiments/14_figures/fig_A_degradation_curve.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# --- PDD colours (matching existing figures) ---
UCI_BLUE  = "#1565C0"
GEF_GREEN = "#2E7D32"
P1_BLACK  = "#000000"
P2_GREY   = "#757575"
P3_DARK   = "#424242"

plt.style.use("seaborn-v0_8-whitegrid")

# --- paths ---
DATA_PATH = "results/15_degradation_curve/degradation_results.csv"
OUT_DIR   = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load ---
df = pd.read_csv(DATA_PATH)

BONFERRONI_ALPHA = 0.0083

PROXY_STYLE = {
    'P1': {'color': P1_BLACK, 'ls': '-',  'lw': 1.8, 'marker': 'o', 'ms': 3.5, 'label': 'P1 Ensemble Variance'},
    'P2': {'color': P2_GREY,  'ls': '--', 'lw': 1.4, 'marker': 's', 'ms': 3.0, 'label': 'P2 PI Width (ARIMA)'},
    'P3': {'color': P3_DARK,  'ls': ':',  'lw': 1.6, 'marker': '^', 'ms': 3.5, 'label': 'P3 Residual Volatility'},
}

UCI_COLLAPSE = {'P1': 81, 'P3': 85}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.subplots_adjust(wspace=0.30)

PANELS = [
    ('uci',    'UCI Portugal',           UCI_BLUE),
    ('gefcom', 'GEFCom2014 New England', GEF_GREEN),
]

for ax, (grid, grid_label, grid_color) in zip(axes, PANELS):
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
    ax.axhspan(-0.08, 0.08, alpha=0.07, color='grey', zorder=0)
    ax.text(70.4, 0.075, 'Non-significant zone (α=0.0083)',
            fontsize=7.5, color='grey', va='top')

    # Collapse annotations — UCI only
    if grid == 'uci':
        for proxy, pct in UCI_COLLAPSE.items():
            style = PROXY_STYLE[proxy]
            pdata = gdf[gdf['proxy'] == proxy].sort_values('pct')
            rho_val = pdata.loc[pdata['pct'] == pct, 'rho'].values
            if len(rho_val) > 0:
                ax.axvline(pct, color=style['color'],
                           ls=':', lw=1.0, alpha=0.7, zorder=2)
                ax.annotate(
                    f'{proxy}: {pct}th pct',
                    xy=(pct, rho_val[0]),
                    xytext=(pct + 1.5, rho_val[0] + 0.05),
                    fontsize=7.5,
                    color=style['color'],
                    arrowprops=dict(arrowstyle='->', color=style['color'], lw=0.8),
                )

    # Grid colour strip on top border
    ax.spines['top'].set_color(grid_color)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Demand Percentile Threshold', fontsize=10)
    ax.set_ylabel('Spearman ρ (proxy vs LSTM MAE)', fontsize=10)
    ax.set_title(grid_label, fontsize=11, pad=8,
                 color=grid_color, fontweight='bold')
    ax.set_xlim(69.5, 98.5)
    ax.set_xticks(range(70, 99, 5))
    ax.tick_params(labelsize=9)

# Shared legend below both panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, -0.06),
    frameon=True,
    edgecolor='#CCCCCC',
    fontsize=9
)

fig.suptitle(
    'Fig A — Proxy Reliability Degradation Across Demand Percentiles (70th–98th)',
    fontsize=11, fontweight='bold', y=1.01
)

# --- save PNG and PDF ---
out_png = f"{OUT_DIR}/fig_A_degradation_curve.png"
out_pdf = f"{OUT_DIR}/fig_A_degradation_curve.pdf"
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
