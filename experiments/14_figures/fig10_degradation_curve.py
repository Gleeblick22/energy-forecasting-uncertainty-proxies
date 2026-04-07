"""
Fig 10 — Degradation Curve
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig10_degradation_curve.pdf
         results/uci/figures/fig10_degradation_curve.png
Run from project root:
    python experiments/14_figures/fig10_degradation_curve.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# --- PDD colours ---
UCI_BLUE  = "#1565C0"
GEF_GREEN = "#2E7D32"
NORM_GREY = "#9E9E9E"

plt.style.use("seaborn-v0_8-whitegrid")

# --- paths ---
DATA_PATH = "results/15_degradation_curve/degradation_results.csv"
OUT_DIR   = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load ---
df = pd.read_csv(DATA_PATH)

PROXY_STYLE = {
    'P1': {'color': UCI_BLUE,  'ls': '-',  'lw': 1.4, 'marker': 'o', 'ms': 2.5, 'label': 'P1 Ensemble Variance'},
    'P2': {'color': NORM_GREY, 'ls': '--', 'lw': 1.1, 'marker': 's', 'ms': 2.0, 'label': 'P2 PI Width (ARIMA)'},
    'P3': {'color': GEF_GREEN, 'ls': ':',  'lw': 1.3, 'marker': '^', 'ms': 2.5, 'label': 'P3 Residual Volatility'},
}

# --- draw panel ---
def draw_panel(ax, grid, panel_label, title, grid_color):
    gdf = df[df['grid'] == grid]

    # Plot all 3 proxies
    for proxy, style in PROXY_STYLE.items():
        pdata = gdf[gdf['proxy'] == proxy].sort_values('pct')
        ax.plot(
            pdata['pct'], pdata['rho'],
            color=style['color'],
            ls=style['ls'],
            lw=style['lw'],
            marker=style['marker'],
            ms=style['ms'],
            markevery=2,
            label=style['label'],
            zorder=3
        )

    # Zero reference line only — no band
    ax.axhline(0, color='black', lw=0.9, ls='-', alpha=0.5, zorder=1)

    # Panel label (a) / (b)
    ax.text(0.02, 0.98, '', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left', color='black')

    # ── UCI annotations ───────────────────────────────────────────
    if grid == 'uci':
        # P1 collapse at 81st — annotation below the collapse point
        p1_data = gdf[gdf['proxy']=='P1'].sort_values('pct')
        rho_81  = p1_data.loc[p1_data['pct']==81, 'rho'].values[0]
        ax.axvline(81, color=UCI_BLUE, ls=':', lw=0.9, alpha=0.7, zorder=2)
        ax.annotate(
            'P1 collapses at 81st pct',
            xy=(81, rho_81),
            xytext=(82, -0.20),
            fontsize=8.5, color=UCI_BLUE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=UCI_BLUE, lw=1.0,
                            connectionstyle='arc3,rad=0.25'),
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec=UCI_BLUE, lw=0.7, alpha=0.95),
            ha='left'
        )

        # P3 collapse at 85th — annotation on right side well below title
        p3_data = gdf[gdf['proxy']=='P3'].sort_values('pct')
        rho_85  = p3_data.loc[p3_data['pct']==85, 'rho'].values[0]
        ax.axvline(85, color=GEF_GREEN, ls=':', lw=0.9, alpha=0.7, zorder=2)
        ax.annotate(
            'P3 collapses at 85th pct',
            xy=(85, rho_85),
            xytext=(86, 0.18),
            fontsize=8.5, color=GEF_GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GEF_GREEN, lw=1.0,
                            connectionstyle='arc3,rad=-0.25'),
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec=GEF_GREEN, lw=0.7, alpha=0.95),
            ha='left'
        )

        # P2 label — flat near zero, placed top right
        ax.text(70.5, -0.03, 'P2: non-significant\nacross all pct',
                fontsize=8, color=NORM_GREY, style='italic',
                ha='left', va='top')

        # Y axis — UCI has wide range -0.30 to +0.25
        ax.set_ylim(-0.35, 0.28)

    # ── GEFCom annotations ────────────────────────────────────────
    if grid == 'gefcom':
        # P1 robust — label at mid percentile, upper area
        p1_data  = gdf[gdf['proxy']=='P1'].sort_values('pct')
        rho_p1_80 = p1_data.loc[p1_data['pct']==80, 'rho'].values[0]
        ax.annotate(
            'P1: significant\nacross all pct',
            xy=(80, rho_p1_80),
            xytext=(70.5, 0.52),
            fontsize=8.5, color=UCI_BLUE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=UCI_BLUE, lw=1.0,
                            connectionstyle='arc3,rad=-0.15'),
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec=UCI_BLUE, lw=0.7, alpha=0.95),
            ha='left'
        )

        # P3 robust — label at different percentile, lower area
        p3_data   = gdf[gdf['proxy']=='P3'].sort_values('pct')
        rho_p3_90 = p3_data.loc[p3_data['pct']==90, 'rho'].values[0]
        ax.annotate(
            'P3: significant\nacross all pct',
            xy=(90, rho_p3_90),
            xytext=(91, 0.52),
            fontsize=8.5, color=GEF_GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GEF_GREEN, lw=1.0,
                            connectionstyle='arc3,rad=0.15'),
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec=GEF_GREEN, lw=0.7, alpha=0.95),
            ha='left'
        )

        # P2 label — placed clearly separate from other text
        ax.text(70.5, 0.03, 'P2: non-significant\nacross all pct',
                fontsize=8, color=NORM_GREY, style='italic',
                ha='left', va='bottom')

        # Y axis — GEFCom all positive, compact range
        ax.set_ylim(-0.05, 0.60)

    # Common formatting
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10, color=grid_color)
    ax.set_xlim(69.5, 98.5)
    ax.set_xticks(range(70, 99, 5))
    ax.set_xlabel('Demand Percentile Threshold', fontsize=12)
    ax.set_ylabel('Spearman ρ (proxy vs LSTM MAE)', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

# --- figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
fig.subplots_adjust(wspace=0.35, bottom=0.18)

draw_panel(axes[0], 'uci',    '', 'UCI — Portugal Grid',           UCI_BLUE)
draw_panel(axes[1], 'gefcom', '', 'GEFCom2014 — New England Grid', GEF_GREEN)

# --- legend using lines ---
legend_handles = [
    mlines.Line2D([], [],
                  color=s['color'], ls=s['ls'], lw=s['lw'],
                  marker=s['marker'], ms=4, label=s['label'])
    for s in PROXY_STYLE.values()
]

fig.legend(
    handles=legend_handles,
    loc='lower center', ncol=3, fontsize=9,
    bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='#CCCCCC'
)

# --- save ---
pdf_path = os.path.join(OUT_DIR, "fig10_degradation_curve.pdf")
png_path = os.path.join(OUT_DIR, "fig10_degradation_curve.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
