"""
Fig 11 — Adaptive P2 vs Static P2 Comparison
Project: When AI Forecasts Are Confidently Wrong
Output:  results/uci/figures/fig11_adaptive_p2.pdf
         results/uci/figures/fig11_adaptive_p2.png
Run from project root:
    python experiments/14_figures/fig11_adaptive_p2.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --- PDD colours ---
UCI_BLUE   = "#1565C0"
GEF_GREEN  = "#2E7D32"
STATIC_CLR = "#757575"

plt.style.use("seaborn-v0_8-whitegrid")

# --- paths ---
CMP_PATH = "results/16_adaptive_p2/adaptive_p2_comparison.csv"
OUT_DIR  = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load ---
cmp = pd.read_csv(CMP_PATH)
uci = cmp[cmp['grid'] == 'uci'].iloc[0]
gc  = cmp[cmp['grid'] == 'gefcom'].iloc[0]

# --- data ---
DATA = {
    'uci': {
        'rho':       (abs(uci['static_p2_rho_extreme']), abs(uci['adaptive_p2_rho_extreme'])),
        'dangerous': (uci['static_p2_dangerous'],        uci['adaptive_p2_dangerous']),
        'winkler':   (uci['static_p2_winkler'],          uci['adaptive_p2_winkler']),
        'color':     UCI_BLUE,
        'label':     'UCI — Portugal Grid',
        'rq6':       'Failure Persists',
        'rq6_color': '#C62828',
    },
    'gefcom': {
        'rho':       (abs(gc['static_p2_rho_extreme']), abs(gc['adaptive_p2_rho_extreme'])),
        'dangerous': (gc['static_p2_dangerous'],        gc['adaptive_p2_dangerous']),
        'winkler':   (gc['static_p2_winkler'],          gc['adaptive_p2_winkler']),
        'color':     GEF_GREEN,
        'label':     'GEFCom2014 — New England Grid',
        'rq6':       'Restores Significance',
        'rq6_color': GEF_GREEN,
    },
}

METRICS = [
    ('rho',       '|Spearman ρ| at Extreme Hours\n(higher = better)', True),
    ('dangerous', 'DANGEROUS Quadrant Rate\n(lower = better)',        False),
    ('winkler',   'Winkler Score\n(lower = better)',                  False),
]

GRIDS   = ['uci', 'gefcom']
W       = 0.28
X_STATI = 0.20
X_ADAP  = 0.60

fig, axes = plt.subplots(3, 2, figsize=(10, 11))
fig.subplots_adjust(hspace=0.60, wspace=0.38, bottom=0.10)

for col, grid in enumerate(GRIDS):
    d = DATA[grid]

    for row, (mkey, ylabel, higher_better) in enumerate(METRICS):
        ax = axes[row][col]

        sv, av = d[mkey]

        # Draw bars
        ax.bar(X_STATI, sv, W, color=STATIC_CLR, alpha=0.75,
               edgecolor='white', linewidth=1.0)
        ax.bar(X_ADAP,  av, W, color=d['color'], alpha=0.85,
               edgecolor='white', linewidth=1.0)

        # Y limit with padding
        top = max(sv, av) * 1.50
        ax.set_ylim(0, top)

        # Value labels — well above bars
        offset = max(sv, av) * 0.08
        ax.text(X_STATI, sv + offset, f'{sv:.4f}',
                ha='center', va='bottom',
                fontsize=9, color=STATIC_CLR, fontweight='bold')
        ax.text(X_ADAP,  av + offset, f'{av:.4f}',
                ha='center', va='bottom',
                fontsize=9, color=d['color'], fontweight='bold')

        # Improvement indicator between bars
        improved = (av > sv) if higher_better else (av < sv)
        pct      = abs(av - sv) / abs(sv) * 100 if sv != 0 else 0
        arrow    = '▼' if improved else '▲'
        ind_col  = d['color'] if improved else '#C62828'
        ax.text(0.40, max(sv, av) + offset * 2.2,
                f'{arrow} {pct:.0f}%',
                ha='center', va='bottom',
                fontsize=9, color=ind_col, fontweight='bold')

        # X axis
        ax.set_xticks([X_STATI, X_ADAP])
        ax.set_xticklabels(['Static P2', 'Adaptive P2'], fontsize=10)
        ax.set_xlim(0, 0.80)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Column title — top row only
        if row == 0:
            ax.set_title(
                f'{d["label"]}\n{d["rq6"]}',
                fontsize=11, fontweight='bold',
                color=d['color'], pad=8
            )

# --- legend ---
static_patch = mpatches.Patch(facecolor=STATIC_CLR, alpha=0.75,
                               label='Static P2 (SARIMA PI Width)')
uci_patch    = mpatches.Patch(facecolor=UCI_BLUE,   alpha=0.85,
                               label='Adaptive P2 — UCI Portugal')
gef_patch    = mpatches.Patch(facecolor=GEF_GREEN,  alpha=0.85,
                               label='Adaptive P2 — GEFCom2014')

fig.legend(handles=[static_patch, uci_patch, gef_patch],
           loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 0.01),
           frameon=True, edgecolor='#CCCCCC')

# --- save ---
pdf_path = os.path.join(OUT_DIR, "fig11_adaptive_p2.pdf")
png_path = os.path.join(OUT_DIR, "fig11_adaptive_p2.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
