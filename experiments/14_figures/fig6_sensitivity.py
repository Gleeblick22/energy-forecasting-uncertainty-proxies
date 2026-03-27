"""
fig6_sensitivity.py — Fig 6 Sensitivity comparison 90th vs 95th percentile
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

rcParams['font.family']     = 'serif'
rcParams['font.size']       = 9
rcParams['axes.labelsize']  = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi']      = 300
rcParams['axes.linewidth']  = 0.8

OUT_PDF = "results/uci/figures/fig6_sensitivity.pdf"
OUT_PNG = "results/uci/figures/fig6_sensitivity.png"

C_UCI  = '#1565C0'
C_GEF  = '#2E7D32'
C_BONF = '#e74c3c'

proxies = ['P1', 'P2', 'P3']

uci_90 = [ 0.0089, -0.0543,  0.0440]
uci_95 = [-0.1103, -0.0863,  0.0945]
gef_90 = [ 0.4815,  0.0182,  0.4554]
gef_95 = [ 0.4232, -0.0047,  0.4838]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.0), sharey=False)
fig.subplots_adjust(wspace=0.45, bottom=0.30, top=0.87)

x     = np.arange(len(proxies))
width = 0.30

def place_rho(ax, xpos, yval, offset=0.018):
    label = f'\u03c1={yval:+.3f}'
    if yval >= 0:
        y  = yval + offset
        va = 'bottom'
    else:
        y  = yval - offset
        va = 'top'
    ax.text(xpos, y, label, ha='center', va=va,
            fontsize=5.5, color='#333', fontstyle='italic')

# ── LEFT: UCI ─────────────────────────────────────────────────────────────────
ax = axes[0]
ax.bar(x - width/2, uci_90, width, color=C_UCI, alpha=1.0,
       edgecolor='white', linewidth=0.5)
ax.bar(x + width/2, uci_95, width, color=C_UCI, alpha=0.40,
       edgecolor='white', linewidth=0.5)
ax.axhline(0,      color='#555', linewidth=0.6, linestyle='-', zorder=0)
ax.axhline(0.0083, color=C_BONF, linewidth=0.8, linestyle='--', zorder=0)

for i in range(len(proxies)):
    place_rho(ax, x[i]-width/2, uci_90[i])
    place_rho(ax, x[i]+width/2, uci_95[i])

ax.set_title('UCI — Portugal (Mediterranean)',
             fontsize=9, fontweight='bold', color=C_UCI, pad=10)
ax.set_xlabel('Uncertainty proxy', labelpad=6)
ax.set_ylabel('Spearman \u03c1 at extreme demand hours')
ax.set_xticks(x)
ax.set_xticklabels(proxies)
ax.set_ylim(-0.32, 0.30)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── RIGHT: GEFCom ─────────────────────────────────────────────────────────────
ax = axes[1]
ax.bar(x - width/2, gef_90, width, color=C_GEF, alpha=1.0,
       edgecolor='white', linewidth=0.5)
ax.bar(x + width/2, gef_95, width, color=C_GEF, alpha=0.40,
       edgecolor='white', linewidth=0.5)
ax.axhline(0,      color='#555', linewidth=0.6, linestyle='-', zorder=0)
ax.axhline(0.0083, color=C_BONF, linewidth=0.8, linestyle='--', zorder=0)

for i in range(len(proxies)):
    place_rho(ax, x[i]-width/2, gef_90[i])
    place_rho(ax, x[i]+width/2, gef_95[i])

ax.set_title('GEFCom2014 — New England (Continental)',
             fontsize=9, fontweight='bold', color=C_GEF, pad=10)
ax.set_xlabel('Uncertainty proxy', labelpad=6)
ax.set_ylabel('Spearman \u03c1 at extreme demand hours')
ax.set_xticks(x)
ax.set_xticklabels(proxies)
ax.set_ylim(-0.10, 0.72)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── SHARED LEGEND ─────────────────────────────────────────────────────────────
legend_elements = [
    Patch(facecolor=C_UCI, alpha=1.0,  label='90th percentile threshold'),
    Patch(facecolor=C_UCI, alpha=0.40, label='95th percentile threshold'),
    Line2D([0],[0], color=C_BONF, linewidth=0.8,
           linestyle='--', label='Bonferroni \u03b1 = 0.0083'),
]
fig.legend(handles=legend_elements,
           loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, 0.12),
           frameon=True, framealpha=0.9,
           edgecolor='#ccc', fontsize=8)

fig.text(0.5, 0.03,
         'Bars above Bonferroni \u03b1 = 0.0083 threshold line are statistically significant (p < 0.0001). '
         'Bars below threshold are not significant.',
         ha='center', va='center', fontsize=7.5,
         color='#555', fontstyle='italic')

plt.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
plt.close()
