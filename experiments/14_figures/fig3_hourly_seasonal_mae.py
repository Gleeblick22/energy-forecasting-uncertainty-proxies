"""
fig4_hourly_seasonal_mae.py
---------------------------
Fig 3 — Hourly + Seasonal MAE combined (2x2 panel)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.family']     = 'serif'
rcParams['font.size']       = 9
rcParams['axes.titlesize']  = 9
rcParams['axes.labelsize']  = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi']      = 300
rcParams['axes.linewidth']  = 0.8
rcParams['grid.linewidth']  = 0.4
rcParams['lines.linewidth'] = 1.2

HOURLY_UCI   = "results/uci/tables/hourly_mae_uci.csv"
HOURLY_GEF   = "results/gefcom/tables/hourly_mae_gefcom.csv"
SEASONAL_UCI = "results/uci/tables/seasonal_mae_uci.csv"
SEASONAL_GEF = "results/gefcom/tables/seasonal_mae_gefcom.csv"
OUT_PDF      = "results/uci/figures/fig4_hourly_seasonal_mae.pdf"
OUT_PNG      = "results/uci/figures/fig4_hourly_seasonal_mae.png"

C_LSTM  = '#1f4e79'
C_ARIMA = '#843c0c'
C_UCI   = '#1565C0'
C_GEF   = '#2E7D32'

plt.style.use("seaborn-v0_8-whitegrid")

h_uci = pd.read_csv(HOURLY_UCI)
h_gef = pd.read_csv(HOURLY_GEF)
s_uci = pd.read_csv(SEASONAL_UCI)
s_gef = pd.read_csv(SEASONAL_GEF)

SEASON_ORDER = ['Spring', 'Summer', 'Autumn', 'Winter']
s_uci = s_uci.set_index('season').reindex(SEASON_ORDER).reset_index()
s_gef = s_gef.set_index('season').reindex(SEASON_ORDER).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
fig.subplots_adjust(hspace=0.52, wspace=0.35)

fig.text(0.01, 0.74, 'Hourly',   va='center', rotation='vertical',
         fontsize=9, fontweight='bold', color='#333')
fig.text(0.01, 0.27, 'Seasonal', va='center', rotation='vertical',
         fontsize=9, fontweight='bold', color='#333')

# UCI title in blue, GEFCom title in green
fig.text(0.30, 0.97, 'UCI — Portugal (Mediterranean)',
         ha='center', fontsize=9, fontweight='bold', color=C_UCI)
fig.text(0.73, 0.97, 'GEFCom2014 — New England (Continental)',
         ha='center', fontsize=9, fontweight='bold', color=C_GEF)

# (a) UCI Hourly
ax = axes[0, 0]
ax.plot(h_uci['hour'], h_uci['lstm_mae'],
        color=C_LSTM,  label='LSTM ensemble', linewidth=1.2)
ax.plot(h_uci['hour'], h_uci['arima_mae'],
        color=C_ARIMA, label='SARIMA', linestyle='--', linewidth=1.0)
ax.set_xlabel('Hour of day')
ax.set_ylabel('MAE (MWh)')
ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
ax.set_xticklabels(['00', '04', '08', '12', '16', '20', '23'])
ax.legend(frameon=True, framealpha=0.9, edgecolor='#ccc', loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate('(a)', xy=(0.04, 0.92), xycoords='axes fraction',
            fontsize=8, fontstyle='italic')

# (b) GEFCom Hourly
ax = axes[0, 1]
ax.plot(h_gef['hour'], h_gef['lstm_mae'],
        color=C_LSTM,  label='LSTM ensemble', linewidth=1.2)
ax.plot(h_gef['hour'], h_gef['arima_mae'],
        color=C_ARIMA, label='SARIMA', linestyle='--', linewidth=1.0)
ax.set_xlabel('Hour of day')
ax.set_ylabel('MAE (MWh)')
ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
ax.set_xticklabels(['00', '04', '08', '12', '16', '20', '23'])
ax.legend(frameon=True, framealpha=0.9, edgecolor='#ccc', loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate('(b)', xy=(0.04, 0.92), xycoords='axes fraction',
            fontsize=8, fontstyle='italic')

# (c) UCI Seasonal
ax = axes[1, 0]
x     = np.arange(len(SEASON_ORDER))
width = 0.32
ax.bar(x - width/2, s_uci['lstm_mae'],  width,
       label='LSTM ensemble', color=C_LSTM,  alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.bar(x + width/2, s_uci['arima_mae'], width,
       label='SARIMA',        color=C_ARIMA, alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.set_xlabel('Season')
ax.set_ylabel('MAE (MWh)')
ax.set_xticks(x)
ax.set_xticklabels(SEASON_ORDER)
ax.legend(frameon=True, framealpha=0.9, edgecolor='#ccc', loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate('(c)', xy=(0.04, 0.92), xycoords='axes fraction',
            fontsize=8, fontstyle='italic')

# (d) GEFCom Seasonal
ax = axes[1, 1]
ax.bar(x - width/2, s_gef['lstm_mae'],  width,
       label='LSTM ensemble', color=C_LSTM,  alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.bar(x + width/2, s_gef['arima_mae'], width,
       label='SARIMA',        color=C_ARIMA, alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.set_xlabel('Season')
ax.set_ylabel('MAE (MWh)')
ax.set_xticks(x)
ax.set_xticklabels(SEASON_ORDER)
ax.legend(frameon=True, framealpha=0.9, edgecolor='#ccc', loc='upper center',
          bbox_to_anchor=(0.5, 0.99))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate('(d)', xy=(0.04, 0.92), xycoords='axes fraction',
            fontsize=8, fontstyle='italic')

plt.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
plt.close()
