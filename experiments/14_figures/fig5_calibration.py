import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("seaborn-v0_8-whitegrid")

OUT_DIR = "results/uci/figures"
os.makedirs(OUT_DIR, exist_ok=True)

P1_COL    = "#1565C0"
P3_COL    = "#E65100"
CONF_COL  = "#00838F"
UCI_TITLE = "#1565C0"
GEF_TITLE = "#2E7D32"

uci    = pd.read_csv("results/uci/tables/confidence_proxies_uci.csv",       index_col=0)
gef    = pd.read_csv("results/gefcom/tables/confidence_proxies_gefcom.csv",  index_col=0)
uci_cf = pd.read_csv("results/uci/tables/conformal_uci.csv",                index_col=0)
gef_cf = pd.read_csv("results/gefcom/tables/conformal_gefcom.csv",           index_col=0)

N_BINS = 10

def rel_curve(proxy_vals, errors, log_x=False):
    bins = pd.qcut(proxy_vals, q=N_BINS, duplicates="drop")
    g    = pd.DataFrame({"p": proxy_vals, "e": errors, "b": bins}).groupby("b")
    x, y = g["p"].mean().values, g["e"].mean().values
    if log_x:
        xn = (np.log1p(x) - np.log1p(x.min())) / (np.log1p(x.max()) - np.log1p(x.min()) + 1e-9)
    else:
        xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return xn, y

def cov_curve(actual, lower, upper, proxy_vals, log_x=False):
    cov  = ((actual >= lower) & (actual <= upper)).astype(float)
    bins = pd.qcut(proxy_vals, q=N_BINS, duplicates="drop")
    g    = pd.DataFrame({"p": proxy_vals, "c": cov, "b": bins}).groupby("b")
    x, y = g["p"].mean().values, g["c"].mean().values
    if log_x:
        xn = (np.log1p(x) - np.log1p(x.min())) / (np.log1p(x.max()) - np.log1p(x.min()) + 1e-9)
    else:
        xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return xn, y

def draw_panel(ax, df, cf, title, title_color, p2_color,
               log_x=False, overall_cov=None):

    # P1
    x1, y1 = rel_curve(df["ensemble_variance"], df["lstm_abs_error"], log_x)
    ax.plot(x1, y1, linestyle="-", color=P1_COL, linewidth=1.1,
            zorder=4, label="P1 — Ensemble Variance")
    ax.scatter(x1, y1, color=P1_COL, s=18, zorder=5, linewidths=0)

    # P3
    x3, y3 = rel_curve(df["resid_volatility"], df["lstm_abs_error"], log_x)
    ax.plot(x3, y3, linestyle="--", color=P3_COL, linewidth=1.1,
            zorder=4, label="P3 — Residual Volatility")
    ax.scatter(x3, y3, color=P3_COL, s=18, marker="s", zorder=5, linewidths=0)

    # P2 box — bottom right near x-axis
    ax.text(0.97, 0.03,
            "P2 (ARIMA PI Width): constant signal — no curve",
            transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom",
            color=p2_color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=p2_color, linewidth=1.2, alpha=0.97),
            zorder=10)

    # conformal right axis
    ax2 = ax.twinx()
    xc, yc = cov_curve(df["actual_load"].values,
                       cf["conformal_lower"].values,
                       cf["conformal_upper"].values,
                       df["ensemble_variance"].values, log_x)
    ax2.plot(xc, yc, linestyle=":", color=CONF_COL, linewidth=1.1,
             zorder=3, label="Conformal coverage")
    ax2.scatter(xc, yc, color=CONF_COL, s=18, marker="^", zorder=4, linewidths=0)

    # target 0.90 line — visible but NO text label
    ax2.axhline(0.90, color=CONF_COL, linewidth=1.2, linestyle="-", alpha=0.7)

    # overall coverage annotation — right side, clean format
    if overall_cov is not None:
        ax2.text(0.97, 0.38,
                 f"overall coverage = {overall_cov:.3f} (< 0.90)",
                 transform=ax2.transAxes, fontsize=7,
                 ha="right", va="top", color=CONF_COL, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                           edgecolor=CONF_COL, linewidth=0.9, alpha=0.93))

    ax2.set_ylabel("Conformal Coverage Rate", fontsize=10, color=CONF_COL)
    ax2.tick_params(axis="y", labelcolor=CONF_COL, labelsize=9)
    ax2.set_ylim(0.25, 1.05)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95)

    xlabel = "Proxy Value Percentile — log-normalised (low → high)" if log_x \
             else "Proxy Value Percentile (low → high)"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Mean Absolute LSTM Error (MWh)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=10)
    ax.tick_params(labelsize=10)
    ax.set_xlim(-0.02, 1.02)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.45)

draw_panel(axes[0], uci, uci_cf,
           "UCI — Portugal Grid", UCI_TITLE, UCI_TITLE,
           log_x=False, overall_cov=0.843)

draw_panel(axes[1], gef, gef_cf,
           "GEFCom2014 — New England Grid", GEF_TITLE, GEF_TITLE,
           log_x=True, overall_cov=0.875)

plt.savefig(os.path.join(OUT_DIR, "fig5_calibration.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "fig5_calibration.png"), dpi=300, bbox_inches="tight")
print("Saved.")
