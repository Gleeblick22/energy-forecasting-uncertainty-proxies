"""
sensitivity_95.py
-----------------
95th percentile demand threshold sensitivity analysis.
Recomputes Spearman correlations and OR metrics using the 95th percentile
as the extreme demand threshold (vs 90th percentile used in main analysis).
Appends results as new rows to results_summary_FINAL.csv.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────
UCI_PROXY    = "results/uci/tables/confidence_proxies_uci.csv"
GEF_PROXY    = "results/gefcom/tables/confidence_proxies_gefcom.csv"
SUMMARY_CSV  = "results/summary/results_summary_FINAL.csv"

DATASETS = [
    ("uci_95",    UCI_PROXY),
    ("gefcom_95", GEF_PROXY),
]

PROXIES = [
    ("P1", "ensemble_variance"),
    ("P2", "pi_width"),
    ("P3", "resid_volatility"),
]

def spearman(x, y):
    if len(x) < 5:
        return np.nan, np.nan, False
    rho, p = stats.spearmanr(x, y)
    return round(rho, 4), round(p, 6), bool(p < 0.0083)

def odds_ratio(proxy, error, proxy_thresh, error_thresh):
    dangerous = ((proxy < proxy_thresh) & (error > error_thresh)).sum()
    return round(dangerous / len(proxy), 4)

def analyse(label, path):
    print(f"\n{'='*60}")
    print(f"Dataset: {label}  |  file: {path}")
    print(f"{'='*60}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    thresh_95 = df["actual_load"].quantile(0.95)
    df["is_extreme_95"] = (df["actual_load"] >= thresh_95).astype(int)

    n_extreme = df["is_extreme_95"].sum()
    n_total   = len(df)
    print(f"  95th pct threshold : {thresh_95:.2f}")
    print(f"  n_extreme (95th)   : {n_extreme}  ({100*n_extreme/n_total:.1f}%)")
    print(f"  n_total            : {n_total}")

    df_extreme = df[df["is_extreme_95"] == 1].copy()
    df_normal  = df[df["is_extreme_95"] == 0].copy()

    row = {"dataset": label,
           "threshold_pct": 95,
           "threshold_value": round(thresh_95, 4),
           "n_extreme": int(n_extreme),
           "n_normal":  int(len(df_normal))}

    error_thresh_all     = df["lstm_abs_error"].quantile(0.75)
    error_thresh_extreme = df_extreme["lstm_abs_error"].quantile(0.75) \
                           if len(df_extreme) > 0 else np.nan

    for pname, pcol in PROXIES:
        proxy_all = df[pcol]
        error_all = df["lstm_abs_error"]

        rho_all, p_all, sig_all = spearman(proxy_all, error_all)

        if len(df_extreme) >= 5:
            rho_ext, p_ext, sig_ext = spearman(
                df_extreme[pcol], df_extreme["lstm_abs_error"])
        else:
            rho_ext, p_ext, sig_ext = np.nan, np.nan, False

        proxy_med_all = proxy_all.median()
        or_all = odds_ratio(proxy_all, error_all,
                            proxy_med_all, error_thresh_all)

        if len(df_extreme) >= 5:
            proxy_med_ext = df_extreme[pcol].median()
            or_ext = odds_ratio(df_extreme[pcol],
                                df_extreme["lstm_abs_error"],
                                proxy_med_ext, error_thresh_extreme)
        else:
            or_ext = np.nan

        print(f"\n  {pname} ({pcol})")
        print(f"    rho_all={rho_all}  p={p_all}  sig={sig_all}")
        print(f"    rho_extreme={rho_ext}  p={p_ext}  sig={sig_ext}")
        print(f"    OR_all={or_all}  OR_extreme={or_ext}")

        row[f"{pname}_rho_all_95"]     = rho_all
        row[f"{pname}_p_all_95"]       = p_all
        row[f"{pname}_sig_all_95"]     = sig_all
        row[f"{pname}_rho_extreme_95"] = rho_ext
        row[f"{pname}_p_extreme_95"]   = p_ext
        row[f"{pname}_sig_extreme_95"] = sig_ext
        row[f"{pname}_or_all_95"]      = or_all
        row[f"{pname}_or_extreme_95"]  = or_ext

    return row

# ── run both datasets ──────────────────────────────────────────────────────
rows = []
for label, path in DATASETS:
    rows.append(analyse(label, path))

results_df = pd.DataFrame(rows)
print("\n\n" + "="*60)
print("SENSITIVITY RESULTS SUMMARY (95th pct threshold)")
print("="*60)
print(results_df.to_string(index=False))

# ── append to results_summary_FINAL.csv ───────────────────────────────────
existing = pd.read_csv(SUMMARY_CSV)
existing = existing[~existing["dataset"].isin(["uci_95", "gefcom_95"])]
combined = pd.concat([existing, results_df], ignore_index=True, sort=False)
combined.to_csv(SUMMARY_CSV, index=False)
print(f"\nAppended to {SUMMARY_CSV}  ({len(combined)} rows total)")
print("\nDone.")
