"""
Adaptive P2 Replacement — Rolling Quantile Regression
Project: When AI Forecasts Are Confidently Wrong
Output:  results/16_adaptive_p2/adaptive_p2_results.csv
         results/16_adaptive_p2/adaptive_p2_comparison.csv
Run from project root:
    python experiments/16_adaptive_p2/adaptive_p2.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import QuantileRegressor
import warnings
import os
warnings.filterwarnings('ignore')

# --- paths ---
UCI_TRAIN    = "data/uci/splits/train_unscaled.csv"
UCI_TEST     = "data/uci/splits/test_unscaled.csv"
GC_TRAIN     = "data/gefcom/splits/train_unscaled.csv"
GC_TEST      = "data/gefcom/splits/test_unscaled.csv"
UCI_PROXY    = "results/uci/tables/confidence_proxies_uci.csv"
GC_PROXY     = "results/gefcom/tables/confidence_proxies_gefcom.csv"
OUT_DIR      = "results/16_adaptive_p2"
os.makedirs(OUT_DIR, exist_ok=True)

# --- config ---
W            = 168       # rolling window — 1 week
LOWER_Q      = 0.05      # 90% PI lower bound
UPPER_Q      = 0.95      # 90% PI upper bound
BONFERRONI   = 0.0083    # same as manuscript

# Features used for quantile regression
UCI_FEATURES = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                 'month_sin', 'month_cos', 'is_weekend', 'lag_1h', 'lag_24h', 'lag_168h']

GC_FEATURES  = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                 'month_sin', 'month_cos', 'is_weekend',
                 'lag_1h', 'lag_24h', 'lag_168h',
                 'temperature_F', 'temperature_lag_24h']

GRIDS = {
    'uci': {
        'train':    UCI_TRAIN,
        'test':     UCI_TEST,
        'proxy':    UCI_PROXY,
        'features': UCI_FEATURES,
    },
    'gefcom': {
        'train':    GC_TRAIN,
        'test':     GC_TEST,
        'proxy':    GC_PROXY,
        'features': GC_FEATURES,
    },
}

# Anchors — static P2 from results_summary_FINAL.csv
STATIC_P2 = {
    'uci':    {'rho_extreme': -0.0543, 'pval': 0.1086, 'winkler': 171.4659, 'dangerous_rate': 0.0425},
    'gefcom': {'rho_extreme':  0.0182, 'pval': 0.5996, 'winkler':  97.4419, 'dangerous_rate': 0.0676},
}

all_results    = []
comparison_rows = []

for grid, cfg in GRIDS.items():
    print(f"\n{'='*60}")
    print(f"GRID: {grid.upper()}")
    print(f"{'='*60}")

    # Load data
    train = pd.read_csv(cfg['train'], index_col=0)
    test  = pd.read_csv(cfg['test'],  index_col=0)
    prx   = pd.read_csv(cfg['proxy'], index_col=0)

    # Align test and proxy on common index
    common = test.index.intersection(prx.index)
    test   = test.loc[common]
    prx    = prx.loc[common]

    # Combine train + test for rolling window
    combined = pd.concat([train, test])
    features  = cfg['features']
    n_train   = len(train)
    n_test    = len(test)

    print(f"Train rows: {n_train} | Test rows: {n_test} | Combined: {len(combined)}")
    print(f"Features: {features}")

    # ── Rolling Quantile Regression ───────────────────────────────
    print(f"\nComputing adaptive P2 with W={W} hours rolling window...")
    adaptive_p2_widths = []

    for t in range(n_train, n_train + n_test):
        window    = combined.iloc[t - W : t]
        X_win     = window[features].values
        y_win     = window['total_load'].values

        X_t = combined.iloc[t][features].values.reshape(1, -1)

        lo = QuantileRegressor(quantile=LOWER_Q, alpha=0, solver='highs').fit(X_win, y_win)
        hi = QuantileRegressor(quantile=UPPER_Q, alpha=0, solver='highs').fit(X_win, y_win)

        width = float((hi.predict(X_t) - lo.predict(X_t))[0])
        adaptive_p2_widths.append(max(width, 0.0))

        if (t - n_train) % 500 == 0:
            print(f"  Processed {t - n_train}/{n_test} test hours...")

    adaptive_p2 = np.array(adaptive_p2_widths)
    print(f"\nAdaptive P2 stats:")
    print(f"  mean = {adaptive_p2.mean():.4f} MWh")
    print(f"  std  = {adaptive_p2.std():.4f} MWh")
    print(f"  min  = {adaptive_p2.min():.4f} MWh")
    print(f"  max  = {adaptive_p2.max():.4f} MWh")

    # Critical check
    if adaptive_p2.std() < 0.01:
        print("  WARNING: std near zero — adaptive P2 may not be working correctly")
    else:
        print("  OK: std > 0 — adaptive P2 is varying")

    # ── Evaluation — same as Phase 7 ─────────────────────────────
    extreme_mask = prx['is_extreme_demand'] == 1
    abs_error    = prx['lstm_abs_error'].values

    # Spearman at extreme hours
    rho_ext, pval_ext = spearmanr(adaptive_p2[extreme_mask], abs_error[extreme_mask])
    sig_ext = pval_ext < BONFERRONI

    # Spearman all hours
    rho_all, pval_all = spearmanr(adaptive_p2, abs_error)

    # 2x2 DANGEROUS quadrant
    err_thresh  = np.percentile(abs_error[extreme_mask], 75)
    conf_thresh = np.median(adaptive_p2[extreme_mask])
    dangerous   = np.sum(
        (adaptive_p2[extreme_mask] <= conf_thresh) &
        (abs_error[extreme_mask]   >= err_thresh)
    )
    dangerous_rate = dangerous / extreme_mask.sum()

    # Winkler score
    # PI = [actual_load - adaptive_p2/2, actual_load + adaptive_p2/2]
    actual = prx["actual_load"].values
    forecast = prx["ensemble_mean"].values
    lo_pi  = forecast - adaptive_p2 / 2
    hi_pi  = forecast + adaptive_p2 / 2
    alpha  = 0.10
    winkler_scores = []
    for i in range(len(actual)):
        w = hi_pi[i] - lo_pi[i]
        if actual[i] < lo_pi[i]:
            w += (2 / alpha) * (lo_pi[i] - actual[i])
        elif actual[i] > hi_pi[i]:
            w += (2 / alpha) * (actual[i] - hi_pi[i])
        winkler_scores.append(w)
    winkler = np.mean(winkler_scores)

    print(f"\nEvaluation Results:")
    print(f"  rho_extreme = {rho_ext:+.4f}  pval = {pval_ext:.4f}  significant = {sig_ext}")
    print(f"  rho_all     = {rho_all:+.4f}  pval = {pval_all:.4f}")
    print(f"  DANGEROUS rate = {dangerous_rate:.4f} ({dangerous_rate*100:.1f}%)")
    print(f"  Winkler score  = {winkler:.4f}")

    # ── Comparison vs static P2 ───────────────────────────────────
    static = STATIC_P2[grid]
    print(f"\nComparison vs Static P2:")
    print(f"  {'Metric':<25} {'Static P2':>12} {'Adaptive P2':>12} {'Better':>10}")
    print(f"  {'-'*60}")
    print(f"  {'rho_extreme':<25} {static['rho_extreme']:>+12.4f} {rho_ext:>+12.4f} {'Adaptive' if abs(rho_ext) > abs(static['rho_extreme']) else 'Static':>10}")
    print(f"  {'significant_extreme':<25} {'No':>12} {str(sig_ext):>12}")
    print(f"  {'DANGEROUS rate':<25} {static['dangerous_rate']:>12.4f} {dangerous_rate:>12.4f} {'Adaptive' if dangerous_rate < static['dangerous_rate'] else 'Static':>10}")
    print(f"  {'Winkler score':<25} {static['winkler']:>12.4f} {winkler:>12.4f} {'Adaptive' if winkler < static['winkler'] else 'Static':>10}")

    # Save per-timestep results
    result_df = pd.DataFrame({
        'index':         test.index,
        'actual_load':   actual,
        'adaptive_p2':   adaptive_p2,
        'lstm_abs_error': abs_error,
        'is_extreme':    prx['is_extreme_demand'].values,
    })
    result_df.to_csv(f"{OUT_DIR}/adaptive_p2_{grid}.csv", index=False)

    all_results.append({
        'grid': grid,
        'adaptive_p2_mean': round(adaptive_p2.mean(), 4),
        'adaptive_p2_std':  round(adaptive_p2.std(),  4),
        'rho_extreme':      round(rho_ext,  4),
        'pval_extreme':     round(pval_ext, 6),
        'significant':      sig_ext,
        'rho_all':          round(rho_all,  4),
        'dangerous_rate':   round(dangerous_rate, 4),
        'winkler':          round(winkler,  4),
    })

    comparison_rows.append({
        'grid': grid,
        'static_p2_rho_extreme':   static['rho_extreme'],
        'adaptive_p2_rho_extreme': round(rho_ext, 4),
        'static_p2_dangerous':     static['dangerous_rate'],
        'adaptive_p2_dangerous':   round(dangerous_rate, 4),
        'static_p2_winkler':       static['winkler'],
        'adaptive_p2_winkler':     round(winkler, 4),
        'rq6_answer':              'YES — adaptive restores significance' if sig_ext else 'NO — failure persists',
    })

# Save summary files
pd.DataFrame(all_results).to_csv(f"{OUT_DIR}/adaptive_p2_results.csv", index=False)
pd.DataFrame(comparison_rows).to_csv(f"{OUT_DIR}/adaptive_p2_comparison.csv", index=False)

print(f"\n{'='*60}")
print("EXTENSION 2 COMPLETE")
print(f"{'='*60}")
for row in comparison_rows:
    print(f"\n{row['grid'].upper()} — RQ6 Answer: {row['rq6_answer']}")
print(f"\nResults saved to {OUT_DIR}/")
