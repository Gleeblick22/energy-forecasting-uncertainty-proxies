"""
Conformal Prediction — Regime-Stratified Analysis
Phase 7.3 — Conformal Extreme Hour Evaluation

Run from: ~/projects/energy-forecasting-uncertainty-proxies/
Command:  python3 experiments/13_cross_dataset/conformal_regime.py

Output:
  results/comparison/conformal_regime_analysis.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ROOT        = Path(".")
ALPHA       = 0.0083   # Bonferroni corrected


def two_by_two(error, proxy, extreme, e_pctile=75, c_pctile=50):
    e_thresh  = np.percentile(error, e_pctile)
    c_thresh  = np.percentile(proxy, c_pctile)
    high_err  = (error >= e_thresh)
    high_conf = (proxy < c_thresh)
    dangerous = (high_err & high_conf)
    n_ext     = int(extreme.sum())
    or_overall = float(dangerous.mean())
    or_extreme = float(dangerous[extreme].mean()) if n_ext > 0 else 0.0
    return {
        "or_overall":      or_overall,
        "or_extreme":      or_extreme,
        "rate_dangerous":  or_overall,
        "elevated":        or_extreme > or_overall,
    }


results = {}

for dataset in ["uci", "gefcom"]:

    log.info("")
    log.info("=" * 56)
    log.info(f"PHASE 7.3 — CONFORMAL REGIME ANALYSIS — {dataset.upper()}")
    log.info("=" * 56)

    # Load proxy file
    proxy_df = pd.read_csv(
        ROOT / f"results/{dataset}/tables/confidence_proxies_{dataset}.csv",
        index_col=0, parse_dates=True
    )

    # Load conformal file
    conf_df = pd.read_csv(
        ROOT / f"results/{dataset}/tables/conformal_{dataset}.csv",
        index_col=0, parse_dates=True
    )

    # Core arrays
    error      = proxy_df["lstm_abs_error"].values
    extreme    = proxy_df["is_extreme_demand"].astype(bool).values
    normal     = ~extreme
    conf_width = conf_df["conformal_width"].values

    n_ext  = int(extreme.sum())
    n_norm = int(normal.sum())
    log.info(f"Extreme hours: {n_ext} | Normal hours: {n_norm}")

    # Spearman correlations
    rho_all,  p_all  = spearmanr(conf_width, error)
    rho_ext,  p_ext  = spearmanr(conf_width[extreme], error[extreme])
    rho_norm, p_norm = spearmanr(conf_width[normal],  error[normal])

    log.info(f"Conformal rho_all={rho_all:.4f}    p={p_all:.6f}    sig={p_all < ALPHA}")
    log.info(f"Conformal rho_extreme={rho_ext:.4f} p={p_ext:.6f}  sig={p_ext < ALPHA}")
    log.info(f"Conformal rho_normal={rho_norm:.4f}  p={p_norm:.6f} sig={p_norm < ALPHA}")

    # Mann-Whitney
    mw_stat, mw_p = mannwhitneyu(
        conf_width[extreme], conf_width[normal],
        alternative="two-sided"
    )
    log.info(f"Conformal Mann-Whitney p={mw_p:.6f}")

    # DANGEROUS rate
    primary = two_by_two(error, conf_width, extreme,
                         e_pctile=75, c_pctile=50)
    log.info(f"Conformal DANGEROUS overall={primary['or_overall']:.4f} "
             f"extreme={primary['or_extreme']:.4f} "
             f"elevated={primary['elevated']}")

    results[dataset] = {
        "dataset":             dataset,
        "conformal_rho_all":   round(rho_all,  4),
        "conformal_p_all":     round(p_all,    6),
        "conformal_sig_all":   bool(p_all  < ALPHA),
        "conformal_rho_ext":   round(rho_ext,  4),
        "conformal_p_ext":     round(p_ext,    6),
        "conformal_sig_ext":   bool(p_ext  < ALPHA),
        "conformal_rho_norm":  round(rho_norm, 4),
        "conformal_p_norm":    round(p_norm,   6),
        "conformal_sig_norm":  bool(p_norm < ALPHA),
        "conformal_mw_p":      round(mw_p,     6),
        "conformal_or_overall": round(primary["or_overall"], 4),
        "conformal_or_extreme": round(primary["or_extreme"], 4),
        "conformal_elevated":   primary["elevated"],
    }

# Print comparison table
log.info("")
log.info("=" * 60)
log.info("COMPARISON: ALL PROXIES + CONFORMAL AT EXTREME HOURS")
log.info("=" * 60)

p1_uci = 0.0089;   p1_gef = 0.4815
p2_uci = -0.0543;  p2_gef = 0.0182
p3_uci = 0.0440;   p3_gef = 0.4554
cf_uci = results["uci"]["conformal_rho_ext"]
cf_gef = results["gefcom"]["conformal_rho_ext"]

log.info(f"{'Proxy':<25} {'UCI ρ_extreme':>15} {'GEFCom ρ_extreme':>18}")
log.info(f"{'-'*60}")
log.info(f"{'P1 (LSTM Ensemble Std)':<25} {p1_uci:>+15.4f} {p1_gef:>+18.4f}")
log.info(f"{'P2 (SARIMA Half-Width)':<25} {p2_uci:>+15.4f} {p2_gef:>+18.4f}")
log.info(f"{'P3 (Resid Volatility)':<25} {p3_uci:>+15.4f} {p3_gef:>+18.4f}")
log.info(f"{'Conformal Prediction':<25} {cf_uci:>+15.4f} {cf_gef:>+18.4f}")
log.info(f"{'(sig UCI/GEF)':<25} "
         f"{str(results['uci']['conformal_sig_ext']):>15} "
         f"{str(results['gefcom']['conformal_sig_ext']):>18}")

# Save
out_path = ROOT / "results/comparison/conformal_regime_analysis.csv"
pd.DataFrame(list(results.values())).to_csv(out_path, index=False)
log.info(f"\nSaved → {out_path}")
log.info("PHASE 7.3 COMPLETE ✓")
