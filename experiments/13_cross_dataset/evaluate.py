"""
Phase 7.1 — Per-Dataset Evaluation
PDD v5 Section 9 + Section 10
Called by cross_dataset.py — do not run directly.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, binomtest, fisher_exact, friedmanchisquare

log = logging.getLogger(__name__)

ROOT             = Path(".")
ALPHA_BONFERRONI = 0.0083   # 0.05/6 — PDD Section 10.2
ALPHA_WINKLER    = 0.05     # PDD Section 10.1


def winkler_score(actual, lower, upper, alpha=0.05):
    """PDD Section 10.1 — lower is better."""
    width   = upper - lower
    penalty = (2 / alpha) * (
        np.maximum(0, lower - actual) +
        np.maximum(0, actual - upper)
    )
    return float(np.mean(width + penalty))


def winkler_per_obs(actual, lower, upper, alpha=0.05):
    """Per-timestep Winkler contributions for Friedman test."""
    width   = upper - lower
    penalty = (2 / alpha) * (
        np.maximum(0, lower - actual) +
        np.maximum(0, actual - upper)
    )
    return width + penalty


def ece_score(proxy, error, n_bins=10):
    """
    Expected Calibration Error — PDD Section 10.1
    Bin by proxy percentile, compute mean error per bin.
    """
    bins   = np.percentile(proxy, np.linspace(0, 100, n_bins + 1))
    result = []
    for i in range(n_bins):
        mask = (proxy >= bins[i]) & (proxy < bins[i + 1])
        if mask.sum() > 0:
            result.append({
                "bin":        i + 1,
                "proxy_mean": float(proxy[mask].mean()),
                "error_mean": float(error[mask].mean()),
                "count":      int(mask.sum())
            })
    return result


def two_by_two(error, proxy, extreme, e_pctile=75, c_pctile=50):
    """
    2x2 classification framework — PDD Section 9.2

    Cells:
      SAFE_CORRECT  — low error, high confidence  (good)
      DANGEROUS     — high error, high confidence (overconfident failure)
      CAUTIOUS      — low error, low confidence   (unnecessarily cautious)
      WARNED        — high error, low confidence  (proxy warned correctly)

    High confidence = proxy BELOW median (low proxy value = model is confident)
    High error      = error ABOVE e_pctile threshold
    """
    e_thresh  = np.percentile(error, e_pctile)
    c_thresh  = np.percentile(proxy, c_pctile)
    high_err  = (error >= e_thresh)
    high_conf = (proxy < c_thresh)

    safe_correct = (~high_err &  high_conf)
    dangerous    = ( high_err &  high_conf)
    cautious     = (~high_err & ~high_conf)
    warned       = ( high_err & ~high_conf)

    n_ext        = int(extreme.sum())
    or_overall   = float(dangerous.mean())
    or_extreme   = float(dangerous[extreme].mean()) if n_ext > 0 else 0.0

    # Fisher exact on 2x2 contingency table
    table = np.array([
        [int(safe_correct.sum()), int(dangerous.sum())],
        [int(cautious.sum()),     int(warned.sum())]
    ])
    _, fisher_p = fisher_exact(table)

    # Binomial test — is extreme OR significantly greater than overall OR?
    n_danger_ext = int(dangerous[extreme].sum())
    binom_p = binomtest(
        n_danger_ext, n_ext, p=or_overall, alternative="greater"
    ).pvalue if n_ext > 0 else 1.0

    return {
        "e_pctile":          e_pctile,
        "c_pctile":          c_pctile,
        "e_threshold":       float(e_thresh),
        "c_threshold":       float(c_thresh),
        "n_safe_correct":    int(safe_correct.sum()),
        "n_dangerous":       int(dangerous.sum()),
        "n_cautious":        int(cautious.sum()),
        "n_warned":          int(warned.sum()),
        "rate_safe_correct": float(safe_correct.mean()),
        "rate_dangerous":    or_overall,
        "rate_cautious":     float(cautious.mean()),
        "rate_warned":       float(warned.mean()),
        "or_overall":        or_overall,
        "or_extreme":        or_extreme,
        "fisher_p":          float(fisher_p),
        "binom_p":           float(binom_p),
    }


def evaluate(dataset):
    log.info("")
    log.info("=" * 56)
    log.info(f"PHASE 7.1 — EVALUATION — {dataset.upper()}")
    log.info("=" * 56)

    # ── Load data ─────────────────────────────────────────────
    proxy_path = ROOT / f"results/{dataset}/tables/confidence_proxies_{dataset}.csv"
    conf_path  = ROOT / f"results/{dataset}/tables/conformal_{dataset}.csv"

    df      = pd.read_csv(proxy_path, index_col=0, parse_dates=True)
    conf_df = pd.read_csv(conf_path,  index_col=0, parse_dates=True)

    log.info(f"Proxy CSV: {len(df):,} rows")
    log.info(f"Conformal CSV: {len(conf_df):,} rows")
    assert len(df) == len(conf_df) == 8592, f"Row count mismatch: proxy={len(df)} conf={len(conf_df)}"

    # ── Core arrays ───────────────────────────────────────────
    actual  = df["actual_load"].values
    pred    = df["ensemble_mean"].values
    extreme = df["is_extreme_demand"].astype(bool).values
    normal  = ~extreme
    error   = df["lstm_abs_error"].values
    a_error = df["arima_abs_error"].values

    proxies = {
        "P1_ensemble_var": df["ensemble_variance"].values,
        "P2_pi_width":     df["pi_width"].values,
        "P3_resid_vol":    df["resid_volatility"].values,
    }

    n_ext  = int(extreme.sum())
    n_norm = int(normal.sum())
    log.info(f"Extreme timesteps: {n_ext} ({n_ext/len(df)*100:.1f}%)  |  Normal: {n_norm}")

    results = {"dataset": dataset}

    # ── BLOCK 1: Error metrics — PDD Section 10.1 ────────────
    log.info("Block 1 — Error metrics...")

    mae_all  = float(np.mean(error))
    mae_ext  = float(np.mean(error[extreme]))
    mae_norm = float(np.mean(error[normal]))
    rmse_all = float(np.sqrt(np.mean((actual - pred) ** 2)))
    rmse_ext = float(np.sqrt(np.mean((actual[extreme] - pred[extreme]) ** 2)))

    nonzero  = actual > 0
    mape_all = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
    n_zero   = int((actual == 0).sum())

    rmse_mae_ratio  = rmse_all / mae_all if mae_all > 0 else 0
    arima_mae_all   = float(np.mean(a_error))
    arima_mae_ext   = float(np.mean(a_error[extreme]))
    arima_rmse_all  = float("nan")   # arima_pred not in proxy CSV — RMSE not computable

    results.update({
        "lstm_mae_all":              round(mae_all, 4),
        "lstm_mae_extreme":          round(mae_ext, 4),
        "lstm_mae_normal":           round(mae_norm, 4),
        "lstm_rmse_all":             round(rmse_all, 4),
        "lstm_rmse_extreme":         round(rmse_ext, 4),
        "lstm_mape_all":             round(mape_all, 4),
        "lstm_rmse_mae_ratio":       round(rmse_mae_ratio, 4),
        "arima_mae_all":             round(arima_mae_all, 4),
        "arima_mae_extreme":         round(arima_mae_ext, 4),
        "arima_rmse_all":            round(arima_rmse_all, 4),
        "n_extreme":                 n_ext,
        "n_normal":                  n_norm,
        "n_zero_load_skipped_mape":  n_zero,
    })

    log.info(f"  LSTM  MAE  all={mae_all:.2f}  extreme={mae_ext:.2f}  normal={mae_norm:.2f} MWh")
    log.info(f"  LSTM  RMSE all={rmse_all:.2f}  extreme={rmse_ext:.2f}")
    log.info(f"  LSTM  MAPE={mape_all:.2f}%  RMSE/MAE={rmse_mae_ratio:.3f}")
    log.info(f"  ARIMA MAE  all={arima_mae_all:.2f}  extreme={arima_mae_ext:.2f} MWh")

    # ── BLOCK 2: Anomalous days — PDD Section 9.1 ────────────
    log.info("Block 2 — Anomalous days (|load - 7d rolling mean| > 2σ)...")
    load_s       = pd.Series(actual, index=df.index)
    roll_mean    = load_s.rolling(168, min_periods=1).mean()
    roll_std     = load_s.rolling(168, min_periods=1).std().fillna(0)
    anomalous    = (np.abs(load_s - roll_mean) > (2 * roll_std)).values
    n_anomalous  = int(anomalous.sum())
    overlap      = int((anomalous & extreme).sum())

    results.update({
        "n_anomalous":                 n_anomalous,
        "n_anomalous_extreme_overlap": overlap,
    })
    log.info(f"  Anomalous: {n_anomalous}  |  overlap with extreme: {overlap}")

    # ── BLOCK 3: Per-proxy analysis ───────────────────────────
    winkler_obs = {}

    for pname, proxy in proxies.items():
        log.info(f"Block 3 — {pname}...")

        # Spearman — 3 regimes
        rho_all,  p_all  = spearmanr(proxy, error)
        rho_ext,  p_ext  = spearmanr(proxy[extreme], error[extreme])
        rho_norm, p_norm = spearmanr(proxy[normal],  error[normal])

        log.info(f"  rho_all={rho_all:.4f} p={p_all:.4f} sig={p_all < ALPHA_BONFERRONI}")
        log.info(f"  rho_extreme={rho_ext:.4f} p={p_ext:.4f} sig={p_ext < ALPHA_BONFERRONI}")
        log.info(f"  rho_normal={rho_norm:.4f} p={p_norm:.4f} sig={p_norm < ALPHA_BONFERRONI}")

        # Mann-Whitney U — proxy dist: extreme vs normal
        mw_stat, mw_p = mannwhitneyu(proxy[extreme], proxy[normal], alternative="two-sided")
        log.info(f"  Mann-Whitney p={mw_p:.4f}")

        # 2x2 PRIMARY — error=75th, confidence=median
        primary     = two_by_two(error, proxy, extreme, e_pctile=75, c_pctile=50)
        # 2x2 SENSITIVITY — error=90th, confidence=25th
        sensitivity = two_by_two(error, proxy, extreme, e_pctile=90, c_pctile=25)

        log.info(f"  PRIMARY   OR_overall={primary['or_overall']:.4f}  OR_extreme={primary['or_extreme']:.4f}  binom_p={primary['binom_p']:.4f}")
        log.info(f"  SENSITIV  OR_overall={sensitivity['or_overall']:.4f}  OR_extreme={sensitivity['or_extreme']:.4f}")

        # Winkler Score — proxy as symmetric interval around ensemble mean
        p_lower = pred - proxy
        p_upper = pred + proxy
        ws      = winkler_score(actual, p_lower, p_upper, alpha=ALPHA_WINKLER)
        w_obs   = winkler_per_obs(actual, p_lower, p_upper, alpha=ALPHA_WINKLER)
        winkler_obs[pname] = w_obs
        log.info(f"  Winkler Score={ws:.4f}")

        # ECE
        ece      = ece_score(proxy, error, n_bins=10)
        ece_mean = float(np.mean([b["error_mean"] for b in ece]))

        results.update({
            f"{pname}_rho_all":           round(rho_all,  4),
            f"{pname}_p_all":             round(p_all,    6),
            f"{pname}_sig_all":           bool(p_all  < ALPHA_BONFERRONI),
            f"{pname}_rho_extreme":       round(rho_ext,  4),
            f"{pname}_p_extreme":         round(p_ext,    6),
            f"{pname}_sig_extreme":       bool(p_ext  < ALPHA_BONFERRONI),
            f"{pname}_rho_normal":        round(rho_norm, 4),
            f"{pname}_p_normal":          round(p_norm,   6),
            f"{pname}_sig_normal":        bool(p_norm < ALPHA_BONFERRONI),
            f"{pname}_mw_p":              round(mw_p,     6),
            f"{pname}_or_overall":        round(primary["or_overall"],  4),
            f"{pname}_or_extreme":        round(primary["or_extreme"],  4),
            f"{pname}_n_dangerous":       primary["n_dangerous"],
            f"{pname}_n_warned":          primary["n_warned"],
            f"{pname}_rate_safe_correct": round(primary["rate_safe_correct"], 4),
            f"{pname}_rate_dangerous":    round(primary["rate_dangerous"],    4),
            f"{pname}_rate_cautious":     round(primary["rate_cautious"],     4),
            f"{pname}_rate_warned":       round(primary["rate_warned"],       4),
            f"{pname}_fisher_p":          round(primary["fisher_p"],  6),
            f"{pname}_binom_p":           round(primary["binom_p"],   6),
            f"{pname}_sens_or_overall":   round(sensitivity["or_overall"], 4),
            f"{pname}_sens_or_extreme":   round(sensitivity["or_extreme"], 4),
            f"{pname}_sens_binom_p":      round(sensitivity["binom_p"],    6),
            f"{pname}_winkler_score":     round(ws, 4),
            f"{pname}_ece_mean":          round(ece_mean, 4),
        })

    # ── BLOCK 4: Conformal Winkler Score (alpha=0.05) ────────
    log.info("Block 4 — Conformal Winkler Score (alpha=0.05)...")
    conf_obs = winkler_per_obs(
        actual,
        conf_df["conformal_lower"].values,
        conf_df["conformal_upper"].values,
        alpha=ALPHA_WINKLER
    )
    ws_conf = float(np.mean(conf_obs))
    winkler_obs["conformal"] = conf_obs
    results["conformal_winkler_score"] = round(ws_conf, 4)
    log.info(f"  Conformal Winkler Score={ws_conf:.4f}")

    # ── BLOCK 5: Friedman test — PDD Section 10.2 ────────────
    log.info("Block 5 — Friedman test across all 4 Winkler Score series...")
    friedman_stat, friedman_p = friedmanchisquare(
        winkler_obs["P1_ensemble_var"],
        winkler_obs["P2_pi_width"],
        winkler_obs["P3_resid_vol"],
        winkler_obs["conformal"]
    )
    results["friedman_stat"] = round(float(friedman_stat), 4)
    results["friedman_p"]    = round(float(friedman_p),    6)
    log.info(f"  Friedman stat={friedman_stat:.4f}  p={friedman_p:.6f}")

    # ── Save evaluation CSV ───────────────────────────────────
    out_path = ROOT / f"results/{dataset}/tables/evaluation_{dataset}.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    log.info(f"Saved -> {out_path}")
    log.info(f"PHASE 7.1 {dataset.upper()} COMPLETE ✓")

    return results
