"""
Temperature Ablation — Proxy Computation + Evaluation

Computes P1 proxy and regime-stratified Spearman
correlation on GEFCom2014 WITHOUT temperature features.
Compares directly to original (WITH temperature).

Run from: ~/projects/energy-forecasting-uncertainty-proxies/
Command:  /usr/bin/python3 experiments/20_ablation_temperature/evaluate_ablation.py
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(".")
MODEL_DIR  = Path("models/gefcom_ablation/lstm")
CONFIG_DIR = Path("models/gefcom_ablation/configs")
SPLIT_DIR  = Path("data/gefcom_ablation/splits")
OUT_DIR    = Path("results/gefcom_ablation/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.0083

log.info("Loading ablation test data...")
scaler  = pickle.load(open(CONFIG_DIR / "scaler_gefcom_ablation.pkl", "rb"))
test_df = pd.read_csv(SPLIT_DIR / "test.csv",
                      index_col=0, parse_dates=True)

feature_cols = [c for c in test_df.columns if c != "total_load"]
n_features   = len(feature_cols)
log.info(f"Features: {n_features} (no temperature)")

log.info("Loading LSTM predictions...")
all_preds_scaled = np.load(MODEL_DIR / "all_predictions.npy")
log.info(f"Predictions shape: {all_preds_scaled.shape}")

n_seeds, n_test = all_preds_scaled.shape

dummy = np.zeros((n_test, n_features + 1))
all_preds_mwh = np.zeros_like(all_preds_scaled)

for s in range(n_seeds):
    dummy[:, -1] = all_preds_scaled[s]
    inv = scaler.inverse_transform(dummy)
    all_preds_mwh[s] = inv[:, -1]

dummy[:, -1] = test_df["total_load"].values[-n_test:]
actual = scaler.inverse_transform(dummy)[:, -1]

log.info("Computing P1 ensemble variance...")
ensemble_mean = all_preds_mwh.mean(axis=0)
ensemble_var  = all_preds_mwh.var(axis=0, ddof=0)
ensemble_std  = np.sqrt(ensemble_var)

log.info("Loading original extreme flags...")
orig_proxy = pd.read_csv(
    ROOT / "results/gefcom/tables/confidence_proxies_gefcom.csv",
    index_col=0, parse_dates=True
)
extreme = orig_proxy["is_extreme_demand"].values[-n_test:].astype(bool)
normal  = ~extreme

log.info(f"Extreme hours: {extreme.sum()} | Normal: {normal.sum()}")

error = np.abs(actual - ensemble_mean)
proxy = ensemble_std

log.info("Computing Spearman correlations...")
rho_all,  p_all  = spearmanr(proxy, error)
rho_ext,  p_ext  = spearmanr(proxy[extreme], error[extreme])
rho_norm, p_norm = spearmanr(proxy[normal],  error[normal])

log.info(f"P1 ABLATION (no temperature):")
log.info(f"  rho_all     = {rho_all:.4f}  p={p_all:.6f}  sig={p_all < ALPHA}")
log.info(f"  rho_extreme = {rho_ext:.4f}  p={p_ext:.6f}  sig={p_ext < ALPHA}")
log.info(f"  rho_normal  = {rho_norm:.4f}  p={p_norm:.6f}  sig={p_norm < ALPHA}")

log.info("")
log.info("="*56)
log.info("ABLATION COMPARISON: WITH vs WITHOUT TEMPERATURE")
log.info("="*56)
log.info(f"{'Metric':<30} {'With Temp':>12} {'No Temp':>12}")
log.info(f"{'-'*56}")
log.info(f"{'P1 rho_all':<30} {0.4396:>+12.4f} {rho_all:>+12.4f}")
log.info(f"{'P1 rho_extreme':<30} {0.4815:>+12.4f} {rho_ext:>+12.4f}")
log.info(f"{'P1 rho_normal':<30} {0.4217:>+12.4f} {rho_norm:>+12.4f}")
log.info(f"{'P1 sig_extreme':<30} {'True':>12} {str(p_ext < ALPHA):>12}")

results = {
    "dataset":                  "gefcom_ablation",
    "temperature_removed":      True,
    "P1_rho_all":               round(rho_all,  4),
    "P1_p_all":                 round(p_all,    6),
    "P1_sig_all":               bool(p_all  < ALPHA),
    "P1_rho_extreme":           round(rho_ext,  4),
    "P1_p_extreme":             round(p_ext,    6),
    "P1_sig_extreme":           bool(p_ext  < ALPHA),
    "P1_rho_normal":            round(rho_norm, 4),
    "P1_p_normal":              round(p_norm,   6),
    "P1_rho_extreme_WITH_temp": 0.4815,
    "P1_rho_extreme_NO_temp":   round(rho_ext,  4),
    "hypothesis_confirmed":     not bool(p_ext < ALPHA),
}

pd.DataFrame([results]).to_csv(
    OUT_DIR / "ablation_results.csv", index=False)
log.info(f"\nSaved → {OUT_DIR}/ablation_results.csv")

log.info("")
log.info("="*56)
log.info("ABLATION VERDICT")
log.info("="*56)

if not (p_ext < ALPHA):
    log.info("OK HYPOTHESIS CONFIRMED:")
    log.info("   P1 reliability COLLAPSES without temperature")
    log.info(f"   rho_extreme: +0.4815 → {rho_ext:.4f} (ns)")
    log.info("   Weather-load coupling IS causal mechanism")
else:
    log.info("FAIL HYPOTHESIS NOT CONFIRMED:")
    log.info("   P1 remains reliable without temperature")
    log.info(f"   rho_extreme: +0.4815 → {rho_ext:.4f} (still sig)")
    log.info("   Other grid characteristics may drive reliability")

log.info("ABLATION EVALUATION COMPLETE ✓")
