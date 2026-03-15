"""
Phase 3 — Proxy Computation
PDD v6 | Section 8 — run for UCI then GEFCom
Change DATASET to switch between datasets.

Output: results/{dataset}/tables/confidence_proxies_{dataset}.csv

Run: python experiments/09_proxies_uci/compute_proxies.py
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET      = "gefcom"
WINDOW       = 168

DATA_DIR  = PROJECT_ROOT / f"data/{DATASET}/splits"
MODEL_DIR = PROJECT_ROOT / f"models/{DATASET}/lstm"
OUT_DIR   = PROJECT_ROOT / f"results/{DATASET}/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load LSTM ensemble predictions (20, 8592) MWh ─────────────────────────────
log.info("Loading all_predictions.npy ...")
all_preds = np.load(MODEL_DIR / "all_predictions.npy")
log.info(f"  Shape: {all_preds.shape}")
assert all_preds.shape[0] == 20

# ── Load actual load aligned to predictions ───────────────────────────────────
test_df    = pd.read_csv(DATA_DIR / "test_unscaled.csv", index_col=0, parse_dates=True)
actual     = test_df["total_load"].values[WINDOW : WINDOW + all_preds.shape[1]]
timestamps = test_df.index[WINDOW : WINDOW + all_preds.shape[1]]

# ── Load extreme flag ─────────────────────────────────────────────────────────
ext_df  = pd.read_csv(DATA_DIR / f"extreme_{DATASET}.csv", index_col=0, parse_dates=True)
extreme = ext_df["is_extreme_90"].values[WINDOW : WINDOW + all_preds.shape[1]]
log.info(f"  Extreme timesteps: {extreme.sum()} / {len(extreme)}")

# ── Load ARIMA if available ───────────────────────────────────────────────────
arima_path = OUT_DIR / "arima_predictions.csv"
arima_ok   = arima_path.exists()
if arima_ok:
    arima_df = pd.read_csv(arima_path, index_col=0, parse_dates=True)
    arima_df = arima_df.iloc[-all_preds.shape[1]:]
    log.info("  ARIMA predictions loaded")
else:
    log.warning("  ARIMA not found — P2 will be NaN (rerun after Phase 4)")

# ── P1: Ensemble Variance ─────────────────────────────────────────────────────
ensemble_mean = all_preds.mean(axis=0)
ensemble_var  = all_preds.var(axis=0, ddof=0)
assert ensemble_var.std() > 0, "FATAL: zero variance"
log.info(f"  P1 var mean: {ensemble_var.mean():.2f} MWh^2")

# ── P2: PI Width ──────────────────────────────────────────────────────────────
if arima_ok:
    pi_width = arima_df["upper_95"].values - arima_df["lower_95"].values
    log.info(f"  P2 PI width mean: {pi_width.mean():.2f} MWh")
else:
    pi_width = np.full(all_preds.shape[1], np.nan)

# ── P3: Residual Volatility ───────────────────────────────────────────────────
residuals = actual - ensemble_mean
resid_vol = np.array([
    np.std(residuals[max(0, t-24):t]) if t > 0 else 0.0
    for t in range(len(residuals))
])
log.info(f"  P3 resid vol mean: {resid_vol.mean():.2f} MWh")

# ── Errors ────────────────────────────────────────────────────────────────────
lstm_abs_error  = np.abs(actual - ensemble_mean)
arima_abs_error = (np.abs(actual - arima_df["arima_pred"].values)
                   if arima_ok else np.full(all_preds.shape[1], np.nan))

log.info(f"  LSTM MAE all:     {lstm_abs_error.mean():.2f} MWh")
log.info(f"  LSTM MAE extreme: {lstm_abs_error[extreme.astype(bool)].mean():.2f} MWh")
log.info(f"  LSTM MAE normal:  {lstm_abs_error[~extreme.astype(bool)].mean():.2f} MWh")

# ── Master CSV ────────────────────────────────────────────────────────────────
master = pd.DataFrame({
    "actual_load":       actual,
    "ensemble_mean":     ensemble_mean,
    "ensemble_variance": ensemble_var,
    "pi_width":          pi_width,
    "residual":          residuals,
    "resid_volatility":  resid_vol,
    "lstm_abs_error":    lstm_abs_error,
    "arima_abs_error":   arima_abs_error,
    "is_extreme_demand": extreme.astype(int),
}, index=timestamps)

if arima_ok:
    master["arima_pred"]     = arima_df["arima_pred"].values
    master["arima_lower_95"] = arima_df["lower_95"].values
    master["arima_upper_95"] = arima_df["upper_95"].values

assert master[["actual_load","ensemble_mean","ensemble_variance",
               "resid_volatility","lstm_abs_error"]].isna().sum().sum() == 0

out_path = OUT_DIR / f"confidence_proxies_{DATASET}.csv"
master.to_csv(out_path)

print("\n" + "="*55)
print(f"GEFCOM PROXY COMPUTATION COMPLETE")
print("="*55)
print(f"  Timesteps:          {len(master)}")
print(f"  Extreme timesteps:  {int(extreme.sum())}")
print(f"  P1 var mean (MWh2): {ensemble_var.mean():.2f}")
print(f"  P2 PI width:        {'N/A rerun after ARIMA' if not arima_ok else f'{pi_width.mean():.2f} MWh'}")
print(f"  P3 resid vol mean:  {resid_vol.mean():.2f} MWh")
print(f"  LSTM MAE all:       {lstm_abs_error.mean():.2f} MWh")
print(f"  LSTM MAE extreme:   {lstm_abs_error[extreme.astype(bool)].mean():.2f} MWh")
print(f"  Output: {out_path}")
print("="*55)
