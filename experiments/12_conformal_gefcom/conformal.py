"""
Phase 6 — Conformal Prediction — UCI
PDD v5 Section 7C
Run from: ~/projects/energy-forecasting-uncertainty-proxies/
"""
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATASET    = "gefcom"
N_FEATURES = 14
N_SEEDS    = 20
WINDOW     = 168
ALPHA      = 0.10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT       = Path(".")

class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2):
        super().__init__()
        self.lstm   = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.linear = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(-1)

def make_sequences(scaled_array, window=168):
    X = []
    for i in range(len(scaled_array) - window):
        X.append(scaled_array[i : i + window])
    return np.array(X, dtype=np.float32)

def inverse_transform_preds(preds_norm, scaler):
    """Inverse transform normalised predictions back to MWh."""
    dummy        = np.zeros((len(preds_norm), scaler.n_features_in_))
    dummy[:, -1] = preds_norm   # total_load is always last in scaler
    return scaler.inverse_transform(dummy)[:, -1]

def get_val_ensemble_mean(val_df, scaler, model_dir, n_seeds, n_features, device, window=168):
    """
    Val CSV has normalised values but columns in wrong order.
    Reorder to match scaler, then pass raw normalised array to model
    (model was trained on normalised data — no need to re-scale).
    """
    # Reorder columns to match scaler feature order
    val_ordered = val_df[scaler.feature_names_in_]   # reorder columns
    scaled      = val_ordered.values                  # already normalised — use directly
    X           = make_sequences(scaled, window)
    X_tensor    = torch.FloatTensor(X).to(device)

    all_preds = []
    for seed in range(n_seeds):
        model = LSTMForecaster(n_features=n_features).to(device)
        model.load_state_dict(torch.load(model_dir / f"seed_{seed}/model.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            preds_norm = model(X_tensor).cpu().numpy()
        preds_mwh = inverse_transform_preds(preds_norm, scaler)
        all_preds.append(preds_mwh)
        log.info(f"  Seed {seed:2d} done")
    return np.array(all_preds).mean(axis=0)

def winkler_score(actual, lower, upper, alpha=0.10):
    width   = upper - lower
    penalty = (2 / alpha) * (np.maximum(0, lower - actual) + np.maximum(0, actual - upper))
    return float(np.mean(width + penalty))

log.info("=" * 56)
log.info("PHASE 6 — CONFORMAL PREDICTION — GEFCOM2014")
log.info("=" * 56)
log.info(f"Device: {DEVICE}  |  Alpha: {ALPHA}  |  Target: {1-ALPHA:.0%} coverage")

scaler = pickle.load(open(ROOT / "models/gefcom/configs/scaler_gefcom.pkl", "rb"))
val_df = pd.read_csv(ROOT / "data/gefcom/splits/val.csv", index_col=0, parse_dates=True)

# Load actual MWh values from proxy CSV — already correct scale and alignment
proxy_df    = pd.read_csv(ROOT / "results/gefcom/tables/confidence_proxies_gefcom.csv", index_col=0, parse_dates=True)
test_actual = proxy_df["actual_load"].values         # MWh, 8592 rows
test_mean   = np.load(ROOT / "models/gefcom/lstm/all_predictions.npy").mean(axis=0)  # MWh, 8592
assert len(test_actual) == len(test_mean) == 8592

log.info(f"Val rows: {len(val_df):,}  |  Test timesteps: {len(test_actual):,}")
log.info(f"Test actual range: {test_actual.min():.1f} – {test_actual.max():.1f} MWh")
log.info(f"Test mean range:   {test_mean.min():.1f} – {test_mean.max():.1f} MWh")
log.info(f"Test MAE (sanity): {np.mean(np.abs(test_actual - test_mean)):.2f} MWh")

# Val actual in MWh — inverse transform the normalised total_load column
val_load_norm    = val_df["total_load"].values[WINDOW:]
dummy_val        = np.zeros((len(val_load_norm), scaler.n_features_in_))
dummy_val[:, -1] = val_load_norm
val_actual_mwh   = scaler.inverse_transform(dummy_val)[:, -1]
log.info(f"Val actual MWh range: {val_actual_mwh.min():.1f} – {val_actual_mwh.max():.1f} MWh")

log.info("Step 1 — Recomputing ensemble mean on validation set...")
val_mean = get_val_ensemble_mean(val_df, scaler, ROOT / "models/gefcom/lstm", N_SEEDS, N_FEATURES, DEVICE)
assert len(val_mean) == len(val_actual_mwh), f"Shape mismatch: {len(val_mean)} vs {len(val_actual_mwh)}"
log.info(f"Val mean range: {val_mean.min():.1f} – {val_mean.max():.1f} MWh")
log.info(f"Val MAE (sanity): {np.mean(np.abs(val_actual_mwh - val_mean)):.2f} MWh")

log.info("Step 2 — Computing split conformal intervals (numpy)...")
val_residuals = val_actual_mwh - val_mean
n_cal         = len(val_residuals)
abs_residuals = np.abs(val_residuals)
q_level       = min(np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal, 1.0)
conformal_q   = np.quantile(abs_residuals, q_level)
log.info(f"n_cal={n_cal}  |  q_level={q_level:.4f}  |  conformal_q={conformal_q:.2f} MWh")

conf_lower = test_mean - conformal_q
conf_upper = test_mean + conformal_q
conf_width = conf_upper - conf_lower

log.info("Step 3 — Computing metrics...")
coverage = float(((test_actual >= conf_lower) & (test_actual <= conf_upper)).mean())
ws       = winkler_score(test_actual, conf_lower, conf_upper, alpha=ALPHA)
log.info(f"Conformal coverage:   {coverage:.4f}  (target: {1-ALPHA:.2f})")
log.info(f"Conformal width mean: {conf_width.mean():.2f} MWh")
log.info(f"Conformal width std:  {conf_width.std():.4f} MWh")
log.info(f"Conformal width min:  {conf_width.min():.2f} MWh")
log.info(f"Conformal width max:  {conf_width.max():.2f} MWh")
log.info(f"Winkler Score:        {ws:.4f}")
if coverage < 0.85:
    log.warning(f"Coverage {coverage:.3f} below 0.85 — seasonal val/test mismatch expected")
else:
    log.info("Coverage check passed ✓")
log.info("Sanity checks passed ✓")

log.info("Step 4 — Saving output...")
conf_df = pd.DataFrame({
    "conformal_lower": conf_lower,
    "conformal_upper": conf_upper,
    "conformal_width": conf_width,
}, index=proxy_df.index)
assert conf_df.isnull().sum().sum() == 0
assert len(conf_df) == 8592
conf_df.to_csv(ROOT / "results/gefcom/tables/conformal_gefcom.csv")
log.info("Saved -> results/gefcom/tables/conformal_gefcom.csv")

log.info("")
log.info("=" * 56)
log.info("GEFCOM2014 CONFORMAL COMPLETE")
log.info("=" * 56)
log.info(f"  Rows:          {len(conf_df):,}")
log.info(f"  Coverage:      {coverage:.4f}")
log.info(f"  Width mean:    {conf_width.mean():.2f} MWh")
log.info(f"  Width std:     {conf_width.std():.4f} MWh")
log.info(f"  Winkler Score: {ws:.4f}")
log.info("Phase 6 complete. Next: Phase 7 evaluation.")
