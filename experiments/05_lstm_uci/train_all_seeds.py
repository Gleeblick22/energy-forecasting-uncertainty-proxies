"""
Phase 2B — Full LSTM Training
PDD v6 (locked) | Implementation Guide v3.0 | Section 7A

Dataset:  UCI (Portugal grid)
Seeds:    0-19 (20 seeds)
Epochs:   100 max, patience=10
Output:   models/uci/lstm/seed_N/model.pt
          models/uci/lstm/all_predictions.npy  shape (20, 8592)
          models/uci/lstm/training_summary.csv
          models/uci/configs/lstm_config.json
          models/uci/configs/scaler_uci.pkl

Run from project root:
    conda activate energy_forecast
    python experiments/05_lstm_uci/train_all_seeds.py

Crash recovery:
    Script detects already-completed seeds from all_predictions.npy
    and resumes from the next seed automatically.

Parallelisation:
    Run UCI and GEFCom simultaneously in two terminals:
    Terminal 1: nohup python experiments/05_lstm_uci/train_all_seeds.py > logs/lstm_uci.log 2>&1 &
    Terminal 2: nohup python experiments/06_lstm_gefcom/train_all_seeds.py > logs/lstm_gefcom.log 2>&1 &
"""

import json
import pickle
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# ── Logging — write to both console and log file ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH     = PROJECT_ROOT / "logs" / "lstm_uci.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — PDD v6 locked values
# ══════════════════════════════════════════════════════════════════════════════
DATASET   = "uci"
WINDOW    = 168        # 1 week lookback (PDD v6 Section 7A)
HIDDEN    = 128        # 2-layer LSTM hidden size (PDD v6 Table 16)
LAYERS    = 2          # stacked LSTM layers
DROPOUT   = 0.2        # training only, disabled at inference
LR        = 0.001      # Adam learning rate
BATCH     = 64         # batch size (PDD v6 Table 16)
MAX_EPOCH = 100        # full training (pilot used 30)
PATIENCE  = 10         # early stopping patience
N_SEEDS   = 20         # full training (pilot used 10)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = PROJECT_ROOT / f"data/{DATASET}/splits"
MODEL_DIR   = PROJECT_ROOT / f"models/{DATASET}/lstm"
CONFIG_DIR  = PROJECT_ROOT / f"models/{DATASET}/configs"
RESULTS_DIR = PROJECT_ROOT / f"results/{DATASET}/tables"

for d in [MODEL_DIR, CONFIG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE CONFIG JSON — all hyperparameters, never hardcoded in downstream scripts
# ══════════════════════════════════════════════════════════════════════════════
config = {
    "dataset":   DATASET,
    "window":    WINDOW,
    "hidden":    HIDDEN,
    "layers":    LAYERS,
    "dropout":   DROPOUT,
    "lr":        LR,
    "batch":     BATCH,
    "max_epoch": MAX_EPOCH,
    "patience":  PATIENCE,
    "n_seeds":   N_SEEDS,
    "output":    "1-step",
    "pdd_version": "v6",
}
with open(CONFIG_DIR / "lstm_config.json", "w") as f:
    json.dump(config, f, indent=2)
log.info(f"Config saved -> {CONFIG_DIR}/lstm_config.json")


# ══════════════════════════════════════════════════════════════════════════════
# SEED CONTROL — all four required (PDD v4 Table 17)
# Must be called before EVERY seed training run
# ══════════════════════════════════════════════════════════════════════════════
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — 2-layer LSTM, hidden=128, output=1-step (PDD v6 Table 16)
# Dropout=0.2 during training, disabled at inference
# ══════════════════════════════════════════════════════════════════════════════
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden, 1)   # 1-step output (PDD v6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(-1)   # (batch,)


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDER — 1-step output, total_load is last column
# ══════════════════════════════════════════════════════════════════════════════
def make_sequences(scaled_array: np.ndarray, window: int = WINDOW):
    X, y = [], []
    for i in range(len(scaled_array) - window):
        X.append(scaled_array[i : i + window])
        y.append(scaled_array[i + window, -1])   # next total_load
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD UNSCALED DATA — fit fresh scaler on train only
# Follows pilot fix: always use _unscaled.csv files
# ══════════════════════════════════════════════════════════════════════════════
log.info(f"Loading {DATASET.upper()} unscaled splits ...")

train_df = pd.read_csv(DATA_DIR / "train_unscaled.csv", index_col=0, parse_dates=True)
val_df   = pd.read_csv(DATA_DIR / "val_unscaled.csv",   index_col=0, parse_dates=True)
test_df  = pd.read_csv(DATA_DIR / "test_unscaled.csv",  index_col=0, parse_dates=True)
extreme  = pd.read_csv(DATA_DIR / f"extreme_{DATASET}.csv")

# Ensure total_load is last column
cols     = [c for c in train_df.columns if c != "total_load"] + ["total_load"]
train_df = train_df[cols]
val_df   = val_df[cols]
test_df  = test_df[cols]
N_FEAT   = len(cols)

log.info(f"Features: {N_FEAT}  |  Columns: {cols}")
log.info(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Sanity checks
assert train_df.isna().sum().sum() == 0, "NaN in train"
assert val_df.isna().sum().sum()   == 0, "NaN in val"
assert test_df.isna().sum().sum()  == 0, "NaN in test"
assert "total_load" in train_df.columns, "total_load missing"

# Fit fresh scaler on train only — save for downstream scripts
scaler    = MinMaxScaler()
scaler.fit(train_df)
pickle.dump(scaler, open(CONFIG_DIR / f"scaler_{DATASET}.pkl", "wb"))
log.info(f"Fresh scaler fitted and saved -> {CONFIG_DIR}/scaler_{DATASET}.pkl")

# Load range for inverse transform
_load_min = scaler.data_min_[-1]
_load_max = scaler.data_max_[-1]
log.info(f"Load range: {_load_min:.2f} – {_load_max:.2f} MWh")

def invert_load(scaled_vals: np.ndarray) -> np.ndarray:
    """Inverse MinMaxScaler for total_load only -> original MWh."""
    return scaled_vals * (_load_max - _load_min) + _load_min

# Scale all splits
tr_scaled = scaler.transform(train_df).astype(np.float32)
va_scaled = scaler.transform(val_df).astype(np.float32)
te_scaled = scaler.transform(test_df).astype(np.float32)

# Build sequences
X_tr, y_tr = make_sequences(tr_scaled)
X_va, y_va = make_sequences(va_scaled)
X_te, _    = make_sequences(te_scaled)
log.info(f"Sequences: X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}")

# Actual MWh values aligned to test sequences
actual       = test_df["total_load"].values[WINDOW : WINDOW + len(X_te)]
extreme_test = extreme["is_extreme_90"].values[WINDOW : WINDOW + len(X_te)]
log.info(f"Test actual shape: {actual.shape}  |  Extreme timesteps: {extreme_test.sum()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# CRASH RECOVERY — detect already completed seeds
# ══════════════════════════════════════════════════════════════════════════════
PRED_PATH = MODEL_DIR / "all_predictions.npy"

if PRED_PATH.exists():
    existing = np.load(PRED_PATH)
    completed_seeds = existing.shape[0]
    all_predictions = list(existing)
    log.info(f"Resuming from seed {completed_seeds} ({completed_seeds} seeds already done)")
else:
    completed_seeds = 0
    all_predictions = []
    log.info("Starting fresh — no existing predictions found")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP — 20 seeds
# ══════════════════════════════════════════════════════════════════════════════
seed_results = []

# Load existing training summary if resuming
SUMMARY_PATH = MODEL_DIR / "training_summary.csv"
if SUMMARY_PATH.exists() and completed_seeds > 0:
    seed_results = pd.read_csv(SUMMARY_PATH).to_dict("records")

for seed in range(completed_seeds, N_SEEDS):
    set_all_seeds(seed)
    log.info(f"{'='*50}")
    log.info(f"SEED {seed} / {N_SEEDS - 1}")
    log.info(f"{'='*50}")

    model     = LSTMForecaster(n_features=N_FEAT, hidden=HIDDEN,
                               n_layers=LAYERS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    tr_losses, va_losses = [], []

    for epoch in range(MAX_EPOCH):

        # ── Train ────────────────────────────────────────────────
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr)
        tr_losses.append(ep_loss)

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                model(torch.FloatTensor(X_va).to(device)),
                torch.FloatTensor(y_va).to(device)
            ).item()
        va_losses.append(val_loss)

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1:3d} — train={ep_loss:.6f}  val={val_loss:.6f}")

        # ── Early stopping ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info(f"  Early stop at epoch {epoch + 1}")
                break

    converged_epoch = len(tr_losses)
    log.info(f"  Converged at epoch {converged_epoch}  |  best_val_loss={best_val_loss:.6f}")

    # ── Save model weights ───────────────────────────────────────
    model.load_state_dict(best_state)
    model_path = MODEL_DIR / f"seed_{seed}" / "model.pt"
    torch.save(best_state, model_path)
    log.info(f"  Model saved -> {model_path}")

    # ── Save per-seed training log ───────────────────────────────
    pd.DataFrame({
        "epoch":      range(1, converged_epoch + 1),
        "train_loss": tr_losses,
        "val_loss":   va_losses,
    }).to_csv(MODEL_DIR / f"seed_{seed}" / "training_log.csv", index=False)

    # ── Test predictions — batched inference (GPU OOM fix) ───────
    model.eval()
    with torch.no_grad():
        preds_list = []
        for i in range(0, len(X_te), 256):
            xb = torch.FloatTensor(X_te[i:i+256]).to(device)
            preds_list.append(model(xb).cpu().numpy())
        preds_norm = np.concatenate(preds_list, axis=0)

    # ── Inverse transform to MWh ─────────────────────────────────
    preds_orig = invert_load(preds_norm)

    mae_seed = float(np.mean(np.abs(actual - preds_orig)))
    log.info(f"  Seed {seed} MAE: {mae_seed:.2f} MWh  |  "
             f"pred range [{preds_orig.min():.1f}, {preds_orig.max():.1f}] MWh")

    # ── Incremental save — crash recovery ────────────────────────
    all_predictions.append(preds_orig)
    np.save(PRED_PATH, np.array(all_predictions))
    log.info(f"  all_predictions.npy updated — shape {np.array(all_predictions).shape}")

    # ── Record seed result ───────────────────────────────────────
    seed_results.append({
        "seed":             seed,
        "converged_epoch":  converged_epoch,
        "best_val_loss":    round(best_val_loss, 6),
        "mae_mwh":          round(mae_seed, 4),
        "pred_min_mwh":     round(float(preds_orig.min()), 2),
        "pred_max_mwh":     round(float(preds_orig.max()), 2),
        "pred_std_mwh":     round(float(preds_orig.std()), 2),
    })

    # Save summary after every seed
    pd.DataFrame(seed_results).to_csv(SUMMARY_PATH, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
all_preds_array = np.array(all_predictions)   # shape (20, n_test_sequences)
np.save(PRED_PATH, all_preds_array)

log.info(f"{'='*50}")
log.info(f"UCI FULL TRAINING COMPLETE")
log.info(f"{'='*50}")
log.info(f"  all_predictions.npy shape: {all_preds_array.shape}")
log.info(f"  Saved -> {PRED_PATH}")

# Final ensemble stats
ensemble_mean = all_preds_array.mean(axis=0)
ensemble_var  = all_preds_array.var(axis=0, ddof=0)
abs_error     = np.abs(actual - ensemble_mean)

log.info(f"  Final ensemble MAE:       {abs_error.mean():.2f} MWh")
log.info(f"  Final ensemble var mean:  {ensemble_var.mean():.2f} MWh^2")

# Training summary
summary_df = pd.DataFrame(seed_results)
log.info(f"\nTraining Summary:\n{summary_df.to_string(index=False)}")
log.info(f"\nMean MAE across seeds: {summary_df['mae_mwh'].mean():.2f} MWh")
log.info(f"MAE std across seeds:  {summary_df['mae_mwh'].std():.2f} MWh")
log.info(f"Mean converged epoch:  {summary_df['converged_epoch'].mean():.1f}")

print("\n" + "="*55)
print("UCI FULL TRAINING COMPLETE")
print("="*55)
print(f"  Seeds completed:          {len(seed_results)}")
print(f"  all_predictions shape:    {all_preds_array.shape}")
print(f"  Ensemble MAE (MWh):       {abs_error.mean():.2f}")
print(f"  Ensemble var mean (MWh²): {ensemble_var.mean():.2f}")
print(f"  Mean converged epoch:     {summary_df['converged_epoch'].mean():.1f}")
print("="*55)
print(f"\nNext: run GEFCom training")
print(f"  python experiments/06_lstm_gefcom/train_all_seeds.py")
print(f"\nOr run both in parallel:")
print(f"  Terminal 2: nohup python experiments/06_lstm_gefcom/train_all_seeds.py > logs/lstm_gefcom.log 2>&1 &")
