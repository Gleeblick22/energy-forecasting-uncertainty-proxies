"""
Temperature Ablation — LSTM Training
20-seed ensemble on GEFCom2014 WITHOUT temperature features

Identical to train_all_seeds.py EXCEPT:
- Uses data/gefcom_ablation/splits/
- Saves to models/gefcom_ablation/lstm/
- 11 input features (was 13)

Run from: ~/projects/energy-forecasting-uncertainty-proxies/
Command:  python3 experiments/20_ablation_temperature/train_lstm_ablation.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SPLIT_DIR  = Path("data/gefcom_ablation/splits")
MODEL_DIR  = Path("models/gefcom_ablation/lstm")
CONFIG_DIR = Path("models/gefcom_ablation/configs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEEDS      = list(range(20))
WINDOW     = 168
HIDDEN     = 128
LAYERS     = 2
DROPOUT    = 0.2
LR         = 0.001
BATCH      = 64
MAX_EPOCHS = 100
PATIENCE   = 10
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")

train_df = pd.read_csv(SPLIT_DIR / "train.csv", index_col=0, parse_dates=True)
val_df   = pd.read_csv(SPLIT_DIR / "val.csv",   index_col=0, parse_dates=True)
test_df  = pd.read_csv(SPLIT_DIR / "test.csv",  index_col=0, parse_dates=True)

cols   = [c for c in train_df.columns if c != "total_load"] + ["total_load"]
N_FEAT = len(cols)
log.info(f"Features: {N_FEAT} | Cols: {cols}")

class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def make_dataset(df, window=168):
    arr  = df[cols].values.astype(np.float32)
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i])
        y.append(arr[i, -1])
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

train_X, train_y = make_dataset(train_df)
val_X,   val_y   = make_dataset(val_df)
test_X,  test_y  = make_dataset(test_df)
log.info(f"Train: {len(train_X)} | Val: {len(val_X)} | Test: {len(test_X)}")

all_preds = []

for seed in SEEDS:
    log.info(f"Seed {seed}/{len(SEEDS)-1}...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = LSTMForecaster(N_FEAT, HIDDEN, LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()
    loader    = DataLoader(TensorDataset(train_X, train_y),
                           batch_size=BATCH, shuffle=True)

    best_val   = float("inf")
    patience   = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X.to(device)),
                               val_y.to(device)).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                log.info(f"  Early stop epoch {epoch+1} "
                         f"val_loss={best_val:.6f}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / f"seed_{seed:02d}.pt")

    model.eval()
    with torch.no_grad():
        preds = model(test_X.to(device)).cpu().numpy()
    all_preds.append(preds)
    log.info(f"  Seed {seed} done. Preds shape: {preds.shape}")

all_preds = np.array(all_preds)
np.save(MODEL_DIR / "all_predictions.npy", all_preds)
log.info(f"Saved all_predictions.npy shape={all_preds.shape}")

pickle.dump({"window": WINDOW, "hidden": HIDDEN, "layers": LAYERS,
             "dropout": DROPOUT, "n_features": N_FEAT,
             "feature_cols": cols},
            open(CONFIG_DIR / "lstm_config.pkl", "wb"))
log.info("ABLATION LSTM TRAINING COMPLETE ✓")
log.info(f"20 seeds trained WITHOUT temperature features")
