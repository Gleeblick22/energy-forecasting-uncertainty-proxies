"""
Ensemble Size Sensitivity — Extension 3
Project: When AI Forecasts Are Confidently Wrong
Output:  results/17_ensemble_sensitivity/sensitivity_results.csv
Run from project root:
    python experiments/17_ensemble_sensitivity/ensemble_sensitivity.py
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
from scipy.stats import spearmanr
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
torch.set_num_threads(8)

# ── Locked hyperparameters — identical to Phase 3 ─────────────────
WINDOW    = 168
HIDDEN    = 128
LAYERS    = 2
DROPOUT   = 0.2
LR        = 0.001
BATCH     = 64
MAX_EPOCH = 50
PATIENCE  = 5

# Seed sets to test — 20 seeds already done in Phase 3
SEED_SETS = [5, 10, 50]  # 20 loaded from existing results

BONFERRONI = 0.0083

# Anchors from results_summary_FINAL.csv — 20-seed baseline
ANCHORS_20 = {
    'uci':    {'rho_extreme': 0.0089, 'pval': 0.7929},
    'gefcom': {'rho_extreme': 0.4815, 'pval': 0.0000},
}

GRIDS = {
    'uci':    {'dataset': 'uci',    'proxy': 'results/uci/tables/confidence_proxies_uci.csv'},
    'gefcom': {'dataset': 'gefcom', 'proxy': 'results/gefcom/tables/confidence_proxies_gefcom.csv'},
}

# ── Model — identical to Phase 3 ──────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=n_features, hidden_size=hidden,
                              num_layers=n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(-1)

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def make_sequences(arr, window=WINDOW):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
        y.append(arr[i+window, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_single_seed(seed, X_tr, y_tr, X_va, y_va, n_feat, device):
    set_all_seeds(seed)
    model     = LSTMForecaster(n_feat, HIDDEN, LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()
    loader    = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=BATCH, shuffle=True, num_workers=0
    )

    best_val  = float('inf')
    patience  = 0
    best_state = None

    for epoch in range(MAX_EPOCH):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(X_va).to(device)
            yv = torch.FloatTensor(y_va).to(device)
            val_loss = loss_fn(model(xv), yv).item()

        if val_loss < best_val:
            best_val   = val_loss
            patience   = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val, epoch + 1

all_results = []

device = torch.device("cpu")
log.info(f"Device: {device}")

for grid, cfg in GRIDS.items():
    dataset   = cfg['dataset']
    DATA_DIR  = PROJECT_ROOT / f"data/{dataset}/splits"

    log.info(f"\n{'='*60}")
    log.info(f"GRID: {grid.upper()}")
    log.info(f"{'='*60}")

    # Load unscaled data
    train_df = pd.read_csv(DATA_DIR / "train_unscaled.csv", index_col=0)
    val_df   = pd.read_csv(DATA_DIR / "val_unscaled.csv",   index_col=0)
    test_df  = pd.read_csv(DATA_DIR / "test_unscaled.csv",  index_col=0)

    # Ensure total_load is last
    cols     = [c for c in train_df.columns if c != "total_load"] + ["total_load"]
    train_df = train_df[cols]
    val_df   = val_df[cols]
    test_df  = test_df[cols]
    N_FEAT   = len(cols)

    # Scale
    scaler    = MinMaxScaler()
    scaler.fit(train_df)
    _load_min = scaler.data_min_[-1]
    _load_max = scaler.data_max_[-1]

    tr_sc = scaler.transform(train_df).astype(np.float32)
    va_sc = scaler.transform(val_df).astype(np.float32)
    te_sc = scaler.transform(test_df).astype(np.float32)

    X_tr, y_tr = make_sequences(tr_sc)
    X_va, y_va = make_sequences(va_sc)
    X_te, _    = make_sequences(te_sc)

    def invert(v):
        return v * (_load_max - _load_min) + _load_min

    # Load proxy file for errors and extreme mask
    prx          = pd.read_csv(cfg['proxy'], index_col=0)
    common       = prx.index[:len(X_te)]
    abs_error    = prx['lstm_abs_error'].values[:len(X_te)]
    extreme_mask = prx['is_extreme_demand'].values[:len(X_te)] == 1

    # ── Add 20-seed baseline from existing results ─────────────────
    anchor = ANCHORS_20[grid]
    all_results.append({
        'grid':        grid,
        'n_seeds':     20,
        'rho_extreme': anchor['rho_extreme'],
        'pval':        anchor['pval'],
        'significant': anchor['pval'] < BONFERRONI,
        'source':      'Phase 3 existing',
    })
    log.info(f"20-seed baseline loaded: rho_extreme={anchor['rho_extreme']:+.4f}")

    # ── Train at 5, 10, 50 seeds ───────────────────────────────────
    for n_seeds in SEED_SETS:
        log.info(f"\n--- Training {n_seeds}-seed ensemble ---")
        ensemble_preds = []

        for seed in range(n_seeds):
            log.info(f"  Seed {seed}/{n_seeds-1} ...")
            model, best_val, stopped = train_single_seed(
                seed, X_tr, y_tr, X_va, y_va, N_FEAT, device
            )
            model.eval()
            with torch.no_grad():
                preds_sc = model(torch.FloatTensor(X_te).to(device)).cpu().numpy()
            preds_mwh = invert(preds_sc)
            ensemble_preds.append(preds_mwh)
            log.info(f"  Seed {seed} done — val_loss={best_val:.6f} epoch={stopped}")

        # P1 = ensemble variance
        p1 = np.var(ensemble_preds, axis=0)

        # Evaluate at extreme hours
        rho, pval = spearmanr(p1[extreme_mask], abs_error[extreme_mask])
        sig       = pval < BONFERRONI

        log.info(f"\n{n_seeds}-seed result: rho_extreme={rho:+.4f}  pval={pval:.4f}  sig={sig}")

        all_results.append({
            'grid':        grid,
            'n_seeds':     n_seeds,
            'rho_extreme': round(float(rho),  4),
            'pval':        round(float(pval), 6),
            'significant': sig,
            'source':      'Extension 3',
        })

# Save
out = PROJECT_ROOT / "results/17_ensemble_sensitivity/sensitivity_results.csv"
pd.DataFrame(all_results).to_csv(out, index=False)
log.info(f"\nSaved {len(all_results)} rows → {out}")

# Print summary
print(f"\n{'='*60}")
print("ENSEMBLE SENSITIVITY — RESULTS SUMMARY")
print(f"{'='*60}")
df = pd.DataFrame(all_results)
for grid in ['uci', 'gefcom']:
    print(f"\n{grid.upper()}:")
    gdf = df[df['grid']==grid].sort_values('n_seeds')
    for _, row in gdf.iterrows():
        sig_str = '✓ SIG' if row['significant'] else '✗ NON-SIG'
        print(f"  {int(row['n_seeds']):>2} seeds: rho={row['rho_extreme']:+.4f}  p={row['pval']:.4f}  {sig_str}")

print(f"\nExtension 3 COMPLETE.")
