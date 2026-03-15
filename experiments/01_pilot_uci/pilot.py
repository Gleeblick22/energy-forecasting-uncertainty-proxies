"""
Phase 2A — Pilot Experiment (fixed)
PDD v6 | Implementation Guide v3.0 | Section 7A + Section 8

Change DATASET = "gefcom" for second pilot run.

Run from project root:
    conda activate energy_forecast
    python experiments/01_pilot_uci/pilot.py

GATE:
    |Spearman rho| > 0.10 on extreme hours  OR  on all hours
    Pass -> proceed to Phase 2B full 20-seed training
    Fail -> fix preprocessing -> recheck -> retry
"""

import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATASET   = "uci"           # "uci" or "gefcom"
WINDOW    = 168             # 1 week input sequence (PDD v6 Section 7A)
HIDDEN    = 128             # hidden size (PDD v6 Table 16 — locked)
LAYERS    = 2               # stacked LSTM layers
DROPOUT   = 0.2             # applied during training only
LR        = 0.001           # Adam learning rate
BATCH     = 64              # batch size (PDD v6 Table 16 — locked)
MAX_EPOCH = 30              # pilot only — full training uses 100
PATIENCE  = 10              # early stopping patience
SEEDS     = list(range(10)) # pilot: seeds 0-9
GATE_RHO  = 0.10            # minimum |Spearman rho| to pass gate

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / f"data/{DATASET}/splits"
RESULTS_FIG  = PROJECT_ROOT / f"results/{DATASET}/figures"
RESULTS_TAB  = PROJECT_ROOT / f"results/{DATASET}/tables"

for d in [RESULTS_FIG, RESULTS_TAB]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SEED CONTROL
# ══════════════════════════════════════════════════════════════════════════════
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — 2-layer LSTM, hidden=128, output=1-step (PDD v6 Table 16)
# ══════════════════════════════════════════════════════════════════════════════
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDER — 1-step output, target = next total_load (last col)
# ══════════════════════════════════════════════════════════════════════════════
def make_sequences(scaled_array: np.ndarray, window: int = WINDOW):
    X, y = [], []
    for i in range(len(scaled_array) - window):
        X.append(scaled_array[i : i + window])
        y.append(scaled_array[i + window, -1])   # total_load is last col
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD UNSCALED DATA — fit fresh scaler on train only
# ══════════════════════════════════════════════════════════════════════════════
log.info(f"Loading {DATASET.upper()} unscaled splits ...")

train_df = pd.read_csv(DATA_DIR / "train_unscaled.csv", index_col=0, parse_dates=True)
val_df   = pd.read_csv(DATA_DIR / "val_unscaled.csv",   index_col=0, parse_dates=True)
test_df  = pd.read_csv(DATA_DIR / "test_unscaled.csv",  index_col=0, parse_dates=True)
extreme  = pd.read_csv(DATA_DIR / f"extreme_{DATASET}.csv")

# CHECK 1 — data loads, no NaN, target column present
assert len(train_df) > 0,                        "CHECK 1 FAILED — train empty"
assert len(test_df)  > 0,                        "CHECK 1 FAILED — test empty"
assert "total_load" in train_df.columns,          "CHECK 1 FAILED — total_load missing"
assert train_df.isna().sum().sum() == 0,          "CHECK 1 FAILED — NaN in train"
assert test_df.isna().sum().sum()  == 0,          "CHECK 1 FAILED — NaN in test"

# Ensure total_load is last column
cols = [c for c in train_df.columns if c != "total_load"] + ["total_load"]
train_df = train_df[cols]
val_df   = val_df[cols]
test_df  = test_df[cols]

N_FEAT = len(cols)
log.info(f"CHECK 1 PASSED — train={len(train_df)}, val={len(val_df)}, "
         f"test={len(test_df)}, n_features={N_FEAT}")

# Fit fresh scaler on TRAIN only — total_load is last col
scaler = MinMaxScaler()
scaler.fit(train_df)

# Save fresh scaler so full training uses the same one
pickle.dump(scaler, open(
    PROJECT_ROOT / f"models/{DATASET}/configs/scaler_{DATASET}_pilot.pkl", "wb"))

# Invert load predictions: use load column min/max from fresh scaler
_load_min = scaler.data_min_[-1]
_load_max = scaler.data_max_[-1]
log.info(f"Fresh scaler load range: {_load_min:.2f} – {_load_max:.2f} MWh")

def invert_load(scaled_vals: np.ndarray) -> np.ndarray:
    """Inverse MinMaxScaler for total_load column only -> original MWh."""
    return scaled_vals * (_load_max - _load_min) + _load_min

# Scale all splits
tr_scaled = scaler.transform(train_df).astype(np.float32)
va_scaled = scaler.transform(val_df).astype(np.float32)
te_scaled = scaler.transform(test_df).astype(np.float32)

# Verify scaling worked
assert tr_scaled[:, -1].min() >= -0.01, "Scaling sanity fail — load below 0"
assert tr_scaled[:, -1].max() <= 1.01,  "Scaling sanity fail — load above 1"
log.info(f"Scaling OK — train load scaled range: "
         f"[{tr_scaled[:,-1].min():.3f}, {tr_scaled[:,-1].max():.3f}]")

# Build sequences
X_tr, y_tr = make_sequences(tr_scaled)
X_va, y_va = make_sequences(va_scaled)
X_te, y_te = make_sequences(te_scaled)
log.info(f"Sequences: X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}")

# Actual MWh values aligned to test sequences
actual = test_df["total_load"].values[WINDOW : WINDOW + len(y_te)]

# Align extreme demand flag
extreme_test = extreme["is_extreme_90"].values[WINDOW : WINDOW + len(y_te)]
log.info(f"Extreme timesteps in test window: {extreme_test.sum()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP — 10 seeds
# ══════════════════════════════════════════════════════════════════════════════
all_preds_orig    = []
train_loss_curves = []
val_loss_curves   = []
seed_results      = []

for seed in SEEDS:
    set_all_seeds(seed)
    log.info(f"--- Seed {seed} ---")

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
        # Train
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

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                model(torch.FloatTensor(X_va).to(device)),
                torch.FloatTensor(y_va).to(device)
            ).item()
        va_losses.append(val_loss)

        # CHECK 2 — loss must decrease by epoch 5
        if epoch == 4:
            assert tr_losses[-1] < tr_losses[0], (
                f"CHECK 2 FAILED seed {seed} — loss not decreasing"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info(f"  Early stop at epoch {epoch + 1}")
                break

    train_loss_curves.append(tr_losses)
    val_loss_curves.append(va_losses)

    # CHECK 2 final
    assert tr_losses[-1] < tr_losses[0], (
        f"CHECK 2 FAILED seed {seed} — loss did not decrease overall"
    )
    log.info(f"  CHECK 2 PASSED — {tr_losses[0]:.6f} -> {tr_losses[-1]:.6f}")

    # CHECK 3 — val/train ratio
    ratio = va_losses[-1] / (tr_losses[-1] + 1e-9)
    assert ratio < 10.0, f"CHECK 3 FAILED seed {seed} — ratio={ratio:.2f}"
    log.info(f"  CHECK 3 PASSED — val/train ratio: {ratio:.2f}")

    # Test predictions — batched to avoid GPU OOM
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_list = []
        for i in range(0, len(X_te), 256):
            xb = torch.FloatTensor(X_te[i:i+256]).to(device)
            preds_list.append(model(xb).cpu().numpy())
        preds_norm = np.concatenate(preds_list, axis=0)

    # CHECK 5 — inverse transform to MWh
    preds_orig = invert_load(preds_norm)

    assert preds_orig.min() > 0, (
        f"CHECK 5 FAILED seed {seed} — negative MWh predictions"
    )
    assert preds_orig.std() > 0, (
        f"CHECK 5 FAILED seed {seed} — constant MWh predictions"
    )
    log.info(f"  CHECK 5 PASSED — range [{preds_orig.min():.1f}, "
             f"{preds_orig.max():.1f}] MWh")

    all_preds_orig.append(preds_orig)
    seed_results.append({
        "seed":            seed,
        "converged_epoch": len(tr_losses),
        "best_val_loss":   round(best_val_loss, 6),
        "pred_min_mwh":    round(float(preds_orig.min()), 2),
        "pred_max_mwh":    round(float(preds_orig.max()), 2),
        "pred_std_mwh":    round(float(preds_orig.std()), 2),
    })

all_preds_orig = np.array(all_preds_orig)   # shape (10, n_test_sequences)


# ══════════════════════════════════════════════════════════════════════════════
# POST-TRAINING CHECKS
# ══════════════════════════════════════════════════════════════════════════════
ensemble_mean = all_preds_orig.mean(axis=0)
ensemble_var  = all_preds_orig.var(axis=0, ddof=0)   # MWh² scale

# CHECK 4 — ensemble variance non-zero
assert ensemble_var.std() > 0, (
    "CHECK 4 FAILED — ensemble variance is zero. Seeds not initialising differently."
)
log.info(f"CHECK 4 PASSED — ensemble var mean={ensemble_var.mean():.2f} MWh², "
         f"std={ensemble_var.std():.2f}")

# CHECK 6 — no NaN/Inf
abs_error = np.abs(actual - ensemble_mean)
assert not np.isnan(ensemble_var).any(), "CHECK 6 FAILED — NaN in ensemble_var"
assert not np.isnan(abs_error).any(),    "CHECK 6 FAILED — NaN in abs_error"
assert not np.isinf(ensemble_var).any(), "CHECK 6 FAILED — Inf in ensemble_var"
assert not np.isinf(abs_error).any(),    "CHECK 6 FAILED — Inf in abs_error"
log.info(f"CHECK 6 PASSED — abs_error mean={abs_error.mean():.2f} MWh, "
         f"max={abs_error.max():.2f} MWh")

# CHECK 7 / GATE
extreme_mask = extreme_test.astype(bool)[:len(actual)]
rho_all,     p_all     = spearmanr(ensemble_var, abs_error)
rho_extreme, p_extreme = spearmanr(ensemble_var[extreme_mask],
                                   abs_error[extreme_mask])
rho_normal,  p_normal  = spearmanr(ensemble_var[~extreme_mask],
                                   abs_error[~extreme_mask])


# ══════════════════════════════════════════════════════════════════════════════
# GATE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"{DATASET.upper()} PILOT — GATE EVALUATION")
print("=" * 60)
print(f"  Spearman rho (all test hours):    {rho_all:.4f}   p={p_all:.4f}")
print(f"  Spearman rho (extreme hours):     {rho_extreme:.4f}   p={p_extreme:.4f}")
print(f"  Spearman rho (normal hours):      {rho_normal:.4f}   p={p_normal:.4f}")
print(f"  Extreme timesteps:                {extreme_mask.sum()}")
print(f"  Ensemble var mean (MWh^2):        {ensemble_var.mean():.2f}")
print(f"  Mean absolute error (MWh):        {abs_error.mean():.2f}")

gate_passed = False
gate_reason = ""

if abs(rho_extreme) >= GATE_RHO:
    gate_passed = True
    gate_reason = f"|rho_extreme| = {abs(rho_extreme):.4f} >= {GATE_RHO}"
elif abs(rho_all) >= GATE_RHO:
    gate_passed = True
    gate_reason = f"|rho_all| = {abs(rho_all):.4f} >= {GATE_RHO} (all hours)"
else:
    gate_reason = (
        f"|rho_extreme|={abs(rho_extreme):.4f} and "
        f"|rho_all|={abs(rho_all):.4f} both < {GATE_RHO}"
    )

if gate_passed:
    print(f"\n  GATE PASSED — {gate_reason}")
    print(f"  -> Proceed to Phase 2B full 20-seed training")
else:
    print(f"\n  GATE NOT PASSED — {gate_reason}")
    print(f"  -> Check: inverse_transform, extreme_mask alignment, seed control")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-whitegrid")

# Figure 1 — Proxy vs Error scatter
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(ensemble_var[~extreme_mask], abs_error[~extreme_mask],
           alpha=0.3, s=5, color="#1565C0", label="Normal demand")
ax.scatter(ensemble_var[extreme_mask], abs_error[extreme_mask],
           alpha=0.6, s=10, color="#C00000", label="Extreme demand")
ax.set_xlabel("Ensemble Variance (MWh²)", fontsize=12)
ax.set_ylabel("Absolute Error (MWh)", fontsize=12)
ax.set_title(
    f"{DATASET.upper()} Pilot — Proxy vs Error\n"
    f"Spearman rho (extreme) = {rho_extreme:.3f}   "
    f"Spearman rho (all) = {rho_all:.3f}",
    fontsize=13,
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(RESULTS_FIG / "pilot_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close()
log.info(f"Scatter saved -> {RESULTS_FIG}/pilot_scatter.pdf")

# Figure 2 — Loss curves (first 3 seeds)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i in range(min(3, len(SEEDS))):
    axes[0].plot(train_loss_curves[i], alpha=0.8, label=f"Seed {i}")
    axes[1].plot(val_loss_curves[i],   alpha=0.8, label=f"Seed {i}")
axes[0].set_title("Training Loss (first 3 seeds)", fontsize=12)
axes[1].set_title("Validation Loss (first 3 seeds)", fontsize=12)
for ax in axes:
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.legend(fontsize=9)
plt.suptitle(f"{DATASET.upper()} Pilot Loss Curves", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_FIG / "pilot_loss_curves.pdf", dpi=300, bbox_inches="tight")
plt.close()
log.info(f"Loss curves saved -> {RESULTS_FIG}/pilot_loss_curves.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
pd.DataFrame(seed_results).to_csv(RESULTS_TAB / "pilot_results.csv", index=False)

pd.DataFrame([{
    "dataset":            DATASET,
    "n_seeds":            len(SEEDS),
    "rho_all":            round(rho_all, 4),
    "p_all":              round(p_all, 4),
    "rho_extreme":        round(rho_extreme, 4),
    "p_extreme":          round(p_extreme, 4),
    "rho_normal":         round(rho_normal, 4),
    "ensemble_var_mean":  round(float(ensemble_var.mean()), 4),
    "abs_error_mean_mwh": round(float(abs_error.mean()), 2),
    "extreme_count":      int(extreme_mask.sum()),
    "gate_passed":        gate_passed,
    "gate_reason":        gate_reason,
}]).to_csv(RESULTS_TAB / "pilot_gate_summary.csv", index=False)

log.info(f"Results       -> {RESULTS_TAB}/pilot_results.csv")
log.info(f"Gate summary  -> {RESULTS_TAB}/pilot_gate_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL PRINT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"{DATASET.upper()} PILOT COMPLETE")
print("=" * 60)
print(f"  Seeds run:              {len(SEEDS)}")
print(f"  Test sequences:         {all_preds_orig.shape[1]}")
print(f"  Extreme timesteps:      {extreme_mask.sum()}")
print(f"  Mean abs error (MWh):   {abs_error.mean():.2f}")
print(f"  Ensemble var mean:      {ensemble_var.mean():.2f} MWh^2")
print(f"  Spearman rho (all):     {rho_all:.4f}   p={p_all:.4f}")
print(f"  Spearman rho (extreme): {rho_extreme:.4f}   p={p_extreme:.4f}")
print(f"  GATE: {'PASSED' if gate_passed else 'NOT PASSED'}")
print("=" * 60)

if gate_passed and DATASET == "uci":
    print("\nNext: run GEFCom pilot")
    print("  cp experiments/01_pilot_uci/pilot.py experiments/02_pilot_gefcom/pilot.py")
    print("  Edit DATASET = 'gefcom' then run")
elif gate_passed and DATASET == "gefcom":
    print("\nBoth pilots passed -> Phase 2B full training.")
else:
    print("\nDo NOT proceed. Fix diagnostics and rerun.")
