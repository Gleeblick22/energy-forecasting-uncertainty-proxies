"""
Phase 7 — Master Runner
PDD v5 Section 9 + Section 10
Implementation Guide v4 Phase 7.1 + Phase 7.2

Folder: experiments/13_cross_dataset/cross_dataset.py
Run from: ~/projects/energy-forecasting-uncertainty-proxies/
Command:  python experiments/13_cross_dataset/cross_dataset.py

Outputs:
  results/uci/tables/evaluation_uci.csv
  results/gefcom/tables/evaluation_gefcom.csv
  results/comparison/cross_dataset.csv
  results/summary/results_summary_FINAL.csv  <- SINGLE SOURCE OF TRUTH
"""

import sys
import logging
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import evaluate
from compare  import cross_dataset_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ROOT = Path(".")

import os
os.makedirs(ROOT / "results/summary",    exist_ok=True)
os.makedirs(ROOT / "results/comparison", exist_ok=True)

log.info("=" * 56)
log.info("PHASE 7 — EVALUATION + CROSS-DATASET COMPARISON")
log.info("PDD v5 Section 9 + Section 10")
log.info("=" * 56)

# ── Phase 7.1 — per-dataset ───────────────────────────────────
uci_results = evaluate("uci")
gef_results = evaluate("gefcom")

# ── Phase 7.2 — cross-dataset ─────────────────────────────────
comparison  = cross_dataset_comparison(uci_results, gef_results)

# ── Save results_summary_FINAL.csv — SINGLE SOURCE OF TRUTH ──
summary_df   = pd.DataFrame([uci_results, gef_results])
summary_path = ROOT / "results/summary/results_summary_FINAL.csv"
summary_df.to_csv(summary_path, index=False)

# ── Final print — all key paper numbers ───────────────────────
log.info("")
log.info("=" * 56)
log.info("PHASE 7 COMPLETE — KEY RESULTS")
log.info("=" * 56)

for label, res in [("UCI", uci_results), ("GEFCOM", gef_results)]:
    log.info(f"")
    log.info(f"  {label}:")
    log.info(f"    LSTM MAE   all={res['lstm_mae_all']:.2f}  extreme={res['lstm_mae_extreme']:.2f}  normal={res['lstm_mae_normal']:.2f} MWh")
    log.info(f"    LSTM RMSE  all={res['lstm_rmse_all']:.2f}  MAPE={res['lstm_mape_all']:.2f}%")
    log.info(f"    ARIMA MAE  all={res['arima_mae_all']:.2f}  extreme={res['arima_mae_extreme']:.2f} MWh")
    for p in ["P1_ensemble_var", "P2_pi_width", "P3_resid_vol"]:
        log.info(f"    {p}:")
        log.info(f"      rho_all={res[f'{p}_rho_all']:.4f}  p={res[f'{p}_p_all']:.4f}  sig={res[f'{p}_sig_all']}")
        log.info(f"      OR_overall={res[f'{p}_or_overall']:.4f}  OR_extreme={res[f'{p}_or_extreme']:.4f}  binom_p={res[f'{p}_binom_p']:.4f}")
        log.info(f"      Winkler Score={res[f'{p}_winkler_score']:.4f}")
    log.info(f"    Conformal Winkler Score (alpha=0.05)={res['conformal_winkler_score']:.4f}")
    log.info(f"    Friedman p={res['friedman_p']:.6f}")

log.info("")
log.info("  RQ4 — Overconfidence elevated at extreme demand?")
for p in ["P1_ensemble_var", "P2_pi_width", "P3_resid_vol"]:
    log.info(f"    {p}: both_elevated={comparison[f'{p}_both_elevated']}")

log.info("")
log.info("  RQ5 — Proxy rankings:")
log.info(f"    UCI:    {comparison['uci_rank_1']} > {comparison['uci_rank_2']} > {comparison['uci_rank_3']}")
log.info(f"    GEFCom: {comparison['gef_rank_1']} > {comparison['gef_rank_2']} > {comparison['gef_rank_3']}")
log.info(f"    Match:  {comparison['proxy_rankings_match']}")

log.info("")
log.info(f"results_summary_FINAL.csv -> {summary_path}")
log.info("This is the single source of truth for all paper numbers.")
log.info("Next: Phase 8 — Generate publication figures")
