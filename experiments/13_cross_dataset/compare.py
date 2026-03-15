"""
Phase 7.2 — Cross-Dataset Comparison
PDD v5 Section 9.3
Called by cross_dataset.py — do not run directly.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(".")


def cross_dataset_comparison(uci, gef):
    log.info("")
    log.info("=" * 56)
    log.info("PHASE 7.2 — CROSS-DATASET COMPARISON")
    log.info("PDD v5 Section 9.3")
    log.info("=" * 56)

    proxies    = ["P1_ensemble_var", "P2_pi_width", "P3_resid_vol"]
    comparison = {}

    # ── RQ5: Proxy ranking by |rho_all| ──────────────────────
    log.info("RQ5 — Proxy ranking by |Spearman rho_all|...")
    uci_ranks = sorted(proxies, key=lambda p: abs(uci[f"{p}_rho_all"]), reverse=True)
    gef_ranks = sorted(proxies, key=lambda p: abs(gef[f"{p}_rho_all"]), reverse=True)
    match     = (uci_ranks == gef_ranks)

    log.info(f"  UCI ranking:    {uci_ranks}")
    log.info(f"  GEFCom ranking: {gef_ranks}")
    log.info(f"  Rankings match: {match}  -> RQ5 ANSWER")

    comparison["uci_rank_1"]          = uci_ranks[0]
    comparison["uci_rank_2"]          = uci_ranks[1]
    comparison["uci_rank_3"]          = uci_ranks[2]
    comparison["gef_rank_1"]          = gef_ranks[0]
    comparison["gef_rank_2"]          = gef_ranks[1]
    comparison["gef_rank_3"]          = gef_ranks[2]
    comparison["proxy_rankings_match"]= match

    # ── RQ4: Overconfidence rate — both datasets ──────────────
    log.info("RQ4 — Overconfidence rate extreme vs overall...")
    for p in proxies:
        u_ext = uci[f"{p}_or_extreme"]
        u_all = uci[f"{p}_or_overall"]
        g_ext = gef[f"{p}_or_extreme"]
        g_all = gef[f"{p}_or_overall"]
        both  = (u_ext > u_all) and (g_ext > g_all)

        log.info(f"  {p}:")
        log.info(f"    UCI    extreme={u_ext:.4f}  overall={u_all:.4f}  elevated={u_ext > u_all}")
        log.info(f"    GEFCom extreme={g_ext:.4f}  overall={g_all:.4f}  elevated={g_ext > g_all}")
        log.info(f"    Both elevated: {both}  -> RQ4 ANSWER")

        comparison[f"{p}_uci_or_extreme"]  = round(u_ext, 4)
        comparison[f"{p}_uci_or_overall"]  = round(u_all, 4)
        comparison[f"{p}_gef_or_extreme"]  = round(g_ext, 4)
        comparison[f"{p}_gef_or_overall"]  = round(g_all, 4)
        comparison[f"{p}_both_elevated"]   = both

    # ── RQ4: Spearman direction consistency ───────────────────
    log.info("RQ4 — Spearman rho direction consistency...")
    for p in proxies:
        u_rho    = uci[f"{p}_rho_all"]
        g_rho    = gef[f"{p}_rho_all"]
        same_dir = (np.sign(u_rho) == np.sign(g_rho))
        log.info(f"  {p}: UCI rho={u_rho:.4f}  GEFCom rho={g_rho:.4f}  same direction={same_dir}")
        comparison[f"{p}_uci_rho_all"]    = round(u_rho, 4)
        comparison[f"{p}_gef_rho_all"]    = round(g_rho, 4)
        comparison[f"{p}_same_direction"] = same_dir

    # ── Winkler Score gap — conformal vs each proxy ───────────
    log.info("Winkler Score gap — proxy vs conformal benchmark...")
    for p in proxies:
        u_gap = uci[f"{p}_winkler_score"] - uci["conformal_winkler_score"]
        g_gap = gef[f"{p}_winkler_score"] - gef["conformal_winkler_score"]
        log.info(f"  {p}: UCI gap={u_gap:.4f}  GEFCom gap={g_gap:.4f}")
        comparison[f"{p}_uci_winkler_gap"] = round(u_gap, 4)
        comparison[f"{p}_gef_winkler_gap"] = round(g_gap, 4)

    comparison["uci_conformal_winkler"] = uci["conformal_winkler_score"]
    comparison["gef_conformal_winkler"] = gef["conformal_winkler_score"]

    # ── Friedman results from 7.1 ─────────────────────────────
    comparison["uci_friedman_stat"] = uci["friedman_stat"]
    comparison["uci_friedman_p"]    = uci["friedman_p"]
    comparison["gef_friedman_stat"] = gef["friedman_stat"]
    comparison["gef_friedman_p"]    = gef["friedman_p"]

    # ── Save ──────────────────────────────────────────────────
    out_path = ROOT / "results/comparison/cross_dataset.csv"
    pd.DataFrame([comparison]).to_csv(out_path, index=False)
    log.info(f"Saved -> {out_path}")
    log.info("PHASE 7.2 COMPLETE ✓")

    return comparison
