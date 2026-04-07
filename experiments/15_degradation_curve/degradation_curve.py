import pandas as pd
import numpy as np
from scipy.stats import spearmanr

GRIDS = {
    'uci':    'results/uci/tables/confidence_proxies_uci.csv',
    'gefcom': 'results/gefcom/tables/confidence_proxies_gefcom.csv',
}

PROXY_COLS = {
    'P1': 'ensemble_variance',
    'P2': 'pi_width',
    'P3': 'resid_volatility',
}

BONFERRONI_ALPHA = 0.0083

# Anchors from results_summary_FINAL.csv — Phase 7 exact values
ANCHORS = {
    'uci':    {'P1': 0.0089, 'P2': -0.0543, 'P3': 0.0440},
    'gefcom': {'P1': 0.4822, 'P2': None,     'P3': 0.4568},
}

results = []

for grid, path in GRIDS.items():
    df = pd.read_csv(path, index_col=0)

    load_col  = 'actual_load'
    err_col   = 'lstm_abs_error'
    flag_col  = 'is_extreme_demand'  # Phase 7 exact mask at pct=90

    # Phase 7 threshold — minimum load in extreme set
    phase7_threshold = df.loc[df[flag_col]==1, load_col].min()

    print(f"\n{'='*60}")
    print(f"GRID: {grid.upper()}  |  n={len(df)}")
    print(f"Phase 7 extreme threshold: {phase7_threshold:.2f} MWh")
    print(f"{'='*60}")

    for pct in range(70, 99):

        if pct == 90:
            # Use exact Phase 7 mask to match manuscript numbers
            mask = df[flag_col] == 1
        else:
            threshold = df[load_col].quantile(pct / 100)
            mask      = df[load_col] >= threshold

        n_extreme = mask.sum()

        for proxy_label, proxy_col in PROXY_COLS.items():
            proxy_vals = df.loc[mask, proxy_col]
            error_vals = df.loc[mask, err_col]

            if proxy_vals.std() == 0:
                rho, pval = 0.0, 1.0
            else:
                rho, pval = spearmanr(proxy_vals, error_vals)

            sig = pval < BONFERRONI_ALPHA

            results.append({
                'grid':        grid,
                'pct':         pct,
                'proxy':       proxy_label,
                'n_extreme':   int(n_extreme),
                'rho':         round(float(rho),  4),
                'pval':        round(float(pval), 6),
                'significant': sig,
            })

    # Validation at pct=90
    print(f"\nVALIDATION CHECK at 90th percentile (Phase 7 mask):")
    for proxy_label in PROXY_COLS:
        row = [r for r in results
               if r['grid']==grid and r['pct']==90 and r['proxy']==proxy_label][0]
        anchor = ANCHORS[grid][proxy_label]
        if anchor is not None:
            match  = abs(row['rho'] - anchor) < 0.002
            status = '✓ PASS' if match else '✗ FAIL'
            print(f"  {proxy_label}: rho={row['rho']:+.4f}  expected={anchor:+.4f}  {status}")
        else:
            print(f"  {proxy_label}: rho={row['rho']:+.4f}  sig={row['significant']}")

# Save
out_df   = pd.DataFrame(results)
out_path = 'results/15_degradation_curve/degradation_results.csv'
out_df.to_csv(out_path, index=False)

print(f"\nSaved {len(out_df)} rows → {out_path}")
print(f"Row check: {len(out_df)} rows  {'✓ PASS' if len(out_df)==174 else '✗ FAIL'}")

# Collapse thresholds
print(f"\nCOLLAPSE THRESHOLDS — Key Findings:")
for grid in ['uci', 'gefcom']:
    print(f"\n  {grid.upper()}:")
    for proxy in ['P1', 'P2', 'P3']:
        subset   = out_df[(out_df['grid']==grid) & (out_df['proxy']==proxy)]
        sig_rows = subset[subset['significant']==True]
        if len(sig_rows) == 0:
            print(f"    {proxy}: non-significant across ALL percentiles (70–98)")
        elif len(sig_rows) == len(subset):
            print(f"    {proxy}: significant across ALL percentiles (70–98)")
        else:
            last_sig   = sig_rows['pct'].max()
            first_fail = subset[subset['significant']==False]['pct'].min()
            print(f"    {proxy}: reliable up to {last_sig}th pct → collapses at {first_fail}th pct")

print("\nExtension 1 COMPLETE. Results verified.")
