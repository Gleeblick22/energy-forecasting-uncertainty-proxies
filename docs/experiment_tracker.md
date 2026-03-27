 
# Experiment Tracker
## Project: When AI Forecasts Are Confidently Wrong: Evaluating Uncertainty Proxies in Energy Load Forecasting
**Last updated:** March 23, 2026
**Project specification version:** v5.0 (architecture locked) 

####---------------------------------------------------------------####

## How to Use This File
After every training run, add a row to the relevant table.
After every completed experiment block, update the Status section.
Never delete rows — mark failed runs as FAILED with a note.

###----------------------------------------------------------------###

## Phase 1 — Data Preprocessing

| Dataset | Script | Status | Date | Notes |
|---------|--------|--------|------|-------|
| UCI | experiments/03_preprocess_uci/preprocess_uci.py |  COMPLETE | Jan 03 2026 | All P1-P8 passed |
| GEFCom | experiments/04_preprocess_gefcom/preprocess_gefcom.py |  COMPLETE | Jan 15 2026 | All G1-G9 passed, 21 gaps interpolated, temperature_F corrected |


### UCI Preprocessing Results (locked)
| Metric | Value |
|--------|-------|
| Raw shape | (140256, 370) |
| Hourly rows | 35,064 |
| Features | 11 |
| Train rows | 23,928 (Jan 8 2011 - Sep 30 2013) |
| Val rows | 2,208 (Oct 1 - Dec 31 2013) |
| Test rows | 8,760 (Jan 1 - Dec 31 2014) |
| Extreme threshold (90th pctile) | 1,357.04 MWh |
| Extreme rows in test | 876 (10.0%) |
| Gaps | 0 |
| Scaler | scaler_uci.pkl (MinMaxScaler, fit on train only) |

### GEFCom Preprocessing Results (locked)
| Metric | Value |
|--------|-------|
| Raw rows (combined) | 57,024 |
| Features | 13 |
| Train rows | 23,928 |
| Val rows | 2,208 (Oct 1 - Dec 31 2009) |
| Test rows | 8,760 (Jan 1 - Dec 31 2010) |
| Extreme threshold (90th pctile) | 237.60 MWh |
| Extreme rows in test | 877 (10.0%) |
| Gaps | 21 structural gaps (216h each), all interpolated |
| Scaler | scaler_gefcom.pkl (MinMaxScaler, independent from UCI) |

###--------------------------------------------------------------------------###

## Phase 2A — Pilot Experiments

### Purpose
Run 10 seeds with full architecture but reduced epochs to verify:
1. Data loads correctly into LSTM dataloader.
2. Training loss decreases — model is learning.
3. Val loss follows train loss — no leakage.
4. Ensemble variance is non-zero — seeds produce different predictions.
5. Predictions are in correct MWh scale after inverse_transform.
6. Spearman rho is computable — no NaN/inf in proxy or error arrays.
7. |Spearman rho| > 0.10 on extreme hours OR visible scatter in proxy vs error plot.

### Locked Hyperparameters (project specification)
| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer stacked LSTM |
| Hidden units | 128 per layer |
| Output | 1-step ahead (t+1 only) |
| Input window | 168 hours (1 week) |
| Dropout | 0.2 (training only, disabled at standard inference) |
| Loss function | MSE |
| Optimizer | Adam, LR=0.001 |
| Batch size | 64 |
| Pilot max epochs | 30 |
| Early stopping patience | 10 |
| Pilot seeds | 0–9 (10 seeds) |
| Proxy checked | Ensemble variance only (P1) |
| Gate | |Spearman rho| > 0.10 on extreme hours OR visible scatter pattern |

### UCI Pilot Signal Check
| Metric | Value | Pass? |
|--------|-------|-------|
| Ensemble variance non-zero |  Yes | yes |
| Spearman rho (all test hours) | 0.2039 | yes |
| Spearman rho (extreme hours only) | > 0.10 | Yes  |
| p-value | < 0.0001 | Yes |
| Scatter pattern visible | Yes | Yes |
| **GATE** | **PASSED Jan 150 2026** | Yes |

### GEFCom Pilot Signal Check
| Metric | Value | Pass? |
|--------|-------|-------|
| Ensemble variance non-zero |  Yes | yes  |
| Spearman rho (all test hours) | 0.3007 | yes |
| Spearman rho (extreme hours only) | > 0.10 | yes |
| p-value | < 0.0001 | yes  |
| Scatter pattern visible | Yes | yes |
| **GATE** | **PASSED Jan 26 2026** | yes |

###----------------------------------------###

## Phase 2B — Full LSTM Training

### Locked Hyperparameters (same as pilot except max epochs=100)
| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer stacked LSTM |
| Hidden units | 128 per layer |
| Output | 1-step ahead (t+1 only) |
| Input window | 168 hours (1 week) |
| Dropout | 0.2 (training only) |
| Loss function | MSE |
| Optimizer | Adam, LR=0.001 |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Seeds | 0–19 (20 seeds, same for both datasets) |

### UCI Full LSTM Runs (experiments/05_lstm_uci/)
**Status:  COMPLETE — Feb 3 2026 — Runtime ~62 min — Device: CUDA**

| Seed | Status | Val Loss | MAE (MWh) | Converged Epoch | Notes |
|------|--------|----------|-----------|-----------------|-------|
| 0 |  DONE | 0.000222 | 19.06 | 41 | |
| 1 |  DONE | 0.000188 | 18.39 | 47 | |
| 2 |  DONE | 0.000193 | 17.86 | 37 | |
| 3 |  DONE | 0.000203 | 18.12 | 44 | |
| 4 |  DONE | 0.000188 | 17.71 | 39 | |
| 5 |  DONE | 0.000189 | 18.89 | 52 | |
| 6 |  DONE | 0.000182 | 18.70 | 62 | |
| 7 |  DONE | 0.000215 | 19.48 | 48 | |
| 8 |  DONE | 0.000176 | 20.35 | 57 | |
| 9 |  DONE | 0.000216 | 18.86 | 44 | |
| 10 |  DONE | 0.000227 | 19.24 | 39 | |
| 11 |  DONE | 0.000231 | 19.80 | 38 | |
| 12 |  DONE | 0.000228 | 19.64 | 43 | |
| 13 |  DONE | 0.000217 | 19.06 | 36 | |
| 14 |  DONE | 0.000230 | 18.80 | 33 | |
| 15 |  DONE | 0.000191 | 18.41 | 44 | |
| 16 |  DONE | 0.000203 | 19.02 | 40 | |
| 17 |  DONE | 0.000210 | 18.82 | 40 | |
| 18 |  DONE | 0.000183 | 18.60 | 50 | |
| 19 |  DONE | 0.000198 | 19.06 | 45 | |

**UCI Full Training Summary:**
| Metric | Value |
|--------|-------|
| Seeds completed | 20/20 |
| all_predictions.npy shape | (20, 8592) |
| Ensemble MAE (MWh) | **17.31** |
| Ensemble var mean (MWh²) | **91.14** |
| Mean per-seed MAE (MWh) | 18.89 ± 0.64 |
| Val loss range | 0.000176 – 0.000231 |
| Mean converged epoch | 44.0 |
| all_predictions.npy saved |  models/uci/lstm/all_predictions.npy |

###--------------------------------------------------------------------------------###

### GEFCom Full LSTM Runs (experiments/06_lstm_gefcom/)
**Status:  COMPLETE WITH NOTE (seeds 11, 13) — Feb 3 2026 — Runtime ~86 min — Device: CUDA**

| Seed | Status | Val Loss | MAE (MWh) | Converged Epoch | Notes |
|------|--------|----------|-----------|-----------------|-------|
| 0 |  DONE | 0.000241 | 4.48 | 63 | |
| 1 |  DONE | 0.000247 | 4.33 | 59 | |
| 2 |  DONE | 0.000173 | 3.92 | 100 | Hit max epochs |
| 3 |  DONE | 0.000223 | 4.03 | 70 | |
| 4 |  DONE | 0.000183 | 4.19 | 90 | |
| 5 |  DONE | 0.000251 | 4.41 | 60 | |
| 6 |  DONE | 0.000174 | 4.04 | 100 | Hit max epochs |
| 7 |  DONE | 0.000180 | 4.18 | 81 | |
| 8 |  DONE | 0.000153 | 3.98 | 100 | Hit max epochs |
| 9 |  DONE | 0.000243 | 4.53 | 61 | |
| 10 |  DONE | 0.000181 | 4.03 | 100 | Hit max epochs |
| 11 | Note DONE | 0.000533 | 5.59 | 21 | Early exit — retained, see note |
| 12 |  DONE | 0.000299 | 4.62 | 39 | |
| 13 | Note  DONE | 0.000566 | 5.88 | 18 | Early exit — retained, see note |
| 14 |  DONE | 0.000217 | 4.24 | 78 | |
| 15 |  DONE | 0.000233 | 4.16 | 64 | |
| 16 |  DONE | 0.000259 | 4.47 | 56 | |
| 17 |  DONE | 0.000213 | 4.15 | 70 | |
| 18 |  DONE | 0.000288 | 4.55 | 49 | |
| 19 |  DONE | 0.000196 | 3.93 | 93 | |

**Seeds 11 and 13 note:**
- Both stopped very early (epochs 18–21) with val_loss ~3× higher than typical.
- Decision: RETAINED in ensemble. Their contribution to ensemble variance is valid
uncertainty signal. Ensemble MAE (3.68 MWh) is robust — diluted by 18 normal seeds.
- Document in paper: "Two seeds converged early (epochs 18–21); retained as their
- variance contribution reflects genuine model uncertainty."

**GEFCom Full Training Summary:**
| Metric | Value |
|--------|-------|
| Seeds completed | 20/20 |
| all_predictions.npy shape | (20, 8592) |
| Ensemble MAE (MWh) | **3.68** |
| Ensemble var mean (MWh²) | **23.89** |
| Mean per-seed MAE (MWh) | 4.39 ± 0.51 |
| Val loss range | 0.000153 – 0.000566 |
| Mean converged epoch | 68.6 |
| all_predictions.npy saved |  models/gefcom/lstm/all_predictions.npy |

###------------------------------------------------------------------------------####

## Phase 3 — Uncertainty Proxy Computation

### Proxies Computed (per dataset)
| Proxy | Name | Source | Formula |
|-------|------|--------|---------|
| P1 | Ensemble Variance | LSTM 20 seeds | var(y_s,t for s=0..19) in original MWh scale |
| P2 | PI Width | ARIMA | upper_95(t) − lower_95(t) in MWh |
| P3 | Residual Volatility | LSTM | std(residuals t-24 to t-1), rolling 24h window in MWh |

### UCI Proxy Results
**Status:  COMPLETE**
| Proxy | Mean Value | Spearman rho (all) | p-value | Spearman rho (extreme) | p-value | Significant? |
|-------|-----------|-------------------|---------|----------------------|---------|-------------|
| P1 Ensemble Variance | 91.14 MWh² | 0.1893 | 0.0000 | 0.0089 | 0.7929 | all: DONE  extreme: NO |
| P2 PI Width (ARIMA) | 78.52 MWh | 0.0477 | 0.0000 | −0.0543 | 0.1086 | all: DONE extreme: NO|
| P3 Residual Volatility | 22.36 MWh | 0.1754 | 0.0000 | 0.0440 | 0.1933 | all:DONE  extreme: NO |

**UCI LSTM Error Split:**
| Split | MAE (MWh) |
|-------|-----------|
| All test | 17.31 |
| Extreme demand | 21.32 (+26.5%) |
| Normal demand | 16.86 |

### GEFCom Proxy Results
**Status:  COMPLETE**
| Proxy | Mean Value | Spearman rho (all) | p-value | Spearman rho (extreme) | p-value | Significant? |
|-------|-----------|-------------------|---------|----------------------|---------|-------------|
| P1 Ensemble Variance | 23.89 MWh² | 0.4396 | 0.0000 | 0.4815 | 0.0000 | all: DONE extreme: YES |
| P2 PI Width (ARIMA) | 42.08 MWh | 0.1551 | 0.0000 | 0.0182 | 0.5996 | all: DONE extreme: NO |
| P3 Residual Volatility | 4.44 MWh | 0.3338 | 0.0000 | 0.4554 | 0.0000 | all: DONE extreme: YES |

**GEFCom LSTM Error Split:**
| Split | MAE (MWh) |
|-------|-----------|
| All test | 3.68 |
| Extreme demand | 5.42 (+55.3%) |
| Normal demand | 3.49 |

---

## Phase 4 — ARIMA Baseline
**Status:  COMPLETE both datasets — Mar 3 2026**

| Dataset | Order | MAE (MWh) | PI Width (MWh) | Coverage | ACF lag24 | ACF lag48 | ACF lag168 | Notes |
|---------|-------|-----------|----------------|----------|-----------|-----------|------------|-------|
| UCI | (2,1,2)(1,1,1,24) | 14.44 | 78.52 | 0.938 | −0.020 DONE | +0.028 DONE | +0.109 DONE | Weekly autocorr expected — accepted |
| GEFCom | (2,1,2)(1,1,1,24) | 6.61 | 42.08 | 0.957 | −0.020 DONE | +0.009 DONE | +0.064 DONE | Weekly autocorr expected — accepted |

**ARIMA vs LSTM comparison:**
| Dataset | ARIMA MAE | LSTM MAE | Winner |
|---------|-----------|----------|--------|
| UCI | 14.44 MWh | 17.31 MWh | ARIMA (stable European grid) |
| GEFCom | 6.61 MWh | 3.68 MWh | LSTM (weather-driven grid) |

**ACF compliance decision:** Ljung-Box over-rejects at n=23,928 (Hyndman 2018).
ACF values at lag 24 and 48 confirm white noise. Lag 168 weekly autocorrelation
is a known SARIMA(m=24) limitation — documented in paper per project specification Section 7B.

**Key output files:**
- results/uci/tables/arima_predictions_uci.csv 
- results/gefcom/tables/arima_predictions_gefcom.csv 
- models/uci/arima/arima_diagnostics_uci.txt 
- models/gefcom/arima/arima_diagnostics_gefcom.txt 

###---------------------------------------------------------------------###

## Phase 5 — Conformal Prediction Benchmark
**Status:  COMPLETE WITH NOTE (UCI undercoverage)**

| Dataset | conformal_q (MWh) | Coverage | Target | Width mean (MWh) | Winkler Score | Status |
|---------|------------------|----------|--------|-----------------|---------------|--------|
| UCI | 30.88 | 0.843 | 0.90 | 61.76 | 113.9588 |  Below target |
| GEFCom | 5.10 | 0.875 | 0.90 | 10.20 | 42.7740 |  Acceptable |

**UCI coverage note (ACCEPTED decision):**
- Val set (Oct–Dec 2013, range 377–1327 MWh) calibrated on mild autumn/winter quarter.
- Test year (Jan–Dec 2014) includes summer peaks to 1740 MWh → undercoverage on extremes.
**Root cause:** seasonal val/test mismatch from calendar-based splitting.
**One sentence in paper limitations.** Benchmark Winkler Score comparison remains valid.

###----------------------------------------------------------------------------###

## Phase 6 — Full Evaluation + Cross-Dataset Comparison
**Status:  COMPLETE — Mar 25 2026**

### UCI — Full Error Metrics
| Model | MAE all (MWh) | MAE extreme (MWh) | MAE normal (MWh) | RMSE all (MWh) | RMSE extreme (MWh) | MAPE | RMSE/MAE |
|-------|--------------|-------------------|-----------------|---------------|-------------------|------|----------|
| LSTM | 17.31 | 21.32 | 16.86 | 25.28 | 28.45 | 5.22% | 1.460 |
| ARIMA | 14.14 | 20.44 | — | — | — | — | — |

Anomalous days (|load − 7d rolling mean| > 2σ): **7 total · 4 overlap with extreme**

### UCI — 2×2 Overconfidence Classification (error thresh=75th pctile, conf thresh=median)
| Proxy | OR_overall | OR_extreme | Elevated at extreme? | binom_p | Winkler Score |
|-------|------------|------------|---------------------|---------|---------------|
| P1 ensemble_var | 0.0914 | 0.0114 |  No (lower at extreme) | 1.0000 | 202.6498 |
| P2 pi_width | 0.0425 | 0.0525 |  Yes | 0.0857 | 171.4659 |
| P3 resid_vol | 0.0935 | 0.0616 |  No | 0.9997 | 220.0946 |
| Conformal | — | — | — | — | 166.1550 |

Friedman test: stat=7099.35, **p<0.0001** 

###--------------------------------------------------------------------###

### GEFCom — Full Error Metrics
| Model | MAE all (MWh) | MAE extreme (MWh) | MAE normal (MWh) | RMSE all (MWh) | RMSE extreme (MWh) | MAPE | RMSE/MAE |
|-------|--------------|-------------------|-----------------|---------------|-------------------|------|----------|
| LSTM | 3.68 | 5.42 | 3.49 | 9.81 | 12.05 | 2.36% | 2.668 |
| ARIMA | 5.74 | 6.08 | — | — | — | — | — |

Anomalous days: **403 total · 81 overlap with extreme**

### GEFCom — 2×2 Overconfidence Classification
| Proxy | OR_overall | OR_extreme | Elevated at extreme? | binom_p | Winkler Score |
|-------|------------|------------|---------------------|---------|---------------|
| P1 ensemble_var | 0.0534 | 0.0179 |  No (lower at extreme) | 1.0000 | 55.1499 |
| P2 pi_width | 0.0676 | 0.1029 |  Yes | 0.0001*** | 97.4419 |
| P3 resid_vol | 0.0729 | 0.0861 |  Yes | 0.0819 | 45.9661 |
| Conformal | — | — | — | — | 75.3455 |

**Notable: P3 (45.97) and P1 (55.15) beat conformal (75.35) on GEFCom.**

Friedman test: stat=11160.31, **p<0.0001** 

####-----------------------------------------------------------------------####

### Cross-Dataset Comparison (RQ4 + RQ5)

**RQ5 — Proxy Rankings by |rho_all|:**
| Dataset | Rank 1 | Rank 2 | Rank 3 |
|---------|--------|--------|--------|
| UCI | P1 (0.1893) | P3 (0.1754) | P2 (0.0477) |
| GEFCom | P1 (0.4396) | P3 (0.3338) | P2 (0.1551) |
| **Match** | **TRUE ** | | |

**RQ4 — Overconfidence elevated at extreme demand (both grids)?**
| Proxy | UCI elevated? | GEFCom elevated? | Both elevated? |
|-------|--------------|-----------------|----------------|
| P1 ensemble_var |  No |  No | **FALSE** |
| P2 pi_width |  Yes |  Yes | **TRUE ** |
| P3 resid_vol |  No |  Yes | **FALSE** |

**Rho direction consistency:**
| Proxy | UCI rho | GEFCom rho | Same direction? |
|-------|---------|------------|----------------|
| P1 | +0.1893 | +0.4396 |  Yes |
| P2 | +0.0477 | +0.1551 |  Yes |
| P3 | +0.1754 | +0.3338 |  Yes |

**Winkler Score gaps (proxy − conformal, negative = proxy better):**
| Proxy | UCI gap | GEFCom gap |
|-------|---------|------------|
| P1 | +36.49 (conformal better) | −20.20  (P1 better) |
| P2 | +5.31 (conformal slightly better) | +22.10 (conformal better) |
| P3 | +53.94 (conformal better) | −29.38  (P3 better) |

####----------------------------------------------------------------------####

## Gates — Final Status

| Gate | Condition | Status |
|------|-----------|--------|
| Phase 1A complete | UCI preprocessing P1–P8 passed |  PASSED  JAN 02 2026|
| Phase 1B complete | GEFCom preprocessing G1–G9 passed | ✅ PASSED  |
| Architecture locked | project specification, hidden=128, output=1-step | ✅ LOCKED  |
| Pilot UCI complete | 10 seeds, rho=0.2039 > 0.10 | ✅ PASSED  |
| Pilot GEFCom complete | 10 seeds, rho=0.3007 > 0.10 | ✅ PASSED  |
| Phase 2B UCI complete | 20 seeds, MAE=17.31 MWh | ✅ PASSED |
| Phase 2B GEFCom complete | 20 seeds, MAE=3.68 MWh | ✅ PASSED |
| Phase 3 proxies complete | P1+P2+P3 computed both datasets | ✅ PASSED  |
| Phase 4 ARIMA UCI | MAE=14.44, Coverage=0.938, ACF24=−0.020 | ✅ PASSED  |
| Phase 4 ARIMA GEFCom | MAE=6.61, Coverage=0.957, ACF24=−0.020 | ✅ PASSED  |
| Phase 5 conformal complete | Both datasets, Winkler computed | ✅ PASSED  |
| Phase 6 evaluation complete | results_summary_FINAL.csv populated | ✅ PASSED Mar 25 2026 |
| Phase 7 figures complete | All 5 publication figures | ⏳ PENDING |
| Paper draft complete | All sections written, within 6 pages | ⏳ PENDING |
| CP8 numbers audit | Every paper number vs results_summary_FINAL.csv | ⏳ PENDING |
| PDF eXpress validation | IEEE format compliant | ⏳ PENDING |
| Submission ready | ICRERA 2026 July deadline | ⏳ PENDING |

####------------------------------------------------------###

## Key Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Option B: independent calendar periods | Calendar-based splits literature-standard (Marino et al. 2016) |  2026 |
| GEFCom test year = 2010 | Last full calendar year in confirmed data range |  2026 |
| Interpolate GEFCom gaps | 21 × 216h structural gaps — linear interpolation standard |  2026 |
| GEFCom scaler independent | study design rule: no information leakage between datasets | 2026 |
| Reinstate pilot experiments | Engineering sanity check before 40 full training runs |  2026 |
| Output = 1-step (t+1) | Proxy and extreme flag must match at hourly resolution |  2026 |
| Hidden = 128 | Consistent with manuscript; matches literature standard |  2026 |
| Batch = 64 | Consistent with manuscript |  2026 |
| Pilot max epochs = 30 | Fast enough to catch bugs, early stopping handles convergence |  2026 |
| temperature_F corrected | GEFCom column mislabelled as temperature_C; values confirmed Fahrenheit |  2026 |
| Rolling one-step ARIMA forecast | Direct multi-step diverges over 8,760 steps |  2026 |
| Kaggle for ARIMA fitting | WSL CommitLimit insufficient; Colab Python 3.12 incompatible |  2026 |
| ACF-based ARIMA compliance | Ljung-Box over-rejects at n=23,928 per Hyndman 2018 |  2026 |
| project specification updated to v5 | Section 7B ACF criterion replaces LB p-value gate |  2026 |
| Shape (20,8592) accepted | 168h lookback window consumes first 168 test rows |  2026 |
| Retain GEFCom seeds 11+13 | Valid variance signal; diluted by 18 normal seeds |  2026 |
| Accept UCI conformal coverage 0.843 | Seasonal val/test mismatch; documented as limitation |  2026 |

###-------------------------------------------------------------------------------------------------------------####

## Known Issues / Notes for Paper

1. GEFCom structural gaps: Oct/Nov/Dec 1–9 missing every year. Must disclose in data section.
2. UCI DST artifact: extra 2015-01-01 00:00 timestamp truncated. No impact on analysis.
3. GEFCom: 13 features used (dew_point_C dropped — not derivable from temperature stations alone).
4. GEFCom temperature column named temperature_F (values confirmed Fahrenheit, range 12.7–97.7°F).
5. Ljung-Box fails large-n — ACF confirms well-specified. Report in paper. Cite Hyndman 2018.
6. ARIMA fitted on Kaggle — notebooks saved in experiments/07_arima_uci/ and 08_arima_gefcom/.
7. Shape (20,8592) vs project specification (20,8736) — 168h window offset. All analysis uses aligned index.
8. GEFCom seeds 11+13 early exit — retained, documented in paper methodology.
9. UCI conformal undercoverage (0.843) — seasonal mismatch, accepted, one sentence in limitations.
10. P3 resid_vol and P1 ensemble_var beat conformal on GEFCom — notable finding, highlight in results.

####-------------------------------------------------------------------------------------------------------############

## Results Summary — Single Source of Truth
*All numbers from results/summary/results_summary_FINAL.csv*

| Metric | UCI | GEFCom |
|--------|-----|--------|
| LSTM MAE all (MWh) | 17.31 | 3.68 |
| LSTM MAE extreme (MWh) | 21.32 | 5.42 |
| LSTM MAE normal (MWh) | 16.86 | 3.49 |
| LSTM RMSE all (MWh) | 25.28 | 9.81 |
| LSTM MAPE | 5.22% | 2.36% |
| ARIMA MAE all (MWh) | 14.14 | 5.74 |
| ARIMA MAE extreme (MWh) | 20.44 | 6.08 |
| ARIMA PI width (MWh) | 78.52 | 42.08 |
| ARIMA coverage | 0.938 | 0.957 |
| P1 rho_all | 0.1893*** | 0.4396*** |
| P1 rho_extreme | 0.0089 ns | 0.4815*** |
| P2 rho_all | 0.0477*** | 0.1551*** |
| P2 rho_extreme | −0.0543 ns | 0.0182 ns |
| P3 rho_all | 0.1754*** | 0.3338*** |
| P3 rho_extreme | 0.0440 ns | 0.4554*** |
| P1 Winkler Score | 202.6498 | 55.1499 |
| P2 Winkler Score | 171.4659 | 97.4419 |
| P3 Winkler Score | 220.0946 | 45.9661 |
| Conformal Winkler Score | 166.1550 | 75.3455 |
| Conformal coverage | **0.843 (LES THEN COMPLIENCE < 90.00>)** | 0.875  |
| Extreme timesteps | 876 (10.2%) | 836 (9.7%) |
| P2 OR_extreme elevated (both) | TRUE | TRUE |
| Proxy rank #1 | P1 | P1 |
| Rankings match across datasets | TRUE | — |
| Friedman p | <0.0001 | <0.0001 |

###--------------------------------------------------------------####
*Tracker maintained by: [DHAN GHALE]*
*Last updated: March 26, 2026*
*project specification · Implementation Guide v4.0 · Roadmap v4.0*
