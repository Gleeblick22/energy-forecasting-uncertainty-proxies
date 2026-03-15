# Conditional Reliability of Uncertainty Proxies Under Extreme Demand
### A Cross-Grid Empirical Analysis of Load Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-under%20review-yellow.svg)]()

> This study empirically evaluates whether uncertainty proxies from AI load forecasting models provide reliable warnings during extreme demand conditions, the hours when grid operators need accurate confidence signals most. Using a 20-seed LSTM ensemble and SARIMA baseline across two geographically distinct grids (UCI Portugal and GEFCom2014 New England), we show that proxy reliability is grid-dependent: ensemble variance (P1) and residual volatility (P3) retain strong correlation with forecast error at extreme demand on the weather-sensitive New England grid (rho=+0.482, rho=+0.455) but collapse entirely on the stable Mediterranean Portuguese grid (rho=+0.009, rho=+0.044). Standard aggregate validation metrics mask this conditional failure. ARIMA prediction interval width (P2) fails systematically on both grids due to structural non-adaptivity. Findings are robust to demand threshold definition (90th vs 95th percentile).

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | Do LSTM forecast errors amplify during extreme demand hours? |
| RQ2 | Do uncertainty proxies correlate with forecast error at extreme demand? |
| RQ3 | Do proxies produce statistically significant overconfidence at extreme demand? |
| RQ4 | Which proxy method scores best under formal interval scoring? |
| RQ5 | Are proxy calibration curves monotonically increasing across demand regimes? |

---

## Datasets

| Property | UCI Electricity Load Diagrams | GEFCom2014 Load Track |
|----------|------------------------------|----------------------|
| Grid | Portugal (Mediterranean) | New England ISO (Continental) |
| Period | 2011-2014 | 2007-2010 |
| Weather | None | 25 temperature stations |
| Test year | 2014 | 2010 |
| Test samples | 8,592 hours | 8,592 hours |
| Extreme threshold (90th pct) | 1,357.04 MWh | 237.60 MWh |
| Extreme hours in test | 876 (10.0%) | 877 (10.0%) |
| Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/321) | [Hong et al. 2016](https://doi.org/10.1016/j.ijforecast.2016.02.001) |

> **Note:** Raw datasets are not included due to licensing. See [Data Setup](#data-setup) below.

---

## Models and Proxies

**Primary forecaster:** Two-layer LSTM, 20-seed ensemble (PyTorch 2.0)
- Hidden size: 128 | Layers: 2 | Dropout: 0.2
- Input window: 168 hours (1 week) | Optimizer: Adam (lr=0.001)
- Early stopping: patience=10

**Baseline forecaster:** SARIMA fitted once per grid, evaluated on rolling window

**Uncertainty proxies evaluated:**

| Proxy | Method | Description |
|-------|--------|-------------|
| P1 | LSTM Ensemble Variance | Variance across 20 seed predictions |
| P2 | ARIMA Prediction Interval Width | Static PI width from single SARIMA fit |
| P3 | Residual Volatility | Rolling 24-hour std of LSTM residuals |
| CP | Conformal Prediction | Split conformal baseline (coverage 84.3% UCI, 87.5% GEFCom) |

---

## Key Results

| Metric | UCI (Portugal) | GEFCom2014 (New England) |
|--------|---------------|--------------------------|
| LSTM MAE — normal demand | 16.86 MWh | 3.49 MW |
| LSTM MAE — extreme demand | 21.32 MWh (+26.4%) | 5.42 MW (+55.3%) |
| P1 Spearman rho — all hours | +0.189 *** | +0.440 *** |
| P1 Spearman rho — extreme hours | +0.009 (ns) | +0.482 *** |
| P3 Spearman rho — extreme hours | +0.044 (ns) | +0.455 *** |
| P2 overconfidence OR (extreme) | 0.053 (ns) | 0.103 *** |
| Best Winkler score | Conformal: 166.2 | P3: 46.0 |
| Sensitivity at 95th pct (P1) | -0.110 (ns) | +0.423 *** |

`***` p < 0.0001 | `ns` not significant after Bonferroni correction (alpha = 0.0083, 6 simultaneous tests)  
UCI load in MWh; GEFCom2014 load in MW.

---

## Project Structure
```
├── data/
│   ├── uci/
│   │   ├── raw/                    # LD2011_2014.txt (not in repo)
│   │   ├── processed/              # Hourly aggregated features
│   │   └── splits/                 # Train / val / test / extreme splits
│   └── gefcom/
│       ├── raw/                    # GEFCom2014 task files (not in repo)
│       ├── processed/              # Hourly aggregated features
│       └── splits/                 # Train / val / test / extreme splits
├── experiments/
│   ├── 01_pilot_uci/               # Gate: Spearman rho > 0.10 on extreme hours
│   ├── 02_pilot_gefcom/
│   ├── 03_preprocess_uci/          # Aggregation, resampling, feature engineering
│   ├── 04_preprocess_gefcom/
│   ├── 05_lstm_uci/                # 20-seed LSTM training
│   ├── 06_lstm_gefcom/
│   ├── 07_arima_uci/               # SARIMA fit and evaluation
│   ├── 08_arima_gefcom/
│   ├── 09_proxies_uci/             # P1, P2, P3 computation
│   ├── 10_proxies_gefcom/
│   ├── 11_conformal_uci/           # Split conformal prediction
│   ├── 12_conformal_gefcom/
│   ├── 13_cross_dataset/           # Cross-grid comparison and evaluation
│   ├── 14_figures/                 # All 5 manuscript figures
│   └── sensitivity_95.py           # 95th percentile threshold sensitivity
├── models/
│   ├── uci/
│   │   ├── lstm/                   # 20 seed checkpoints + all_predictions.npy
│   │   ├── arima/                  # SARIMA diagnostics
│   │   └── configs/                # LSTM config, scaler
│   └── gefcom/
│       ├── lstm/
│       ├── arima/
│       └── configs/
├── results/
│   ├── uci/
│   │   ├── figures/                # fig1-fig5 (PDF + PNG)
│   │   └── tables/                 # Proxy CSVs, conformal, evaluation
│   ├── gefcom/
│   ├── comparison/                 # Cross-dataset summary
│   └── summary/
│       └── results_summary_FINAL.csv   # Single source of truth (4 rows x 93 cols)
├── docs/
│   ├── research_log.md             # Session-by-session decisions
│   ├── experiment_tracker.md       # Phase status and gate results
│   └── data_cards.md               # Full dataset documentation
├── logs/                           # LSTM and ARIMA training logs
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Data Setup

**UCI dataset:**
1. Download `LD2011_2014.txt` from [UCI ML Repository](https://archive.ics.uci.edu/dataset/321)
2. Place at `data/uci/raw/LD2011_2014.txt`

**GEFCom2014 dataset:**
1. Download from [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) or the official competition archive
2. Place task files at `data/gefcom/raw/GEFCom2014 Data/Load/`

---

## Reproducing Results

### 1. Install dependencies
```bash
pip install -r requirements.txt
# PyTorch requires separate installation:
# https://pytorch.org/get-started/locally/
```

### 2. Preprocess
```bash
python3 experiments/03_preprocess_uci/preprocess_uci.py
python3 experiments/04_preprocess_gefcom/preprocess_gefcom.py
```

### 3. Train LSTM ensemble (20 seeds per grid)
```bash
python3 experiments/05_lstm_uci/train_all_seeds.py
python3 experiments/06_lstm_gefcom/train_all_seeds.py
```

### 4. Fit SARIMA baselines
```bash
python3 experiments/07_arima_uci/arima_uci.py
python3 experiments/08_arima_gefcom/arima_gefcom.py
```

### 5. Compute proxies and conformal intervals
```bash
python3 experiments/09_proxies_uci/compute_proxies.py
python3 experiments/10_proxies_gefcom/compute_proxies.py
python3 experiments/11_conformal_uci/conformal.py
python3 experiments/12_conformal_gefcom/conformal.py
```

### 6. Cross-dataset analysis and figures
```bash
python3 experiments/13_cross_dataset/cross_dataset.py
python3 experiments/14_figures/fig1_error_regime.py
python3 experiments/14_figures/fig2_proxy_scatter.py
python3 experiments/14_figures/fig3_heatmaps.py
python3 experiments/14_figures/fig4_ranking.py
python3 experiments/14_figures/fig5_calibration.py
```

### 7. Sensitivity analysis
```bash
python3 experiments/sensitivity_95.py
```

All verified results are stored in `results/summary/results_summary_FINAL.csv`.

---

## Requirements

- Python 3.10+
- PyTorch 2.0.1 (CUDA optional)
- See `requirements.txt` for full dependency list

---

## Citation

> Citation details will be added upon acceptance.

---

## Authors

**Dhan Ghale** — lead author  
Co-authors listed in the paper.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- UCI Electricity Load Diagrams: Trindade, A. (2015). UCI Machine Learning Repository.
- GEFCom2014: Hong, T. et al. (2016). Probabilistic energy forecasting. *International Journal of Forecasting*, 32(3), 896-913.
