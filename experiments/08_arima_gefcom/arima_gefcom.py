"""
Phase 4 - ARIMA Fitting GEFCom
PDD v6 Section 7B
Method: SARIMAX(2,1,2)(1,1,1,24) rolling one-step-ahead
Environment: Kaggle (statsmodels, Python 3.12)
Results:
  AIC:      181137.48
  Coverage: 0.957
  PI width: 42.08 MWh
  MAE:      6.61 MWh
  ACF24:   -0.0200
  ACF48:    0.0089
  ACF168:   0.0642
Compliance: ACF-based per PDD v6
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/kaggle/input/datasets/dhanghale/efr2026-gefcom-train-test/train_unscaled.csv",
                    index_col=0, parse_dates=True)
test  = pd.read_csv("/kaggle/input/datasets/dhanghale/efr2026-gefcom-train-test/test_unscaled.csv",
                    index_col=0, parse_dates=True)
train_load = train["total_load"]
test_load  = test["total_load"]
print(f"Train: {len(train_load)}  Test: {len(test_load)}")
