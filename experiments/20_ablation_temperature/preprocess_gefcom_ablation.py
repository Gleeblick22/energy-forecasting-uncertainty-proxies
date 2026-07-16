"""
Temperature Ablation — GEFCom2014 Preprocessing
Addresses Reviewer #3 Comment 5: Controlled ablation 
removing temperature features to test weather-load 
coupling hypothesis.

Identical to preprocess_gefcom.py EXCEPT:
- temperature_F removed from features
- temperature_lag_24h removed from features
- Feature count: 11 (was 13)

Run from: ~/projects/energy-forecasting-uncertainty-proxies/
Command:  python3 experiments/20_ablation_temperature/preprocess_gefcom_ablation.py
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(".")
PROC_DIR   = ROOT / "data/gefcom_ablation/processed"
SPLIT_DIR  = ROOT / "data/gefcom_ablation/splits"
CONFIG_DIR = ROOT / "models/gefcom_ablation/configs"
PROC_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2006-01-09"
TRAIN_END   = "2008-12-31"
VAL_START   = "2009-01-01"
VAL_END     = "2009-12-31"
TEST_START  = "2010-01-01"
TEST_END    = "2010-12-31"

log.info("Loading existing processed GEFCom data...")
src      = ROOT / "data/gefcom/processed/hourly_load_weather.csv"
combined = pd.read_csv(src, index_col=0, parse_dates=True)
log.info(f"Loaded {len(combined):,} rows")

log.info("Building features WITHOUT temperature...")
feat = pd.DataFrame(index=combined.index)
feat["total_load"] = combined["total_load"]

hour  = feat.index.hour
dow   = feat.index.dayofweek
month = feat.index.month

feat["hour_sin"]      = np.sin(2*np.pi*hour/24)
feat["hour_cos"]      = np.cos(2*np.pi*hour/24)
feat["day_sin"]       = np.sin(2*np.pi*dow/7)
feat["day_cos"]       = np.cos(2*np.pi*dow/7)
feat["month_sin"]     = np.sin(2*np.pi*month/12)
feat["month_cos"]     = np.cos(2*np.pi*month/12)
feat["is_weekend"]    = (dow >= 5).astype(int)
feat["is_holiday_MA"] = 0
feat["lag_1h"]        = feat["total_load"].shift(1)
feat["lag_24h"]       = feat["total_load"].shift(24)
feat["lag_168h"]      = feat["total_load"].shift(168)

n_before = len(feat)
feat.dropna(inplace=True)
log.info(f"Dropped {n_before - len(feat)} NaN rows")

feature_cols = [c for c in feat.columns if c != "total_load"]
assert len(feature_cols) == 11, \
    f"Expected 11 features got {len(feature_cols)}: {feature_cols}"
log.info(f"Features ({len(feature_cols)}): {feature_cols}")

feat.to_csv(PROC_DIR / "features.csv")

train = feat[TRAIN_START:TRAIN_END]
val   = feat[VAL_START:VAL_END]
test  = feat[TEST_START:TEST_END]
log.info(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

all_cols = feature_cols + ["total_load"]
scaler   = MinMaxScaler()
scaler.fit(train[all_cols])

for name, df in [("train", train), ("val", val), ("test", test)]:
    df_s          = df.copy()
    df_s[all_cols] = scaler.transform(df[all_cols])
    df_s.to_csv(SPLIT_DIR / f"{name}.csv")
    log.info(f"Saved {name} → {SPLIT_DIR}/{name}.csv")

pickle.dump(scaler, open(CONFIG_DIR / "scaler_gefcom_ablation.pkl", "wb"))
log.info("ABLATION PREPROCESSING COMPLETE ✓")
log.info(f"Removed: temperature_F, temperature_lag_24h")
log.info(f"Features remaining: {feature_cols}")
