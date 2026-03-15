

import pickle, logging
import numpy as np, pandas as pd, holidays
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE  = PROJECT_ROOT / "data/uci/raw/LD2011_2014.txt"
PROC_DIR  = PROJECT_ROOT / "data/uci/processed"
SPLIT_DIR = PROJECT_ROOT / "data/uci/splits"
CFG_DIR   = PROJECT_ROOT / "models/uci/configs"
for d in [PROC_DIR, SPLIT_DIR, CFG_DIR]: d.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = "2011-01-08", "2013-09-30"
VAL_START,   VAL_END   = "2013-10-01", "2013-12-31"
TEST_START,  TEST_END  = "2014-01-01", "2014-12-31"

log.info("P1 — Loading raw file ...")
assert RAW_FILE.exists(), f"Raw file not found: {RAW_FILE}"
df_raw = pd.read_csv(RAW_FILE, sep=";", index_col=0, parse_dates=True, decimal=",", low_memory=False)
assert df_raw.shape == (140256, 370), f"P1 FAILED — got {df_raw.shape}"
log.info(f"P1 OK — {df_raw.shape}")

log.info("P2 — Aggregating ...")
df_raw["total_load"] = df_raw.sum(axis=1)

log.info("P3 — Resampling to hourly ...")
hourly = df_raw["total_load"].resample("H").sum() / 1000.0
hourly = hourly.loc[:"2014-12-31 23:00:00"]
assert len(hourly) == 35064, f"P3 FAILED — got {len(hourly)}"
log.info(f"P3 OK — {len(hourly)} rows")

log.info("P4 — Gap detection ...")
nan_mask = hourly.isna()
gap_records = []
if nan_mask.any():
    gap_groups = (nan_mask != nan_mask.shift()).cumsum()[nan_mask]
    for gid, group in gap_groups.groupby(gap_groups):
        gl = len(group)
        gap_records.append({"gap_start": group.index[0], "gap_end": group.index[-1], "length_hrs": gl, "filled": gl <= 3})
else:
    log.info("P4 — No gaps")
pd.DataFrame(gap_records).to_csv(PROC_DIR / "gap_log.csv", index=False)
hourly_filled = hourly.ffill(limit=3)
hourly_filled.name = "total_load"
hourly_filled.to_frame().to_csv(PROC_DIR / "hourly_load.csv")
log.info(f"P4 OK — {len(gap_records)} gaps")

log.info("P5 — Building features ...")
feat = pd.DataFrame(index=hourly_filled.index)
feat["total_load"] = hourly_filled.values
hour, dow, month = feat.index.hour, feat.index.dayofweek, feat.index.month
feat["hour_sin"]  = np.sin(2*np.pi*hour/24)
feat["hour_cos"]  = np.cos(2*np.pi*hour/24)
feat["day_sin"]   = np.sin(2*np.pi*dow/7)
feat["day_cos"]   = np.cos(2*np.pi*dow/7)
feat["month_sin"] = np.sin(2*np.pi*month/12)
feat["month_cos"] = np.cos(2*np.pi*month/12)
feat["is_weekend"]    = (dow >= 5).astype(int)
pt_hols = holidays.Portugal(years=range(2011,2015))
feat["is_holiday_PT"] = feat.index.normalize().isin(pt_hols).astype(int)
feat["lag_1h"]   = feat["total_load"].shift(1)
feat["lag_24h"]  = feat["total_load"].shift(24)
feat["lag_168h"] = feat["total_load"].shift(168)
n_before = len(feat)
feat.dropna(inplace=True)
log.info(f"P5 — Dropped {n_before - len(feat)} NaN rows")
assert feat.isna().sum().sum() == 0
feature_cols = [c for c in feat.columns if c != "total_load"]
assert len(feature_cols) == 11, f"P5 FAILED — got {len(feature_cols)}"
log.info(f"P5 OK — {feature_cols}")
feat.to_csv(PROC_DIR / "features.csv")

log.info("P6 — Splitting ...")
train = feat[TRAIN_START:TRAIN_END]
val   = feat[VAL_START:VAL_END]
test  = feat[TEST_START:TEST_END]
assert len(set(train.index) & set(test.index)) == 0
assert len(test) == 8760, f"P6 FAILED — test={len(test)} need 8760"
log.info(f"P6 OK — train={len(train):,} val={len(val):,} test={len(test):,}")

log.info("P7 — Scaling ...")
all_cols = feature_cols + ["total_load"]
scaler = MinMaxScaler()
scaler.fit(train[all_cols])
train_s = train.copy(); train_s[all_cols] = scaler.transform(train[all_cols])
val_s   = val.copy();   val_s[all_cols]   = scaler.transform(val[all_cols])
test_s  = test.copy();  test_s[all_cols]  = scaler.transform(test[all_cols])
assert train_s[all_cols].min().min() >= 0.0
assert train_s[all_cols].max().max() <= 1.0
with open(CFG_DIR / "scaler_uci.pkl", "wb") as f:
    pickle.dump(scaler, f)
log.info("P7 OK — scaler saved")

log.info("P8 — Extreme flag ...")
test_load_orig = test["total_load"]
threshold_90 = np.percentile(test_load_orig, 90)
threshold_95 = np.percentile(test_load_orig, 95)
extreme_mask  = (test_load_orig >= threshold_90).astype(int)
extreme_count = extreme_mask.sum()
pd.DataFrame({
    "timestamp":     test_load_orig.index,
    "total_load":    test_load_orig.values,
    "is_extreme_90": extreme_mask.values,
    "threshold_90":  threshold_90,
    "threshold_95":  threshold_95,
}).to_csv(SPLIT_DIR / "extreme_uci.csv", index=False)
log.info(f"P8 OK — threshold={threshold_90:.2f} MWh  extreme={extreme_count}")

train_s.to_csv(SPLIT_DIR / "train.csv")
val_s.to_csv(SPLIT_DIR   / "val.csv")
test_s.to_csv(SPLIT_DIR  / "test.csv")
train.to_csv(SPLIT_DIR   / "train_unscaled.csv")
val.to_csv(SPLIT_DIR     / "val_unscaled.csv")
test.to_csv(SPLIT_DIR    / "test_unscaled.csv")

print("\n" + "="*50)
print("UCI PREPROCESSING COMPLETE — P1-P8")
print("="*50)
print(f"  Train={len(train):,}  Val={len(val):,}  Test={len(test):,}")
print(f"  Extreme={extreme_count} ({100*extreme_count/len(test):.1f}%)  90th={threshold_90:.2f} MWh")
print(f"  Gaps={len(gap_records)}")
print("="*50)
