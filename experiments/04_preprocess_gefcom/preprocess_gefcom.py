import pickle, logging
import numpy as np, pandas as pd, holidays
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = PROJECT_ROOT / "data/gefcom/raw/GEFCom2014 Data/Load"
PROC_DIR  = PROJECT_ROOT / "data/gefcom/processed"
SPLIT_DIR = PROJECT_ROOT / "data/gefcom/splits"
CFG_DIR   = PROJECT_ROOT / "models/gefcom/configs"
for d in [PROC_DIR, SPLIT_DIR, CFG_DIR]: d.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = "2007-01-08", "2009-09-30"
VAL_START,   VAL_END   = "2009-10-01", "2009-12-31"
TEST_START,  TEST_END  = "2010-01-01", "2010-12-31"
W_COLS = [f"w{i}" for i in range(1, 26)]

def parse_gefcom_timestamp(ts_series):
    dates = []
    for ts in ts_series:
        date_part, time_part = ts.strip().split(" ")
        hour  = int(time_part.split(":")[0])
        year  = int(date_part[-4:])
        md    = date_part[:-4]
        if len(md) >= 3:
            day   = int(md[-2:])
            month = int(md[:-2])
        else:
            day   = int(md[-1:])
            month = int(md[:-1]) if md[:-1] else 1
        dates.append(pd.Timestamp(year=year, month=month, day=day, hour=hour))
    return pd.DatetimeIndex(dates)

log.info("G1 — Loading all task train files ...")
frames = []
for task_num in range(1, 16):
    fpath = RAW_DIR / f"Task {task_num}" / f"L{task_num}-train.csv"
    assert fpath.exists(), f"Missing: {fpath}"
    df = pd.read_csv(fpath)
    df = df[df["LOAD"].notna() & (df["LOAD"] != "")]
    df["LOAD"] = pd.to_numeric(df["LOAD"], errors="coerce")
    df = df[df["LOAD"].notna()]
    df["datetime"] = parse_gefcom_timestamp(df["TIMESTAMP"])
    df = df[["datetime", "LOAD"] + W_COLS].copy()
    frames.append(df)
    log.info(f"  Task {task_num:2d}: {len(df):,} rows")

log.info("G2 — Loading solution file (Dec 2011) ...")
sol_path = RAW_DIR / "Solution to Task 15" / "solution15_L_temperature.csv"
assert sol_path.exists(), f"Missing: {sol_path}"
sol = pd.read_csv(sol_path)
sol["datetime"] = pd.to_datetime(sol["date"]) + pd.to_timedelta(sol["hour"].astype(int), unit="h")
sol = sol[["datetime", "LOAD"] + W_COLS].copy()
frames.append(sol)
log.info(f"  Solution: {len(sol):,} rows")

log.info("G3 — Concatenating ...")
combined = pd.concat(frames, ignore_index=True)
combined = combined.sort_values("datetime").drop_duplicates(subset="datetime")
combined = combined.set_index("datetime").sort_index()
log.info(f"G3 OK — {len(combined):,} rows  {combined.index[0]} -> {combined.index[-1]}")

log.info("G4 — Averaging weather stations ...")
combined[W_COLS] = combined[W_COLS].apply(pd.to_numeric, errors="coerce")
combined["temperature_F"] = combined[W_COLS].mean(axis=1)
combined = combined[["LOAD", "temperature_F"]].rename(columns={"LOAD": "total_load"})

log.info("G5 — Checking gaps ...")
full_idx = pd.date_range(start=combined.index[0], end=combined.index[-1], freq="H")
combined = combined.reindex(full_idx)
nan_rows = combined["total_load"].isna().sum()
gap_records = []
if nan_rows > 0:
    nan_mask = combined["total_load"].isna()
    gap_groups = (nan_mask != nan_mask.shift()).cumsum()[nan_mask]
    for gid, group in gap_groups.groupby(gap_groups):
        gl = len(group)
        gap_records.append({"gap_start": group.index[0], "gap_end": group.index[-1], "length_hrs": gl, "filled": gl <= 3})
    log.info(f"G5 — {len(gap_records)} gap(s)")
else:
    log.info("G5 — No gaps")
pd.DataFrame(gap_records).to_csv(PROC_DIR / "gap_log.csv", index=False)
combined["total_load"]    = combined["total_load"].interpolate(method="time", limit=240, limit_direction="forward")
combined["temperature_F"] = combined["temperature_F"].interpolate(method="time", limit=240, limit_direction="forward")
combined.to_csv(PROC_DIR / "hourly_load_weather.csv")
log.info("G5 OK")

log.info("G6 — Building features ...")
feat = pd.DataFrame(index=combined.index)
feat["total_load"]    = combined["total_load"]
feat["temperature_F"] = combined["temperature_F"]
hour, dow, month = feat.index.hour, feat.index.dayofweek, feat.index.month
feat["hour_sin"]  = np.sin(2*np.pi*hour/24);  feat["hour_cos"]  = np.cos(2*np.pi*hour/24)
feat["day_sin"]   = np.sin(2*np.pi*dow/7);    feat["day_cos"]   = np.cos(2*np.pi*dow/7)
feat["month_sin"] = np.sin(2*np.pi*month/12); feat["month_cos"] = np.cos(2*np.pi*month/12)
feat["is_weekend"]    = (dow >= 5).astype(int)
us_hols = holidays.US(state="MA", years=range(2005, 2012))
feat["is_holiday_MA"] = feat.index.normalize().isin(us_hols).astype(int)
feat["lag_1h"]              = feat["total_load"].shift(1)
feat["lag_24h"]             = feat["total_load"].shift(24)
feat["lag_168h"]            = feat["total_load"].shift(168)
feat["temperature_lag_24h"] = feat["temperature_F"].shift(24)
n_before = len(feat)
feat.dropna(inplace=True)
log.info(f"G6 — Dropped {n_before - len(feat)} NaN rows")
assert feat.isna().sum().sum() == 0
feature_cols = [c for c in feat.columns if c != "total_load"]
assert len(feature_cols) == 13, f"G6 FAILED — got {len(feature_cols)}: {feature_cols}"
log.info(f"G6 OK — {feature_cols}")
feat.to_csv(PROC_DIR / "features.csv")

log.info("G7 — Splitting ...")
train = feat[TRAIN_START:TRAIN_END]
val   = feat[VAL_START:VAL_END]
test  = feat[TEST_START:TEST_END]
assert len(set(train.index) & set(test.index)) == 0
assert len(test) == 8760, f"G7 FAILED — test={len(test)} need 8760"
log.info(f"G7 OK — train={len(train):,} val={len(val):,} test={len(test):,}")

log.info("G8 — Scaling (INDEPENDENT from UCI) ...")
all_cols = feature_cols + ["total_load"]
scaler_gf = MinMaxScaler()
scaler_gf.fit(train[all_cols])
train_s = train.copy(); train_s[all_cols] = scaler_gf.transform(train[all_cols])
val_s   = val.copy();   val_s[all_cols]   = scaler_gf.transform(val[all_cols])
test_s  = test.copy();  test_s[all_cols]  = scaler_gf.transform(test[all_cols])
assert train_s[all_cols].min().min() >= -1e-6, "G8 FAILED — scaled train below 0"
assert train_s[all_cols].max().max() <= 1.0 + 1e-6, "G8 FAILED — scaled train above 1"
with open(CFG_DIR / "scaler_gefcom.pkl", "wb") as f:
    pickle.dump(scaler_gf, f)
log.info("G8 OK — scaler_gefcom.pkl saved")

log.info("G9 — Extreme flag ...")
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
}).to_csv(SPLIT_DIR / "extreme_gefcom.csv", index=False)
log.info(f"G9 OK — threshold={threshold_90:.2f} MWh  extreme={extreme_count}")

train_s.to_csv(SPLIT_DIR / "train.csv"); val_s.to_csv(SPLIT_DIR / "val.csv"); test_s.to_csv(SPLIT_DIR / "test.csv")
train.to_csv(SPLIT_DIR / "train_unscaled.csv"); val.to_csv(SPLIT_DIR / "val_unscaled.csv"); test.to_csv(SPLIT_DIR / "test_unscaled.csv")

print("\n" + "="*50)
print("GEFCOM PREPROCESSING COMPLETE — G1-G9")
print("="*50)
print(f"  Full range: {combined.index[0].date()} -> {combined.index[-1].date()}")
print(f"  Train={len(train):,}  Val={len(val):,}  Test={len(test):,}")
print(f"  Extreme={extreme_count} ({100*extreme_count/len(test):.1f}%)  90th={threshold_90:.2f} MWh")
print(f"  Gaps={len(gap_records)}")
print(f"  Scaler: scaler_gefcom.pkl (INDEPENDENT from UCI)")
print("="*50)
