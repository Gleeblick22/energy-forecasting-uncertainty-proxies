

# Data Cards
## Project: Conditional Reliability of Uncertainty Proxies Under Extreme Demand
**Last updated:** Jan 2 - 15, 2026


## Dataset 1 — UCI Electricity Load Diagrams 2011-2014

### Source
- Repository: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/dataset/321
- Original collector: ERSE (Portuguese energy regulator)
- License: CC BY 4.0

### Description
Electricity consumption of 370 individual clients in Portugal,
recorded at 15-minute intervals. Aggregated to hourly total load (MWh)
for this study.

### Raw File
- Filename: LD2011_2014.txt
- Size: 679 MB
- Format: semicolon-separated, decimal comma
- Shape: (140256, 370) — 140,256 timestamps x 370 clients
- Location: data/uci/raw/LD2011_2014.txt (READ-ONLY)

### Date Range
- Raw: Jan 1 2011 00:00 to Dec 31 2014 23:45 (15-min intervals)
- After hourly resample: Jan 1 2011 00:00 to Dec 31 2014 23:00
- Total hourly rows: 35,064

### Known Issues
- One extra timestamp (2015-01-01 00:00) present in raw file — DST artifact
  Fix: truncate to 2014-12-31 23:00 after resampling
- No weather data — temperature not available for this dataset

### Preprocessing Applied
- Aggregated 370 client columns to single total_load (sum, MWh)
- Resampled 15-min to hourly (sum / 1000 to convert kWh -> MWh)
- 0 gaps detected in hourly series
- 168 NaN rows dropped (lag warm-up period)

### Features (11)
| Feature | Description |
|---------|-------------|
| total_load | Target variable, MWh per hour |
| hour_sin / hour_cos | Cyclic hour of day encoding |
| day_sin / day_cos | Cyclic day of week encoding |
| month_sin / month_cos | Cyclic month of year encoding |
| is_weekend | Binary flag (1 = Sat/Sun) |
| is_holiday_PT | Binary flag, Portuguese public holidays |
| lag_1h | Load 1 hour prior |
| lag_24h | Load 24 hours prior |
| lag_168h | Load 168 hours prior (1 week) |

### Splits
| Split | Start | End | Rows |
|-------|-------|-----|------|
| Train | 2011-01-08 | 2013-09-30 | 23,928 |
| Val   | 2013-10-01 | 2013-12-31 | 2,208  |
| Test  | 2014-01-01 | 2014-12-31 | 8,760  |

**Split rationale:** Test = full calendar year 2014 (Marino et al. 2016)

### Extreme Demand Definition
- Threshold: 90th percentile of test set total_load
- Value: 1,357.04 MWh
- Extreme hours in test: 876 (10.0%)
- Saved to: data/uci/splits/extreme_uci.csv

### Scaler
- Type: MinMaxScaler (sklearn)
- Fit on: train split only
- Saved to: models/uci/configs/scaler_uci.pkl

### Paper Citation
Trindade, A. (2025). ElectricityLoadDiagrams20112014. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/321



## Dataset 2 — GEFCom2014 Load Track

### Source
- Competition: Global Energy Forecasting Competition 2014
- URL: https://www.kaggle.com/datasets/cthngon/gefcom2014-dataset
- Original paper: Hong et al. (2016), IJF 32(3), 896-913
- License: Public competition dataset

### Description
Hourly electricity load for a single zone (New England ISO aggregate)
with 25 temperature station readings. Released as a rolling-window
competition where each task revealed one additional month of data.

### Raw Files
- Location: data/gefcom/raw/GEFCom2014 Data/Load/
- Structure:
  - Task 1/L1-train.csv: 85,441 rows (base history, Jan 2005 - Jan 10 2010)
  - Task 2-15/Lx-train.csv: 672-744 rows each (monthly increments)
  - Solution to Task 15/solution15_L_temperature.csv: 744 rows (Dec 2011)
- Columns: ZONEID, TIMESTAMP, LOAD, w1...w25

### Timestamp Format
Format: MDDYYYY H:MM (month first, then 2-digit day, then 4-digit year)
Examples:
  1012005 1:00  = Oct 1 2005 01:00
  9302005 23:00 = Sep 30 2005 23:00
  1202005 0:00  = Jan 20 2005 00:00

### Date Range
- Full combined range: Jan 1 2005 to Dec 31 2011
- Total rows after combine: 57,024
- Single zone only (Zone 1, New England aggregate)

### Known Issues
- Structural gaps: Oct/Nov/Dec 1-9 missing every year (21 gaps x 216 hours)
  Cause: single-digit days in months 10/11/12 excluded from original dataset
  Fix: time-based linear interpolation (limit=240 hours)
- No dew point data — only 25 temperature station readings available
- Single zone only (full competition had 20 zones; public track = Zone 1)

### Preprocessing Applied
- Concatenated all 15 task files + solution in chronological order
- Deduplicated on datetime index
- Averaged 25 weather stations -> single temperature_F feature
- Reindexed to full hourly range, applied time interpolation for gaps
- 168 NaN rows dropped (lag warm-up period)

### Features (13)
| Feature | Description |
|---------|-------------|
| total_load | Target variable, MWh per hour |
| temperature_F | Mean of 25 station temperatures (°F) |
| hour_sin / hour_cos | Cyclic hour of day encoding |
| day_sin / day_cos | Cyclic day of week encoding |
| month_sin / month_cos | Cyclic month of year encoding |
| is_weekend | Binary flag (1 = Sat/Sun) |
| is_holiday_MA | Binary flag, Massachusetts public holidays |
| lag_1h | Load 1 hour prior |
| lag_24h | Load 24 hours prior |
| lag_168h | Load 168 hours prior (1 week) |
| temperature_lag_24h | Temperature 24 hours prior |

### Splits
| Split | Start | End | Rows |
|-------|-------|-----|------|
| Train | 2007-01-08 | 2009-09-30 | 23,928 |
| Val   | 2009-10-01 | 2009-12-31 | 2,208  |
| Test  | 2010-01-01 | 2010-12-31 | 8,760  |

Split rationale: Test = full calendar year 2010 (Marino et al. 2016)

### Extreme Demand Definition
- Threshold: 90th percentile of test set total_load
- Value: 237.60 MWh
- Extreme hours in test: 877 (10.0%)
- Saved to: data/gefcom/splits/extreme_gefcom.csv

### Scaler
- Type: MinMaxScaler (sklearn)
- Fit on: train split only — INDEPENDENT from UCI scaler
- Saved to: models/gefcom/configs/scaler_gefcom.pkl

### Paper Citation
Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., & Hyndman, R. J. (2016).
Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond.
International Journal of Forecasting, 32(3), 896-913.
DOI: 10.1016/j.ijforecast.2016.02.001



## Cross-Dataset Comparison Notes

| Property | UCI | GEFCom |
|----------|-----|--------|
| Grid | Portugal (Mediterranean) | New England (Continental) |
| Weather | None | 25 temperature stations |
| Clients | 370 aggregated | Single zone aggregate |
| Seasonality | Mild | Strong (cold winters, hot summers) |
| Test year | 2014 | 2010 |
| Features | 11 | 13 |
| Extreme threshold | 1,357 MWh | 237.60 MWh |

The two grids have different characteristics by design.
Cross-dataset comparison evaluates whether proxy reliability patterns
generalise across different grid types and weather regimes.
