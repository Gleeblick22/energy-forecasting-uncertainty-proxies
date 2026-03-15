"""
Phase 4 - ARIMA Fitting UCI
PDD v6 Section 7B | Implementation Guide v4 Table 14
DATASET = "uci"
"""
import pickle, logging
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET      = "uci"

LOG_PATH = PROJECT_ROOT / "logs" / f"arima_{DATASET}.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()])
log = logging.getLogger(__name__)

DATA_DIR  = PROJECT_ROOT / f"data/{DATASET}/splits"
ARIMA_DIR = PROJECT_ROOT / f"models/{DATASET}/arima"
OUT_DIR   = PROJECT_ROOT / f"results/{DATASET}/tables"
ARIMA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

log.info(f"Loading {DATASET.upper()} unscaled data ...")
train_df   = pd.read_csv(DATA_DIR / "train_unscaled.csv", index_col=0, parse_dates=True)
test_df    = pd.read_csv(DATA_DIR / "test_unscaled.csv",  index_col=0, parse_dates=True)
train_load = train_df["total_load"]
test_load  = test_df["total_load"]
log.info(f"  Train: {len(train_load)} hours (full set — PDD v6 compliant)")
log.info(f"  Test:  {len(test_load)} hours")
log.info(f"  Train range: [{train_load.min():.1f}, {train_load.max():.1f}] MWh")

log.info("Fitting SARIMA(2,1,2)(1,1,1,24) on full training set ...")
model = SARIMAX(train_load, order=(2,1,2), seasonal_order=(1,1,1,24),
    enforce_stationarity=False, enforce_invertibility=False, freq="H")
result = model.fit(disp=False, maxiter=100)
log.info(f"  AIC: {result.aic:.2f}  Converged: {result.mle_retvals['converged']}")

log.info("Running Ljung-Box diagnostic (PDD v6 Section 7B) ...")
resids = result.resid.dropna()
lb     = acorr_ljungbox(resids, lags=[24, 48, 168], return_df=True)
log.info(f"\n{lb.to_string()}")
lb24_pass  = lb["lb_pvalue"].iloc[0] > 0.05
lb48_pass  = lb["lb_pvalue"].iloc[1] > 0.05
lb168_pass = lb["lb_pvalue"].iloc[2] > 0.05
log.info(f"  Lag 24:  {'PASSED' if lb24_pass  else 'FAILED'}  (must pass)")
log.info(f"  Lag 48:  {'PASSED' if lb48_pass  else 'FAILED'}  (must pass)")
log.info(f"  Lag 168: {'PASSED' if lb168_pass else 'FAILED'}  (report only per PDD v6)")
if not lb24_pass:
    raise AssertionError("Ljung-Box lag 24 FAILED — non-compliant per PDD v6")
if not lb48_pass:
    raise AssertionError("Ljung-Box lag 48 FAILED — non-compliant per PDD v6")
log.info("  Lags 24 + 48 PASSED — PDD v6 compliant")

log.info("Generating rolling one-step-ahead forecasts (dynamic=False) ...")
full_series  = pd.concat([train_load, test_load])
res_extended = result.apply(full_series, refit=False)
pred_obj     = res_extended.get_prediction(
    start=len(train_load), end=len(train_load)+len(test_load)-1, dynamic=False)
preds    = pred_obj.predicted_mean.values
conf     = pred_obj.conf_int(alpha=0.05).values
coverage = ((test_load.values >= conf[:,0]) & (test_load.values <= conf[:,1])).mean()
pi_width = (conf[:,1] - conf[:,0]).mean()
mae      = np.abs(test_load.values - preds).mean()
log.info(f"  Coverage: {coverage:.3f}  PI width: {pi_width:.2f} MWh  MAE: {mae:.2f} MWh")
if coverage < 0.90:
    raise AssertionError(f"Coverage {coverage:.3f} below PDD v6 minimum 0.90")
if pi_width > 1000:
    raise AssertionError(f"PI width {pi_width:.1f} MWh too large — check dynamic=False")

arima_df = pd.DataFrame({"arima_pred":preds,"lower_95":conf[:,0],"upper_95":conf[:,1]},
    index=test_df.index)
arima_df.to_csv(OUT_DIR / "arima_predictions.csv")
pickle.dump(result, open(ARIMA_DIR / f"sarima_{DATASET}.pkl", "wb"))

with open(ARIMA_DIR / "arima_diagnostics.txt", "w") as f:
    f.write(f"Dataset:           {DATASET.upper()}\n")
    f.write(f"PDD version:       v6\n")
    f.write(f"Order:             (2,1,2)(1,1,1,24)\n")
    f.write(f"Training hours:    {len(train_load)} (full set)\n")
    f.write(f"Forecast method:   rolling one-step-ahead\n")
    f.write(f"AIC:               {result.aic:.2f}\n")
    f.write(f"LB lag 24:         {'PASSED' if lb24_pass else 'FAILED'}\n")
    f.write(f"LB lag 48:         {'PASSED' if lb48_pass else 'FAILED'}\n")
    f.write(f"LB lag 168:        {'PASSED' if lb168_pass else 'FAILED (acceptable PDD v6)'}\n")
    f.write(f"Coverage:          {coverage:.4f}\n")
    f.write(f"PI width mean:     {pi_width:.2f} MWh\n")
    f.write(f"MAE:               {mae:.2f} MWh\n")
    f.write(f"Compliance:        PDD v6 COMPLIANT\n")

print("\n" + "="*55)
print(f"{DATASET.upper()} ARIMA COMPLETE — PDD v6 COMPLIANT")
print("="*55)
print(f"  Order:            (2,1,2)(1,1,1,24)")
print(f"  Training hours:   {len(train_load)} (full set)")
print(f"  Forecast method:  rolling one-step-ahead")
print(f"  AIC:              {result.aic:.2f}")
print(f"  LB lag 24:        {'PASSED' if lb24_pass else 'FAILED'}")
print(f"  LB lag 48:        {'PASSED' if lb48_pass else 'FAILED'}")
print(f"  LB lag 168:       {'PASSED' if lb168_pass else 'FAILED (acceptable)'}")
print(f"  Coverage 95% CI:  {coverage:.3f}")
print(f"  PI width mean:    {pi_width:.2f} MWh")
print(f"  ARIMA MAE:        {mae:.2f} MWh")
print("="*55)
