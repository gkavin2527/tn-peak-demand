"""Configuration for the Tamil Nadu daily electricity demand forecasting dataset."""

from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

# Per-city-year weather chunks, written the moment each API call succeeds so a
# 60-request backfill can resume instead of starting over. Disposable: the
# chunks are re-derivable from the archive API, so this dir is gitignored.
CACHE = DATA / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

RAW_POSOCO = DATA / "POSOCO_data.csv"      # cached full download
RAW_DEMAND = DATA / "demand_daily.csv"     # TN + Southern Region slice
RAW_WEATHER = DATA / "weather_daily.csv"
RAW_PEAK = DATA / "peak_demand.csv"        # Tamil Nadu peak demand from Grid-India
FORECAST_LOG = DATA / "weather_forecast_log.csv"      # forecasts we logged live
FORECAST_ARCHIVE = DATA / "weather_forecast_archive.csv"  # Open-Meteo past runs
MASTER = DATA / "master.csv"
TN_HOLIDAYS = DATA / "tn_holidays.csv"   # resolved calendar, committed
PREDICTIONS_LOG = DATA / "predictions_log.csv"   # every forecast we issue

# Generated artefacts both tracks read. Committed, so the paper's tables are
# produced from files rather than retyped from a terminal.
RESULTS = Path(__file__).parent / "results"
MODELS = Path(__file__).parent / "models"
FEATURE_GROUPS = Path(__file__).parent / "feature_groups.json"

PEAK_TARGET = "tn_peak_demand_mw"          # primary target: TN peak demand in MW

# ---------------------------------------------------------------------------
# Modelling contract (roadmap Section C.2). Both tracks read these; change
# only by agreement, because a silent edit here invalidates both sides' numbers.
# ---------------------------------------------------------------------------

# Demand-data latency, in days. MEASURED, not assumed: Grid-India publishes
# day D's PSP report on D+1 (1,052 of 1,052 observations, 2015-2026), landing
# ~07:30-08:30 IST. A forecast issued the evening of day D therefore holds
# demand only through D-1, and the target is D+1 - a two-day gap.
#
# The practical consequence is easy to miss: peak_lag_1 is NOT available for a
# genuine day-ahead forecast. src/contract.py shifts every demand/energy/grid
# feature by (H - 1) so that nothing newer than target_day - H is used.
# Re-measure with: python measure_publish_time.py
H = 2
H_IDEALISED = 1          # reported alongside H as the "if data were instant" row

# Per-source horizons. Not every input arrives on the same schedule, and
# pretending otherwise leaves features NaN at inference that were present in
# training - a silent train/serve inconsistency the model cannot warn you about.
#
#   demand  Grid-India PSP report, published D+1              -> H
#   energy  same report (build_dataset backfills tn_energy_gwh
#           from tn_energy_met_mu, and the two agree to 0.02 GWh) -> H
#   grid    sr_* come from the POSOCO/CICERO slice, which runs a
#           further day behind the PSP report (demand_daily.csv
#           had data through 11-08 when peak_demand.csv had 11-09;
#           60 of the last 61 days vs 61)                      -> H + 1
#
# Found by smoke-testing predict_next_day.py: all four sr_*_lag1 columns came
# back NaN on the row being predicted. Re-check with:
#   python -c "import pandas as pd; #     print(pd.read_csv('data/peak_demand.csv').date.max(), #           pd.read_csv('data/demand_daily.csv').date.max())"
H_BY_GROUP = {"demand": H, "energy": H, "grid": H + 1}

# Walk-forward design (D5). Optuna tunes on the three expanding folds; the
# test period is scored once, at the end, after hyperparameters are frozen.
TUNING_FOLDS = [("2021-12-31", "2022-12-31"),
                ("2022-12-31", "2023-12-31"),
                ("2023-12-31", "2024-12-31")]
TEST_START = "2025-01-01"
REFIT_DAYS = 7           # weekly refit in evaluation; monthly allowed in Optuna

# Peak-surge subset definitions (D4). Primary is a per-year quantile because a
# fixed MW threshold drifts with demand growth.
SURGE_QUANTILE = 0.05
SURGE_LEGACY_MW = 19000
SUMMER_MONTHS = (3, 4, 5, 6)      # March-June, the heat-driven season

RANDOM_SEED = 42

# Grid India (POSOCO) daily reports, parsed and republished as a single CSV by
# Robbie Andrew (CICERO). Updated daily. Cite both him and Grid India.
POSOCO_URL = "https://robbieandrew.github.io/india/data/POSOCO_data.csv"

# Columns we keep. States expose only EnergyMet; peak demand in MW exists at
# regional level only, so Southern Region is our proxy for peak behaviour.
# Not the target - the target is PEAK_TARGET, from the Grid-India PSP reports.
# This is the column health_check.py watches to tell whether the upstream
# download actually carried new rows.
TARGET_COL = "Tamil Nadu: EnergyMet"

KEEP = {
    "Tamil Nadu: EnergyMet":  "tn_energy_gwh",       # same-day; lagged into master
    "Kerala: EnergyMet":      "kerala_energy_gwh",
    "Karnataka: EnergyMet":   "karnataka_energy_gwh",
    "Andhra Pradesh: EnergyMet": "ap_energy_gwh",
    "SR: MaximumDemand":      "sr_max_demand_mw",
    "SR: DemandMet":          "sr_demand_met_mw",
    "SR: PeakShortage":       "sr_peak_shortage_mw",
    "SR: EnergyMet":          "sr_energy_gwh",
    "SR: WindGen":            "sr_wind_gwh",
    "SR: SolarGen":           "sr_solar_gwh",
    "India: EnergyMet":       "india_energy_gwh",
}

START_DATE = "2013-01-01"   # weather backfill start; demand goes back to 2013
TZ = "Asia/Kolkata"

CITIES = [
    {"name": "chennai",    "lat": 13.0827, "lon": 80.2707, "weight": 0.36},
    {"name": "coimbatore", "lat": 11.0168, "lon": 76.9558, "weight": 0.22},
    {"name": "madurai",    "lat":  9.9252, "lon": 78.1198, "weight": 0.16},
    {"name": "trichy",     "lat": 10.7905, "lon": 78.7047, "weight": 0.14},
    {"name": "salem",      "lat": 11.6643, "lon": 78.1460, "weight": 0.12},
]

CDD_BASE = 25.0
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "apparent_temperature"]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Archived past model runs, resolved by lead time: requesting
# temperature_2m_previous_day2 for date D returns what the run two days before
# D predicted for D. This is the real historical forecast our live log cannot
# reach back to - the log starts 2026-08-04, this reaches mid-2021. It is what
# makes an operational backtest possible instead of a reanalysis fantasy.
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_START = "2021-06-01"   # earlier dates return all-null hours
MAX_LEAD = 3                         # previous_day1..3 are available

# Horizons to log each morning: today + the next 3 days. Next-day forecasting
# only needs h=1, but logging to h=3 costs the same single request per city and
# is the only way to ever measure skill decay - the forecast log cannot be
# backfilled from any archive.
FORECAST_DAYS = 4

# Same-day quantities from the Grid-India PSP report, which is published the
# morning AFTER the day it describes. They are not knowable before day T begins,
# so build_dataset.py lags them into master.csv and drops the same-day column.
# tn_energy_gwh correlates 0.971 with the target: left in, it flatters a
# backtest by ~24% and silently breaks every forecast made with it.
# They remain available raw in demand_daily.csv and peak_demand.csv.
LEAKY_SAMEDAY = ("tn_energy_gwh", "tn_shortage_at_peak_mw",
                 "sr_max_demand_mw", "sr_demand_met_mw", "sr_peak_shortage_mw",
                 "sr_energy_gwh", "sr_wind_gwh", "sr_solar_gwh",
                 "india_energy_gwh", "kerala_energy_gwh",
                 "karnataka_energy_gwh", "ap_energy_gwh")

# Weather LEVELS are leak-free (they come from a forecast at prediction time)
# but add nothing once the autoregressive lags are present: dropping them
# improves backtest MAE from 591 to 545 MW. Exposed as a group so you can
# exclude them in one line - see README, "Choosing features".
WEATHER_LEVEL_COLS = ("t_mean", "t_max", "t_min", "rh_mean", "app_t_mean",
                      "app_t_max", "hours_above_32", "cdd", "cdd_max",
                      "cdd_apparent", "thi",
                      "cdd_lag1", "cdd_max_lag1", "thi_lag1")
