"""
Measure what forecast weather costs you, and backtest against it.

    python forecast_skew.py

The problem
-----------
master.csv is built on reanalysis - the weather that actually happened. In
production you have a forecast, and the two differ systematically rather than
just noisily. A model trained and tested on reanalysis reports a number it
cannot reproduce on any real morning.

weather_forecast_log.csv is our own live log, which is ground truth about our
pipeline but starts 2026-08-04 - a single season, far too short to backtest.
weather_forecast_archive.csv fills that in: Open-Meteo's archived past model
runs, resolved by lead time, back to mid-2021. Build it once with

    python fetch_weather.py archive-forecast

What this script does
---------------------
1. Reports forecast bias and MAE per variable, per lead, per season, and
   PER YEAR - because the bias is not stationary. It has drifted from +0.49 C
   in 2021 to -0.55 C in 2026 as Open-Meteo's models changed underneath us. A
   correction fitted on the whole archive would be ~1 C wrong today, which is
   why drift_table() reports a trailing-window estimate as the one to use.
2. Cross-checks the archive against our own live log where they overlap - if
   Open-Meteo's archived runs disagree with what we logged live, the archive is
   not a valid stand-in and everything below is void.
3. Runs the operational backtest: train on reanalysis, score with forecast
   weather substituted in, exactly as a real morning would. The gap between
   that and baseline.py is the honest cost of not knowing tomorrow's weather.

Substitution is lead-aware. Predicting day T at lead L, you hold the forecast
for T issued L days earlier, but yesterday's weather is already observed - so
d_t_max = forecast(T) - actual(T-1), not forecast minus forecast. Getting this
wrong would overstate the degradation.
"""

import threadlimit  # noqa: F401  - must precede numpy/sklearn

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from baseline import ANCHOR, DROP, FOLDS, SEEDS
from build_dataset import state_weather
from config import (FORECAST_ARCHIVE, FORECAST_LOG, MASTER, PEAK_TARGET,
                    RAW_WEATHER, WEATHER_LEVEL_COLS)
from src.contract import swap_weather

TARGET = PEAK_TARGET
LEVELS = ("t_mean", "t_max", "t_min", "rh_mean", "app_t_mean", "app_t_max",
          "hours_above_32", "cdd", "cdd_max", "cdd_apparent", "thi")
DELTA_OF = ("t_mean", "t_max", "rh_mean", "app_t_max", "cdd", "cdd_max", "thi")
ANOM_OF = ("t_max", "cdd_max", "thi")
SEASON = {1: "NE monsoon", 2: "hot", 3: "hot", 4: "hot", 5: "hot", 6: "SW monsoon",
          7: "SW monsoon", 8: "SW monsoon", 9: "SW monsoon", 10: "NE monsoon",
          11: "NE monsoon", 12: "NE monsoon"}


def load_state_forecast():
    """State-weighted forecast weather per lead: {lead: DataFrame}."""
    if not FORECAST_ARCHIVE.exists():
        sys.exit("no weather_forecast_archive.csv - run "
                 "'python fetch_weather.py archive-forecast' first")
    raw = pd.read_csv(FORECAST_ARCHIVE)
    out = {}
    for lead, g in raw.groupby("lead"):
        w = state_weather(g.drop(columns=["lead"]))
        out[int(lead)] = w.set_index("date").sort_index()
    return out


def drift_table(fc, actual, trailing_days=90):
    """Bias per year, and the trailing-window estimate to actually correct by.

    Open-Meteo's forecast models have been upgraded repeatedly over the
    archive, and the reanalysis it is scored against moved from ERA5 to ERA5T
    for recent dates. The result is a bias that walks: positive early, negative
    now. Anyone bias-correcting from a constant should read this table first.
    """
    lead = min(fc)
    j = fc[lead].join(actual, how="inner", suffixes=("_fc", "_ar"))
    j.index = pd.to_datetime(j.index)

    print(f"\n=== bias DRIFT, lead {lead} - the bias is not stationary ===")
    print("  year   n      t_mean   t_max  rh_mean")
    for y, g in j.groupby(j.index.year):
        cells = [f"{(g[f'{v}_fc'] - g[f'{v}_ar']).mean():+7.2f}"
                 for v in ("t_mean", "t_max", "rh_mean")]
        print(f"  {y}  {len(g):<6,}" + "".join(cells))

    cut = j.index.max() - pd.Timedelta(days=trailing_days)
    recent = j[j.index > cut]
    print(f"\n  trailing {trailing_days}d (since {cut:%Y-%m-%d}, n={len(recent)}) "
          f"- USE THIS, not the full-archive mean:")
    for v in ("t_mean", "t_max", "rh_mean", "cdd_max"):
        d = recent[f"{v}_fc"] - recent[f"{v}_ar"]
        full = (j[f"{v}_fc"] - j[f"{v}_ar"]).mean()
        print(f"    {v:<9} bias {d.mean():+6.2f}  MAE {d.abs().mean():5.2f}"
              f"   (full archive {full:+.2f})")


def bias_table(fc, actual):
    print("=== forecast vs reanalysis, state-weighted ===")
    print("  lead  n      " + "".join(f"{v:>22}" for v in
                                      ("t_mean", "t_max", "rh_mean", "cdd_max")))
    for lead in sorted(fc):
        j = fc[lead].join(actual, how="inner", suffixes=("_fc", "_ar"))
        cells = []
        for v in ("t_mean", "t_max", "rh_mean", "cdd_max"):
            d = j[f"{v}_fc"] - j[f"{v}_ar"]
            cells.append(f"{d.mean():+7.2f} / {d.abs().mean():5.2f}  ")
        print(f"  h={lead}  {len(j):<6,}" + "".join(f"{c:>22}" for c in cells))
    print("  (bias / MAE)")

    print("\n=== t_max bias by season (lead 1) - is the cold bias seasonal? ===")
    j = fc[min(fc)].join(actual, how="inner", suffixes=("_fc", "_ar"))
    j["season"] = pd.to_datetime(j.index).month.map(SEASON)
    for s, g in j.groupby("season"):
        d = g["t_max_fc"] - g["t_max_ar"]
        dc = g["cdd_max_fc"] - g["cdd_max_ar"]
        print(f"  {s:<12} n={len(g):<5,} t_max bias {d.mean():+5.2f} "
              f"MAE {d.abs().mean():4.2f}   cdd_max bias {dc.mean():+5.2f}")


def cross_check(fc):
    """Does Open-Meteo's archive match what we logged live? If not, stop."""
    if not FORECAST_LOG.exists():
        print("\n(no live log to cross-check against)")
        return
    log = pd.read_csv(FORECAST_LOG, parse_dates=["date", "issued_on"])
    log["lead"] = (log["date"] - log["issued_on"]).dt.days
    log = log[log["lead"] >= 1]
    if log.empty:
        return
    print("\n=== cross-check: our live log vs the archived runs ===")
    for lead, g in log.groupby("lead"):
        if lead not in fc:
            continue
        ours = state_weather(g.drop(columns=["lead", "issued_on"])).set_index("date")
        j = ours.join(fc[lead], how="inner", suffixes=("_log", "_arc"))
        if j.empty:
            continue
        d = j["t_max_log"] - j["t_max_arc"]
        verdict = "consistent" if d.abs().mean() < 1.0 else "DIVERGENT"
        print(f"  h={lead}  n={len(j):<4,} t_max  bias {d.mean():+5.2f}  "
              f"MAE {d.abs().mean():4.2f}   {verdict}")


# substitute() used to live here. It is now src.contract.swap_weather - the
# shared contract function - because two private copies of the same formula is
# exactly how Regime A and Regime B drift apart. The G0 identity test
# (test_contract.py) pins swap_weather against build_dataset.py, and finding
# that pin failing is what surfaced the gap-propagation bug in the weather
# anomaly columns.


def backtest(master, fc, actual):
    cols = [c for c in master.columns
            if c not in {"date", TARGET, "holiday_name", *WEATHER_LEVEL_COLS, *DROP}]
    print("\n=== operational backtest: train on reanalysis, score on forecast ===")
    print("  weather at scoring time        " +
          " | ".join(f"{e[:4]:>5}" for _, e in FOLDS) + "       mean")

    variants = [("reanalysis (baseline.py)", None)]
    variants += [(f"forecast, lead {l}", l) for l in sorted(fc)]

    for label, lead in variants:
        if lead is None:
            scored = master
        else:
            swapped = swap_weather(master.set_index("date"), fc[lead],
                                   observed=actual)
            scored = swapped.reset_index()
        maes, bias = [], None
        for cutoff, end in FOLDS:
            tr = master[master["date"] <= cutoff]
            te = scored[(scored["date"] > cutoff) & (scored["date"] <= end)]
            if te.empty:
                maes.append(np.nan)
                continue
            y = tr[TARGET] - tr[ANCHOR]
            ok = y.notna()
            p = np.mean([HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.06, random_state=s)
                .fit(tr.loc[ok, cols], y[ok]).predict(te[cols])
                for s in range(SEEDS)], axis=0) + te[ANCHOR].values
            yt = te[TARGET].values
            g = np.isfinite(p) & np.isfinite(yt)
            e = p[g] - yt[g]
            maes.append(np.abs(e).mean())
            bias = e.mean()
        shown = [f"{v:5.0f}" if np.isfinite(v) else "    -" for v in maes]
        mean = np.nanmean(maes)
        print(f"  {label:<30} " + " | ".join(shown) +
              f"   MAE {mean:6.1f}   bias {bias:+7.1f}")


def main():
    fc = load_state_forecast()
    actual = state_weather(pd.read_csv(RAW_WEATHER)).set_index("date").sort_index()
    master = pd.read_csv(MASTER, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    span = pd.to_datetime(sorted(fc[min(fc)].index))
    print(f"forecast archive: {span.min():%Y-%m-%d} -> {span.max():%Y-%m-%d}, "
          f"leads {sorted(fc)}\n")

    bias_table(fc, actual)
    drift_table(fc, actual)
    cross_check(fc)
    backtest(master, fc, actual)


if __name__ == "__main__":
    main()
