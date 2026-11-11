"""
Build the modelling table.

    python build_dataset.py

Target: tn_peak_demand_mw - Tamil Nadu daily peak electricity demand, in MW.

Every column written to master.csv is computable before the target day begins.
Same-day quantities from the Grid-India PSP report (energy met, shortage at
peak, regional totals) are published the morning after the day they describe,
so they enter only as _lag1 and the same-day column is dropped - see
config.LEAKY_SAMEDAY. Weather is the one same-day input that stays, because at
prediction time you substitute a forecast for it.
"""

import numpy as np
import pandas as pd

from config import (CDD_BASE, CITIES, LEAKY_SAMEDAY, MASTER, PEAK_TARGET,
                    RAW_DEMAND, RAW_PEAK, RAW_WEATHER)

WEIGHTS = {c["name"]: c["weight"] for c in CITIES}
TARGET = PEAK_TARGET
ENERGY_COL = "tn_energy_gwh"


def state_weather(long: pd.DataFrame) -> pd.DataFrame:
    df = long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["w"] = df["city"].map(WEIGHTS)
    if df["w"].isna().any():
        raise ValueError(f"unweighted cities: {df[df.w.isna()].city.unique().tolist()}")

    vals = [c for c in df.columns
            if c not in ("date", "city", "w", "issued_on", "source", "lead")]

    def wavg(g):
        # renormalise per column, not once for the group: if a city reports
        # t_max but not rh_mean, dividing by the full weight sum would drag
        # that day's humidity toward zero instead of ignoring the gap
        out = {}
        for c in vals:
            v = g[c]
            w = g["w"].where(v.notna())
            total = w.sum()
            out[c] = float((v * w).sum() / total) if total > 0 else np.nan
        return pd.Series(out)

    out = df.groupby("date")[["w"] + vals].apply(wavg).reset_index()

    out["cdd"] = (out["t_mean"] - CDD_BASE).clip(lower=0)
    out["cdd_max"] = (out["t_max"] - CDD_BASE).clip(lower=0)
    out["cdd_apparent"] = (out["app_t_mean"] - CDD_BASE).clip(lower=0)
    out["thi"] = out["t_mean"] - (0.55 - 0.0055 * out["rh_mean"]) * (out["t_mean"] - 14.5)
    return weather_derivatives(out)


def weather_derivatives(wx: pd.DataFrame) -> pd.DataFrame:
    """Lagged, day-over-day and anomaly weather columns.

    Computed HERE, on the weather series, rather than later on the merged
    modelling frame. That distinction is not cosmetic. The merged frame is
    reindexed over master.csv's span and therefore carries NaN on the 157 days
    with no demand report - but the weather on those days is perfectly well
    known. Deriving these columns there punched demand's gaps into weather:
    `d_t_max` went NaN the day after every gap, and the trailing mean behind
    `t_max_vs_roll7` quietly averaged fewer than 7 days (min_periods=3 hid it),
    discarding real observations for no reason.

    It also made build_dataset.py and src.contract.swap_weather disagree, which
    is how the G0 identity test found this: the two computed the same column by
    two different routes. Keeping one definition, on the gap-free series, is
    what makes the Regime A / Regime B comparison trustworthy.
    """
    wx = wx.sort_values("date").reset_index(drop=True)
    span = pd.date_range(wx["date"].min(), wx["date"].max(), freq="D")
    full = wx.set_index("date").reindex(span)

    for col in ("cdd", "cdd_max", "thi"):
        full[f"{col}_lag1"] = full[col].shift(1)

    for col in ("t_mean", "t_max", "rh_mean", "app_t_max", "cdd", "cdd_max", "thi"):
        full[f"d_{col}"] = full[col] - full[col].shift(1)

    for col in ("t_max", "cdd_max", "thi"):
        full[f"{col}_vs_roll7"] = (
            full[col] - full[col].shift(1).rolling(7, min_periods=3).mean())

    return full.rename_axis("date").reset_index()


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    # tn_holidays, not holidays.India directly: the package resolves Maha
    # Shivaratri in only 6 of 15 years, so is_holiday would otherwise mean
    # "a holiday the package knew about" and vary by year. See tn_holidays.py.
    from tn_holidays import calendar as tn_calendar
    tn = tn_calendar(sorted({d.year for d in dates}))
    hol = set(tn.keys())

    cal = pd.DataFrame({"date": dates})
    d = cal["date"]
    cal["dow"] = d.dt.day_of_week
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    # Saturday still runs industry (-364 MW vs a working day); Sunday does not
    # (-2,207 MW). is_weekend averages the two into a day type that exists on
    # neither, so give the model the cycle as well as the flag.
    cal["dow_sin"] = np.sin(2 * np.pi * cal["dow"] / 7)
    cal["dow_cos"] = np.cos(2 * np.pi * cal["dow"] / 7)
    cal["is_holiday"] = d.dt.date.map(lambda x: int(x in hol))
    cal["day_before_holiday"] = (d + pd.Timedelta(days=1)).dt.date.map(lambda x: int(x in hol))
    cal["day_after_holiday"] = (d - pd.Timedelta(days=1)).dt.date.map(lambda x: int(x in hol))
    cal["holiday_name"] = d.dt.date.map(lambda x: tn.get(x, ""))
    cal["pongal_window"] = ((d.dt.month == 1) & d.dt.day.between(13, 17)).astype(int)

    doy = d.dt.day_of_year
    for k in (1, 2):
        cal[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cal[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

    cal["trend"] = (d - d.min()).dt.days
    return cal


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Attach backward-looking features on a calendar-continuous index.

    Shifting positionally would be wrong: missing days in the series would cause
    positional lag_1 to quietly shift 2+ days ago. Reindexing onto a complete
    date range first guarantees every lag represents the exact calendar day it claims.

    Nothing is forward-filled. An earlier version carried the last known value
    across gaps with ffill(limit=2), which left ~60 rows per lag column holding
    a value from two or three days back while still being named lag_1. A NaN
    says "this day is missing" and every gradient-boosting implementation
    handles it; a quietly wrong number does not.
    """
    span = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    full = df.set_index("date").reindex(span)

    # 1. Target Autoregressive features (Peak Demand in MW)
    y = full[TARGET]
    for lag in (1, 2, 3, 7, 14, 364):
        full[f"peak_lag_{lag}"] = y.shift(lag)
    for win, min_p in ((7, 3), (30, 14)):
        full[f"peak_roll_mean_{win}"] = y.shift(1).rolling(win, min_periods=min_p).mean()
        # left NaN, not 0.0: a zero here would claim the series was perfectly
        # flat over the window when what we mean is that we cannot say
        full[f"peak_roll_std_{win}"] = y.shift(1).rolling(win, min_periods=min_p).std()

    full["peak_lag_1_diff"] = full["peak_lag_1"] - full["peak_lag_2"]
    full["peak_lag_1_vs_week"] = full["peak_lag_1"] - full["peak_lag_7"]

    # 2. Daily Energy Consumption features (tn_energy_gwh)
    if ENERGY_COL in full.columns:
        e = full[ENERGY_COL]
        for lag in (1, 2, 7, 14):
            full[f"energy_lag_{lag}"] = e.shift(lag)
        for win, min_p in ((7, 3), (30, 14)):
            full[f"energy_roll_mean_{win}"] = e.shift(1).rolling(win, min_periods=min_p).mean()
            full[f"energy_roll_std_{win}"] = e.shift(1).rolling(win, min_periods=min_p).std()

    # 3. Weather lags, deltas and anomalies are NOT built here - see
    #    weather_derivatives(), which builds them on the gap-free weather
    #    series before the merge. Building them on `full` would propagate
    #    demand's 157 missing days into weather columns that are perfectly
    #    well observed on those days.
    #
    #    Why the weather block is shaped the way it is: levels alone are
    #    near-redundant, because peak_roll_mean_7 already encodes how hot this
    #    part of the season is (backtest MAE is marginally BETTER with every
    #    weather level deleted). What a next-day model is missing is the
    #    *change* - corr(d_peak, d_t_mean) is +0.15 against ~0 for the level.
    #    Both terms stay leak-free: today's weather is a forecast at prediction
    #    time, yesterday's is observed.

    # 4. Regional context lags
    for col in ("sr_max_demand_mw", "sr_wind_gwh", "sr_solar_gwh", "sr_peak_shortage_mw"):
        if col in full.columns:
            full[f"{col}_lag1"] = full[col].shift(1)

    # Re-filter back to the observed rows
    return (full.loc[pd.DatetimeIndex(df["date"])]
                .rename_axis("date").reset_index())


def merged_frame(extend_to=None) -> pd.DataFrame:
    """Peak + energy + weather + calendar, merged but NOT yet lagged.

    `extend_to` appends calendar-continuous rows out to that date, with the
    target and demand columns left NaN. Inference needs this: the target day
    has no demand row yet, but it still needs a feature row, and the lag block
    for it has to be built by the same add_lags() the training data went
    through. An earlier version of predict_next_day.py hand-rolled those
    shifts and got the energy and grid lags wrong - there was no row at
    target-1 to shift from, so they silently came out NaN. Sharing this
    function is what keeps training and inference on one definition.
    """
    # 1. Load peak demand
    peak = pd.read_csv(RAW_PEAK, parse_dates=["date"])
    peak_cols = [c for c in ["date", TARGET, "tn_shortage_at_peak_mw", "tn_energy_met_mu"] if c in peak.columns]
    peak = (peak[peak_cols]
            .dropna(subset=["date", TARGET])
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date"))

    # 2. Load daily energy demand & weather
    demand = pd.read_csv(RAW_DEMAND, parse_dates=["date"])
    # Historically backfill regional metrics before tracking began:
    # Utility solar in Southern Region was negligible/zero prior to Sept 2016
    if "sr_solar_gwh" in demand.columns:
        demand["sr_solar_gwh"] = demand["sr_solar_gwh"].fillna(0.0)
    # Peak MW was recorded as Demand Met prior to April 2017
    if "sr_max_demand_mw" in demand.columns and "sr_demand_met_mw" in demand.columns:
        demand["sr_max_demand_mw"] = demand["sr_max_demand_mw"].fillna(demand["sr_demand_met_mw"])

    weather = state_weather(pd.read_csv(RAW_WEATHER))

    # 3. Merge peak demand, energy demand, weather, and calendar features
    df = (peak.merge(demand, on="date", how="left")
              .merge(weather, on="date", how="inner")
              .sort_values("date"))

    # If energy demand slice is lagging upstream, backfill from Grid-India peak report
    if "tn_energy_met_mu" in df.columns:
        if ENERGY_COL in df.columns:
            df[ENERGY_COL] = df[ENERGY_COL].fillna(df["tn_energy_met_mu"])
        else:
            df[ENERGY_COL] = df["tn_energy_met_mu"]
        df = df.drop(columns=["tn_energy_met_mu"])

    # A row exists because the TARGET exists. An earlier version dropped rows
    # whose energy was missing, which discarded 6 days where peak demand had
    # been published perfectly well, and made the row count depend on a column
    # that no longer even ships. Missing energy now just means NaN energy lags.
    # Extend to a future target before the calendar join, so the appended
    # rows get real calendar features rather than NaN.
    if extend_to is not None:
        extend_to = pd.Timestamp(extend_to)
        if extend_to > df["date"].max():
            span = pd.date_range(df["date"].min(), extend_to, freq="D")
            df = (df.set_index("date").reindex(span)
                    .rename_axis("date").reset_index())
            # weather is known (forecast) for the future row; re-join it so the
            # appended dates carry levels instead of NaN
            wx_cols = [c for c in weather.columns if c != "date"]
            df = df.drop(columns=wx_cols).merge(weather, on="date", how="left")

    df = df.merge(calendar_features(df["date"]), on="date", how="left")

    # Clean non-holiday string labels.
    #
    # "No Holiday", not "None": pandas' default read_csv na_values list
    # includes the literal string "None" (and ""), so the previous sentinel
    # came back as NaN on a plain pd.read_csv - 4,577 rows that were never
    # actually missing were counted as missing values on every read.
    #
    # tn_shortage_at_peak_mw is deliberately NOT filled with 0.0 here. The PSP
    # reports only carry it from 2024 onward; filling the earlier 79% with
    # zeros asserted that Tamil Nadu ran no deficit at peak through the
    # 2013-2015 load-shedding years, which is false rather than merely missing.
    df["holiday_name"] = df["holiday_name"].fillna("No Holiday").replace("", "No Holiday")

    return df.sort_values("date").reset_index(drop=True)


def build() -> pd.DataFrame:
    df = merged_frame()
    df = add_lags(df).reset_index(drop=True)

    # 4. Drop every same-day column published after the target day.
    #    Their information survives as the _lag1 / _lag7 / rolling features
    #    built above. Keeping tn_energy_gwh here (r=0.971 with the target)
    #    was worth ~140 MW of fictional accuracy.
    leaky = [c for c in LEAKY_SAMEDAY if c in df.columns]
    df = df.drop(columns=leaky)

    # 5. Order columns cleanly: target first, then weather, calendar, and lags
    front = ["date", TARGET]
    other = [c for c in df.columns if c not in front]
    df = df[front + other]

    df.to_csv(MASTER, index=False)
    return df


def report(df: pd.DataFrame):
    print(f"rows          {len(df):,}")
    print(f"date range    {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print(f"features      {len(df.columns) - 2}")

    span = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    gaps = span.difference(pd.DatetimeIndex(df["date"]))
    print(f"missing days  {len(gaps)}")

    y = df[TARGET]
    print(f"target MW     min {y.min():,.0f}  mean {y.mean():,.0f}  max {y.max():,.0f}")

    dropped = [c for c in LEAKY_SAMEDAY if c not in df.columns]
    print(f"same-day cols dropped as unavailable at prediction time: {len(dropped)}")

    keys = [c for c in ("cdd", "cdd_max", "d_t_max", "d_cdd", "t_max_vs_roll7",
                        "peak_lag_1", "peak_lag_7", "peak_roll_mean_7",
                        "energy_lag_1", "trend") if c in df.columns]
    print("\ncorrelation with target:")
    print(df[[TARGET] + keys].corr()[TARGET].drop(TARGET).round(3).to_string())

    print("\nmean target by day type:")
    print(f"  working day  {y[(df.is_weekend == 0) & (df.is_holiday == 0)].mean():,.1f} MW")
    print(f"  weekend      {y[df.is_weekend == 1].mean():,.1f} MW")
    print(f"  holiday      {y[df.is_holiday == 1].mean():,.1f} MW")
    print(f"  Pongal week  {y[df.pongal_window == 1].mean():,.1f} MW")

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("\nnulls (head-only, matching lag windows):")
        print(nulls.to_string())


if __name__ == "__main__":
    report(build())
    print(f"\nwrote {MASTER}")
