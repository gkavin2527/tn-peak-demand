"""
The integration contract between Track A (modelling) and Track B (operations).

Roadmap C.2. Three functions, frozen at Gate G0:

    load_master(H)                  -> DataFrame indexed by target date   [B]
    swap_weather(df, weather_df)    -> DataFrame, target-day weather only [B]
    run_walk_forward(...)           -> DataFrame of predictions           [A]

Track B owns the first two and they live here. `run_walk_forward` is Track A's
and belongs in src/walkforward.py; until it lands, this module re-exports a
stand-in from src/operational/walkforward.py with the identical signature so
Phase 5 is not blocked (roadmap C.3).

Why load_master exists at all
-----------------------------
master.csv is indexed by the day the demand was MEASURED. A forecaster is
indexed by the day being PREDICTED, and the gap between those two is H.

H is measured, not assumed: Grid-India publishes day D's PSP report on D+1
(46 of 46 observed daily runs never saw same-day data; see
results/publish_time.json). So a forecast issued the evening of day D holds
demand only through D-1, while the target is D+1 - a two-day gap, H=2.

The consequence everyone trips over: **peak_lag_1 is not available for a
genuine day-ahead forecast.** Reading master.csv directly and training on
peak_lag_1 produces a model that cannot be run in production, and a backtest
roughly as optimistic as the same-day-energy leak that was already removed
from this dataset once.

Lag naming convention under H
-----------------------------
load_master shifts the demand/energy/grid block by H-1 days. Column names are
deliberately NOT renamed, so Track A's feature lists work unchanged across
values of H. The convention is:

    lag numbers are relative to the newest AVAILABLE day for that source,
    not to target_date - 1.

So at H=2, `peak_lag_1` holds the observation from target - 2, and
`peak_roll_mean_7` averages target-2 .. target-8. Renaming was considered and
rejected: shifting {1,2,3,7,14,364} by one collides with itself at 2 and 3.
Instead of trusting the names, `verify_no_leakage()` proves mechanically that
no shifted column contains information from later than target - H, and the
G0 identity test calls it.

Sources do not all arrive together
----------------------------------
`config.H_BY_GROUP` gives each group its own horizon, because they come from
different publications:

    demand, energy   Grid-India PSP report, published D+1      -> H     (2)
    grid (sr_*)      POSOCO/CICERO slice, a further day behind -> H + 1 (3)

Shifting them uniformly looked right and was not: every `sr_*_lag1` column
came back NaN on the row being predicted, because the grid slice simply had
not published `target - 2` yet. In training those columns are present, at
inference they were absent - a train/serve inconsistency nothing would have
warned about, found only by running predict_next_day.py end to end.

Energy escapes this because build_dataset backfills `tn_energy_gwh` from the
PSP report's own `tn_energy_met_mu` when the slice lags, and the two agree to
0.02 GWh over 1,050 overlapping days.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import (FEATURE_GROUPS, FORECAST_ARCHIVE, H as CONFIG_H,
                    H_BY_GROUP, MASTER, PEAK_TARGET, RAW_DEMAND, RAW_WEATHER)

__all__ = ["load_master", "swap_weather", "run_walk_forward", "group_offsets",
           "feature_groups", "model_features", "verify_no_leakage",
           "load_phase5_data", "verify_phase5_integrity", "PHASE5_FEATURE_SETS"]


def feature_groups() -> dict:
    """The parsed contract file, comments stripped."""
    spec = json.loads(FEATURE_GROUPS.read_text())
    return {k: v for k, v in spec.items() if not k.startswith("_")}


def _spec():
    s = feature_groups()
    g = s["groups"]
    return s, g


def group_offsets(H: int | None = None) -> dict:
    """Extra days to shift each source group, keyed by group name.

    Each source gets its own horizon (config.H_BY_GROUP): the grid (sr_*)
    columns come from a slice that publishes a day after the PSP report, so
    shifting them by the same amount as demand leaves them NaN on exactly the
    row being predicted.

    Shared by load_master() and predict_next_day.py so inference and training
    cannot disagree about staleness - the sort of duplication that produced
    the all-NaN sr_* columns in the first place.
    """
    H = int(CONFIG_H if H is None else H)
    spec, _ = _spec()
    offsets = {}
    for gname in spec["shifted_by_H"]:
        group_h = H_BY_GROUP.get(gname, H)
        if H != CONFIG_H:
            # honour an explicitly requested H (e.g. the idealised H=1 row)
            # while preserving each source's relative staleness
            group_h = max(1, group_h - (CONFIG_H - H))
        offsets[gname] = group_h - 1
    return offsets


def load_master(H: int | None = None, path=None) -> pd.DataFrame:
    """master.csv re-indexed for forecasting, with demand shifted by H-1.

    Returns a frame indexed by **target date**, with the target in column `y`
    and every feature computable before that target day begins.

    H defaults to config.H (the measured value). Pass H=1 to produce the
    idealised "if demand arrived instantly" variant the roadmap asks to report
    alongside the real one.

    Each group is shifted by its own horizon (config.H_BY_GROUP), so the grid
    columns - which publish a day later than demand - are shifted one day
    further rather than being left NaN at inference.

    The returned frame carries `.attrs`:
        H               the horizon applied
        lag_offset      extra days the demand block was shifted (H - 1)
        newest_demand   "target - H"
        group_offsets   {group: extra days shifted}
        shift_detail    {group: {columns, offset, newest_day}}
    """
    H = int(CONFIG_H if H is None else H)
    if H < 1:
        raise ValueError(f"H must be >= 1, got {H}")

    spec, groups = _spec()
    df = pd.read_csv(path or MASTER, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    offsets = group_offsets(H)

    # Shift on a calendar-continuous index, never positionally: master.csv has
    # 157 missing days, and a positional shift would quietly borrow from 2-3
    # days earlier across every gap. Same trap build_dataset.add_lags documents.
    span = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    full = df.set_index("date").reindex(span)
    shifted = {}
    for gname, off in offsets.items():
        cols = [c for c in groups[gname] if c in full.columns]
        if off and cols:
            full[cols] = full[cols].shift(off)
        shifted[gname] = {"columns": cols, "offset": off,
                          "newest_day": f"target - {off + 1}"}
    df = full.loc[pd.DatetimeIndex(df["date"])].rename_axis("date").reset_index()

    out = df.rename(columns={PEAK_TARGET: "y"}).set_index("date")
    out.attrs.update(H=H, lag_offset=offsets.get("demand", H - 1),
                     newest_demand=f"target - {H}",
                     group_offsets=offsets,
                     shifted_columns=[c for v in shifted.values()
                                      for c in v["columns"]],
                     shift_detail=shifted)
    return out


def swap_weather(df: pd.DataFrame, weather_df: pd.DataFrame,
                 observed: pd.DataFrame | None = None,
                 required_dates: pd.DatetimeIndex | list | None = None,
                 mask_unswapped: bool = False) -> pd.DataFrame:
    """Replace ONLY the target-day weather columns, leaving all else untouched.

    This is what turns a Regime A (reanalysis) frame into a Regime B
    (forecast) frame. Everything outside `weather_target_day` -- demand lags,
    calendar, lagged weather -- is passed through byte-identical, which is what
    the G0 identity test checks.

    df              output of load_master(), indexed by target date
    weather_df      state-weighted weather for the target day, indexed by date.
                    Needs the level columns; deltas/anomalies are derived.
    observed        state-weighted OBSERVED weather, indexed by date, used for the
                    yesterday-side of deltas and anomalies. Defaults to
                    weather_df, which is only correct if you are passing
                    reanalysis (i.e. the identity case).
    required_dates  Optional iterable of target dates required to be in weather_df.
                    Raises ValueError if any required target date lacks forecast coverage,
                    preventing silent fallback to ERA5 reanalysis in Regime B.
    mask_unswapped  If True, target-day weather columns are set to NaN for any dates in df
                    that do not exist in weather_df, preventing silent retention of ERA5.

    Lead-awareness is the subtle part. Predicting day T, the forecast supplies
    T, but T-1 has already happened and is observed. So

        d_t_max        = forecast(T) - observed(T-1)
        t_max_vs_roll7 = forecast(T) - mean(observed[T-7 .. T-1])

    Treating both sides as forecast would overstate the degradation; taking
    both from observed would understate it.
    """
    spec, groups = _spec()
    rules = spec["weather_swap_rules"]
    obs = weather_df if observed is None else observed

    out = df.copy()

    # Validate 100% coverage for required scored target dates
    if required_dates is not None:
        req_idx = pd.DatetimeIndex(required_dates)
        missing_req = list(req_idx.difference(weather_df.index))
        # Also check for NaN values on required dates present in weather_df
        present_req = req_idx.intersection(weather_df.index)
        check_cols = [c for c in ("t_max", "t_mean", "t_min") if c in weather_df.columns]
        nan_req = []
        if check_cols and len(present_req) > 0:
            sub = weather_df.loc[present_req, check_cols]
            nan_req = list(sub[sub.isna().any(axis=1)].index)
        all_missing = sorted(set(missing_req + nan_req))
        if len(all_missing) > 0:
            missing_sample = [str(pd.Timestamp(d).date()) for d in all_missing[:5]]
            raise ValueError(
                f"Missing forecast weather coverage for {len(all_missing)} required target date(s): "
                f"{missing_sample}... Silent fallback to ERA5 reanalysis in Regime B is strictly forbidden."
            )

    idx = out.index.intersection(weather_df.index)
    if len(idx) == 0:
        raise ValueError("swap_weather: no overlapping dates")

    # If mask_unswapped is enabled, set target-day weather columns to NaN for dates
    # missing from weather_df so that ERA5 values cannot silently persist into Regime B
    if mask_unswapped:
        missing_dates = out.index.difference(weather_df.index)
        if len(missing_dates) > 0:
            target_cols = [c for c in groups["weather_target_day"] if c in out.columns]
            out.loc[missing_dates, target_cols] = np.nan

    for col in rules["levels"]["columns"]:
        if col in out.columns and col in weather_df.columns:
            out.loc[idx, col] = weather_df.loc[idx, col].to_numpy()

    for col in rules["deltas"]["columns"]:
        base = col[2:]                      # d_t_max -> t_max
        if col in out.columns and base in weather_df.columns:
            prev = obs[base].reindex(idx - pd.Timedelta(days=1)).to_numpy()
            out.loc[idx, col] = weather_df.loc[idx, base].to_numpy() - prev

    for col in rules["anomalies"]["columns"]:
        base = col.replace("_vs_roll7", "")
        if col in out.columns and base in weather_df.columns:
            # The trailing mean has to be computed on a calendar index that
            # INCLUDES the target dates, not just the observed ones. At
            # inference the target day is tomorrow and is absent from `obs`, so
            # reindexing a mean computed only over obs would land on a missing
            # label and return NaN - silently dropping three features on
            # exactly the row being predicted. Reindexing onto the union first
            # means shift(1) still reaches the real previous days.
            span = pd.date_range(min(obs.index.min(), idx.min())
                                 - pd.Timedelta(days=8),
                                 max(obs.index.max(), idx.max()), freq="D")
            trailing = (obs[base].reindex(span)
                        .shift(1).rolling(7, min_periods=3).mean()
                        .reindex(idx).to_numpy())
            out.loc[idx, col] = weather_df.loc[idx, base].to_numpy() - trailing

    out.attrs.update(df.attrs)
    out.attrs["weather_source"] = getattr(weather_df, "attrs", {}).get(
        "source", "swapped")
    return out


def verify_no_leakage(df: pd.DataFrame, raw_master=None) -> dict:
    """Prove no shifted column carries information newer than target - H.

    Does not trust column names. For each demand/energy/grid column, finds
    which calendar offset of the raw target (or energy) series it actually
    equals, and checks that offset is at least H.

    Returns a report; raises AssertionError on a violation.
    """
    H = df.attrs.get("H", CONFIG_H)
    spec, groups = _spec()
    raw = pd.read_csv(raw_master or MASTER, parse_dates=["date"])
    y = raw.set_index("date")[PEAK_TARGET]

    findings, violations = {}, []
    # Only the plain peak_lag_* columns map to a single raw observation; the
    # rolling and difference columns are combinations, so we bound them by the
    # newest raw day any of their inputs could have touched.
    for col in groups["demand"]:
        if col not in df.columns or not col.startswith("peak_lag_"):
            continue
        matched = None
        for k in range(0, 400):
            exp = y.reindex(df.index - pd.Timedelta(days=k)).to_numpy()
            act = df[col].to_numpy()
            both = ~(pd.isna(exp) | pd.isna(act))
            if both.sum() > 50 and np.allclose(exp[both], act[both], atol=1e-6):
                matched = k
                break
        findings[col] = matched
        if matched is not None and matched < H:
            violations.append(
                f"{col} resolves to target - {matched}, which is newer than "
                f"target - {H}")

    if violations:
        raise AssertionError("H-shift leakage: " + "; ".join(violations))

    return {"H": H, "resolved_offsets": findings,
             "min_offset": min((v for v in findings.values() if v is not None),
                               default=None),
             "ok": True}


def model_features(df: pd.DataFrame, drop_weather_levels: bool = True,
                   drop_trend: bool = True) -> list[str]:
    """Convenience feature list. Not contract - a measured recommendation.

    See README 'Choosing features': weather levels are redundant with the
    autoregressive block, and `trend` hurts trees in an extrapolating regime.
    """
    spec, _ = _spec()
    rec = spec["recommended_drops"]
    exclude = {"y", PEAK_TARGET, *spec["non_numeric"]}
    if drop_weather_levels:
        exclude |= set(rec["weather_levels"])
    if drop_trend:
        exclude |= set(rec["trend"])
    return [c for c in df.columns if c not in exclude]


def run_walk_forward(*args, **kwargs):
    """Track A's engine (src/walkforward.py); falls back to Track B's shim.

    Signature (frozen at G0):
        run_walk_forward(model_factory, df, start, end,
                         refit_days=7, half_life_years=None) -> DataFrame
        columns: date, y_true, y_pred, refit_date
    """
    try:
        from src.walkforward import run_walk_forward as impl   # Track A
    except ImportError:
        from src.operational.walkforward import run_walk_forward as impl
    return impl(*args, **kwargs)


# --------------------------------------------------------------------------
# Canonical Phase 5 Feature Contract & Dataset Gateway
# --------------------------------------------------------------------------

PHASE5_FEATURE_SETS: dict[str, list[str]] = {
    # A0: Weather-free baseline (19 features, H=2, exact Track A contract)
    "A0_weather_free": [
        # Calendar (11)
        "dow", "is_weekend", "is_holiday", "day_before_holiday",
        "day_after_holiday", "pongal_window",
        "sin1", "cos1", "sin2", "cos2", "trend",
        # Peak lags H >= 2 (5) - literal offsets T-2, T-3, T-7, T-14, T-364
        "peak_lag_2", "peak_lag_3", "peak_lag_7", "peak_lag_14", "peak_lag_364",
        # Energy lags H >= 2 (3) - literal offsets T-2, T-7, T-14
        "energy_lag_2", "energy_lag_7", "energy_lag_14",
    ],

    # A1: Weather levels (temperature only, no humidity)
    "A1_weather_levels": [
        "dow", "is_weekend", "is_holiday", "day_before_holiday",
        "day_after_holiday", "pongal_window",
        "sin1", "cos1", "sin2", "cos2", "trend",
        "peak_lag_2", "peak_lag_3", "peak_lag_7", "peak_lag_14", "peak_lag_364",
        "energy_lag_2", "energy_lag_7", "energy_lag_14",
        "t_max", "t_mean", "t_min", "cdd_max", "hours_above_32",
    ],

    # A2: Weather levels + deltas + anomalies (canonical decision gate specification: 26 features)
    "A2_weather_deltas_anomalies": [
        "dow", "is_weekend", "is_holiday", "day_before_holiday",
        "day_after_holiday", "pongal_window",
        "sin1", "cos1", "sin2", "cos2", "trend",
        "peak_lag_2", "peak_lag_3", "peak_lag_7", "peak_lag_14", "peak_lag_364",
        "energy_lag_2", "energy_lag_7", "energy_lag_14",
        "t_max", "t_mean", "t_min", "cdd_max", "hours_above_32",
        "d_cdd_max", "cdd_max_vs_roll7",
    ],
}


def load_phase5_data(feature_set: str | list[str] = "A0_weather_free",
                     weather_source: str = "none",
                     forecast_lead: int = 1,
                     path: str | Path | None = None,
                     clean_na: bool = True,
                     date_range: tuple[str, str] | None = None,
                     required_dates: pd.DatetimeIndex | list | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Canonical Phase 5 dataset preparation gateway.

    Implements the agreed architectural decision for Phase 5 research:
    1. Ingests master.csv using Track A's literal lag representation.
    2. Preserves literal physical calendar offsets (peak_lag_2 is T-2,
       peak_lag_7 is T-7, peak_lag_14 is T-14, peak_lag_364 is T-364).
    3. Enforces H=2 by strictly excluding unavailable T-1 demand/energy features
       (peak_lag_1, energy_lag_1).
    4. Preserves 7-day cyclical weekday alignment without shifting lag columns.
    5. Provides identical non-weather features for Regime A and Regime B, with
       weather being swapped cleanly via swap_weather().

    Parameters:
        feature_set: Name of approved feature set ('A0_weather_free', 'A1_weather_levels',
                     'A2_weather_deltas_anomalies') or explicit list of column names.
        weather_source: 'none' (for A0), 'reanalysis' (for Regime A), or 'forecast' (for Regime B).
        forecast_lead: Forecast lead day for Regime B (default: 1 day ahead).
        path: Optional override path for master.csv.
        clean_na: Whether to drop rows where any requested feature or target is NaN (default True).
        date_range: Optional (start_date, end_date) tuple to filter dates.
        required_dates: Optional list of target dates required to have forecast coverage.
                        If provided, raises ValueError if any target date lacks forecast.

    Returns:
        (df, features): DataFrame indexed by date with target in both 'tn_peak_demand_mw'
                        and 'y', and the resolved feature list.
    """
    from build_dataset import state_weather

    if isinstance(feature_set, str):
        if feature_set not in PHASE5_FEATURE_SETS:
            raise KeyError(f"Unknown Phase 5 feature set: {feature_set}. "
                           f"Available: {list(PHASE5_FEATURE_SETS.keys())}")
        features = list(PHASE5_FEATURE_SETS[feature_set])
    else:
        features = list(feature_set)

    # Ingest literal master.csv
    raw = pd.read_csv(path or MASTER, parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    raw = raw.set_index("date")

    # Handle weather source
    if weather_source == "forecast":
        if not FORECAST_ARCHIVE.exists():
            raise FileNotFoundError(f"Missing forecast archive: {FORECAST_ARCHIVE}. "
                                    "Run fetch_weather.py archive-forecast")
        fc_raw = pd.read_csv(FORECAST_ARCHIVE)
        fc_lead = fc_raw[fc_raw["lead"] == int(forecast_lead)].drop(columns=["lead"])
        fc_wx = state_weather(fc_lead).set_index("date").sort_index()
        fc_wx.attrs["source"] = f"forecast_lead_{int(forecast_lead)}"

        obs_raw = pd.read_csv(RAW_WEATHER)
        obs_wx = state_weather(obs_raw).set_index("date").sort_index()
        obs_wx.attrs["source"] = "era5_reanalysis"

        # Enforce 100% forecast coverage for required evaluation dates
        req = required_dates
        if req is None and date_range is not None:
            start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            req = raw[(raw.index >= start_d) & (raw.index <= end_d)].index

        if req is not None:
            req_idx = pd.DatetimeIndex(req)
            missing_idx = list(req_idx.difference(fc_wx.index))
            present_idx = req_idx.intersection(fc_wx.index)
            check_cols = [c for c in ("t_max", "t_mean", "t_min") if c in fc_wx.columns]
            nan_idx = []
            if check_cols and len(present_idx) > 0:
                sub = fc_wx.loc[present_idx, check_cols]
                nan_idx = list(sub[sub.isna().any(axis=1)].index)
            missing_fc = sorted(set(missing_idx + nan_idx))
            if len(missing_fc) > 0:
                sample = [str(pd.Timestamp(d).date()) for d in missing_fc[:5]]
                raise ValueError(
                    f"Missing forecast weather coverage for {len(missing_fc)} required target date(s): "
                    f"{sample}... Silent fallback to ERA5 reanalysis in Regime B is strictly forbidden."
                )

        raw = swap_weather(raw, fc_wx, observed=obs_wx, required_dates=req, mask_unswapped=True)
    elif weather_source in ("none", "reanalysis"):
        pass
    else:
        raise ValueError(f"Unknown weather_source: '{weather_source}'. "
                         "Must be 'none', 'reanalysis', or 'forecast'.")

    df = raw.reset_index()

    # Ensure target column exists
    target_col = PEAK_TARGET
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataset")
    df["y"] = df[target_col]

    # Filter date range if specified
    if date_range is not None:
        start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df[(df["date"] >= start_d) & (df["date"] <= end_d)].reset_index(drop=True)

    # Check required columns
    needed = ["date", target_col, "y"] + [f for f in features if f not in ("y",)]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Requested features missing in dataset: {missing_cols}")

    # Clean NaNs across requested features and target
    if clean_na:
        df = df.dropna(subset=needed).reset_index(drop=True)

    # Set index to DatetimeIndex
    out = df.set_index("date")
    out["date"] = out.index

    # Set metadata
    out.attrs.update(
        H=2,
        lag_representation="literal_master",
        feature_set=feature_set if isinstance(feature_set, str) else "custom",
        features=features,
        weather_source=weather_source,
        forecast_lead=forecast_lead if weather_source == "forecast" else None,
        newest_demand="target - 2",
        newest_energy="target - 2",
    )

    return out, features


def verify_phase5_integrity(df: pd.DataFrame, features: list[str], H: int = 2,
                            raw_master: str | Path | None = None) -> dict:
  """Mechanically proves that a Phase 5 modelling dataset satisfies all integrity,
  leakage-prevention, and weekday-alignment contracts.
  """
  import re

  raw_path = raw_master or MASTER
  raw = pd.read_csv(raw_path, parse_dates=["date"]).sort_values("date")
  y_raw = raw.set_index("date")[PEAK_TARGET]
  if "tn_energy_gwh" in raw.columns:
      e_raw = raw.set_index("date")["tn_energy_gwh"]
  elif RAW_DEMAND.exists():
      e_raw = pd.read_csv(RAW_DEMAND, parse_dates=["date"]).sort_values("date").set_index("date")["tn_energy_gwh"]
  else:
      e_raw = None

  df_idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["date"])

  # 1. Zero forbidden T-1 lags
  lag1_matches = [
      f for f in features
      if re.search(r"(_lag_?1$|_lag_?1_)", f) and ("peak" in f or "energy" in f or "sr_" in f)
  ]
  if lag1_matches:
      raise AssertionError(f"Forbidden T-1 lag leakage detected: {lag1_matches}. "
                           f"Under H={H}, newest permitted information is T-2.")

  # 2. Target and same-day actual energy isolation
  forbidden_targets = ["tn_peak_demand_mw", "y", "tn_energy_gwh", "tn_shortage_at_peak_mw"]
  target_leaks = [f for f in features if f in forbidden_targets]
  if target_leaks:
      raise AssertionError(f"Target or same-day context found in feature list: {target_leaks}")

  # 3. Information-age and weekday alignment verification
  resolved_offsets = {}
  weekday_alignments = {}
  for col in features:
      if not col.startswith("peak_lag_") and not col.startswith("energy_lag_"):
          continue

      series = y_raw if "peak" in col else e_raw
      if series is None:
          continue

      matched_k = None
      for k in range(0, 400):
          exp = series.reindex(df_idx - pd.Timedelta(days=k)).to_numpy()
          act = df[col].to_numpy()
          both = ~(pd.isna(exp) | pd.isna(act))
          if both.sum() > 50 and np.allclose(exp[both], act[both], atol=1e-4):
              matched_k = k
              break

      resolved_offsets[col] = matched_k
      if matched_k is not None:
          if matched_k < H:
              raise AssertionError(f"{col} resolves to target - {matched_k}, violating H={H}!")
          # Check literal weekday alignment
          if col in ("peak_lag_7", "energy_lag_7") and matched_k != 7:
              raise AssertionError(f"{col} broke weekly weekday alignment! Expected T-7, got T-{matched_k}")
          if col in ("peak_lag_14", "energy_lag_14") and matched_k != 14:
              raise AssertionError(f"{col} broke bi-weekly weekday alignment! Expected T-14, got T-{matched_k}")
          if col == "peak_lag_364" and matched_k != 364:
              raise AssertionError(f"{col} broke annual seasonal weekday alignment (52*7)! Expected T-364, got T-{matched_k}")
          if matched_k % 7 == 0:
              weekday_alignments[col] = f"Preserved (Exact Weekday Match: target - {matched_k}d)"

  # 4. Humidity-dependent feature restrictions
  forbidden_humidity = [
      "rh_mean", "app_t_mean", "app_t_max", "cdd_apparent", "thi",
      "d_rh_mean", "d_app_t_max", "d_thi", "thi_vs_roll7"
  ]
  hum_leaks = [f for f in features if f in forbidden_humidity]
  if hum_leaks:
      raise AssertionError(f"Humidity-dependent features forbidden in Phase 5 due to archive coverage: {hum_leaks}")

  # 5. Monotonic chronological ordering
  if not df_idx.is_monotonic_increasing:
      raise AssertionError("Dates are not strictly monotonically increasing!")

  # 6. Zero nulls in requested feature matrix
  null_counts = df[features].isna().sum()
  if null_counts.sum() > 0:
      bad = null_counts[null_counts > 0].to_dict()
      raise AssertionError(f"Null values detected in feature matrix: {bad}")

  return {
      "ok": True,
      "H": H,
      "features_count": len(features),
      "resolved_offsets": resolved_offsets,
      "weekday_alignments": weekday_alignments,
  }

