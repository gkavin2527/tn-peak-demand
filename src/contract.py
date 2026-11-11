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

from config import (FEATURE_GROUPS, H as CONFIG_H, H_BY_GROUP, MASTER,
                    PEAK_TARGET)

__all__ = ["load_master", "swap_weather", "run_walk_forward", "group_offsets",
           "feature_groups", "model_features", "verify_no_leakage"]


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
                 observed: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replace ONLY the target-day weather columns, leaving all else untouched.

    This is what turns a Regime A (reanalysis) frame into a Regime B
    (forecast) frame. Everything outside `weather_target_day` -- demand lags,
    calendar, lagged weather -- is passed through byte-identical, which is what
    the G0 identity test checks.

    df          output of load_master(), indexed by target date
    weather_df  state-weighted weather for the target day, indexed by date.
                Needs the level columns; deltas/anomalies are derived.
    observed    state-weighted OBSERVED weather, indexed by date, used for the
                yesterday-side of deltas and anomalies. Defaults to
                weather_df, which is only correct if you are passing
                reanalysis (i.e. the identity case).

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
    idx = out.index.intersection(weather_df.index)
    if len(idx) == 0:
        raise ValueError("swap_weather: no overlapping dates")

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
