"""
Issue tomorrow's Tamil Nadu peak demand forecast.

    python predict_next_day.py                 # forecast tomorrow, log it
    python predict_next_day.py --date 2026-11-12
    python predict_next_day.py --dry-run       # print, do not append to the log

Roadmap Phase 7.1. Runs in the evening, after the forecast logger, and appends
to data/predictions_log.csv with the issue time, H and model version so that
performance can be audited later against what was actually knowable.

What makes this honest
----------------------
1. **H is respected.** Grid-India publishes day D's demand on D+1, so an
   evening-of-D run targeting D+1 has demand only through D-1. Features come
   from `load_master(H)` with H=2 measured, which shifts the whole demand
   block. Using master.csv directly would silently use peak_lag_1 - data that
   does not exist yet - and the forecast would be unservable.

2. **Target-day weather is a forecast, not an actual.** It is read from
   weather_forecast_log.csv for the target date and substituted through
   `swap_weather`, the same function the Phase 5 experiment uses. So tonight's
   prediction is produced by exactly the pipeline that was measured at
   ~524 MW MAE, not by the ~481 MW reanalysis pipeline that cannot be run in
   production.

3. **Intervals are pre-calibrated.** Nothing is fitted at inference beyond the
   model refit itself; the interval multipliers come from
   results/intervals.json.

The model is still Track B's stand-in (roadmap C.3). At G2, point
--champion at models/champion_config.json and the rest of this script is
unchanged.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy/sklearn

import argparse
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_dataset import state_weather
from config import (FORECAST_LOG, H, PEAK_TARGET, PREDICTIONS_LOG,
                    RANDOM_SEED, RAW_WEATHER)
from src.contract import (feature_groups, group_offsets, model_features,
                          swap_weather)
from src.operational import intervals as IV

ANCHOR = "peak_lag_1"          # newest available demand day under load_master(H)
MODEL_VERSION = "standin-hgb-differenced-v1"
TRAIN_TAIL_YEARS = None        # None = full history; set to e.g. 8 to recency-trim
# Highest forecast lead allowed to stand in for a missing observed day. 0 is
# today's own analysis; 1 tolerates a logger that missed one run. Above that the
# substitute stops being "a day that already happened" and becomes a forecast.
MAX_FILL_LEAD = 1


def target_weather(target: pd.Timestamp):
    """State-weighted FORECAST weather for the target day, plus provenance."""
    if not FORECAST_LOG.exists():
        raise SystemExit("no weather_forecast_log.csv - run "
                         "'python fetch_weather.py forecast' first")
    log = pd.read_csv(FORECAST_LOG, parse_dates=["date", "issued_on"])
    if "issue_slot" not in log.columns:
        log["issue_slot"] = "morning"
    rows = log[log["date"] == target]
    if rows.empty:
        raise SystemExit(
            f"no logged forecast for {target:%Y-%m-%d}. Run "
            f"'python fetch_weather.py forecast --slot=evening' first; the "
            f"logger must run before inference.")

    # Newest issue date, and within it the EVENING run when we have one.
    # Inference is an evening job (that is what makes H=2 rather than H=3),
    # so the evening forecast is the one actually in hand at issue time; the
    # morning run is twelve hours staler.
    issued = rows["issued_on"].max()
    rows = rows[rows["issued_on"] == issued]
    slot = "evening" if (rows["issue_slot"] == "evening").any() else "morning"
    rows = rows[rows["issue_slot"] == slot]
    lead = int((target - issued).days)

    wx = state_weather(rows.drop(columns=["issued_on", "issue_slot"]))
    wx = wx.set_index("date")
    wx.attrs["source"] = f"live_log_{slot}_lead_{lead}"
    return wx, issued, lead, slot


def best_known_observed(target: pd.Timestamp) -> pd.DataFrame:
    """Observed weather, with the most recent days filled from the forecast log.

    The delta features are `forecast(T) - observed(T-1)`. Forecasting tomorrow
    from this evening, T-1 is *today* - and today is not in weather_daily.csv,
    because `fetch_weather.py update` deliberately excludes the current day as
    partial. Left alone, every `d_*` column comes back NaN on exactly the row
    being predicted. (Fourth instance of that failure mode here; the
    train/serve parity check in test_contract.py is what caught it.)

    The substitute is the forecast log's **lead-0** row for that day: at 20:00
    IST today's peak has already happened, so lead-0 is mostly observation with
    only the late evening still forecast. Measured against the archive over 99
    days of lead-0 rows captured by the 07:00 job - the pessimistic case, since
    far less of the day was observed at issue time:

        t_max    bias -0.08  MAE 0.31 degC
        t_mean   bias -0.38  MAE 0.39 degC
        rh_mean  bias +1.95  MAE 2.00 points

    At the measured ~180 MW per degC that is roughly 55 MW of induced noise on
    a 525 MW MAE. Acceptable, and far better than dropping seven features. The
    evening run should beat those figures outright; there is not enough
    evening-slot history to say by how much yet.

    **Lead is capped, and that cap is the point.** An earlier version took
    simply the newest `issued_on`, which is wrong when the logger has been
    down: a row for today issued three days ago is a lead-3 *forecast* wearing
    the label "today's analysis", and the accuracy above would not remotely
    apply to it. So each missing day is filled from its lowest available lead,
    nothing above MAX_FILL_LEAD is used, and the lead actually used is recorded
    and printed. If no low-lead row exists the day stays NaN and the parity
    check fails loudly - which is the correct outcome, because a silently
    stale fill is worse than a visible gap.
    """
    obs = state_weather(pd.read_csv(RAW_WEATHER)).set_index("date").sort_index()

    log = pd.read_csv(FORECAST_LOG, parse_dates=["date", "issued_on"])
    if "issue_slot" not in log.columns:
        log["issue_slot"] = "morning"
    log["lead"] = (log["date"] - log["issued_on"]).dt.days

    gap = log[(log["date"] < target) & (log["date"] > obs.index.max())
              & (log["lead"] >= 0) & (log["lead"] <= MAX_FILL_LEAD)]
    if gap.empty:
        return obs

    # lowest lead wins; within a lead, the evening run (more of the day
    # observed) beats the morning one
    gap = gap.assign(_slot_rank=(gap["issue_slot"] != "evening").astype(int))
    gap = (gap.sort_values(["lead", "_slot_rank", "issued_on"])
              .drop_duplicates(subset=["date", "city"], keep="first"))

    fill = state_weather(gap.drop(columns=["issued_on", "issue_slot", "lead",
                                           "_slot_rank"]))
    fill = fill.set_index("date").sort_index()
    fill = fill[~fill.index.isin(obs.index)]
    if fill.empty:
        return obs

    used = (gap.drop_duplicates("date").set_index("date")["lead"]
               .reindex(fill.index))
    merged = pd.concat([obs, fill]).sort_index()
    merged.attrs["filled_from_forecast"] = [
        f"{d:%Y-%m-%d}(lead{int(l)})" for d, l in used.items()]
    merged.attrs["max_fill_lead"] = int(used.max()) if len(used) else None
    return merged


def build_features(target: pd.Timestamp):
    """Feature row for the target day: demand shifted by H, weather forecast.

    Built from build_dataset.merged_frame(extend_to=target) and the real
    add_lags(), not a hand-rolled copy. Sharing that code is what guarantees
    the feature row served tonight is constructed identically to the rows the
    model trained on - a reimplementation drifts, and a drifted inference row
    is indistinguishable from a model that has silently stopped working.
    """
    from build_dataset import add_lags, merged_frame
    from config import LEAKY_SAMEDAY

    merged = merged_frame(extend_to=target)
    if target not in set(merged["date"]):
        raise SystemExit(f"could not extend the feature frame to {target:%Y-%m-%d}")

    df = add_lags(merged).reset_index(drop=True)
    df = df.drop(columns=[c for c in LEAKY_SAMEDAY if c in df.columns])
    df = df.rename(columns={PEAK_TARGET: "y"}).set_index("date").sort_index()

    # Apply the same per-group shifts load_master() applies, from the same
    # helper - at H=2 the newest demand observation is target-2, and the grid
    # slice is a day staler still. Duplicating this logic is what left every
    # sr_*_lag1 column NaN on the predicted row.
    groups = feature_groups()["groups"]
    offsets = group_offsets(H)
    for gname, off in offsets.items():
        cols = [c for c in groups[gname] if c in df.columns]
        if off and cols:
            df[cols] = df[cols].shift(off)
    df.attrs.update(H=H, lag_offset=offsets.get("demand", H - 1),
                    group_offsets=offsets, newest_demand=f"target - {H}")

    obs = best_known_observed(target)
    wx, issued, lead, slot = target_weather(target)
    df = swap_weather(df, wx, observed=obs)
    df.attrs["observed_filled_from_forecast"] = obs.attrs.get(
        "filled_from_forecast", [])
    df.attrs["max_fill_lead"] = obs.attrs.get("max_fill_lead")
    return df, issued, lead, slot


def fit_and_predict(df: pd.DataFrame, target: pd.Timestamp):
    feats = model_features(df)
    train = df[df.index < target]
    if TRAIN_TAIL_YEARS:
        train = train[train.index >= target - pd.DateOffset(years=TRAIN_TAIL_YEARS)]

    y = train["y"] - train[ANCHOR]
    ok = y.notna() & np.isfinite(y)
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          random_state=RANDOM_SEED)
    model.fit(train.loc[ok, feats], y[ok])

    row = df.loc[[target], feats]
    missing = row.columns[row.isna().any()].tolist()
    anchor_val = df.loc[target, ANCHOR]
    if not np.isfinite(anchor_val):
        # Distinguish the two causes, because the fix is different. Usually the
        # pipeline is simply behind - the morning job has not ingested the
        # report for target-H yet - and running it resolves this. Rarely, the
        # demand series has a genuine hole there (157 such days exist) and no
        # amount of fetching will help.
        anchor_day = target - pd.Timedelta(days=H)
        newest = df["y"].last_valid_index()
        if newest is not None and newest < anchor_day:
            raise SystemExit(
                f"cannot forecast {target:%Y-%m-%d}: the newest demand "
                f"observation is {newest:%Y-%m-%d}, but H={H} needs "
                f"{anchor_day:%Y-%m-%d}.\n"
                f"The data pipeline is {(anchor_day - newest).days} day(s) "
                f"behind. Run:\n"
                f"  python update_peak_demand.py {anchor_day:%Y-%m-%d}\n"
                f"  python build_dataset.py")
        raise SystemExit(
            f"cannot forecast {target:%Y-%m-%d}: {ANCHOR} is NaN because the "
            f"demand series has a genuine gap at {anchor_day:%Y-%m-%d}. "
            f"Refusing to issue a differenced forecast without its anchor.")

    yhat = float(model.predict(row)[0] + anchor_val)
    return yhat, int(ok.sum()), missing, float(anchor_val)


def append_log(record: dict):
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([record])
    if PREDICTIONS_LOG.exists():
        old = pd.read_csv(PREDICTIONS_LOG)
        new = pd.concat([old, new], ignore_index=True)
        # one row per (target_date, issued_at-day): re-running the same evening
        # replaces rather than duplicates
        new = new.drop_duplicates(subset=["target_date", "issue_day"], keep="last")
    new.to_csv(PREDICTIONS_LOG, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="target date (default: tomorrow)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--champion", default=None,
                    help="models/champion_config.json (G2 hand-off)")
    args = ap.parse_args()

    target = pd.Timestamp(args.date or (date.today() + timedelta(days=1)))
    now = pd.Timestamp.now()

    print(f"=== Tamil Nadu peak demand forecast ===")
    print(f"target      {target:%Y-%m-%d} ({target:%A})")
    print(f"issued      {now:%Y-%m-%d %H:%M}")
    if args.champion:
        print(f"model       {args.champion} (requested)")
        print("            NOTE: champion loading lands at G2; using stand-in")
    print(f"model       {MODEL_VERSION}")
    print(f"H           {H} (measured; newest demand = target - {H})")

    df, issued, lead, slot = build_features(target)
    yhat, n_train, missing, anchor_val = fit_and_predict(df, target)

    print(f"weather     {slot} forecast issued {issued:%Y-%m-%d}, lead {lead}")
    filled = df.attrs.get("observed_filled_from_forecast") or []
    if filled:
        print(f"            yesterday-side filled from the forecast for "
              f"{', '.join(filled)} (not yet in the archive)")
        if (df.attrs.get("max_fill_lead") or 0) > 0:
            print(f"WARNING     fill used a lead>0 forecast - the logger likely "
                  f"missed a run; today's own analysis was unavailable")
    print(f"train rows  {n_train:,}")
    print(f"{ANCHOR:<11} {anchor_val:,.0f} MW  (the level the change is added to)")
    if missing:
        print(f"WARNING     {len(missing)} feature(s) NaN: {missing[:6]}")

    try:
        calib = IV.load()
        lo90, hi90 = IV.apply(yhat, target, calib, 0.90)
        lo95, hi95 = IV.apply(yhat, target, calib, 0.95)
        have_iv = True
    except FileNotFoundError as e:
        print(f"\nWARNING     {e}")
        lo90 = hi90 = lo95 = hi95 = float("nan")
        have_iv = False

    print(f"\n  FORECAST  {yhat:,.0f} MW")
    if have_iv:
        print(f"  90% PI    {lo90:,.0f} - {hi90:,.0f} MW  "
              f"(+/-{(hi90 - lo90) / 2:,.0f})")
        print(f"  95% PI    {lo95:,.0f} - {hi95:,.0f} MW  "
              f"(+/-{(hi95 - lo95) / 2:,.0f})")
    print(f"\n  horizon   day-ahead with one-day demand-data latency (H={H})")
    print(f"  expected  ~524 MW MAE / ~3.0% MAPE under forecast weather")
    print(f"            (results/operational_gap.json; NOT the 481 MW")
    print(f"             reanalysis figure, which is unservable)")

    record = {
        "target_date": f"{target:%Y-%m-%d}",
        "issue_day": f"{now:%Y-%m-%d}",
        "issued_at": now.isoformat(timespec="seconds"),
        "predicted_mw": round(yhat, 1),
        "lo90": round(lo90, 1), "hi90": round(hi90, 1),
        "lo95": round(lo95, 1), "hi95": round(hi95, 1),
        "H": H,
        "anchor_col": ANCHOR, "anchor_mw": round(anchor_val, 1),
        "weather_source": f"live_log_{slot}_lead_{lead}",
        "weather_issue_slot": slot,
        "weather_issued_on": f"{issued:%Y-%m-%d}",
        "observed_filled_from_forecast": ";".join(
            df.attrs.get("observed_filled_from_forecast") or []),
        "model_version": MODEL_VERSION,
        "model_is_champion": False,
        "n_train_rows": n_train,
        "features_nan": ";".join(missing) if missing else "",
        "actual_mw": "",        # filled later by track_predictions.py
        "error_mw": "",
    }

    if args.dry_run:
        print(f"\n--dry-run: not appending to {PREDICTIONS_LOG}")
    else:
        append_log(record)
        print(f"\nlogged -> {PREDICTIONS_LOG}")


if __name__ == "__main__":
    main()
