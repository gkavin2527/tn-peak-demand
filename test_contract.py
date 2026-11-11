"""
Gate G0 acceptance checks for the Track A / Track B contract.

    python test_contract.py

The roadmap's G0 condition is: "Identity test: swapping ERA5 weather into
swap_weather reproduces the Regime A predictions exactly."

That is the check that matters, because swap_weather is the only thing
standing between the retrospective and operational results. If it perturbs
anything outside the target-day weather columns -- or rebuilds the deltas
differently from build_dataset.py -- then the measured operational gap is
partly an artifact of the swap, and Phase 5's headline number is wrong.

Also checks the H-shift, which is the other way Phase 5 could silently lie:
if load_master(H=2) left any demand feature resolving to target-1, the
operational model would be using data it cannot have.

Train/serve parity (sections 6 and 7)
-------------------------------------
The identity test caught the weather-anomaly gap bug. Two others got through
it and were found only because someone ran predict_next_day.py by hand:

  * all four sr_*_lag1 columns came back NaN on the predicted row, because the
    grid slice publishes a day later than the PSP report and all three source
    groups were being shifted uniformly;
  * swap_weather computed its trailing mean on an index that excluded the
    target date, silently dropping three more features.

Both share a shape: **a feature present during training and absent at
serving**. A model cannot report that - it just quietly stops using the
column. Being found by luck is not a control, so sections 6 and 7 assert it
directly: build the real inference row and require no unexpected NaN, and
check config.H_BY_GROUP against each source's observed freshness so that a
change in upstream publishing fails loudly instead of silently blanking
features.
"""

import threadlimit  # noqa: F401  - must precede numpy/sklearn

import sys

import numpy as np
import pandas as pd

from build_dataset import state_weather
from config import H, MASTER, RAW_WEATHER
from src.contract import (feature_groups, load_master, model_features,
                          swap_weather, verify_no_leakage)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    print("=== G0: contract acceptance ===\n")
    spec = feature_groups()

    # --- 1. feature_groups.json covers master.csv exactly ------------------
    print("feature_groups.json")
    cols = set(pd.read_csv(MASTER, nrows=1).columns) - {"date", spec["target"]}
    grouped = {c for g in spec["groups"].values() for c in g}
    check("every master.csv column is grouped", not (cols - grouped),
          f"ungrouped: {sorted(cols - grouped)}" if cols - grouped else "62/62")
    check("no group names a column master.csv lacks", not (grouped - cols),
          f"phantom: {sorted(grouped - cols)}" if grouped - cols else "")
    dupes = [c for c in grouped
             if sum(c in g for g in spec["groups"].values()) > 1]
    check("no column in two groups", not dupes, str(dupes) if dupes else "")

    # --- 2. load_master ----------------------------------------------------
    print("\nload_master")
    a = load_master(H=1)
    b = load_master(H=H)
    check("indexed by date, target renamed to y",
          a.index.name == "date" and "y" in a.columns)
    check(f"H={H} recorded in attrs", b.attrs.get("H") == H,
          f"lag_offset={b.attrs.get('lag_offset')}")
    check("H=1 is identical to raw master (no shift applied)",
          a.attrs["lag_offset"] == 0)
    shifted = b.attrs["shifted_columns"]
    check("demand/energy/grid columns were shifted", len(shifted) == 24,
          f"{len(shifted)} columns")
    check("calendar and weather were NOT shifted",
          a[spec["groups"]["calendar"][:1] + spec["groups"]["weather_lag"]]
          .equals(b[spec["groups"]["calendar"][:1] + spec["groups"]["weather_lag"]]))

    # --- 3. the H-shift is real, not just named ---------------------------
    print("\nH-shift leakage audit (does not trust column names)")
    try:
        rep1 = verify_no_leakage(a)
        check(f"H=1: no demand feature newer than target-1",
              rep1["min_offset"] >= 1, f"min offset {rep1['min_offset']}")
    except AssertionError as e:
        check("H=1 leakage audit", False, str(e))
    try:
        rep = verify_no_leakage(b)
        check(f"H={H}: no demand feature newer than target-{H}",
              rep["min_offset"] >= H, f"min offset {rep['min_offset']}")
        check("peak_lag_1 resolves one day older under H-shift",
              rep["resolved_offsets"].get("peak_lag_1") == H,
              f"peak_lag_1 -> target-{rep['resolved_offsets'].get('peak_lag_1')}")
    except AssertionError as e:
        check(f"H={H} leakage audit", False, str(e))

    # --- 4. THE identity test ---------------------------------------------
    print("\nG0 identity test: swap_weather(reanalysis) must be a no-op")
    era = state_weather(pd.read_csv(RAW_WEATHER)).set_index("date").sort_index()
    swapped = swap_weather(b, era, observed=era)

    check("shape unchanged", swapped.shape == b.shape,
          f"{b.shape} -> {swapped.shape}")
    check("index unchanged", swapped.index.equals(b.index))

    untouched = [c for c in b.columns
                 if c not in spec["groups"]["weather_target_day"]]
    diffs = []
    for c in untouched:
        if pd.api.types.is_numeric_dtype(b[c]):
            if not np.allclose(b[c].to_numpy(), swapped[c].to_numpy(),
                               atol=0, rtol=0, equal_nan=True):
                diffs.append(c)
        elif not b[c].equals(swapped[c]):
            diffs.append(c)
    check("every non-weather column byte-identical", not diffs,
          f"perturbed: {diffs[:5]}" if diffs else f"{len(untouched)} columns")

    # The weather columns themselves must come back equal too, because the
    # weather we swapped in IS the weather they were built from. Any mismatch
    # means swap_weather's delta/anomaly formulas disagree with
    # build_dataset.py's -- which would contaminate the Regime B numbers.
    wx = [c for c in spec["groups"]["weather_target_day"] if c in b.columns]
    mismatch = {}
    for c in wx:
        o, n = b[c].to_numpy(float), swapped[c].to_numpy(float)
        both = ~(np.isnan(o) | np.isnan(n))
        if both.sum() and not np.allclose(o[both], n[both], atol=1e-6):
            mismatch[c] = float(np.nanmax(np.abs(o[both] - n[both])))
    check("weather columns reproduce build_dataset.py's formulas",
          not mismatch, f"max abs diff: {mismatch}" if mismatch else
          f"{len(wx)} columns agree to 1e-6")

    # --- 5. prediction-level identity -------------------------------------
    print("\nprediction-level identity (the roadmap's literal wording)")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src.contract import run_walk_forward
    feats = model_features(b)
    factory = lambda: HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.1, random_state=0)
    common = dict(start="2026-01-01", end="2026-11-09", refit_days=30,
                  features=feats, differenced=True, anchor="peak_lag_1")
    pa = run_walk_forward(factory, b, **common)
    pb = run_walk_forward(factory, swapped, **common)
    merged = pa.merge(pb, on="date", suffixes=("_a", "_b"))
    worst = float(np.nanmax(np.abs(merged.y_pred_a - merged.y_pred_b)))
    check("Regime A predictions reproduced exactly", worst < 1e-6,
          f"max |delta| = {worst:.3e} MW over {len(merged)} days")

    # --- 6. H_BY_GROUP vs what the sources actually publish ----------------
    # The grid bug in one check: if the relative staleness of the sources ever
    # changes, this fails and tells you to update H_BY_GROUP, instead of the
    # features silently going NaN at inference.
    print("\nsource freshness vs config.H_BY_GROUP")
    from config import H_BY_GROUP, RAW_DEMAND, RAW_PEAK
    pk = pd.read_csv(RAW_PEAK, parse_dates=["date"]).dropna(subset=[spec["target"]])
    dd = pd.read_csv(RAW_DEMAND, parse_dates=["date"])
    newest_demand = pk["date"].max()
    grid_col = next((c for c in ("sr_max_demand_mw", "sr_demand_met_mw")
                     if c in dd.columns), None)
    newest_grid = dd.dropna(subset=[grid_col])["date"].max()
    observed_extra = int((newest_demand - newest_grid).days)
    declared_extra = H_BY_GROUP["grid"] - H_BY_GROUP["demand"]
    check("grid lag behind demand matches H_BY_GROUP",
          observed_extra == declared_extra,
          f"observed {observed_extra}d (peak_demand {newest_demand:%Y-%m-%d} vs "
          f"grid {newest_grid:%Y-%m-%d}), declared {declared_extra}d"
          + ("" if observed_extra == declared_extra else
             "  -> update config.H_BY_GROUP"))

    # --- 7. train/serve parity: the real inference row --------------------
    print("\ntrain/serve parity (builds the actual inference feature row)")
    import predict_next_day as P

    target = newest_demand + pd.Timedelta(days=H)
    try:
        serve, issued, lead, slot = P.build_features(target)
    except SystemExit as e:
        check("inference row builds", False, str(e)[:160])
        serve = None

    if serve is not None:
        train_cols = set(model_features(b))
        serve_cols = set(model_features(serve))
        check("serving frame exposes the same features as training",
              train_cols == serve_cols,
              f"train-only: {sorted(train_cols - serve_cols)[:4]} "
              f"serve-only: {sorted(serve_cols - train_cols)[:4]}"
              if train_cols != serve_cols else f"{len(train_cols)} columns")

        feats = model_features(serve)
        row = serve.loc[[target], feats]
        nan_cols = row.columns[row.isna().any()].tolist()
        check("no feature is NaN on the row being predicted", not nan_cols,
              f"NaN: {nan_cols}" if nan_cols else
              f"all {len(feats)} features populated for {target:%Y-%m-%d}")

        check("anchor is populated", np.isfinite(serve.loc[target, "peak_lag_1"]),
              f"peak_lag_1 = {serve.loc[target, 'peak_lag_1']:,.0f} MW")

        # The yesterday-side of the weather deltas is filled from the forecast
        # log when the archive has not caught up. That is fine at lead 0 (a day
        # that has already happened) and indefensible at lead 3 (a forecast
        # wearing the label "observed"), so the lead actually used is asserted,
        # not assumed. P.MAX_FILL_LEAD is the contract.
        fill_lead = serve.attrs.get("max_fill_lead")
        filled = serve.attrs.get("observed_filled_from_forecast") or []
        check("observed-side fill stays within MAX_FILL_LEAD",
              fill_lead is None or fill_lead <= P.MAX_FILL_LEAD,
              f"filled {filled or 'nothing'}, cap {P.MAX_FILL_LEAD}")

        # the serving row must come from the same construction as training:
        # compare an overlapping historical row built both ways
        probe = b.index[b.index < target][-30]
        common = sorted(train_cols & serve_cols)
        tv = b.loc[probe, common].astype(float).to_numpy()
        sv = serve.loc[probe, common].astype(float).to_numpy()
        both = ~(np.isnan(tv) | np.isnan(sv))
        worst_col, worst = None, 0.0
        if both.sum():
            d = np.abs(tv[both] - sv[both])
            if d.size:
                worst = float(d.max())
                worst_col = [c for c, m in zip(common, both) if m][int(d.argmax())]
        check("serving path reproduces training features on a shared date",
              worst < 1e-6,
              f"{probe:%Y-%m-%d}: max diff {worst:.2e}"
              + (f" on {worst_col}" if worst >= 1e-6 else ""))

    # --- summary ----------------------------------------------------------
    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{'=' * 60}")
    if failed:
        print(f"G0 NOT MET - {len(failed)} of {len(results)} checks failed")
        for _, name, detail in failed:
            print(f"  FAIL  {name}  {detail}")
        sys.exit(1)
    print(f"G0 MET - all {len(results)} checks passed")
    print("Contract is safe to freeze. swap_weather does not perturb anything")
    print(f"outside target-day weather, and H={H} shifting is mechanically verified.")


if __name__ == "__main__":
    main()
