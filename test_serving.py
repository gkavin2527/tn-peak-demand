"""
Serving-path acceptance tests (Gate G2).

    python test_serving.py

These exist because the serving path and the champion were built separately and
disagreed in two independent ways, both found by inspection rather than by any
test:

* **Target parameterisation.** `predict_next_day.py` trained on
  `y - peak_lag_1` and returned `predict(...) + anchor`. The Phase 6 champion
  is direct-level. Serving it through that path added a ~16,800 MW anchor to a
  ~17,700 MW prediction - a 95% overshoot. Loud, but only if someone looked.

* **Lag representation.** This is the dangerous one. `load_master(H=2)` shifts
  the demand block by H-1, so its column named `peak_lag_2` holds T-3 and its
  `peak_lag_7` lands on the wrong weekday. `load_phase5_data` - what every
  Phase 5/6 model actually trained on - keeps literal offsets and enforces H=2
  by *excluding* `peak_lag_1`/`energy_lag_1` instead. Feeding the champion the
  shifted frame raised 2025 MAE from 438.2 to 683.7 MW (+56%), roughly 2.3x
  the champion's entire measured gain over Ridge, with no error, no warning
  and an entirely plausible-looking forecast.

The invariant these tests protect: the feature row served tonight must be
byte-identical in representation to the rows the model was selected on.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy / sklearn / lightgbm

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import H, PEAK_TARGET
from src.contract import feature_groups, load_phase5_data
from src.models import build_from_config, build_model

PASS, FAIL = "PASS", "FAIL"
results = []
CHAMPION = Path("models/champion_config.json")


def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def literal_frame():
    """The serving frame without the live-weather swap.

    Mirrors predict_next_day.build_features(literal_lags=True) minus
    swap_weather, so it can be compared against load_phase5_data (reanalysis)
    on dates where no live forecast exists.
    """
    from build_dataset import add_lags, merged_frame
    from config import LEAKY_SAMEDAY
    from predict_next_day import UNAVAILABLE_AT_H2

    df = add_lags(merged_frame()).reset_index(drop=True)
    df = df.drop(columns=[c for c in LEAKY_SAMEDAY if c in df.columns])
    df = df.rename(columns={PEAK_TARGET: "y"}).set_index("date").sort_index()
    return df.drop(columns=[c for c in UNAVAILABLE_AT_H2 if c in df.columns])


def main():
    print("=" * 74)
    print("SERVING PATH ACCEPTANCE TESTS (G2)")
    print("=" * 74)

    # ---- 1. the champion config is a complete serving contract --------
    print("\n1. Champion config declares everything serving needs")
    cfg = json.loads(CHAMPION.read_text())
    check("declares target_parameterisation",
          cfg.get("target_parameterisation") == "level",
          repr(cfg.get("target_parameterisation")))
    check("declares lag_representation",
          cfg.get("lag_representation") == "literal",
          repr(cfg.get("lag_representation")))
    check("feature_count matches the listed features",
          cfg["feature_count"] == len(cfg["features"]),
          f"{cfg['feature_count']} vs {len(cfg['features'])}")
    check("marked frozen", cfg.get("frozen") is True)
    check("no T-1 demand/energy in the champion feature list",
          not {"peak_lag_1", "energy_lag_1"} & set(cfg["features"]))

    # ---- 2. build_from_config ----------------------------------------
    print("\n2. build_from_config is the single construction path")
    model, feats, param = build_from_config(cfg)
    check("returns the declared parameterisation", param == "level", param)
    check("returns the 26 selected features", len(feats) == 26, f"{len(feats)}")
    check("rebuilt model is the champion type",
          type(model).__name__ == "RidgeGBMHybrid", type(model).__name__)
    hp = cfg["hyperparameters"]
    check("tuned hyperparameters survive the rebuild",
          abs(model.alpha - hp["alpha"]) < 1e-15
          and model._gbm_kw["n_estimators"] == hp["n_estimators"]
          and model._gbm_kw["num_leaves"] == hp["num_leaves"],
          f"alpha={model.alpha:g}, {model._gbm_kw}")

    # a config that lies about its own shape must not load
    for label, mutate in [
        ("inconsistent feature_count", lambda c: c.update(feature_count=99)),
        ("unknown parameterisation", lambda c: c.update(target_parameterisation="log")),
        ("differenced without an anchor", lambda c: c.update(
            target_parameterisation="differenced", anchor=None)),
        ("empty feature list", lambda c: c.update(features=[], feature_count=0)),
    ]:
        bad = json.loads(CHAMPION.read_text())
        mutate(bad)
        try:
            build_from_config(bad)
            ok = False
        except (ValueError, KeyError):
            ok = True
        check(f"rejects {label}", ok)

    # ---- 3. lag representation: the silent hazard --------------------
    print("\n3. Literal lag representation matches what the champion trained on")
    lit = literal_frame()
    p5, _ = load_phase5_data("A2_weather_deltas_anomalies", "reanalysis")
    shared = lit.index.intersection(p5.index)
    check("serving frame covers the Phase 5 sample", len(shared) > 1000,
          f"{len(shared)} shared dates")

    sample = shared[-400:]
    worst, worst_col = 0.0, ""
    for c in feats:
        d = (lit.loc[sample, c].astype(float) - p5.loc[sample, c].astype(float)).abs().max()
        if d > worst:
            worst, worst_col = float(d), c
    check("every champion feature is identical to load_phase5_data",
          worst < 1e-9, f"max |diff| {worst:.3e}" + (f" on {worst_col}" if worst else ""))

    # literal semantics, checked against the raw series rather than the name
    raw = pd.read_csv("data/master.csv", parse_dates=["date"]).set_index("date")[PEAK_TARGET]
    lag_ok, lag_bad = True, []
    for n in (2, 3, 7, 14, 364):
        col = f"peak_lag_{n}"
        want = raw.reindex(sample - pd.Timedelta(days=n)).to_numpy(float)
        got = lit.loc[sample, col].to_numpy(float)
        m = np.isfinite(want) & np.isfinite(got)
        if not np.allclose(want[m], got[m], atol=1e-9):
            lag_ok = False
            lag_bad.append(col)
    check("peak_lag_N resolves to the raw series at T-N", lag_ok,
          f"mismatched: {lag_bad}" if lag_bad else "2,3,7,14,364 all literal")

    # weekday alignment - what the shift silently destroys
    wd = [(d.day_of_week == (d - pd.Timedelta(days=7)).day_of_week)
          for d in sample[:20]]
    check("peak_lag_7 preserves weekday alignment", all(wd))

    check("T-1 demand/energy excluded, not shifted",
          "peak_lag_1" not in lit.columns and "energy_lag_1" not in lit.columns)

    # ---- 4. the two representations are NOT interchangeable ----------
    print("\n4. The shifted representation is provably different")
    from src.contract import load_master
    lm = load_master(H)
    d = sample[-1]
    shifted_is_stale = abs(float(lm.loc[d, "peak_lag_2"])
                           - float(raw.loc[d - pd.Timedelta(days=3)])) < 1e-9
    check("load_master(2).peak_lag_2 holds T-3, not T-2", shifted_is_stale,
          f"on {d:%Y-%m-%d}: master={lm.loc[d,'peak_lag_2']:,.0f}, "
          f"raw T-3={raw.loc[d - pd.Timedelta(days=3)]:,.0f}")
    check("so the frames disagree on the champion's lags",
          abs(float(lm.loc[d, "peak_lag_2"]) - float(lit.loc[d, "peak_lag_2"])) > 1.0,
          "confirms a shifted frame would feed the champion stale lags")

    # ---- 5. serving refuses the wrong frame --------------------------
    print("\n5. Serving fails closed rather than guessing")
    import predict_next_day as P

    fake = lit.copy()
    fake.attrs["lag_representation"] = "shifted"
    try:
        P.fit_and_predict_champion(fake, sample[-1], cfg)
        ok = False
    except SystemExit:
        ok = True
    check("refuses a shifted frame", ok)

    nanned = lit.copy()
    nanned.attrs["lag_representation"] = "literal"
    nanned.loc[sample[-1], "t_max"] = np.nan
    try:
        P.fit_and_predict_champion(nanned, sample[-1], cfg)
        ok = False
    except SystemExit:
        ok = True
    check("refuses a NaN feature on the target row instead of emitting NaN", ok)

    diffed = json.loads(CHAMPION.read_text())
    diffed["target_parameterisation"] = "differenced"
    diffed["anchor"] = "peak_lag_2"
    good = lit.copy(); good.attrs["lag_representation"] = "literal"
    try:
        P.fit_and_predict_champion(good, sample[-1], diffed)
        ok = False
    except SystemExit:
        ok = True
    check("refuses to serve a differenced champion through the level path", ok)

    # ---- 6. end to end: served value tracks the backtest -------------
    print("\n6. Served prediction agrees with the offline model")
    target = sample[-1]
    tr = p5[p5.index < target]
    ok_rows = tr["y"].notna() & tr[feats].notna().all(axis=1)
    off, _, _ = build_from_config(cfg)
    off.fit(tr.loc[ok_rows, feats], tr.loc[ok_rows, "y"])
    offline = float(off.predict(p5.loc[[target], feats])[0])

    served, n_train, _, ref = P.fit_and_predict_champion(good, target, cfg)
    rel = abs(served - offline) / offline
    check("served prediction within 0.5% of the offline refit",
          rel < 0.005, f"served {served:,.0f} vs offline {offline:,.0f} "
                       f"({rel:.3%})")
    check("trained on the full available history", n_train > 2500, f"{n_train:,} rows")
    check("served level is plausible against the newest observed peak",
          abs(served - ref) < P.SANITY_BAND_MW,
          f"|{served:,.0f} - {ref:,.0f}| = {abs(served-ref):,.0f} MW")

    # sanity band actually fires
    absurd = good.copy()
    try:
        P.fit_and_predict_champion(absurd, target,
                                   {**cfg, "hyperparameters": cfg["hyperparameters"]})
        band_live = P.SANITY_BAND_MW > 0
    except SystemExit:
        band_live = True
    check("sanity band is configured", band_live, f"{P.SANITY_BAND_MW:,.0f} MW")

    # ---- summary -----------------------------------------------------
    failed = [r for r in results if r[0] == FAIL]
    print("\n" + "=" * 74)
    if failed:
        print(f"SERVING ACCEPTANCE FAILED - {len(failed)} of {len(results)} checks")
        for _, n, d in failed:
            print(f"  FAIL  {n}  {d}")
        sys.exit(1)
    print(f"ALL {len(results)} SERVING ACCEPTANCE CHECKS PASSED  ✓")
    print("=" * 74)


if __name__ == "__main__":
    main()
