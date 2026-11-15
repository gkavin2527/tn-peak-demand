"""
Calibrate the prediction intervals that predict_next_day.py serves.

    python calibrate_intervals.py

Roadmap D6 / Phase 7.1. Walks the CHAMPION forward over a trailing window,
then fits rolling split-conformal relative residuals per season and writes
results/intervals.json.

W2-3 rewrote this. It previously calibrated a HistGradientBoostingRegressor
stand-in, differenced against peak_lag_1, on a load_master() frame - none of
which is the champion. Intervals fitted on a different model's residuals do
not describe the forecast being served, and the width is the part users act
on.

Two things that must stay right, both of which the old version got wrong for
the champion:

  * TRAIN on Regime A, SCORE on Regime B. The Regime B frame only spans the
    forecast archive (2021-06 onward), so training on it would cut the
    training set to a fraction of the available history - the same trap
    documented in run_phase6_benchmark.run_walk_forward. Residuals must come
    from forecast weather, because that is what inference faces.
  * LITERAL lags and a LEVEL target, read from champion_config.json rather
    than assumed. See test_serving.py.

Intentionally separate from inference. Inference tonight must not refit
anything - it loads these multipliers and applies them. Re-run weekly (the
workflow does), or whenever the champion changes.
"""

import threadlimit  # noqa: F401  - must precede numpy/sklearn/lightgbm

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import REFIT_DAYS, RESULTS
from src.contract import PHASE5_FEATURE_SETS, load_phase5_data
from src.envinfo import capture_environment
from src.models import build_from_config, build_model
from src.operational import intervals as IV

CALIB_START = "2024-01-01"      # trailing window to calibrate on
CHAMPION = Path("models/champion_config.json")


def champion_walk_forward(cfg, train_df, score_df, feats, start, end,
                          refit_days=REFIT_DAYS):
    """Expanding-window walk-forward with periodic refit, train A / score B.

    Returns run_walk_forward's contract columns (date, y_true, y_pred,
    refit_date) so src.operational.intervals consumes it unchanged.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    dates = score_df.index[(score_df.index >= start) & (score_df.index <= end)]
    dates = dates[score_df.loc[dates, feats + ["y"]].notna().all(axis=1)]
    if len(dates) == 0:
        raise SystemExit("no scorable dates in the calibration window - is the "
                         "forecast archive populated?")

    rows, model, refit_at = [], None, None
    for d in dates:
        if model is None or (d - refit_at).days >= refit_days:
            # Strictly past data only. Training weather is ERA5 throughout;
            # only the scored row's weather is a forecast.
            tr = train_df[train_df.index < d]
            ok = tr["y"].notna() & tr[feats].notna().all(axis=1)
            if int(ok.sum()) < 500:
                continue
            model = build_model(cfg["model_type"],
                                **cfg["hyperparameters"]).fit(
                tr.loc[ok, feats], tr.loc[ok, "y"])
            refit_at = d
        yhat = float(model.predict(score_df.loc[[d], feats])[0])
        rows.append({"date": d, "y_true": float(score_df.loc[d, "y"]),
                     "y_pred": yhat, "refit_date": refit_at})
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CHAMPION.read_text())
    _, feats, param = build_from_config(cfg)
    if param != "level":
        raise SystemExit(f"champion is {param}; this calibrator serves 'level'")
    if cfg.get("lag_representation") != "literal":
        raise SystemExit("champion is not on the literal lag representation")

    fset = cfg["feature_set"]
    assert feats == list(PHASE5_FEATURE_SETS[fset]), "feature list drifted"

    train_df, _ = load_phase5_data(fset, weather_source="reanalysis")
    score_df, _ = load_phase5_data(fset, weather_source="forecast",
                                   forecast_lead=1)
    end = score_df.index.max()

    print(f"champion    {cfg['model_type']}  {fset} ({len(feats)} features)")
    print(f"train       Regime A (ERA5), {len(train_df):,} rows available")
    print(f"score       Regime B (archived lead-1 forecast)")
    print(f"walk-forward {CALIB_START} -> {end:%Y-%m-%d}, refit {REFIT_DAYS}d ...")

    preds = champion_walk_forward(cfg, train_df, score_df, feats,
                                  CALIB_START, end)
    err = preds.y_pred - preds.y_true
    print(f"  {len(preds)} predictions, "
          f"{preds.refit_date.nunique()} refits")
    print(f"  MAE {err.abs().mean():.1f} MW   bias {err.mean():+.1f} MW")

    calib = IV.calibrate(preds)
    cov = IV.coverage_report(preds, calib)

    print(f"\ncalibration window: {calib['window'][0]} -> {calib['window'][1]}")
    for s, v in calib["seasons"].items():
        hw = v["relative_halfwidth"]
        print(f"  {s:<8} n={v['n']:<4} "
              + "  ".join(f"{lvl}: +/-{q:.2%}" for lvl, q in hw.items()))

    print("\ncoverage (realised vs nominal - within +/-3 pts is the bar):")
    ok_all = True
    for lvl, c in cov.items():
        gap = 100 * (c["realised_coverage"] - c["nominal"])
        flag = "OK" if abs(gap) <= 3 else "OFF"
        ok_all &= abs(gap) <= 3
        print(f"  {lvl}: realised {c['realised_coverage']:.1%} "
              f"(nominal {c['nominal']:.0%}, {gap:+.1f} pts)  "
              f"mean width {c['mean_width_mw']:,.0f} MW  [{flag}]")

    calib["model"] = {
        "champion": cfg["model_type"],
        "feature_set": fset,
        "hyperparameters": cfg["hyperparameters"],
        "target_parameterisation": param,
        "lag_representation": cfg["lag_representation"],
        "config_generated_at": cfg.get("generated_at"),
    }
    calib["walk_forward"] = {
        "train_regime": "A (ERA5)", "score_regime": "B (lead-1 forecast)",
        "start": CALIB_START, "end": f"{end:%Y-%m-%d}",
        "refit_days": REFIT_DAYS, "n_predictions": int(len(preds)),
        "n_refits": int(preds.refit_date.nunique()),
        "mae_mw": float(err.abs().mean()), "bias_mw": float(err.mean()),
        "note": ("training uses ERA5 throughout; only the scored row's "
                 "weather is a forecast, matching how the model is served"),
    }
    calib["coverage"] = cov
    calib["coverage_within_3pts"] = bool(ok_all)

    # ---- a second, 2026-free calibration for the paper ---------------
    #
    # The operational calibration above uses a trailing 365-day window, which
    # is correct for SERVING - you want interval widths set by recent
    # residuals. But that window now reaches into 2026, and 2026 is the
    # one-shot confirmation set (W0-5). Quoting 2026 interval coverage against
    # widths fitted on 2026 would be circular.
    #
    # So a parallel calibration is fitted on residuals up to 2025-12-31 only.
    # W0-5 evaluates 2026 coverage against THESE widths, which makes it a
    # genuine out-of-sample coverage claim. Serving continues to use the
    # operational set.
    paper_cut = pd.Timestamp("2025-12-31")
    paper_preds = preds[pd.to_datetime(preds.date) <= paper_cut]
    if len(paper_preds) >= 120:
        paper_calib = IV.calibrate(paper_preds)
        paper_cov = IV.coverage_report(paper_preds, paper_calib)
        print(f"\npaper calibration (<= {paper_cut:%Y-%m-%d}, keeps 2026 clean):")
        print(f"  window {paper_calib['window'][0]} -> {paper_calib['window'][1]}, "
              f"n={len(paper_preds)}")
        for s_, v in paper_calib["seasons"].items():
            print(f"  {s_:<8} n={v['n']:<4} " + "  ".join(
                f"{lvl}: +/-{q:.2%}"
                for lvl, q in v["relative_halfwidth"].items()))
        calib["paper_calibration_2026_free"] = {
            "why": ("operational widths use a trailing window that now "
                    "includes 2026, the one-shot confirmation set; these "
                    "widths are fitted on residuals up to 2025-12-31 so that "
                    "W0-5 can report 2026 coverage out-of-sample"),
            "cutoff": f"{paper_cut:%Y-%m-%d}",
            "n_predictions": int(len(paper_preds)),
            "window": paper_calib["window"],
            "seasons": paper_calib["seasons"],
            "in_window_coverage": paper_cov,
            "use_for": "W0-5 2026 coverage reporting",
        }
    else:
        print(f"\nWARNING  only {len(paper_preds)} predictions on or before "
              f"{paper_cut:%Y-%m-%d}; skipping the 2026-free calibration")

    calib["environment"] = capture_environment()

    IV.save(calib) if hasattr(IV, "save") else (
        (RESULTS / "intervals.json").write_text(
            json.dumps(calib, indent=2, default=str) + "\n"))
    preds.to_csv(RESULTS / "intervals_walkforward_predictions.csv", index=False)
    print(f"\nwrote {RESULTS / 'intervals.json'}")
    print(f"      {RESULTS / 'intervals_walkforward_predictions.csv'}")
    if not ok_all:
        print("\nWARNING  at least one level is off nominal by >3 points - "
              "report the realised figure, do not quote the nominal one")


if __name__ == "__main__":
    main()
