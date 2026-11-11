"""
Calibrate the prediction intervals that predict_next_day.py serves.

    python calibrate_intervals.py

Roadmap D6 / Phase 7.1's dependency. Runs the stand-in model through a
walk-forward over the trailing window, then fits rolling split-conformal
relative residuals per season and writes results/intervals.json.

Intentionally separate from inference. Inference tonight must not refit
anything - it loads these multipliers and applies them. Re-run this weekly
(the workflow does), or whenever the champion changes at G2.
"""

import threadlimit  # noqa: F401  - must precede numpy/sklearn

from sklearn.ensemble import HistGradientBoostingRegressor

from config import H, RANDOM_SEED, REFIT_DAYS, RESULTS
from src.contract import load_master, model_features, run_walk_forward
from src.operational import intervals as IV

CALIB_START = "2024-01-01"      # two years of walk-forward to calibrate on
ANCHOR = "peak_lag_1"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_master(H=H)
    feats = model_features(df)
    end = df.index.max()

    print(f"walk-forward {CALIB_START} -> {end:%Y-%m-%d}, H={H}, "
          f"refit {REFIT_DAYS}d ...")
    preds = run_walk_forward(
        lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                              random_state=RANDOM_SEED),
        df, CALIB_START, end, refit_days=REFIT_DAYS,
        features=feats, differenced=True, anchor=ANCHOR)
    print(f"  {len(preds)} predictions")

    calib = IV.calibrate(preds)
    cov = IV.coverage_report(preds, calib)

    print(f"\ncalibration window: {calib['window'][0]} -> {calib['window'][1]} "
          f"(trailing {calib['window_days']}d)")
    for season, s in calib["seasons"].items():
        widths = "  ".join(f"{lv}: +/-{q * 100:.2f}%"
                           for lv, q in s["relative_halfwidth"].items())
        print(f"  {season:<7} {s['note']:<28} {widths}")

    print("\nrealised coverage on the full walk-forward:")
    for lv, c in cov.items():
        by = "  ".join(f"{k} {v * 100:.1f}%" for k, v in c["by_season"].items())
        print(f"  nominal {float(lv) * 100:.0f}%  realised "
              f"{c['realised_coverage'] * 100:5.1f}%  mean width "
              f"{c['mean_width_mw']:>6.0f} MW   ({by})")
    print("\n  Coverage is approximate by construction - a demand series is not")
    print("  exchangeable, so the conformal guarantee does not transfer exactly.")

    path = IV.save(calib, cov)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
