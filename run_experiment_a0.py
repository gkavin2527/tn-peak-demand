"""
Experiment A0: Faithful Weather-Free Track A Baseline Reproduction & Integration.

Phase 5, Step 1 & 2.
Evaluates the verified Track A Ridge model (alpha=0.001) under two protocols:
1. Protocol 1 (Track A Exact Protocol):
   Train: 2014-03-30 to 2025-12-31 (trainval, 4,234 rows)
   Test : 2026-01-01 to 2026-11-09 (313 rows, matching historical benchmark)
          plus full 2026 window through 2026-11-10 (314 rows)
2. Protocol 2 (Track B Operational Protocol):
   Weekly-refit expanding walk-forward (refit_days=7) from 2026-01-01 onward,
   providing the official weather-free operational baseline (M0) for Phase 5.

Features: Exactly 19 leakage-free features (Calendar + Lags H>=2). Zero weather features.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import MASTER, RESULTS
from src.contract import load_phase5_data, verify_phase5_integrity
from src.operational import metrics as M
from src.operational.walkforward import run_walk_forward
from src.ridge import TrackARidge

# --------------------------------------------------------------------------
# 1. Feature Contract Definition (Verified Track A, H=2)
# --------------------------------------------------------------------------

CALENDAR = [
    "dow", "is_weekend", "is_holiday", "day_before_holiday",
    "day_after_holiday", "pongal_window",
    "sin1", "cos1", "sin2", "cos2", "trend",
]
PEAK_LAGS = ["peak_lag_2", "peak_lag_3", "peak_lag_7", "peak_lag_14", "peak_lag_364"]
ENERGY_LAGS = ["energy_lag_2", "energy_lag_7", "energy_lag_14"]

FEATURES = CALENDAR + PEAK_LAGS + ENERGY_LAGS   # Exactly 19 features
TARGET = "tn_peak_demand_mw"

HISTORICAL_BENCHMARK = {
    "protocol": "Track A Static Hold-Out (2026-01-01 to 2026-11-09)",
    "n_trainval": 4234,
    "n_test": 313,
    "mae": 733.3797,
    "rmse": 929.7411,
    "mape": 3.9883,
    "r2": 0.7092,
}


def run_integrity_checks(df: pd.DataFrame, features: list[str]) -> None:
    """Rigorous pre-execution leakage and feature contract assertions."""
    print("=" * 72)
    print("RUNNING AUTOMATED INTEGRITY CHECKS")
    print("=" * 72)

    # 1. Zero lag-1 features
    lag1 = [f for f in features if re.search(r"(_lag_?1$|_lag_?1_)", f)]
    assert len(lag1) == 0, f"Integrity Failure: Found forbidden lag-1 features: {lag1}"
    print("✓ Check 1: Zero lag-1 features present (H=2 horizon satisfied).")

    # 2. Zero weather features
    weather_keywords = ["temp", "t_max", "t_min", "t_mean", "rh", "app_t", "cdd", "thi", "hours_above"]
    weather_found = [f for f in features if any(w in f for w in weather_keywords)]
    assert len(weather_found) == 0, f"Integrity Failure: Weather features found: {weather_found}"
    print("✓ Check 2: Zero weather features present (Weather-free A0 contract satisfied).")

    # 3. Exactly 19 features
    assert len(features) == 19, f"Expected 19 features, got {len(features)}"
    print("✓ Check 3: Exactly 19 predictive features configured.")

    # 4. Strict monotonic chronological ordering
    assert df["date"].is_monotonic_increasing, "Integrity Failure: Dates not strictly monotonic!"
    print("✓ Check 4: Strictly monotonic chronological dates.")

    # 5. Zero missing values
    assert df[features + [TARGET]].isna().sum().sum() == 0, "Null values detected in modelling set!"
    print("✓ Check 5: 100% complete dataset with zero missing values.")
    print()


def main():
    print("=" * 72)
    print("EXPERIMENT A0: WEATHER-FREE TRACK A BASELINE REPRODUCTION & INTEGRATION")
    print("=" * 72)

    # Ingest through canonical Phase 5 dataset gateway
    df, features = load_phase5_data(feature_set="A0_weather_free", weather_source="none")
    rep = verify_phase5_integrity(df, features, H=2)
    print("Canonical Phase 5 Gateway: Loaded and verified successfully  ✓")
    print(f"  Feature set          : {df.attrs.get('feature_set')}")
    print(f"  Lag representation   : {df.attrs.get('lag_representation')}")
    print(f"  Information horizon  : H={df.attrs.get('H')} ({df.attrs.get('newest_demand')})")
    print()

    year = df["date"].dt.year
    trainval = df[year <= 2025].reset_index(drop=True)
    test_hist = df[(year == 2026) & (df["date"] <= "2026-11-09")].reset_index(drop=True)
    test_full = df[year == 2026].reset_index(drop=True)

    print(f"Dataset summary:")
    print(f"  Total clean rows : {len(df):,} ({df['date'].min().date()} → {df['date'].max().date()})")
    print(f"  TrainVal rows    : {len(trainval):,} ({trainval['date'].min().date()} → {trainval['date'].max().date()})")
    print(f"  Test (historical): {len(test_hist):,} rows ({test_hist['date'].min().date()} → {test_hist['date'].max().date()})")
    print(f"  Test (full 2026) : {len(test_full):,} rows ({test_full['date'].min().date()} → {test_full['date'].max().date()})")
    print()

    # ----------------------------------------------------------------------
    # Protocol 1: Track A Static Retrain on 2014-2025 -> 2026 Test
    # ----------------------------------------------------------------------
    print("-" * 72)
    print("PROTOCOL 1: TRACK A STATIC RETRAIN ON 2014-2025 -> 2026 TEST")
    print("-" * 72)

    model_p1 = TrackARidge(alpha=0.001)
    model_p1.fit(trainval[FEATURES], trainval[TARGET])

    # 1A. Historical test window (313 dates through 2026-11-09)
    preds_hist = model_p1.predict(test_hist[FEATURES])
    df_eval_hist = pd.DataFrame({
        "date": test_hist["date"],
        "y_true": test_hist[TARGET].to_numpy(),
        "y_pred": preds_hist
    })
    res_p1_hist = M.evaluate(df_eval_hist, label="Track A Static 2026 (Historical 313)")

    # 1B. Full current test window (314 dates through 2026-11-10)
    preds_full = model_p1.predict(test_full[FEATURES])
    df_eval_full = pd.DataFrame({
        "date": test_full["date"],
        "y_true": test_full[TARGET].to_numpy(),
        "y_pred": preds_full
    })
    res_p1_full = M.evaluate(df_eval_full, label="Track A Static 2026 (Full 314)")

    print(f"\nHistorical Benchmark Comparison (313 rows, 2026-01-01 → 2026-11-09):")
    fmt = "  {:<12} {:>14} {:>14} {:>14}"
    print(fmt.format("Metric", "Historical", "A0 (Current)", "Delta"))
    print("  " + "-" * 56)
    d_mae = res_p1_hist["overall"]["mae"] - HISTORICAL_BENCHMARK["mae"]
    d_rmse = res_p1_hist["overall"]["rmse"] - HISTORICAL_BENCHMARK["rmse"]
    d_mape = res_p1_hist["overall"]["mape"] - HISTORICAL_BENCHMARK["mape"]
    d_r2 = res_p1_hist["overall"]["r2"] - HISTORICAL_BENCHMARK["r2"]
    print(fmt.format("MAE (MW)", f"{HISTORICAL_BENCHMARK['mae']:.2f}", f"{res_p1_hist['overall']['mae']:.2f}", f"{d_mae:+.2f}"))
    print(fmt.format("RMSE (MW)", f"{HISTORICAL_BENCHMARK['rmse']:.2f}", f"{res_p1_hist['overall']['rmse']:.2f}", f"{d_rmse:+.2f}"))
    print(fmt.format("MAPE (%)", f"{HISTORICAL_BENCHMARK['mape']:.2f}%", f"{res_p1_hist['overall']['mape']:.2f}%", f"{d_mape:+.2f}%"))
    print(fmt.format("R²", f"{HISTORICAL_BENCHMARK['r2']:.4f}", f"{res_p1_hist['overall']['r2']:.4f}", f"{d_r2:+.4f}"))
    print()

    # ----------------------------------------------------------------------
    # Protocol 2: Track B Weekly-Refit Operational Walk-Forward
    # ----------------------------------------------------------------------
    print("-" * 72)
    print("PROTOCOL 2: TRACK B WEEKLY-REFIT OPERATIONAL WALK-FORWARD (2026)")
    print("-" * 72)

    df_wf = df[["date"] + FEATURES + [TARGET]].rename(columns={TARGET: "y"}).set_index("date")

    wf_preds = run_walk_forward(
        model_factory=lambda: TrackARidge(alpha=0.001),
        df=df_wf,
        start=test_full["date"].min(),
        end=test_full["date"].max(),
        refit_days=7,
        features=FEATURES,
        differenced=False
    )

    res_wf = M.evaluate(wf_preds, label="Ridge Weather-Free Weekly Refit (2026)")
    print("\nOverall Walk-Forward Metrics:")
    print(f"  Predictions : {res_wf['overall']['n']}")
    print(f"  MAE         : {res_wf['overall']['mae']:.2f} MW (95% CI: {res_wf['overall']['mae_ci95']})")
    print(f"  RMSE        : {res_wf['overall']['rmse']:.2f} MW")
    print(f"  MAPE        : {res_wf['overall']['mape']:.2f}%")
    print(f"  Bias        : {res_wf['overall']['bias']:+.2f} MW")
    print(f"  R²          : {res_wf['overall']['r2']:.4f}")

    if "summer_mar_jun" in res_wf:
        print(f"  Summer MAE  : {res_wf['summer_mar_jun']['mae']:.2f} MW (n={res_wf['summer_mar_jun']['n']})")
    if "surge_per_year" in res_wf:
        print(f"  Surge MAE   : {res_wf['surge_per_year']['mae']:.2f} MW (n={res_wf['surge_per_year']['n']})")

    # ----------------------------------------------------------------------
    # Save Artifacts
    # ----------------------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    out_json = RESULTS / "experiment_a0_results.json"
    out_csv = RESULTS / "experiment_a0_predictions.csv"

    # Merge predictions for output CSV
    pred_export = pd.DataFrame({
        "date": test_full["date"],
        "actual_mw": test_full[TARGET],
        "pred_static_mw": preds_full,
        "error_static_mw": preds_full - test_full[TARGET],
    }).merge(
        wf_preds[["date", "y_pred", "refit_date"]].rename(
            columns={"y_pred": "pred_walkforward_mw"}),
        on="date", how="left"
    )
    pred_export["error_walkforward_mw"] = pred_export["pred_walkforward_mw"] - pred_export["actual_mw"]
    pred_export.to_csv(out_csv, index=False)

    summary_payload = {
        "experiment": "A0_weather_free_ridge_baseline",
        "horizon_H": 2,
        "features_count": len(FEATURES),
        "features": FEATURES,
        "model": "TrackARidge(alpha=0.001)",
        "protocol_1_static_historical_313": res_p1_hist,
        "protocol_1_static_full_314": res_p1_full,
        "protocol_2_walkforward_weekly_refit": res_wf,
        "benchmark_comparison": {
            "historical_benchmark": HISTORICAL_BENCHMARK,
            "reproduced_mae": res_p1_hist["overall"]["mae"],
            "delta_mae_mw": d_mae,
            "delta_mae_pct": (d_mae / HISTORICAL_BENCHMARK["mae"]) * 100,
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("EXPERIMENT A0 COMPLETE — ARTIFACTS WRITTEN:")
    print(f"  Metrics JSON : {out_json}")
    print(f"  Predictions  : {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()

