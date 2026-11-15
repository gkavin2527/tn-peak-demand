"""
Phase 5 — Experiment A1: Weather Levels Feature Evaluation and Selection.

Evaluates basic temperature-related weather features to determine whether
weather levels improve next-day Tamil Nadu peak-demand forecasting over A0.

Methodology Rules:
1. H=2 operational information constraint preserved.
2. Canonical gateway load_phase5_data() and verify_phase5_integrity() used.
3. Feature selection strictly uses historical walk-forward validation folds:
   - Fold 1: Val 2023 (Train <= 2022)
   - Fold 2: Val 2024 (Train <= 2023)
   - Fold 3: Val 2025 (Train <= 2024)
4. Single selection metric: Validation MAE across the 3 historical folds.
5. Humidity-dependent features strictly excluded (rh_mean, app_t_*, thi, cdd_apparent).
6. 2026 performance reported solely for diagnostic comparison (not for selection).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS
from src.contract import (
    PHASE5_FEATURE_SETS,
    load_phase5_data,
    verify_phase5_integrity,
)
from src.ridge import TrackARidge

# --------------------------------------------------------------------------
# Candidate Weather Feature Sets
# --------------------------------------------------------------------------

BASE_FEATURES = PHASE5_FEATURE_SETS["A0_weather_free"]

CANDIDATES = [
    ("A0: Baseline (No Weather)", []),
    ("A1.1: + t_max", ["t_max"]),
    ("A1.2: + t_mean", ["t_mean"]),
    ("A1.3: + t_min", ["t_min"]),
    ("A1.4: + cdd_max", ["cdd_max"]),
    ("A1.5: + hours_above_32", ["hours_above_32"]),
    ("A1.6: + (t_max, t_min)", ["t_max", "t_min"]),
    ("A1.7: + (t_max, cdd_max)", ["t_max", "cdd_max"]),
    ("A1.8: + (t_max, hours_above_32)", ["t_max", "hours_above_32"]),
    ("A1.9: + (cdd_max, hours_above_32)", ["cdd_max", "hours_above_32"]),
    ("A1.10: + (t_max, t_min, hours_above_32)", ["t_max", "t_min", "hours_above_32"]),
    ("A1.11: + (t_max, t_min, cdd_max)", ["t_max", "t_min", "cdd_max"]),
    ("A1.12: Full A1 (All 5 Weather Levels)", ["t_max", "t_mean", "t_min", "cdd_max", "hours_above_32"]),
]

FOLDS = [
    {"name": "Fold 1 (Val 2023)", "train_end": 2022, "val_year": 2023},
    {"name": "Fold 2 (Val 2024)", "train_end": 2023, "val_year": 2024},
    {"name": "Fold 3 (Val 2025)", "train_end": 2024, "val_year": 2025},
]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    ss_r = float(np.sum((y_true - y_pred) ** 2))
    ss_t = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2)


def main():
    print("=" * 80)
    print("PHASE 5 — EXPERIMENT A1: WEATHER LEVELS EVALUATION & SELECTION")
    print("=" * 80)

    # 1. Ingest Full Reanalysis Table via Canonical Gateway
    # Regime A reanalysis is used for feature selection because it covers the full
    # historical validation span (2014-2025) under stationary atmospheric physics.
    df_full, full_a1_feats = load_phase5_data(
        feature_set="A1_weather_levels",
        weather_source="reanalysis",
        clean_na=True
    )
    
    # Run pre-execution mechanical assertions
    rep = verify_phase5_integrity(df_full, full_a1_feats, H=2)
    print("Canonical Gateway Pre-Execution Integrity Check: PASSED  ✓")
    print(f"  Total Clean Rows    : {len(df_full):,} ({df_full.index.min().date()} → {df_full.index.max().date()})")
    print(f"  Base Features       : {len(BASE_FEATURES)} (A0 weather-free contract)")
    print(f"  Full A1 Features    : {len(full_a1_feats)} (A0 + 5 weather levels)")
    print()

    # 2. Historical Walk-Forward Validation Feature Selection
    print("-" * 80)
    print("SECTION 1: HISTORICAL WALK-FORWARD VALIDATION (2023, 2024, 2025)")
    print("Selection Rule: Lowest Mean Validation MAE Across Folds 1, 2, 3")
    print("-" * 80)

    year = df_full.index.year
    ablation_records = []
    baseline_val_mae = None

    fmt_hdr = f"{'Candidate Set':<38} {'F1 (2023)':>10} {'F2 (2024)':>10} {'F3 (2025)':>10} {'Mean MAE':>10} {'Delta A0':>10}"
    print(fmt_hdr)
    print("-" * 92)

    for name, wx_cols in CANDIDATES:
        cur_feats = BASE_FEATURES + wx_cols
        fold_metrics = []
        for f in FOLDS:
            tr = df_full[year <= f["train_end"]]
            va = df_full[year == f["val_year"]]

            model = TrackARidge(alpha=0.001)
            model.fit(tr[cur_feats], tr["y"])
            preds = model.predict(va[cur_feats])
            m = evaluate_predictions(va["y"].to_numpy(), preds)
            fold_metrics.append(m)

        f1_mae = fold_metrics[0]["mae"]
        f2_mae = fold_metrics[1]["mae"]
        f3_mae = fold_metrics[2]["mae"]
        mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
        mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
        mean_mape = float(np.mean([m["mape"] for m in fold_metrics]))
        mean_r2 = float(np.mean([m["r2"] for m in fold_metrics]))

        if baseline_val_mae is None:
            baseline_val_mae = mean_mae
        delta = mean_mae - baseline_val_mae

        ablation_records.append({
            "candidate": name,
            "weather_features": wx_cols,
            "total_features": len(cur_feats),
            "f1_2023_mae": f1_mae,
            "f2_2024_mae": f2_mae,
            "f3_2025_mae": f3_mae,
            "mean_val_mae": mean_mae,
            "mean_val_rmse": mean_rmse,
            "mean_val_mape": mean_mape,
            "mean_val_r2": mean_r2,
            "delta_vs_a0_mw": delta,
            "delta_vs_a0_pct": (delta / baseline_val_mae) * 100,
        })

        print(f"{name:<38} {f1_mae:>10.2f} {f2_mae:>10.2f} {f3_mae:>10.2f} {mean_mae:>10.2f} {delta:>+10.2f}")

    # Determine winning feature set from validation MAE only
    best_candidate = min(ablation_records, key=lambda x: x["mean_val_mae"])
    print("-" * 92)
    print(f"WINNING CANDIDATE ON VALIDATION FOLDS: {best_candidate['candidate']}")
    print(f"  Mean Val MAE : {best_candidate['mean_val_mae']:.2f} MW ({best_candidate['delta_vs_a0_mw']:+.2f} MW / {best_candidate['delta_vs_a0_pct']:+.1f}%)")
    print()

    # 3. Diagnostic Test Evaluation (Year 2026)
    # Strictly for diagnostic reporting — not used for feature selection
    print("-" * 80)
    print("SECTION 2: 2026 DIAGNOSTIC TEST EVALUATION (STATIC RETRAIN 2014-2025)")
    print("Note: Reported for verification; NOT used for feature selection")
    print("-" * 80)

    tr_2025 = df_full[year <= 2025]
    te_2026_hist = df_full[(year == 2026) & (df_full.index <= "2026-11-09")]
    te_2026_full = df_full[year == 2026]

    # Evaluate A0 on 2026
    m_a0 = TrackARidge(alpha=0.001).fit(tr_2025[BASE_FEATURES], tr_2025["y"])
    p_a0_hist = m_a0.predict(te_2026_hist[BASE_FEATURES])
    p_a0_full = m_a0.predict(te_2026_full[BASE_FEATURES])
    m_a0_hist = evaluate_predictions(te_2026_hist["y"].to_numpy(), p_a0_hist)
    m_a0_full = evaluate_predictions(te_2026_full["y"].to_numpy(), p_a0_full)

    # Evaluate Winner on 2026
    win_feats = BASE_FEATURES + best_candidate["weather_features"]
    m_win = TrackARidge(alpha=0.001).fit(tr_2025[win_feats], tr_2025["y"])
    p_win_hist = m_win.predict(te_2026_hist[win_feats])
    p_win_full = m_win.predict(te_2026_full[win_feats])
    m_win_hist = evaluate_predictions(te_2026_hist["y"].to_numpy(), p_win_hist)
    m_win_full = evaluate_predictions(te_2026_full["y"].to_numpy(), p_win_full)

    # Surge Days (Top 5% Demand in 2026)
    q95 = te_2026_full["y"].quantile(0.95)
    surge_mask = te_2026_full["y"] >= q95
    surge_n = int(surge_mask.sum())
    surge_a0_mae = float(np.mean(np.abs(te_2026_full.loc[surge_mask, "y"] - p_a0_full[surge_mask])))
    surge_win_mae = float(np.mean(np.abs(te_2026_full.loc[surge_mask, "y"] - p_win_full[surge_mask])))
    surge_a0_bias = float(np.mean(p_a0_full[surge_mask] - te_2026_full.loc[surge_mask, "y"]))
    surge_win_bias = float(np.mean(p_win_full[surge_mask] - te_2026_full.loc[surge_mask, "y"]))

    print(f"{'Evaluation Set':<36} {'Model':<16} {'MAE (MW)':>10} {'RMSE (MW)':>11} {'MAPE (%)':>9} {'R²':>8}")
    print("-" * 92)
    print(f"{'Historical 313 Days (through 11-09)':<36} {'A0 (No Wx)':<16} {m_a0_hist['mae']:>10.2f} {m_a0_hist['rmse']:>11.2f} {m_a0_hist['mape']:>8.2f}% {m_a0_hist['r2']:>8.4f}")
    print(f"{'Historical 313 Days (through 11-09)':<36} {'A1 (Winning Wx)':<16} {m_win_hist['mae']:>10.2f} {m_win_hist['rmse']:>11.2f} {m_win_hist['mape']:>8.2f}% {m_win_hist['r2']:>8.4f}")
    print(f"{'Full 314 Days (through 11-10)':<36} {'A0 (No Wx)':<16} {m_a0_full['mae']:>10.2f} {m_a0_full['rmse']:>11.2f} {m_a0_full['mape']:>8.2f}% {m_a0_full['r2']:>8.4f}")
    print(f"{'Full 314 Days (through 11-10)':<36} {'A1 (Winning Wx)':<16} {m_win_full['mae']:>10.2f} {m_win_full['rmse']:>11.2f} {m_win_full['mape']:>8.2f}% {m_win_full['r2']:>8.4f}")
    print("-" * 92)
    print(f"Top 5% Surge Days (n={surge_n}, threshold={q95:.1f} MW):")
    print(f"  A0 MAE = {surge_a0_mae:.2f} MW (Bias = {surge_a0_bias:+.2f} MW)")
    print(f"  A1 MAE = {surge_win_mae:.2f} MW (Bias = {surge_win_bias:+.2f} MW)  ->  Delta: {surge_win_mae - surge_a0_mae:+.2f} MW")
    print()

    # 4. Save Artifacts
    RESULTS.mkdir(exist_ok=True)
    out_json = RESULTS / "experiment_a1_results.json"
    out_csv = RESULTS / "experiment_a1_predictions.csv"

    pred_df = pd.DataFrame({
        "date": te_2026_full.index,
        "actual_mw": te_2026_full["y"].to_numpy(),
        "pred_a0_mw": p_a0_full,
        "error_a0_mw": p_a0_full - te_2026_full["y"].to_numpy(),
        "pred_a1_mw": p_win_full,
        "error_a1_mw": p_win_full - te_2026_full["y"].to_numpy(),
        "is_surge_day": surge_mask.to_numpy(),
    })
    pred_df.to_csv(out_csv, index=False)

    summary_payload = {
        "experiment": "A1_weather_levels",
        "horizon_H": 2,
        "selection_protocol": "3-fold historical walk-forward (2023, 2024, 2025)",
        "selection_metric": "mean_val_mae",
        "winning_candidate": best_candidate["candidate"],
        "winning_weather_features": best_candidate["weather_features"],
        "total_winning_features": best_candidate["total_features"],
        "validation_ablation_results": ablation_records,
        "diagnostic_2026_results": {
            "historical_313_days": {
                "A0": m_a0_hist,
                "A1": m_win_hist,
                "delta_mae_mw": m_win_hist["mae"] - m_a0_hist["mae"],
            },
            "full_314_days": {
                "A0": m_a0_full,
                "A1": m_win_full,
                "delta_mae_mw": m_win_full["mae"] - m_a0_full["mae"],
            },
            "surge_days": {
                "n": surge_n,
                "threshold_mw": float(q95),
                "A0_mae": surge_a0_mae,
                "A1_mae": surge_win_mae,
                "delta_mae_mw": surge_win_mae - surge_a0_mae,
            }
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)

    print("=" * 80)
    print("EXPERIMENT A1 COMPLETE — ARTIFACTS WRITTEN:")
    print(f"  Results JSON : {out_json}")
    print(f"  Predictions  : {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

