"""
Phase 5 — Experiment A3: Heat Persistence / Multi-Day Heat Effects Evaluation.

Evaluates whether sustained multi-day heat contains additional predictive information
beyond the selected A2 weather features (frozen A1 features + d_t_max + t_max_vs_roll7).

Methodology Rules:
1. H=2 operational information constraint strictly preserved.
2. Canonical gateway load_phase5_data() and verify_phase5_integrity() used.
3. Feature selection strictly uses historical walk-forward validation folds:
   - Fold 1: Val 2023 (Train <= 2022)
   - Fold 2: Val 2024 (Train <= 2023)
   - Fold 3: Val 2025 (Train <= 2024)
4. Single selection metric: Mean Validation MAE across the 3 historical folds.
5. All persistence baselines computed on continuous daily weather series with zero leakage.
6. 2026 performance reported solely for diagnostic comparison (not for selection).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import state_weather
from config import RAW_WEATHER, RESULTS
from src.contract import (
    PHASE5_FEATURE_SETS,
    load_phase5_data,
    verify_phase5_integrity,
)
from src.ridge import TrackARidge

# --------------------------------------------------------------------------
# Frozen A2 Feature Set (26 features)
# --------------------------------------------------------------------------
A1_BASE_FEATURES = PHASE5_FEATURE_SETS["A1_weather_levels"]
A2_FROZEN_FEATURES = A1_BASE_FEATURES + ["d_t_max", "t_max_vs_roll7"]

# --------------------------------------------------------------------------
# Candidate Persistence Feature Sets for A3
# --------------------------------------------------------------------------
CANDIDATES = [
    ("Frozen A2 Baseline (26 feats)", []),
    ("A3.1: + cdd_max_lag1 (Yesterday CDD)", ["cdd_max_lag1_candidate"]),
    ("A3.2: + t_max_roll2_lag1 (2-Day Antecedent Mean)", ["t_max_roll2_lag1"]),
    ("A3.3: + t_max_roll3_lag1 (3-Day Antecedent Mean)", ["t_max_roll3_lag1"]),
    ("A3.4: + heat_wave_run_35_lag1 (Antecedent Heatwave Run)", ["heat_wave_run_35_lag1"]),
    ("A3.5: + heat_wave_run_35 (Active Heatwave Run)", ["heat_wave_run_35"]),
    ("A3.6: + (heat_wave_run_35_lag1, t_max_roll2_lag1)", ["heat_wave_run_35_lag1", "t_max_roll2_lag1"]),
    ("A3.7: + (heat_wave_run_35, t_max_roll3_lag1)", ["heat_wave_run_35", "t_max_roll3_lag1"]),
    ("A3.8: + consec_hot_36 (Severe Consecutive Heat)", ["consec_hot_36"]),
    ("A3.9: + excess_heat_35_roll3 (3-Day Excess Heat DD)", ["excess_heat_35_roll3"]),
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


def compute_persistence_features(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Computes all heat persistence candidates on the continuous daily weather series.
    
    Guarantees:
    - Zero calendar gaps in rolling/shift operations.
    - Historical lag windows strictly use observations through T-1 or earlier.
    - Zero future data leakage into trailing aggregates.
    """
    raw_w = pd.read_csv(RAW_WEATHER)
    sw = state_weather(raw_w).set_index("date")
    span = pd.date_range(sw.index.min(), sw.index.max(), freq="D")
    sw = sw.reindex(span)

    # 1. cdd_max_lag1: Yesterday's cooling degree days
    sw["cdd_max_lag1_candidate"] = sw["cdd_max"].shift(1)

    # 2. t_max_roll2_lag1: Antecedent 2-day mean of observed maximum temperature [T-2, T-1]
    sw["t_max_roll2_lag1"] = sw["t_max"].shift(1).rolling(2, min_periods=2).mean()

    # 3. t_max_roll3_lag1: Antecedent 3-day mean of observed maximum temperature [T-3, T-2, T-1]
    sw["t_max_roll3_lag1"] = sw["t_max"].shift(1).rolling(3, min_periods=3).mean()

    # 4. heat_wave_run_35: Number of consecutive days t_max >= 35.0 C ending at target day T (capped at 3)
    c35 = (sw["t_max"] >= 35.0).astype(int)
    runs_35 = []
    cur = 0
    for val in c35:
        cur = (cur + 1) if val == 1 else 0
        runs_35.append(min(cur, 3))
    sw["heat_wave_run_35"] = runs_35

    # 5. heat_wave_run_35_lag1: Antecedent consecutive days t_max >= 35.0 C through T-1 (capped at 3)
    sw["heat_wave_run_35_lag1"] = sw["heat_wave_run_35"].shift(1)

    # 6. consec_hot_36: Severe consecutive heat indicator I(t_max(T)>=36 and t_max(T-1)>=36)
    sw["consec_hot_36"] = ((sw["t_max"] >= 36.0) & (sw["t_max"].shift(1) >= 36.0)).astype(int)

    # 7. excess_heat_35_roll3: 3-day cumulative excess degrees above 35 C over [T-2, T-1, T]
    sw["excess_heat_35_roll3"] = (sw["t_max"] - 35.0).clip(lower=0).rolling(3, min_periods=3).sum()

    out_cols = [
        "cdd_max_lag1_candidate",
        "t_max_roll2_lag1",
        "t_max_roll3_lag1",
        "heat_wave_run_35_lag1",
        "heat_wave_run_35",
        "consec_hot_36",
        "excess_heat_35_roll3",
    ]
    return sw[out_cols].reindex(date_index)


def main():
    print("=" * 80)
    print("PHASE 5 — EXPERIMENT A3: HEAT PERSISTENCE & MULTI-DAY HEAT EFFECTS")
    print("=" * 80)

    # 1. Ingest Canonical Data Gateway and Derive Persistence Features
    df_full, _ = load_phase5_data(
        feature_set="A2_weather_deltas_anomalies",
        weather_source="reanalysis",
        clean_na=True
    )

    # Pre-execution mechanical assertions on base features
    rep = verify_phase5_integrity(df_full, A2_FROZEN_FEATURES, H=2)
    print("Canonical Gateway Pre-Execution Integrity Check: PASSED  ✓")
    print(f"  Total Clean Rows    : {len(df_full):,} ({df_full.index.min().date()} → {df_full.index.max().date()})")
    print(f"  Frozen A2 Features  : {len(A2_FROZEN_FEATURES)} (A1 + d_t_max + t_max_vs_roll7)")
    print()

    # Compute candidate persistence features gap-free
    persist_df = compute_persistence_features(df_full.index)
    for col in persist_df.columns:
        df_full[col] = persist_df[col].to_numpy()

    # Drop any edge rows where 3-day rolling history is NaN at the very start of dataset (2013)
    all_candidate_cols = list(persist_df.columns)
    clean_df = df_full.dropna(subset=A2_FROZEN_FEATURES + all_candidate_cols).copy()
    print(f"  Candidate Matrix    : {len(clean_df):,} rows with full persistence history")
    print()

    # 2. Historical Walk-Forward Validation Feature Selection
    print("-" * 80)
    print("SECTION 1: HISTORICAL WALK-FORWARD VALIDATION (2023, 2024, 2025)")
    print("Selection Rule: Lowest Mean Validation MAE Across Folds 1, 2, 3")
    print("-" * 80)

    year = clean_df.index.year
    ablation_records = []
    baseline_val_mae = None

    fmt_hdr = f"{'Candidate Set':<48} {'F1 (2023)':>10} {'F2 (2024)':>10} {'F3 (2025)':>10} {'Mean MAE':>10} {'Delta A2':>10}"
    print(fmt_hdr)
    print("-" * 102)

    for name, extra_cols in CANDIDATES:
        cur_feats = A2_FROZEN_FEATURES + extra_cols
        fold_metrics = []
        for f in FOLDS:
            tr = clean_df[year <= f["train_end"]]
            va = clean_df[year == f["val_year"]]

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
            "extra_features": extra_cols,
            "total_features": len(cur_feats),
            "f1_2023_mae": f1_mae,
            "f2_2024_mae": f2_mae,
            "f3_2025_mae": f3_mae,
            "mean_val_mae": mean_mae,
            "mean_val_rmse": mean_rmse,
            "mean_val_mape": mean_mape,
            "mean_val_r2": mean_r2,
            "delta_vs_a2_mw": delta,
            "delta_vs_a2_pct": (delta / baseline_val_mae) * 100,
        })

        print(f"{name:<48} {f1_mae:>10.2f} {f2_mae:>10.2f} {f3_mae:>10.2f} {mean_mae:>10.2f} {delta:>+10.2f}")

    # Determine winning candidate strictly from historical validation MAE
    best_candidate = min(ablation_records, key=lambda x: x["mean_val_mae"])
    print("-" * 102)
    print(f"WINNING CANDIDATE ON VALIDATION FOLDS: {best_candidate['candidate']}")
    print(f"  Mean Val MAE : {best_candidate['mean_val_mae']:.2f} MW ({best_candidate['delta_vs_a2_mw']:+.2f} MW / {best_candidate['delta_vs_a2_pct']:+.2f}%)")
    print()

    # 3. Diagnostic Test Evaluation (Year 2026)
    print("-" * 80)
    print("SECTION 2: 2026 DIAGNOSTIC TEST EVALUATION (STATIC RETRAIN 2014-2025)")
    print("Note: Reported for verification; NOT used for feature selection")
    print("-" * 80)

    tr_2025 = clean_df[year <= 2025]
    te_2026_hist = clean_df[(year == 2026) & (clean_df.index <= "2026-11-09")]
    te_2026_full = clean_df[year == 2026]

    # Evaluate Frozen A2 baseline on 2026
    m_a2 = TrackARidge(alpha=0.001).fit(tr_2025[A2_FROZEN_FEATURES], tr_2025["y"])
    p_a2_hist = m_a2.predict(te_2026_hist[A2_FROZEN_FEATURES])
    p_a2_full = m_a2.predict(te_2026_full[A2_FROZEN_FEATURES])
    m_a2_hist = evaluate_predictions(te_2026_hist["y"].to_numpy(), p_a2_hist)
    m_a2_full = evaluate_predictions(te_2026_full["y"].to_numpy(), p_a2_full)

    # Evaluate Selected A3 on 2026
    win_feats = A2_FROZEN_FEATURES + best_candidate["extra_features"]
    m_win = TrackARidge(alpha=0.001).fit(tr_2025[win_feats], tr_2025["y"])
    p_win_hist = m_win.predict(te_2026_hist[win_feats])
    p_win_full = m_win.predict(te_2026_full[win_feats])
    m_win_hist = evaluate_predictions(te_2026_hist["y"].to_numpy(), p_win_hist)
    m_win_full = evaluate_predictions(te_2026_full["y"].to_numpy(), p_win_full)

    # Surge Days (Top 5% Demand in 2026)
    q95 = te_2026_full["y"].quantile(0.95)
    surge_mask = te_2026_full["y"] >= q95
    surge_n = int(surge_mask.sum())
    surge_a2_mae = float(np.mean(np.abs(te_2026_full.loc[surge_mask, "y"] - p_a2_full[surge_mask])))
    surge_win_mae = float(np.mean(np.abs(te_2026_full.loc[surge_mask, "y"] - p_win_full[surge_mask])))
    surge_a2_bias = float(np.mean(p_a2_full[surge_mask] - te_2026_full.loc[surge_mask, "y"]))
    surge_win_bias = float(np.mean(p_win_full[surge_mask] - te_2026_full.loc[surge_mask, "y"]))

    print(f"{'Evaluation Set':<36} {'Model':<16} {'MAE (MW)':>10} {'RMSE (MW)':>11} {'MAPE (%)':>9} {'R²':>8}")
    print("-" * 92)
    print(f"{'Historical 313 Days (through 11-09)':<36} {'A2 (Frozen)':<16} {m_a2_hist['mae']:>10.2f} {m_a2_hist['rmse']:>11.2f} {m_a2_hist['mape']:>8.2f}% {m_a2_hist['r2']:>8.4f}")
    print(f"{'Historical 313 Days (through 11-09)':<36} {'A3 (Selected)':<16} {m_win_hist['mae']:>10.2f} {m_win_hist['rmse']:>11.2f} {m_win_hist['mape']:>8.2f}% {m_win_hist['r2']:>8.4f}")
    print(f"{'Full 314 Days (through 11-10)':<36} {'A2 (Frozen)':<16} {m_a2_full['mae']:>10.2f} {m_a2_full['rmse']:>11.2f} {m_a2_full['mape']:>8.2f}% {m_a2_full['r2']:>8.4f}")
    print(f"{'Full 314 Days (through 11-10)':<36} {'A3 (Selected)':<16} {m_win_full['mae']:>10.2f} {m_win_full['rmse']:>11.2f} {m_win_full['mape']:>8.2f}% {m_win_full['r2']:>8.4f}")
    print("-" * 92)
    print(f"Top 5% Surge Days (n={surge_n}, threshold={q95:.1f} MW):")
    print(f"  A2 MAE = {surge_a2_mae:.2f} MW (Bias = {surge_a2_bias:+.2f} MW)")
    print(f"  A3 MAE = {surge_win_mae:.2f} MW (Bias = {surge_win_bias:+.2f} MW)  ->  Delta: {surge_win_mae - surge_a2_mae:+.2f} MW")
    print()

    # 4. Save Artifacts
    RESULTS.mkdir(exist_ok=True)
    out_json = RESULTS / "experiment_a3_results.json"
    out_csv = RESULTS / "experiment_a3_predictions.csv"

    pred_df = pd.DataFrame({
        "date": te_2026_full.index,
        "actual_mw": te_2026_full["y"].to_numpy(),
        "pred_a2_mw": p_a2_full,
        "error_a2_mw": p_a2_full - te_2026_full["y"].to_numpy(),
        "pred_a3_mw": p_win_full,
        "error_a3_mw": p_win_full - te_2026_full["y"].to_numpy(),
        "is_surge_day": surge_mask.to_numpy(),
    })
    pred_df.to_csv(out_csv, index=False)

    summary_payload = {
        "experiment": "A3_heat_persistence",
        "horizon_H": 2,
        "selection_protocol": "3-fold historical walk-forward (2023, 2024, 2025)",
        "selection_metric": "mean_val_mae",
        "baseline_a2_mae": baseline_val_mae,
        "winning_candidate": best_candidate["candidate"],
        "winning_extra_features": best_candidate["extra_features"],
        "total_winning_features": best_candidate["total_features"],
        "validation_ablation_results": ablation_records,
        "diagnostic_2026_results": {
            "historical_313_days": {
                "A2": m_a2_hist,
                "A3": m_win_hist,
                "delta_mae_mw": m_win_hist["mae"] - m_a2_hist["mae"],
            },
            "full_314_days": {
                "A2": m_a2_full,
                "A3": m_win_full,
                "delta_mae_mw": m_win_full["mae"] - m_a2_full["mae"],
            },
            "surge_days": {
                "n": surge_n,
                "threshold_mw": float(q95),
                "A2_mae": surge_a2_mae,
                "A3_mae": surge_win_mae,
                "delta_mae_mw": surge_win_mae - surge_a2_mae,
            }
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)

    print("=" * 80)
    print("EXPERIMENT A3 COMPLETE — ARTIFACTS WRITTEN:")
    print(f"  Results JSON : {out_json}")
    print(f"  Predictions  : {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

