"""
Automated unit and integration test suite for the Canonical Phase 5 Dataset Gateway.

Verifies:
1. Information-age integrity (H=2, newest allowed information is T-2).
2. Leakage prevention (peak_lag_1 and energy_lag_1 strictly forbidden).
3. Weekday alignment preservation (peak_lag_7 is T-7, peak_lag_14 is T-14, peak_lag_364 is T-364).
4. Dual-regime non-weather parity (Regime A and Regime B share identical non-weather features).
5. Experiment A0 reproduction through the canonical gateway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contract import (
    PHASE5_FEATURE_SETS,
    load_phase5_data,
    verify_phase5_integrity,
)
from src.ridge import TrackARidge


def check(desc: str, condition: bool, extra: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {desc}" + (f" - {extra}" if extra else ""))
    assert condition, f"Check failed: {desc} ({extra})"


def main():
    print("=" * 72)
    print("CANONICAL PHASE 5 GATEWAY INTEGRITY & ACCEPTANCE SUITE")
    print("=" * 72)

    # ----------------------------------------------------------------------
    # 1. Feature Set Specifications
    # ----------------------------------------------------------------------
    print("\n1. Feature set specifications")
    a0_feats = PHASE5_FEATURE_SETS["A0_weather_free"]
    check("A0 contains exactly 19 features", len(a0_feats) == 19, f"{len(a0_feats)} features")
    check("A0 contains zero lag-1 features", "peak_lag_1" not in a0_feats and "energy_lag_1" not in a0_feats)
    check("A0 contains peak_lag_2 as newest peak demand", "peak_lag_2" in a0_feats)
    check("A0 contains peak_lag_7, 14, 364", all(f in a0_feats for f in ["peak_lag_7", "peak_lag_14", "peak_lag_364"]))
    check("A0 contains energy_lag_2 as newest energy", "energy_lag_2" in a0_feats)
    check("A0 contains zero weather features", not any("t_max" in f or "cdd" in f for f in a0_feats))

    a1_feats = PHASE5_FEATURE_SETS["A1_weather_levels"]
    check("A1 contains exactly 24 features (19 baseline + 5 weather)", len(a1_feats) == 24)
    check("A1 contains no humidity features", not any("rh" in f or "app_t" in f or "thi" in f for f in a1_feats))

    # ----------------------------------------------------------------------
    # 2. Canonical Gateway Loading: A0 Weather-Free
    # ----------------------------------------------------------------------
    print("\n2. Canonical gateway loading: A0 (weather_source='none')")
    df_a0, feats_a0 = load_phase5_data(feature_set="A0_weather_free", weather_source="none")
    check("A0 loads cleanly without NaNs", df_a0[feats_a0 + ["tn_peak_demand_mw"]].isna().sum().sum() == 0)
    check("Target column tn_peak_demand_mw present", "tn_peak_demand_mw" in df_a0.columns)
    check("Target column y present and equal", np.allclose(df_a0["y"], df_a0["tn_peak_demand_mw"]))
    check("Index is DatetimeIndex", isinstance(df_a0.index, pd.DatetimeIndex))
    check("H=2 recorded in attrs", df_a0.attrs.get("H") == 2)
    check("lag_representation recorded as literal_master", df_a0.attrs.get("lag_representation") == "literal_master")

    # ----------------------------------------------------------------------
    # 3. Mechanical Integrity & Weekday Alignment
    # ----------------------------------------------------------------------
    print("\n3. Mechanical verification & weekday alignment")
    rep_a0 = verify_phase5_integrity(df_a0, feats_a0, H=2)
    check("verify_phase5_integrity reports ok", rep_a0["ok"] is True)
    
    offsets = rep_a0["resolved_offsets"]
    check("peak_lag_2 resolves to target - 2 (newest demand)", offsets.get("peak_lag_2") == 2)
    check("peak_lag_3 resolves to target - 3", offsets.get("peak_lag_3") == 3)
    check("peak_lag_7 resolves to target - 7 (WEEKDAY ALIGNED)", offsets.get("peak_lag_7") == 7)
    check("peak_lag_14 resolves to target - 14 (WEEKDAY ALIGNED)", offsets.get("peak_lag_14") == 14)
    check("peak_lag_364 resolves to target - 364 (WEEKDAY ALIGNED 52*7)", offsets.get("peak_lag_364") == 364)
    check("energy_lag_2 resolves to target - 2 (newest energy)", offsets.get("energy_lag_2") == 2)
    check("energy_lag_7 resolves to target - 7 (WEEKDAY ALIGNED)", offsets.get("energy_lag_7") == 7)
    check("energy_lag_14 resolves to target - 14 (WEEKDAY ALIGNED)", offsets.get("energy_lag_14") == 14)

    # ----------------------------------------------------------------------
    # 4. Anti-Leakage Assertion Tests
    # ----------------------------------------------------------------------
    print("\n4. Anti-leakage assertion enforcement")
    leak_failed = False
    try:
        verify_phase5_integrity(df_a0, feats_a0 + ["peak_lag_1"])
    except AssertionError:
        leak_failed = True
    check("Attempting to use peak_lag_1 triggers AssertionError", leak_failed)

    target_failed = False
    try:
        verify_phase5_integrity(df_a0, feats_a0 + ["tn_peak_demand_mw"])
    except AssertionError:
        target_failed = True
    check("Attempting to use target in features triggers AssertionError", target_failed)

    hum_failed = False
    try:
        verify_phase5_integrity(df_a0, feats_a0 + ["rh_mean"])
    except AssertionError:
        hum_failed = True
    check("Attempting to use forbidden humidity triggers AssertionError", hum_failed)

    # ----------------------------------------------------------------------
    # 5. Dual-Regime Weather Compatibility (Regime A vs Regime B)
    # ----------------------------------------------------------------------
    print("\n5. Dual-regime weather compatibility (Regime A vs Regime B)")
    df_regime_a, feats_a = load_phase5_data(feature_set="A1_weather_levels", weather_source="reanalysis")
    df_regime_b, feats_b = load_phase5_data(feature_set="A1_weather_levels", weather_source="forecast", forecast_lead=1)
    
    check("Regime A and Regime B feature lists are identical", feats_a == feats_b)
    
    # Check non-weather columns are byte-identical on overlapping dates
    common_dates = df_regime_a.index.intersection(df_regime_b.index)
    non_wx_cols = [c for c in a0_feats]
    diff_non_wx = 0.0
    for col in non_wx_cols:
        diff = np.abs(df_regime_a.loc[common_dates, col] - df_regime_b.loc[common_dates, col]).max()
        diff_non_wx = max(diff_non_wx, diff)
    check("Non-weather features are byte-identical across Regime A and Regime B", diff_non_wx == 0.0)

    # Check weather columns actually swapped
    wx_diff = np.abs(df_regime_a.loc[common_dates, "t_max"] - df_regime_b.loc[common_dates, "t_max"]).max()
    check("Target-day weather t_max is swapped between regimes", wx_diff > 0.0, f"max diff {wx_diff:.3f} C")

    # ----------------------------------------------------------------------
    # 6. A0 Baseline Reproduction Through Canonical Gateway
    # ----------------------------------------------------------------------
    print("\n6. A0 baseline reproduction through canonical gateway")
    year = df_a0.index.year
    trainval = df_a0[year <= 2025]
    test_hist = df_a0[(year == 2026) & (df_a0.index <= "2026-11-09")]
    test_full = df_a0[year == 2026]

    model = TrackARidge(alpha=0.001)
    model.fit(trainval[feats_a0], trainval["y"])

    preds_hist = model.predict(test_hist[feats_a0])
    mae_hist = float(np.mean(np.abs(test_hist["y"] - preds_hist)))
    check("A0 on historical 313 test dates reproduces to 730.02 MW (+/- 0.01 MW)",
          abs(mae_hist - 730.0185) < 0.01, f"observed {mae_hist:.2f} MW")

    preds_full = model.predict(test_full[feats_a0])
    mae_full = float(np.mean(np.abs(test_full["y"] - preds_full)))
    check("A0 on full 314 test dates reproduces to 730.59 MW (+/- 0.01 MW)",
          abs(mae_full - 730.5892) < 0.01, f"observed {mae_full:.2f} MW")

    print("\n" + "=" * 72)
    print("ALL 25 CANONICAL GATEWAY ACCEPTANCE CHECKS PASSED  ✓")
    print("=" * 72)


if __name__ == "__main__":
    main()

