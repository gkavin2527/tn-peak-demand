"""
Phase 6 acceptance tests.

    python test_phase6_benchmark.py

Mirrors test_phase5_gateway.py in spirit: every claim the benchmark makes is
asserted mechanically rather than trusted. Run this before committing any
change to src/models.py, src/evaluation.py or run_phase6_benchmark.py.

The checks exist because of specific failures, not hypothetically:

* **Phase 5 reproduction** - the benchmark must reproduce the committed
  Regime A/B Ridge numbers (571.61 / 611.95 MW on 1,066 dates) exactly. If a
  refactor changes them, Phase 6 results stop being comparable with Phase 5
  and the whole experimental story breaks.

* **Train/score separation** - the first version of the runner restricted
  TRAINING to the common scoring mask. Because the Regime B frame only spans
  the forecast archive, that silently cut fold 1's training set from ~2,900
  rows to 364 and moved every number in the table. Nothing warned.

* **Shuffle control on the hybrid** - a -105 MW gain is large enough to be a
  leak. Destroying the feature->residual relationship must destroy the gain.

* **Shared sample** - trees tolerate NaN, linear models do not. Without one
  mask the comparison measures sample composition, not model skill.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy / sklearn / lightgbm

import sys

import numpy as np
import pandas as pd

from src.contract import PHASE5_FEATURE_SETS, load_phase5_data
from src.evaluation import compute_metrics, moving_block_bootstrap, surge_mask
from src.models import (MODEL_ZOO, REFERENCE_MODEL, available_models,
                        build_model)
from run_phase6_benchmark import (FEATURE_SET, FOLDS, common_mask,
                                  load_regime, run_walk_forward)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    print("=" * 74)
    print("PHASE 6 BENCHMARK ACCEPTANCE TESTS")
    print("=" * 74)

    feats = list(PHASE5_FEATURE_SETS[FEATURE_SET])

    # ---- 1. frozen specification -------------------------------------
    print("\n1. Frozen A2 specification")
    check("A2 is the canonical 26-feature set", len(feats) == 26,
          f"{len(feats)} features")
    extras = [f for f in feats if f not in PHASE5_FEATURE_SETS["A1_weather_levels"]]
    check("A2 extras are the gate-selected pair",
          sorted(extras) == ["cdd_max_vs_roll7", "d_cdd_max"], str(sorted(extras)))
    check("no T-1 demand/energy leaks into A2",
          not {"peak_lag_1", "energy_lag_1"} & set(feats))

    # ---- 2. model zoo interface --------------------------------------
    print("\n2. Model zoo")
    avail = available_models()
    check("all zoo models importable", len(avail) == len(MODEL_ZOO),
          f"{len(avail)}/{len(MODEL_ZOO)}")
    check("reference model is in the zoo", REFERENCE_MODEL in MODEL_ZOO)

    df_a = load_regime("A", feats)
    small = df_a.head(500)
    iface_ok, bad = True, []
    for name in avail:
        try:
            m = build_model(name).fit(small[feats], small["y"])
            p = np.asarray(m.predict(small[feats].head(10)), float)
            if p.shape != (10,) or not np.isfinite(p).all():
                iface_ok = False; bad.append(name)
        except Exception as e:
            iface_ok = False; bad.append(f"{name}({type(e).__name__})")
    check("every model honours fit/predict and returns finite output",
          iface_ok, f"failed: {bad}" if bad else f"{len(avail)} models")

    check("naive_persistence echoes peak_lag_2 (T-2, not T-1)",
          MODEL_ZOO["naive_persistence"]["factory"]().column == "peak_lag_2")
    check("seasonal_naive echoes peak_lag_7 (same weekday)",
          MODEL_ZOO["seasonal_naive"]["factory"]().column == "peak_lag_7")

    # ---- 3. shared sample --------------------------------------------
    print("\n3. Shared evaluation sample")
    scored_a = common_mask({"A": df_a}, feats)
    check("Regime-A-only scoring covers all three validation years",
          len(scored_a) == 1085, f"{len(scored_a)} dates")
    per_model_n = {}
    for name in ["ridge", "lightgbm", "ridge_gbm_hybrid"]:
        p = run_walk_forward(df_a, df_a, feats, name, scored_a)
        per_model_n[name] = len(p)
    check("every model scores the identical number of rows",
          len(set(per_model_n.values())) == 1, str(per_model_n))

    # ---- 4. train/score separation -----------------------------------
    print("\n4. Train / score separation (the bug that moved every number)")
    df_b = load_regime("B", feats)
    scored_ab = common_mask({"A": df_a, "B": df_b}, feats)
    p_b = run_walk_forward(df_a, df_b, feats, "ridge", scored_ab)
    n_train_f1 = int(p_b[p_b.val_year == 2023]["n_train"].iloc[0])
    check("Regime B trains on the full ERA5 history, not the forecast span",
          n_train_f1 > 2500, f"fold-1 n_train = {n_train_f1:,} (Phase 5: 2,897)")
    check("Regime B frame really is shorter than the training frame",
          len(df_b) < len(df_a), f"B={len(df_b):,} rows vs A={len(df_a):,}")

    # ---- 5. Phase 5 reproduction -------------------------------------
    print("\n5. Reproduction of the committed Phase 5 Regime A/B result")
    check("common scored dates match Phase 5", len(scored_ab) == 1066,
          f"{len(scored_ab)} dates")
    mae_a = compute_metrics(*run_walk_forward(df_a, df_a, feats, "ridge", scored_ab)
                            [["y_true", "y_pred"]].to_numpy().T)["mae"]
    mae_b = compute_metrics(*p_b[["y_true", "y_pred"]].to_numpy().T)["mae"]
    check("Ridge Regime A reproduces 571.61 MW", abs(mae_a - 571.61) < 0.05,
          f"{mae_a:.2f} MW")
    check("Ridge Regime B reproduces 611.95 MW", abs(mae_b - 611.95) < 0.05,
          f"{mae_b:.2f} MW")
    check("forecast degradation reproduces +40.34 MW",
          abs((mae_b - mae_a) - 40.34) < 0.1, f"{mae_b - mae_a:+.2f} MW")

    # ---- 6. hybrid is signal, not leakage ----------------------------
    print("\n6. Hybrid shuffle control (is the -105 MW gain real?)")
    import lightgbm as lgb
    from src.ridge import TrackARidge

    def hybrid_mae(shuffle: bool) -> float:
        maes = []
        for f in FOLDS:
            tr = df_a[df_a.index.year <= f["train_end"]]
            va = df_a[(df_a.index.year == f["val_year"]) & df_a.index.isin(scored_a)]
            r = TrackARidge(alpha=0.001).fit(tr[feats], tr["y"])
            resid = tr["y"].to_numpy(float) - r.predict(tr[feats])
            if shuffle:
                resid = np.random.default_rng(42).permutation(resid)
            g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03,
                                  num_leaves=7, min_child_samples=30,
                                  subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=42, n_jobs=1, verbose=-1)
            g.fit(tr[feats], resid)
            pred = r.predict(va[feats]) + g.predict(va[feats])
            maes.append(np.abs(pred - va["y"].to_numpy(float)).mean())
        return float(np.mean(maes))

    ridge_mae = np.mean([
        compute_metrics(
            df_a[(df_a.index.year == f["val_year"]) & df_a.index.isin(scored_a)]["y"],
            TrackARidge(alpha=0.001)
            .fit(df_a[df_a.index.year <= f["train_end"]][feats],
                 df_a[df_a.index.year <= f["train_end"]]["y"])
            .predict(df_a[(df_a.index.year == f["val_year"])
                          & df_a.index.isin(scored_a)][feats]))["mae"]
        for f in FOLDS])
    real, shuf = hybrid_mae(False), hybrid_mae(True)
    check("hybrid beats Ridge by a wide margin", real < ridge_mae - 50,
          f"{ridge_mae:.1f} -> {real:.1f} MW ({real - ridge_mae:+.1f})")
    check("shuffling residuals destroys the gain (so it is signal, not leakage)",
          abs(shuf - ridge_mae) < 25,
          f"shuffled {shuf:.1f} MW ({shuf - ridge_mae:+.1f} vs Ridge)")

    # ---- 7. evaluation primitives ------------------------------------
    print("\n7. Evaluation primitives")
    y = np.array([100.0, 200.0, 300.0]); p = np.array([110.0, 190.0, 330.0])
    m = compute_metrics(y, p)
    check("MAE arithmetic", abs(m["mae"] - 50 / 3) < 1e-9, f"{m['mae']:.4f}")
    check("bias arithmetic", abs(m["bias"] - 10.0) < 1e-9, f"{m['bias']:+.1f}")
    bs = moving_block_bootstrap(np.full(200, -10.0))
    check("bootstrap recovers a constant difference",
          abs(bs["mean"] + 10.0) < 1e-9 and bs["ci_95_high"] < 0,
          f"mean {bs['mean']:+.1f}, CI [{bs['ci_95_low']:+.1f},{bs['ci_95_high']:+.1f}]")
    sm = surge_mask(pd.date_range("2024-01-01", periods=100),
                    np.arange(100.0))
    check("surge mask selects ~top 5% within the year",
          4 <= sm.sum() <= 6, f"{sm.sum()} of 100 days")

    # ---- summary ------------------------------------------------------
    failed = [r for r in results if r[0] == FAIL]
    print("\n" + "=" * 74)
    if failed:
        print(f"PHASE 6 ACCEPTANCE FAILED - {len(failed)} of {len(results)} checks")
        for _, n, d in failed:
            print(f"  FAIL  {n}  {d}")
        sys.exit(1)
    print(f"ALL {len(results)} PHASE 6 ACCEPTANCE CHECKS PASSED  ✓")
    print("=" * 74)


if __name__ == "__main__":
    main()
