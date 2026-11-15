"""
W1-3 - recency weighting: exponential half-life swept, then validated honestly.

    python run_phase7_recency.py

THE PROBLEM. Tamil Nadu's peak is not stationary - 2026 runs ~10% above 2025.
The oldest training rows describe a materially different system from the one
being forecast, yet every committed Phase 5/6 fit weighted 2013 exactly like
2025.

THE INSTRUMENT. `half_life_years` on RidgeGBMHybrid / QuantileHybrid applies
weights 0.5^(age / half_life), with age taken from the frame's DatetimeIndex
rather than row position - master.csv has 157 missing days, so position is not
age. `None` is uniform and remains the default, so unweighted results
reproduce byte-identically.

PROTOCOL. Two stages, because a sweep over the full 2023-25 sample is another
selection round against data that has already driven four:

  stage 1  sweep half-life on 2023-25 for visibility (reported, not adopted)
  stage 2  re-select half-life on inner folds 2023/2024 ONLY, score 2025 once

Stage 2 decides. A half-life that only helps in stage 1 is selection noise.
Run on both the plain hybrid and the quantile head from W1-2, since recency
weighting and the quantile objective could easily be fixing the same drift
twice - if so, their gains will not add.
"""

from __future__ import annotations

import threadlimit  # noqa: F401

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import PHASE5_FEATURE_SETS
from src.envinfo import capture_environment
from src.evaluation import diebold_mariano, moving_block_bootstrap, surge_mask
from src.models import build_model
from experiments.run_phase6_benchmark import FOLDS
from experiments.run_phase7_festival import load_set, walk_forward

RESULTS = Path("results")
CHAMPION = Path("models/champion_config.json")
FEATURE_SET = "A2_weather_deltas_anomalies"
HALF_LIVES = [None, 1.0, 2.0, 3.0, 5.0, 8.0]
INNER = [{"train_end": 2022, "val_year": 2023},
         {"train_end": 2023, "val_year": 2024}]


def metrics(preds):
    y, p = preds.y_true.to_numpy(float), preds.y_pred.to_numpy(float)
    e = p - y
    sm = surge_mask(pd.DatetimeIndex(preds.date), y)
    return {"n": int(len(y)), "mae": float(np.abs(e).mean()),
            "bias": float(e.mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "surge_mae": float(np.abs(e[sm]).mean()),
            "surge_bias": float(e[sm].mean()), "surge_n": int(sm.sum())}


def main():
    cfg = json.loads(CHAMPION.read_text())
    hp = dict(cfg["hyperparameters"])
    feats = list(PHASE5_FEATURE_SETS[FEATURE_SET])
    frames = {r: load_set(FEATURE_SET, r) for r in ("A", "B")}

    val_years = {f["val_year"] for f in FOLDS}
    scored = None
    for r in ("A", "B"):
        ok = frames[r][feats + ["y"]].notna().all(axis=1)
        d = frames[r].index[ok & frames[r].index.year.isin(val_years)]
        scored = d if scored is None else scored.intersection(d)
    scored = pd.DatetimeIndex(sorted(scored))

    print("=" * 78)
    print("W1-3  RECENCY WEIGHTING")
    print("=" * 78)
    print(f"{FEATURE_SET} ({len(feats)} features), {len(scored)} scored dates")

    # growth context, so the premise is checked rather than asserted
    y = frames["A"]["y"]
    ann = y.groupby(y.index.year).max()
    recent = ann.loc[2022:]
    print("\nannual peak maximum (the premise for weighting at all):")
    for yr, v in recent.items():
        print(f"  {yr}  {v:>9,.0f} MW")
    print(f"  2025->2026 change: "
          f"{100 * (ann.get(2026, np.nan) / ann.get(2025, np.nan) - 1):+.1f}%")

    out = {}
    for model_name in ("ridge_gbm_hybrid", "quantile_hybrid"):
        # the quantile head uses the tau W1-2 selected on inner folds
        extra = {}
        if model_name == "quantile_hybrid":
            surge = json.loads((RESULTS / "phase7_surge.json").read_text())
            extra = {k: v for k, v in
                     surge["honest_validation"]["inner_selected_params"].items()}
        base_params = {**hp, **extra}

        print("\n" + "-" * 78)
        print(f"STAGE 1  {model_name} - sweep on 2023-25 (visibility only)")
        print("-" * 78)
        print(f"{'half-life':>10} {'MAE_A':>9} {'MAE_B':>9} {'dMAE_B':>9} "
              f"{'bias_B':>9} {'surgeBias':>10}")
        rows = []
        ref_mae = None
        for hl in HALF_LIVES:
            params = {**base_params, "half_life_years": hl}
            m = {r: metrics(walk_forward(frames["A"], frames[r], feats,
                                         model_name, scored, params))
                 for r in ("A", "B")}
            if hl is None:
                ref_mae = m["B"]["mae"]
            d = m["B"]["mae"] - ref_mae
            label = "none" if hl is None else f"{hl:g} yr"
            print(f"{label:>10} {m['A']['mae']:>9.2f} {m['B']['mae']:>9.2f} "
                  f"{d:>+9.2f} {m['B']['bias']:>+9.1f} "
                  f"{m['B']['surge_bias']:>+10.1f}")
            rows.append({"half_life_years": hl, "regime_a": m["A"],
                         "regime_b": m["B"], "delta_mae_b_vs_uniform": d})

        # ---- stage 2: inner-fold selection, 2025 scored once ----------
        print(f"\nSTAGE 2  {model_name} - half-life from 2023/2024 only, "
              f"2025 scored once")
        inner_scores = {}
        for hl in HALF_LIVES:
            params = {**base_params, "half_life_years": hl}
            maes = []
            for f in INNER:
                tr = frames["A"][frames["A"].index.year <= f["train_end"]]
                dates = scored[scored.year == f["val_year"]]
                ok = tr["y"].notna() & tr[feats].notna().all(axis=1)
                mod = build_model(model_name, **params).fit(tr.loc[ok, feats],
                                                            tr.loc[ok, "y"])
                va = frames["A"].loc[frames["A"].index.isin(dates)]
                maes.append(float(np.abs(mod.predict(va[feats])
                                         - va["y"].to_numpy(float)).mean()))
            inner_scores[hl] = float(np.mean(maes))
        sel = min(inner_scores, key=inner_scores.get)
        print("  inner-fold MAE: " + "  ".join(
            f"{'none' if k is None else f'{k:g}y'}={v:.1f}"
            for k, v in inner_scores.items()))
        print(f"  -> selected half-life: {'none' if sel is None else f'{sel:g} yr'}")

        outer = scored[scored.year == 2025]
        tr = frames["A"][frames["A"].index.year <= 2024]
        ok = tr["y"].notna() & tr[feats].notna().all(axis=1)
        va = frames["B"].loc[frames["B"].index.isin(outer)]
        yt = va["y"].to_numpy(float)
        errs = {}
        for label, hl in (("uniform", None), ("weighted", sel)):
            params = {**base_params, "half_life_years": hl}
            mod = build_model(model_name, **params).fit(tr.loc[ok, feats],
                                                        tr.loc[ok, "y"])
            errs[label] = np.asarray(mod.predict(va[feats]), float) - yt
        sm = surge_mask(pd.DatetimeIndex(va.index), yt)
        for label, e in errs.items():
            print(f"    {label:<9} 2025 Regime B: MAE {np.abs(e).mean():7.2f}  "
                  f"bias {e.mean():+7.1f}   surge bias {e[sm].mean():+7.1f}")

        if sel is None:
            print("    -> inner folds preferred UNIFORM weighting; nothing to test")
            verdict = {"selected_half_life": None, "survives": False,
                       "note": "inner folds preferred uniform weighting"}
        else:
            d = np.abs(errs["weighted"]) - np.abs(errs["uniform"])
            bs = moving_block_bootstrap(d)
            dm = diebold_mariano(errs["weighted"], errs["uniform"], loss="abs")
            surv = bs["ci_95_high"] < 0
            print(f"    paired on 2025 (n={len(d)}): {d.mean():+.2f} MW  "
                  f"95% CI [{bs['ci_95_low']:+.2f}, {bs['ci_95_high']:+.2f}]  "
                  f"DM p={dm['p_value']:.3f}")
            print(f"    -> {'SURVIVES' if surv else 'does NOT survive'} on "
                  f"untouched 2025")
            verdict = {"selected_half_life": sel,
                       "delta_mae_mw": float(d.mean()),
                       "ci_95_low": bs["ci_95_low"], "ci_95_high": bs["ci_95_high"],
                       "dm_p_two_sided": dm["p_value"],
                       "survives": bool(surv)}

        out[model_name] = {
            "base_params": base_params,
            "stage1_sweep_2023_25": rows,
            "stage2_inner_fold_mae": {str(k): v for k, v in inner_scores.items()},
            "stage2_verdict_2025": verdict,
            "scored_2025": {
                label: {"mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                        "surge_bias": float(e[sm].mean()), "n": int(len(e))}
                for label, e in errs.items()},
        }

    payload = {
        "phase": "w1_3_recency_weighting",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_set": FEATURE_SET,
        "half_lives_tested": [("none" if h is None else h) for h in HALF_LIVES],
        "protocol": {
            "stage1": "sweep on full 2023-25 sample - reported, NOT adopted",
            "stage2": ("half-life re-selected on inner folds 2023/2024 only, "
                       "2025 scored once under Regime B - this decides"),
            "weighting": "0.5^(age_days / (365.25 * half_life_years))",
            "age_source": ("frame DatetimeIndex, not row position - 157 "
                           "missing days make position a bad age proxy"),
            "default": "None (uniform); committed results reproduce exactly",
        },
        "annual_peak_max": {str(k): float(v) for k, v in ann.items()},
        "results": out,
        "environment": capture_environment(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase7_recency.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {RESULTS / 'phase7_recency.json'}")


if __name__ == "__main__":
    main()
