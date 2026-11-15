"""
Phase 6 - full model benchmark on the frozen A2 weather specification.

    python run_phase6_benchmark.py                  # Regime A, all models
    python run_phase6_benchmark.py --models ridge,lightgbm
    python run_phase6_benchmark.py --regime-b --models ridge,lightgbm_linear

Research question
-----------------
Phase 5 established WHAT information the model gets (the frozen 26-feature A2
set) and what forecast weather costs (+40.34 MW, +7.06%). Phase 6 asks the
remaining question:

    Which model class best exploits that fixed information set?

Everything except the estimator is held constant: the A2 feature list, H=2,
the three expanding walk-forward folds, the training windows and - critically -
the scored rows.

Design decisions, and why
-------------------------
**Regime A for model comparison, Regime B for the finalists.** Model selection
and weather realism are separate questions. Phase 5 already answered the
weather question with Ridge; re-answering it for nine candidates multiplies
runs without adding knowledge. But the A->B degradation is not guaranteed to
be model-independent - a tree can split on forecast noise in ways a linear
model cannot - so the top candidates are re-run under Regime B before anything
is called champion. `--regime-b` does that.

**One shared row mask across every model.** Tree ensembles tolerate NaN and
linear models do not. Left alone, LightGBM would train and score on rows Ridge
silently dropped, and the comparison would be measuring sample composition
rather than model skill. The mask is computed once from the A2 feature block
plus the target, and applied to all candidates.

**Paired bootstrap against Ridge, not an unpaired MAE table.** Ranking by mean
MAE alone invites reading a 3 MW gap as a result. Each candidate is compared
to the incumbent day-by-day with a 7-day moving-block bootstrap, the same
protocol as the Phase 5 decision gate, so the Phase 6 verdicts are directly
comparable to the Phase 5 ones.

**Surge-day bias is reported for every model.** The project's known weakness is
systematic under-prediction on the highest-demand days. A model that wins on
mean MAE while being several hundred MW low on every record day is not the
better operational choice, and MAE alone cannot show that.

Nothing here tunes anything. Tuning only the finalists is Phase 6b.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy / sklearn / lightgbm

import argparse
import json
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config import RANDOM_SEED, RESULTS
from src.contract import PHASE5_FEATURE_SETS, load_phase5_data
from src.evaluation import (BLOCK_DAYS, BOOT_SEED, N_BOOT, compute_metrics,
                            moving_block_bootstrap, subset_metrics, surge_mask)
from src.models import MODEL_ZOO, REFERENCE_MODEL, build_model

FOLDS = [
    {"name": "Fold 1 (Val 2023)", "train_end": 2022, "val_year": 2023},
    {"name": "Fold 2 (Val 2024)", "train_end": 2023, "val_year": 2024},
    {"name": "Fold 3 (Val 2025)", "train_end": 2024, "val_year": 2025},
]
FEATURE_SET = "A2_weather_deltas_anomalies"
SUMMER = (3, 4, 5, 6)


def load_regime(regime: str, features: list[str]):
    """Regime A (reanalysis) or Regime B (archived lead-1 forecast) frame.

    Regime B requires forecast coverage on every scored date; the gateway
    raises rather than silently leaving ERA5 weather in place. That guard is
    the Phase 5 anti-leakage fix and must not be bypassed here.
    """
    if regime == "A":
        df, feats = load_phase5_data(FEATURE_SET, weather_source="reanalysis")
    else:
        val_dates = []
        base, _ = load_phase5_data(FEATURE_SET, weather_source="reanalysis")
        for f in FOLDS:
            val_dates += list(base.index[base.index.year == f["val_year"]])
        df, feats = load_phase5_data(FEATURE_SET, weather_source="forecast",
                                     forecast_lead=1)
    assert feats == features, "feature list drifted from the canonical A2 set"
    return df


def common_mask(frames: dict, features: list[str]) -> pd.DatetimeIndex:
    """VALIDATION dates valid in every frame under consideration.

    Returns the intersection of rows whose A2 features and target are all
    finite, restricted to the three validation years. One mask for all models
    and both regimes, so no model is ever scored on a different sample.

    This constrains scoring only - see run_walk_forward for why training must
    not be filtered by it.
    """
    val_years = {f["val_year"] for f in FOLDS}
    idx = None
    for df in frames.values():
        ok = df[features + ["y"]].notna().all(axis=1)
        d = df.index[ok & df.index.year.isin(val_years)]
        idx = d if idx is None else idx.intersection(d)
    return pd.DatetimeIndex(sorted(idx))


def run_walk_forward(train_df: pd.DataFrame, score_df: pd.DataFrame,
                     features: list[str], model_name: str,
                     scored: pd.DatetimeIndex) -> pd.DataFrame:
    """Expanding walk-forward. Returns one row per scored validation date.

    TRAINING and SCORING are deliberately separated, and getting this wrong is
    easy. Both regimes train on the SAME full ERA5 history (`train_df`); only
    the validation rows differ, taken from `score_df`. That is what makes the
    A/B difference attributable to target-day weather rather than to training
    volume - and it matches how Phase 5 ran (n_train 2,897 / 3,262 / 3,626).

    An earlier version of this function restricted training to `scored` as
    well. Because the Regime B frame only spans the forecast archive
    (2021-06 onward, 1,920 rows), that silently cut fold 1's training set from
    ~2,900 rows to 364 and changed every number in the table. The scoring mask
    constrains WHICH DAYS ARE SCORED, never what the model may learn from.
    """
    rows = []
    for f in FOLDS:
        tr = train_df[train_df.index.year <= f["train_end"]]
        va = score_df[(score_df.index.year == f["val_year"])
                      & score_df.index.isin(scored)]
        if va.empty:
            continue
        model = build_model(model_name).fit(tr[features], tr["y"])
        pred = model.predict(va[features])
        rows.append(pd.DataFrame({
            "date": va.index, "fold": f["name"], "val_year": f["val_year"],
            "y_true": va["y"].to_numpy(float), "y_pred": np.asarray(pred, float),
            "n_train": len(tr),
        }))
    return pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)


def evaluate(preds: pd.DataFrame) -> dict:
    """Overall, per-fold, summer and surge-day metrics for one model."""
    y, p, d = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(), preds["date"]
    out = {"overall": compute_metrics(y, p), "folds": {}}
    for f in FOLDS:
        m = preds["val_year"] == f["val_year"]
        if m.any():
            out["folds"][str(f["val_year"])] = compute_metrics(y[m.to_numpy()],
                                                               p[m.to_numpy()])
    summer = pd.to_datetime(d).dt.month.isin(SUMMER).to_numpy()
    out["summer_mar_jun"] = subset_metrics(d, y, p, summer, "summer_mar_jun")
    surge = surge_mask(d, y)
    out["surge_top5pct_per_year"] = subset_metrics(d, y, p, surge, "surge_top5pct")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated subset; default = whole zoo")
    ap.add_argument("--regime-b", action="store_true",
                    help="also score under archived lead-1 forecast weather")
    ap.add_argument("--out", default="phase6_benchmark.json")
    args = ap.parse_args()

    features = list(PHASE5_FEATURE_SETS[FEATURE_SET])
    names = ([m.strip() for m in args.models.split(",")] if args.models
             else list(MODEL_ZOO))
    for n in names:
        if n not in MODEL_ZOO:
            raise SystemExit(f"unknown model '{n}'; available: {list(MODEL_ZOO)}")

    print("=" * 78)
    print("PHASE 6 - MODEL BENCHMARK ON FROZEN A2 SPECIFICATION")
    print("=" * 78)
    print(f"  feature set   : {FEATURE_SET} ({len(features)} features)")
    print(f"  validation    : expanding walk-forward, val 2023 / 2024 / 2025")
    print(f"  primary metric: MAE   reference model: {REFERENCE_MODEL}")
    print(f"  bootstrap     : {BLOCK_DAYS}-day moving block, n={N_BOOT}, seed={BOOT_SEED}")
    print(f"  models        : {len(names)}")

    frames = {"A": load_regime("A", features)}
    if args.regime_b:
        frames["B"] = load_regime("B", features)
        print("  regimes       : A (reanalysis) + B (archived lead-1 forecast)")
    else:
        print("  regimes       : A (reanalysis) only")

    scored = common_mask(frames, features)
    print(f"\n  shared SCORED dates: {len(scored):,} "
          f"({scored.min():%Y-%m-%d} -> {scored.max():%Y-%m-%d})")
    print(f"  training frame     : Regime A (ERA5), {len(frames['A']):,} rows "
          f"- identical for both regimes")
    per_fold = {str(f["val_year"]): int((scored.year == f["val_year"]).sum())
                for f in FOLDS}
    print(f"  per fold           : {per_fold}")
    print("  (identical rows for every model and regime - no composition effects)")

    results, preds_long, timings = {}, [], {}
    for regime, df in frames.items():
        results[regime] = {}
        print(f"\n{'-' * 78}\nREGIME {regime}\n{'-' * 78}")
        print(f"  {'model':<22}{'MAE':>9}{'RMSE':>9}{'MAPE':>8}{'bias':>9}"
              f"{'surgeBias':>11}{'sec':>7}")
        for name in names:
            t0 = time.perf_counter()
            # train on ERA5 always; score on this regime's weather
            p = run_walk_forward(frames["A"], df, features, name, scored)
            dt = time.perf_counter() - t0
            ev = evaluate(p)
            results[regime][name] = ev
            timings[f"{regime}:{name}"] = round(dt, 2)
            p = p.assign(model=name, regime=regime)
            preds_long.append(p)
            o, s = ev["overall"], ev["surge_top5pct_per_year"]
            print(f"  {name:<22}{o['mae']:>9.2f}{o['rmse']:>9.2f}"
                  f"{o['mape']:>7.2f}%{o['bias']:>+9.1f}"
                  f"{s.get('bias', float('nan')):>+11.1f}{dt:>7.1f}")

    # ---- paired comparison against the incumbent -------------------------
    print(f"\n{'-' * 78}\nPAIRED vs {REFERENCE_MODEL.upper()} "
          f"(Regime A, negative = candidate better)\n{'-' * 78}")
    allp = pd.concat(preds_long, ignore_index=True)
    ref = allp[(allp.model == REFERENCE_MODEL) & (allp.regime == "A")] \
        .set_index("date").sort_index()
    comparisons = {}
    print(f"  {'model':<22}{'dMAE':>9}{'95% CI':>22}{'p':>9}  verdict")
    for name in names:
        if name == REFERENCE_MODEL:
            continue
        cand = allp[(allp.model == name) & (allp.regime == "A")] \
            .set_index("date").sort_index()
        d = (cand["y_pred"] - cand["y_true"]).abs() - \
            (ref["y_pred"] - ref["y_true"]).abs()
        bs = moving_block_bootstrap(d.to_numpy())
        comparisons[name] = bs
        sig = bs["ci_95_high"] < 0
        worse = bs["ci_95_low"] > 0
        verdict = ("BETTER (CI excludes 0)" if sig else
                   "worse (CI excludes 0)" if worse else "no difference")
        print(f"  {name:<22}{bs['mean']:>+9.2f}"
              f"  [{bs['ci_95_low']:>+8.2f},{bs['ci_95_high']:>+8.2f}]"
              f"{bs['p_value_one_sided']:>9.4f}  {verdict}")

    # ---- ranking ---------------------------------------------------------
    rank = sorted(results["A"].items(), key=lambda kv: kv[1]["overall"]["mae"])
    print(f"\n{'-' * 78}\nRANKING BY MEAN VALIDATION MAE (Regime A)\n{'-' * 78}")
    for i, (name, ev) in enumerate(rank, 1):
        fam = MODEL_ZOO[name]["family"]
        print(f"  {i}. {name:<22} {ev['overall']['mae']:>8.2f} MW   ({fam})")
    champion = rank[0][0]
    runner_up = rank[1][0] if len(rank) > 1 else None
    print(f"\n  provisional leader : {champion}")
    print(f"  runner-up          : {runner_up}")
    print("  NOTE: leader is provisional. Confirm with the paired CI above and")
    print("        with Regime B before calling anything champion.")

    payload = {
        "phase": "6_model_benchmark",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_set": FEATURE_SET,
        "feature_count": len(features),
        "features": features,
        "H": 2,
        "validation_protocol": "expanding walk-forward, val 2023/2024/2025",
        "primary_metric": "MAE",
        "reference_model": REFERENCE_MODEL,
        "regimes_evaluated": list(frames),
        "scored_dates": {
            "total": int(len(scored)),
            "per_fold": per_fold,
            "first": str(scored.min().date()), "last": str(scored.max().date()),
            "sha256": __import__("hashlib").sha256(
                ",".join(d.strftime("%Y-%m-%d") for d in scored).encode()
            ).hexdigest(),
        },
        "bootstrap": {"method": "moving_block", "block_days": BLOCK_DAYS,
                      "n_boot": N_BOOT, "seed": BOOT_SEED,
                      "direction": "d = |e_candidate| - |e_ridge|; negative = candidate better"},
        "model_notes": {k: v["note"] for k, v in MODEL_ZOO.items() if k in names},
        "results": results,
        "paired_vs_reference": comparisons,
        "ranking_regime_a": [{"model": n, "mae": ev["overall"]["mae"],
                              "family": MODEL_ZOO[n]["family"]}
                             for n, ev in rank],
        "provisional_leader": champion,
        "runner_up": runner_up,
        "fit_seconds": timings,
        "environment": {"python": platform.python_version(),
                        "seed": RANDOM_SEED or 42},
        "caveats": [
            "Models use untuned conservative defaults; only finalists are tuned (Phase 6b).",
            "Regime A is a reanalysis benchmark and is not operationally achievable; "
            "quote Regime B for deployable accuracy.",
            "2026 is NOT touched here. It was inspected during Track A diagnostics "
            "and is a confirmation set, not a blind test.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / args.out
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    out_csv = RESULTS / args.out.replace(".json", "_predictions.csv")
    allp.to_csv(out_csv, index=False)
    print(f"\nwrote {out_json}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
