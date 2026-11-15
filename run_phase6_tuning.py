"""
Phase 6b - hyperparameter tuning of the top-2 benchmark candidates.

    python run_phase6_tuning.py                  # tune both, then Regime B
    python run_phase6_tuning.py --trials 30
    python run_phase6_tuning.py --models ridge   # tune one

Why only two models
-------------------
The roadmap is explicit: rank by mean validation MAE, take the top two, tune
only those. Tuning the whole zoo would multiply the search space over model
families that the benchmark already showed are the wrong shape for this
problem (all four pure GBMs lost to plain Ridge with CIs excluding zero), and
every extra tuned candidate is another chance to overfit the selection
process itself.

The two carried forward:

  ridge_gbm_hybrid   benchmark leader, 465.88 MW
  ridge              incumbent, 571.03 MW - and a specific open question:
                     OLS and Ridge came out IDENTICAL to two decimals, so at
                     alpha=0.001 the L2 penalty is doing nothing at all.
                     alpha has never actually been tuned; it was inherited.

Protocol
--------
* Objective = **mean validation MAE across the three expanding folds**, the
  same quantity the benchmark ranked on and the same one the Phase 5 decision
  gate used. Not RMSE, not a single fold.
* Regime A (reanalysis) for tuning. Weather realism is a separate question,
  settled afterwards by scoring the tuned finalists under Regime B.
* The shared scored-date mask from the benchmark, so tuned and untuned numbers
  are directly comparable.
* <=50 trials per model, TPE sampler, fixed seed.
* **2026 is never touched here.** It is a confirmation set, inspected once
  after the champion is frozen - not a tuning signal.

Champion selection
------------------
The champion is chosen on the **Regime B** number, not Regime A. Regime A is a
reanalysis benchmark that cannot be reproduced on any real evening; Regime B
is the deployable configuration. The benchmark already showed the A->B
degradation is model-dependent (hybrid +12.98% vs Ridge +7.06%), which is
exactly why the ranking must be confirmed in the operational regime rather
than assumed to carry over.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy / sklearn / lightgbm

import argparse
import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd

from config import MODELS, RANDOM_SEED, RESULTS
from src.contract import PHASE5_FEATURE_SETS
from src.evaluation import compute_metrics, moving_block_bootstrap
from src.models import REFERENCE_MODEL
from src.envinfo import capture_environment
from src.ridge import TrackARidge
from run_phase6_benchmark import (FEATURE_SET, FOLDS, common_mask, evaluate,
                                  load_regime, run_walk_forward)

SEED = RANDOM_SEED or 42
TOP2 = ["ridge_gbm_hybrid", "ridge"]


# --------------------------------------------------------------------------
# tunable constructors
# --------------------------------------------------------------------------

def _ridge(params):
    return TrackARidge(alpha=params["alpha"])


def _hybrid(params):
    from src.models import RidgeGBMHybrid
    return RidgeGBMHybrid(alpha=params["alpha"],
                          n_estimators=params["n_estimators"],
                          learning_rate=params["learning_rate"],
                          num_leaves=params["num_leaves"])


SPACES = {
    "ridge": {
        "build": _ridge,
        # alpha spans 8 orders of magnitude because the benchmark showed the
        # inherited 0.001 is indistinguishable from OLS - the useful range,
        # if there is one, is somewhere above it.
        "suggest": lambda t: {"alpha": t.suggest_float("alpha", 1e-6, 1e3, log=True)},
    },
    "ridge_gbm_hybrid": {
        "build": _hybrid,
        # The residual GBM is kept deliberately small. It is correcting a
        # residual, not learning the series: a strong learner here would
        # overwrite Ridge's extrapolation with leaf averages and reintroduce
        # the very failure mode the hybrid exists to avoid.
        "suggest": lambda t: {
            "alpha": t.suggest_float("alpha", 1e-6, 1e3, log=True),
            "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": t.suggest_int("num_leaves", 3, 31),
        },
    },
}


def fold_mae(model_builder, params, df, features, scored) -> float:
    """Mean MAE across the three expanding folds for one parameter set."""
    maes = []
    for f in FOLDS:
        tr = df[df.index.year <= f["train_end"]]
        va = df[(df.index.year == f["val_year"]) & df.index.isin(scored)]
        if va.empty:
            continue
        m = model_builder(params).fit(tr[features], tr["y"])
        p = np.asarray(m.predict(va[features]), float)
        maes.append(float(np.abs(p - va["y"].to_numpy(float)).mean()))
    return float(np.mean(maes))


def tune(name, df, features, scored, n_trials) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = SPACES[name]

    def objective(trial):
        return fold_mae(space["build"], space["suggest"](trial),
                        df, features, scored)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"phase6b_{name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "model": name,
        "n_trials": n_trials,
        "best_params": study.best_params,
        "best_mean_fold_mae": float(study.best_value),
        "trials": [{"number": t.number, "value": t.value, "params": t.params}
                   for t in study.trials if t.value is not None],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50,
                    help="Optuna trials per model (roadmap cap: 50)")
    ap.add_argument("--models", default=",".join(TOP2))
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",")]

    feats = list(PHASE5_FEATURE_SETS[FEATURE_SET])
    df_a = load_regime("A", feats)
    df_b = load_regime("B", feats)
    scored = common_mask({"A": df_a, "B": df_b}, feats)

    print("=" * 76)
    print("PHASE 6b - TUNING THE TOP-2 CANDIDATES")
    print("=" * 76)
    print(f"  feature set   : {FEATURE_SET} ({len(feats)} features)")
    print(f"  objective     : mean validation MAE across folds 2023/2024/2025")
    print(f"  tuning regime : A (reanalysis)   budget: {args.trials} trials/model")
    print(f"  scored dates  : {len(scored):,} (shared with Regime B)")
    print(f"  2026          : NOT touched - confirmation set only")

    tuned, summary = {}, []
    for name in names:
        print(f"\n--- tuning {name} ({args.trials} trials) ---")
        res = tune(name, df_a, feats, scored, args.trials)
        tuned[name] = res
        vals = [t["value"] for t in res["trials"]]
        print(f"  best mean fold MAE : {res['best_mean_fold_mae']:.2f} MW")
        print(f"  best params        : {res['best_params']}")
        print(f"  trial MAE range    : {min(vals):.2f} - {max(vals):.2f} MW")

    # ---- score tuned models in BOTH regimes --------------------------
    print(f"\n{'=' * 76}\nTUNED FINALISTS - BOTH REGIMES (shared {len(scored):,} dates)\n{'=' * 76}")
    print(f"  {'model':<22}{'regimeA':>10}{'regimeB':>10}{'degrade':>10}"
          f"{'degr%':>8}{'surgeB_B':>11}")
    final, preds_all = {}, []
    for name in names:
        space = SPACES[name]
        params = tuned[name]["best_params"]
        builder = lambda p=params, s=space: s["build"](p)

        row = {}
        for regime, frame in (("A", df_a), ("B", df_b)):
            rows = []
            for f in FOLDS:
                tr = df_a[df_a.index.year <= f["train_end"]]      # always ERA5
                va = frame[(frame.index.year == f["val_year"])
                           & frame.index.isin(scored)]
                m = builder().fit(tr[feats], tr["y"])
                rows.append(pd.DataFrame({
                    "date": va.index, "fold": f["name"], "val_year": f["val_year"],
                    "y_true": va["y"].to_numpy(float),
                    "y_pred": np.asarray(m.predict(va[feats]), float),
                    "n_train": len(tr)}))
            p = pd.concat(rows, ignore_index=True).sort_values("date")
            row[regime] = evaluate(p)
            preds_all.append(p.assign(model=name, regime=regime, tuned=True))

        a, b = row["A"]["overall"]["mae"], row["B"]["overall"]["mae"]
        row["degradation_mw"] = b - a
        row["degradation_pct"] = (b / a - 1) * 100
        final[name] = row
        print(f"  {name:<22}{a:>10.2f}{b:>10.2f}{b - a:>+10.2f}"
              f"{row['degradation_pct']:>7.2f}%"
              f"{row['B']['surge_top5pct_per_year']['bias']:>+11.1f}")

    # ---- champion on the OPERATIONAL number --------------------------
    champ = min(final, key=lambda n: final[n]["B"]["overall"]["mae"])
    other = [n for n in names if n != champ]
    print(f"\n  CHAMPION (selected on Regime B, the deployable regime): {champ}")

    comparison = None
    if other:
        allp = pd.concat(preds_all, ignore_index=True)
        c = allp[(allp.model == champ) & (allp.regime == "B")].set_index("date").sort_index()
        o = allp[(allp.model == other[0]) & (allp.regime == "B")].set_index("date").sort_index()
        d = (c["y_pred"] - c["y_true"]).abs() - (o["y_pred"] - o["y_true"]).abs()
        comparison = moving_block_bootstrap(d.to_numpy())
        print(f"  paired vs {other[0]} in Regime B: {comparison['mean']:+.2f} MW "
              f"[{comparison['ci_95_low']:+.2f}, {comparison['ci_95_high']:+.2f}]  "
              f"p={comparison['p_value_one_sided']:.4f}")

    # ---- artifacts ---------------------------------------------------
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    champion_config = {
        "phase": "6b_tuning",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "champion": champ,
        "selected_on": "Regime B (archived lead-1 forecast weather) mean validation MAE",
        "selection_rationale": (
            "Regime A is a reanalysis benchmark and is not reproducible on any "
            "real forecast evening. The benchmark showed A->B degradation is "
            "model-dependent, so the operational regime decides."),
        "model_type": champ,
        "hyperparameters": tuned[champ]["best_params"],
        "feature_set": FEATURE_SET,
        "feature_count": len(feats),
        # Serving contract. predict_next_day.py reads both rather than
        # inferring them; getting either wrong is silent. See test_serving.py.
        "target_parameterisation": "level",
        "lag_representation": "literal",
        "serving_notes": {
            "target_parameterisation": (
                "Model predicts tn_peak_demand_mw directly. Do NOT add an "
                "anchor. Adding peak_lag_1 would roughly double the forecast."),
            "lag_representation": (
                "peak_lag_N / energy_lag_N are LITERAL calendar offsets (T-N), "
                "as produced by load_phase5_data and build_dataset.add_lags. "
                "Do NOT apply config.H_BY_GROUP shifts when serving: "
                "load_master(H=2) shifts the demand block one further day, so "
                "its 'peak_lag_2' holds T-3 and its 'peak_lag_7' lands on the "
                "wrong weekday. H=2 is honoured by EXCLUDING peak_lag_1 and "
                "energy_lag_1, not by shifting."),
        },
        "features": feats,
        "H": 2,
        "validation_protocol": "expanding walk-forward, val 2023/2024/2025",
        "primary_metric": "MAE",
        "tuning": {"sampler": "TPE", "n_trials": args.trials,
                   "objective": "mean validation MAE across 3 folds",
                   "regime": "A", "seed": SEED},
        "validation_performance": {
            "regime_a_mae": final[champ]["A"]["overall"]["mae"],
            "regime_b_mae": final[champ]["B"]["overall"]["mae"],
            "degradation_mw": final[champ]["degradation_mw"],
            "degradation_pct": final[champ]["degradation_pct"],
        },
        # Versions that produced these numbers. The champion is re-fitted
        # rather than unpickled, so the config alone does not pin the result.
        "environment": capture_environment(),
        "frozen": True,
        "note_2026": ("2026 has NOT been used in tuning or selection. It was "
                      "inspected during Track A diagnostics, so it is a "
                      "confirmation set, not a blind test."),
    }
    (MODELS / "champion_config.json").write_text(
        json.dumps(champion_config, indent=2, default=str) + "\n")

    payload = {
        "phase": "6b_tuning_and_champion_regime_ab",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_set": FEATURE_SET, "H": 2,
        "scored_dates": {"total": int(len(scored)),
                         "first": str(scored.min().date()),
                         "last": str(scored.max().date())},
        "tuning": {k: {kk: vv for kk, vv in v.items() if kk != "trials"}
                   for k, v in tuned.items()},
        "tuning_trials": {k: v["trials"] for k, v in tuned.items()},
        "finalists_both_regimes": final,
        "champion": champ,
        "champion_vs_runner_up_regime_b": comparison,
        "caveats": [
            "Champion selected on Regime B; Regime A is a diagnostic upper bound.",
            "2026 untouched - confirmation set, not a blind test.",
            "Forecast issue hour unverified; archived lead-1 aligned to valid date.",
        ],
        "environment": {"python": platform.python_version(), "seed": SEED},
    }
    (RESULTS / "champion_regime_ab.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    pd.concat(preds_all, ignore_index=True).to_csv(
        RESULTS / "champion_regime_ab_predictions.csv", index=False)

    print(f"\nwrote {MODELS / 'champion_config.json'}")
    print(f"wrote {RESULTS / 'champion_regime_ab.json'}")
    print(f"wrote {RESULTS / 'champion_regime_ab_predictions.csv'}")


if __name__ == "__main__":
    main()
