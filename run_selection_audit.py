"""
W0-1 - quantify selection optimism by replaying the whole pipeline honestly.

    python run_selection_audit.py [--trials 50]

THE PROBLEM. The same 2023-2025 validation set drove four sequential
selection decisions: the A1->A2 weather gate, the A2->A3.4 gate, the 9-model
ranking, and 100 Optuna trials. Every number the project reports is therefore
a quantity that selection was optimised against, and until now there was no
estimate of how much of the reported skill is selection noise. A referee will
ask. This script answers.

THE METHOD. Hold 2025 out completely, then rerun selection using only data
through 2024:

    inner folds (selection):  train <=2022 -> val 2023
                              train <=2023 -> val 2024
    outer holdout (scored once): train <=2024 -> 2025

Inside the inner folds the script re-selects, in the same order and with the
same decision regimes the real pipeline used:

    1. feature set     A0 / A1 / A2 / A4        (Regime A, as the a1/a2 gates ran)
    2. model           all 9 zoo candidates     (Regime A, as the benchmark ran)
    3. hyperparameters Optuna TPE, top-2 only   (Regime A objective, as recorded
                                                  in champion_config tuning.regime)
    4. champion        best of the top-2        (Regime B, the operational cell)

Then the resulting champion is fitted on <=2024 and scored ONCE on 2025.

    optimism = honest_2025_MAE - reported_2025_fold_MAE

POSITIVE optimism means the reported figure is flatteringly low: selection
bought skill that does not survive on data it never saw. NEGATIVE optimism
means the reported figure is conservative - the honest pipeline did at least
as well, so the reported number is not inflated by selection.

READING THE RESULT. A small optimism means the project's numbers are robust
and that can now be said with evidence. A large one means the honest figure
is the one to report - and the paper is stronger for having measured it
rather than having been asked. Either outcome is a result.

WHAT THIS CANNOT RECOVER. 2026 was already inspected during Track A feature
diagnostics (diagnostic_2026_results blocks in experiment_a1/a2/a3), so it is
a confirmation set, not a blind test, and this script does not touch it. The
estimate below is also a LOWER BOUND on total optimism: it replays the
selection rounds that are reproducible from the committed code, not the
informal looking-at-results that shaped the search spaces themselves.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy / sklearn / lightgbm

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import PHASE5_FEATURE_SETS, load_phase5_data
from src.envinfo import capture_environment
from src.evaluation import compute_metrics, moving_block_bootstrap
from src.models import MODEL_ZOO, build_model

RESULTS = Path("results")
CHAMPION = Path("models/champion_config.json")

CANDIDATE_SETS = ["A0_weather_free", "A1_weather_levels",
                  "A2_weather_deltas_anomalies", "A4_festival"]

# Selection sees these; 2025 is the outer holdout and is scored once.
INNER_FOLDS = [{"train_end": 2022, "val_year": 2023},
               {"train_end": 2023, "val_year": 2024}]
OUTER = {"train_end": 2024, "val_year": 2025}


def frames_for(feature_set: str) -> dict:
    out = {}
    for regime, source in (("A", "reanalysis"), ("B", "forecast")):
        df, feats = load_phase5_data(feature_set, weather_source=source,
                                     forecast_lead=1)
        want = list(PHASE5_FEATURE_SETS[feature_set])
        assert feats == want, f"{feature_set} drifted"
        out[regime] = df
    return out


def fit_score(train_df, score_df, feats, model_name, hp, dates):
    """Fit on train_df, return (y_true, y_pred) on `dates` of score_df."""
    ok = train_df["y"].notna() & train_df[feats].notna().all(axis=1)
    m = build_model(model_name, **(hp or {})).fit(train_df.loc[ok, feats],
                                                  train_df.loc[ok, "y"])
    va = score_df.loc[score_df.index.isin(dates)]
    return va["y"].to_numpy(float), np.asarray(m.predict(va[feats]), float)


def inner_mae(fr, feats, model_name, hp, regime, scored):
    """Mean MAE across the two INNER folds. Trains on Regime A throughout."""
    maes = []
    for f in INNER_FOLDS:
        tr = fr["A"][fr["A"].index.year <= f["train_end"]]
        dates = scored[scored.year == f["val_year"]]
        if len(dates) == 0:
            continue
        yt, yp = fit_score(tr, fr[regime], feats, model_name, hp, dates)
        maes.append(float(np.abs(yp - yt).mean()))
    return float(np.mean(maes)) if maes else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50,
                    help="Optuna trials per finalist (match Phase 6b: 50)")
    args = ap.parse_args()

    cfg = json.loads(CHAMPION.read_text())
    print("=" * 74)
    print("W0-1  SELECTION OPTIMISM AUDIT")
    print("=" * 74)
    print("outer holdout 2025 - never seen by any selection step below")
    print(f"inner folds   {[f['val_year'] for f in INNER_FOLDS]}")

    # ---- the scoring sample, taken from the REPORTED protocol ---------
    # The reported 2025 fold was scored on the Phase 6 common mask (A2, both
    # regimes finite). The honest run must score the same dates or the
    # comparison is confounded by sample composition. A0/A1 are subsets of A2
    # and A4 adds only never-NaN calendar columns, so every candidate set is
    # finite wherever A2 is.
    a2 = frames_for("A2_weather_deltas_anomalies")
    f2 = list(PHASE5_FEATURE_SETS["A2_weather_deltas_anomalies"])
    val_years = {f["val_year"] for f in INNER_FOLDS} | {OUTER["val_year"]}
    scored = None
    for r in ("A", "B"):
        ok = a2[r][f2 + ["y"]].notna().all(axis=1)
        d = a2[r].index[ok & a2[r].index.year.isin(val_years)]
        scored = d if scored is None else scored.intersection(d)
    scored = pd.DatetimeIndex(sorted(scored))
    outer_dates = scored[scored.year == OUTER["val_year"]]
    print(f"scoring sample {len(scored)} dates; outer 2025 = {len(outer_dates)} dates")

    cache = {s: frames_for(s) for s in CANDIDATE_SETS}
    audit = {"rounds": {}}

    # ---- round 1: feature set, on Regime A (as the a1/a2 gates ran) ---
    print("\n" + "-" * 74)
    print("ROUND 1  feature set  (inner folds, Regime A)")
    print("-" * 74)
    r1 = {}
    for s in CANDIDATE_SETS:
        feats = list(PHASE5_FEATURE_SETS[s])
        r1[s] = inner_mae(cache[s], feats, "ridge", None, "A", scored)
        print(f"  {s:<32}{len(feats):>3} feats   {r1[s]:>8.2f} MW")
    sel_set = min(r1, key=r1.get)
    print(f"  -> selected {sel_set}")
    audit["rounds"]["1_feature_set"] = {
        "decided_on": "Regime A, mean inner-fold MAE, ridge probe",
        "candidates": r1, "selected": sel_set}

    feats = list(PHASE5_FEATURE_SETS[sel_set])
    fr = cache[sel_set]

    # ---- round 2: model ranking, on Regime A -------------------------
    print("\n" + "-" * 74)
    print("ROUND 2  model ranking  (inner folds, Regime A)")
    print("-" * 74)
    r2 = {}
    for name in MODEL_ZOO:
        try:
            r2[name] = inner_mae(fr, feats, name, None, "A", scored)
        except ImportError as e:
            print(f"  [SKIP] {name}: {e}")
            continue
        print(f"  {name:<24}{r2[name]:>8.2f} MW")
    top2 = sorted(r2, key=r2.get)[:2]
    print(f"  -> finalists {top2}")
    audit["rounds"]["2_model"] = {
        "decided_on": "Regime A, mean inner-fold MAE, default hyperparameters",
        "candidates": r2, "finalists": top2}

    # ---- round 3: tuning, Regime A objective -------------------------
    print("\n" + "-" * 74)
    print(f"ROUND 3  tuning  ({args.trials} TPE trials per finalist, Regime A)")
    print("-" * 74)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def space(trial, name):
        if name == "ridge_gbm_hybrid":
            return {"alpha": trial.suggest_float("alpha", 1e-6, 1e3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 3, 31)}
        if name in ("ridge", "ols"):
            return {"alpha": trial.suggest_float("alpha", 1e-6, 1e3, log=True)}
        if name in ("lightgbm", "lightgbm_linear"):
            return {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 3, 31)}
        if name in ("xgboost", "catboost"):
            return {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True)}
        return {}

    # Baselines echo a column; there is nothing to tune and Optuna would
    # spend 50 trials rediscovering the same number.
    PARAMETERLESS = {"naive_persistence", "seasonal_naive"}

    tuned = {}
    for name in top2:
        if name in PARAMETERLESS:
            tuned[name] = {"best_params": {}, "best_value": r2[name]}
            print(f"  {name:<24}{r2[name]:>8.2f} MW   (no hyperparameters)")
            continue
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
        st.optimize(lambda t, n=name: inner_mae(fr, feats, n, space(t, n), "A", scored),
                    n_trials=args.trials, show_progress_bar=False)
        tuned[name] = {"best_params": st.best_params, "best_value": st.best_value}
        print(f"  {name:<24}{st.best_value:>8.2f} MW   {st.best_params}")
    audit["rounds"]["3_tuning"] = {
        "decided_on": f"Regime A inner-fold objective, {args.trials} TPE trials, seed 42",
        "results": tuned}

    # ---- round 4: champion, on Regime B (operational cell) -----------
    print("\n" + "-" * 74)
    print("ROUND 4  champion  (inner folds, Regime B)")
    print("-" * 74)
    r4 = {}
    for name in top2:
        r4[name] = inner_mae(fr, feats, name, tuned[name]["best_params"], "B", scored)
        print(f"  {name:<24}{r4[name]:>8.2f} MW")
    sel_model = min(r4, key=r4.get)
    sel_hp = tuned[sel_model]["best_params"]
    print(f"  -> champion {sel_model}  {sel_hp}")
    audit["rounds"]["4_champion"] = {
        "decided_on": "Regime B, mean inner-fold MAE", "candidates": r4,
        "selected": sel_model, "hyperparameters": sel_hp}

    # ---- the outer holdout, scored ONCE ------------------------------
    print("\n" + "=" * 74)
    print("OUTER HOLDOUT 2025 - scored once")
    print("=" * 74)
    tr = fr["A"][fr["A"].index.year <= OUTER["train_end"]]
    honest, reported, optimism = {}, {}, {}
    for regime in ("A", "B"):
        yt, yp = fit_score(tr, fr[regime], feats, sel_model, sel_hp, outer_dates)
        honest[regime] = compute_metrics(yt, yp)
        reported[regime] = cfg["validation_performance"][
            f"regime_{regime.lower()}_mae"]
        rep_fold = json.loads(
            (RESULTS / "champion_regime_ab.json").read_text()
        )["finalists_both_regimes"]["ridge_gbm_hybrid"][regime]["folds"]["2025"]["mae"]
        opt = honest[regime]["mae"] - rep_fold
        optimism[regime] = {
            "reported_2025_fold_mae": rep_fold,
            "honest_2025_mae": honest[regime]["mae"],
            # Sign: positive = the reported figure is flatteringly low.
            "optimism_mw": opt,
            "optimism_sign_note": (
                "positive => reported is optimistic; negative => reported is "
                "conservative and selection bought no unearned skill"),
            "reported_overall_val_mae": reported[regime],
        }
        print(f"\nRegime {regime}:")
        print(f"  reported 2025 fold (selection saw 2025) : {rep_fold:8.2f} MW")
        print(f"  honest   2025      (selection did not)  : {honest[regime]['mae']:8.2f} MW")
        print(f"  optimism (honest - reported)            : {opt:+8.2f} MW"
              f"   {'<- reported is OPTIMISTIC' if opt > 0 else '<- reported is CONSERVATIVE'}")

    # Paired comparison on the SAME dates: reported champion vs honest champion.
    rep_feats = list(PHASE5_FEATURE_SETS[cfg["feature_set"]])
    rep_fr = cache[cfg["feature_set"]]
    rep_tr = rep_fr["A"][rep_fr["A"].index.year <= OUTER["train_end"]]
    yt_r, yp_r = fit_score(rep_tr, rep_fr["B"], rep_feats, cfg["model_type"],
                           cfg["hyperparameters"], outer_dates)
    yt_h, yp_h = fit_score(tr, fr["B"], feats, sel_model, sel_hp, outer_dates)
    assert np.allclose(yt_r, yt_h), "paired comparison is misaligned"
    d = np.abs(yp_h - yt_h) - np.abs(yp_r - yt_r)
    bs = moving_block_bootstrap(d)
    print(f"\npaired on 2025 (n={len(d)}), honest champion - reported champion:")
    print(f"  delta MAE {d.mean():+.2f} MW   95% CI "
          f"[{bs['ci_95_low']:+.2f}, {bs['ci_95_high']:+.2f}]")

    same = (sel_set == cfg["feature_set"] and sel_model == cfg["model_type"])
    print(f"\ninner selection reproduced the committed choice: {same}")
    print(f"  feature set  {cfg['feature_set']}  ->  {sel_set}")
    print(f"  model        {cfg['model_type']}  ->  {sel_model}")

    audit.update({
        "phase": "w0_1_selection_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "outer_holdout": OUTER["val_year"],
            "inner_folds": INNER_FOLDS,
            "n_scored_outer": len(outer_dates),
            "trials_per_finalist": args.trials,
            "scoring_sample": ("Phase 6 common mask (A2 finite in both "
                               "regimes), restricted to the holdout year"),
        },
        "selected_by_honest_pipeline": {
            "feature_set": sel_set, "model": sel_model,
            "hyperparameters": sel_hp,
            "matches_committed_champion": bool(same),
        },
        "committed_champion": {
            "feature_set": cfg["feature_set"], "model": cfg["model_type"],
            "hyperparameters": cfg["hyperparameters"],
        },
        "optimism": optimism,
        "honest_metrics_2025": honest,
        "paired_vs_reported_champion_regime_b": {
            "delta_mae_mw": float(d.mean()),
            "ci_95_low": bs["ci_95_low"], "ci_95_high": bs["ci_95_high"],
            "n": int(len(d)),
        },
        "caveats": [
            "2026 is untouched here and remains a confirmation set, not a "
            "blind test - it was inspected during Track A diagnostics.",
            "This is a LOWER BOUND on total optimism: it replays the "
            "selection rounds reproducible from committed code, not the "
            "informal inspection that shaped the search spaces.",
            "The A2->A3.4 gate is not replayed because the A3 variants were "
            "never registered in PHASE5_FEATURE_SETS.",
        ],
        "environment": capture_environment(),
    })
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selection_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n")
    print(f"\nwrote {RESULTS / 'selection_audit.json'}")


if __name__ == "__main__":
    main()
