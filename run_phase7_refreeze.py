"""
Phase 7 champion re-freeze.

    python run_phase7_refreeze.py [--trials 50]

W1-2 established that a quantile residual head beats the mean-targeting one by
29.45 MW on untouched 2025 (95% CI [-55.11, -3.09], DM p=0.025). W1-3
established that recency weighting does not help. The champion therefore needs
re-freezing, once, before the paper assets (W2) and the single 2026
confirmation (W0-5) are produced against it.

PROTOCOL - deliberately identical to Phase 6b, so the new champion is
comparable with the committed numbers rather than being measured differently:

  * tune each finalist with TPE on the standard 3 folds (val 2023/24/25),
    Regime A objective, seed 42 - matching champion_config's tuning.regime
  * select between finalists on Regime B mean validation MAE, the operational
    cell
  * feature set stays A2 (26 features). A4_festival is a documented extension
    per the W1-1 gate, not the headline champion.

The honest, selection-free estimates for whatever this picks are produced
separately by run_selection_audit.py; they are carried into the config so the
serving path and the paper quote the same pair of numbers.
"""

from __future__ import annotations

import threadlimit  # noqa: F401

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import PHASE5_FEATURE_SETS
from src.envinfo import capture_environment
from src.evaluation import (compute_metrics, diebold_mariano,
                            moving_block_bootstrap, surge_mask)
from src.models import build_model
from run_phase6_benchmark import FOLDS
from run_phase7_festival import load_set, walk_forward

RESULTS = Path("results")
MODELS = Path("models")
FEATURE_SET = "A2_weather_deltas_anomalies"
SEED = 42

SPACES = {
    "ridge_gbm_hybrid": lambda t: {
        "alpha": t.suggest_float("alpha", 1e-6, 1e3, log=True),
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": t.suggest_int("num_leaves", 3, 31)},
    "quantile_hybrid": lambda t: {
        "alpha": t.suggest_float("alpha", 1e-6, 1e3, log=True),
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": t.suggest_int("num_leaves", 3, 31),
        "quantile": t.suggest_float("quantile", 0.50, 0.90)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

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
    print("PHASE 7 CHAMPION RE-FREEZE")
    print("=" * 78)
    print(f"{FEATURE_SET} ({len(feats)} features), {len(scored)} scored dates")
    print(f"{args.trials} TPE trials per finalist, Regime A objective, seed {SEED}")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def fold_mae(params, model_name, regime="A"):
        maes = []
        for f in FOLDS:
            tr = frames["A"][frames["A"].index.year <= f["train_end"]]
            va = frames[regime][(frames[regime].index.year == f["val_year"])
                                & frames[regime].index.isin(scored)]
            if va.empty:
                continue
            ok = tr["y"].notna() & tr[feats].notna().all(axis=1)
            m = build_model(model_name, **params).fit(tr.loc[ok, feats],
                                                      tr.loc[ok, "y"])
            maes.append(float(np.abs(m.predict(va[feats])
                                     - va["y"].to_numpy(float)).mean()))
        return float(np.mean(maes))

    tuned = {}
    for name, space in SPACES.items():
        print(f"\n  tuning {name} ...")
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(lambda t, n=name, s=space: fold_mae(s(t), n),
                    n_trials=args.trials, show_progress_bar=False)
        tuned[name] = {"best_params": st.best_params, "best_value": st.best_value}
        print(f"    best Regime A fold MAE {st.best_value:.2f}")
        print(f"    {st.best_params}")

    # ---- select on Regime B, the operational cell ---------------------
    print("\n" + "-" * 78)
    print(f"{'finalist':<22}{'Regime A':>12}{'Regime B':>12}{'A->B':>10}")
    print("-" * 78)
    preds, mets = {}, {}
    for name in SPACES:
        hp = tuned[name]["best_params"]
        for r in ("A", "B"):
            p = walk_forward(frames["A"], frames[r], feats, name, scored, hp)
            preds[(name, r)] = p
            mets[(name, r)] = compute_metrics(p.y_true.to_numpy(),
                                              p.y_pred.to_numpy())
        a, b = mets[(name, "A")]["mae"], mets[(name, "B")]["mae"]
        print(f"{name:<22}{a:>12.2f}{b:>12.2f}{b - a:>+10.2f}")

    champ = min(SPACES, key=lambda n: mets[(n, "B")]["mae"])
    other = [n for n in SPACES if n != champ][0]
    print(f"\n  champion: {champ}")

    # paired, against the model it displaces
    pc, po = preds[(champ, "B")], preds[(other, "B")]
    assert (pc.date.to_numpy() == po.date.to_numpy()).all()
    ec = pc.y_pred.to_numpy() - pc.y_true.to_numpy()
    eo = po.y_pred.to_numpy() - po.y_true.to_numpy()
    bs = moving_block_bootstrap(np.abs(ec) - np.abs(eo))
    dm = diebold_mariano(ec, eo, loss="abs")
    print(f"  vs {other} on Regime B: {np.mean(np.abs(ec)-np.abs(eo)):+.2f} MW  "
          f"95% CI [{bs['ci_95_low']:+.2f}, {bs['ci_95_high']:+.2f}]  "
          f"DM p={dm['p_value']:.2e}")

    # surge / segment picture for the new champion
    sm = surge_mask(pd.DatetimeIndex(pc.date), pc.y_true.to_numpy())
    print(f"  surge top-5%: MAE {np.abs(ec[sm]).mean():.1f}  "
          f"bias {ec[sm].mean():+.1f} (n={int(sm.sum())})")

    # ---- write the config -------------------------------------------
    hp = tuned[champ]["best_params"]
    prev = json.loads((MODELS / "champion_config.json").read_text())
    shutil.copy(MODELS / "champion_config.json",
                MODELS / "champion_config.phase6b.json")

    surge = json.loads((RESULTS / "phase7_surge.json").read_text())
    audit = json.loads((RESULTS / "selection_audit.json").read_text())

    cfg = {
        "phase": "7_refreeze",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "champion": champ,
        "selected_on": ("Regime B (archived lead-1 forecast weather) mean "
                        "validation MAE"),
        "selection_rationale": (
            "Regime A is a reanalysis benchmark and is not reproducible on any "
            "real forecast evening, so the operational regime decides. The "
            "quantile residual head replaced the mean-targeting one because "
            "MAE is minimised by the conditional median, not the mean, and the "
            "previous champion carried a -69.6 MW bias."),
        "supersedes": {
            "config": "models/champion_config.phase6b.json",
            "model": prev["model_type"],
            "regime_b_mae": prev["validation_performance"]["regime_b_mae"],
        },
        "model_type": champ,
        "hyperparameters": hp,
        "feature_set": FEATURE_SET,
        "feature_count": len(feats),
        "target_parameterisation": "level",
        "lag_representation": "literal",
        "serving_notes": prev["serving_notes"],
        "features": feats,
        "H": 2,
        "validation_protocol": "expanding walk-forward, val 2023/2024/2025",
        "primary_metric": "MAE",
        "tuning": {"sampler": "TPE", "n_trials": args.trials,
                   "objective": "mean validation MAE across 3 folds",
                   "regime": "A", "seed": SEED,
                   "finalists": list(SPACES)},
        "validation_performance": {
            "regime_a_mae": mets[(champ, "A")]["mae"],
            "regime_b_mae": mets[(champ, "B")]["mae"],
            "degradation_mw": mets[(champ, "B")]["mae"] - mets[(champ, "A")]["mae"],
            "degradation_pct": 100 * (mets[(champ, "B")]["mae"]
                                      / mets[(champ, "A")]["mae"] - 1),
            "regime_b_bias": mets[(champ, "B")]["bias"],
            "regime_b_surge_mae": float(np.abs(ec[sm]).mean()),
            "regime_b_surge_bias": float(ec[sm].mean()),
        },
        "vs_displaced_model_regime_b": {
            "model": other,
            "delta_mae_mw": float(np.mean(np.abs(ec) - np.abs(eo))),
            "ci_95_low": bs["ci_95_low"], "ci_95_high": bs["ci_95_high"],
            "dm_p_two_sided": dm["p_value"],
        },
        # Carried so serving and the paper quote the same unbiased figures
        # rather than re-deriving them and drifting apart.
        "honest_estimates": {
            "method": ("hyperparameters re-selected on inner folds 2023/2024 "
                       "only, 2025 scored once - see run_phase7_surge.py "
                       "honest_validation and run_selection_audit.py"),
            "quantile_head_2025_regime_b_mae":
                surge["honest_validation"]["scored_2025"]["quantile_head"]["mae"],
            "prior_champion_2025_regime_b_mae":
                surge["honest_validation"]["scored_2025"]["champion"]["mae"],
            "paired_delta_mw":
                surge["honest_validation"]["paired_vs_champion_2025"]["delta_mae_mw"],
            "paired_ci_95": [
                surge["honest_validation"]["paired_vs_champion_2025"]["ci_95_low"],
                surge["honest_validation"]["paired_vs_champion_2025"]["ci_95_high"]],
            "prior_champion_selection_optimism_mw":
                audit["optimism"]["B"]["optimism_mw"],
        },
        "known_limitations": [
            ("Surge-day bias is reduced but NOT solved: on untouched 2025 the "
             "inner-selected quantile level cuts surge |bias| by only 7%. A "
             "global quantile shift cannot fix a conditional problem - see "
             "results/phase7_surge.json."),
            ("Recency weighting was tested across half-lives 1-8 years and is "
             "not adopted; it is redundant with the trend feature and Ridge "
             "extrapolation - see results/phase7_recency.json."),
            ("A4_festival reduces Diwali bias +1,519 -> +259 MW but its "
             "overall gain does not clear the paired-CI bar on this cell; it "
             "is reported as an extension, not adopted."),
        ],
        "environment": capture_environment(),
        "frozen": True,
        "note_2026": ("2026 has NOT been used in tuning or selection. It was "
                      "inspected during Track A diagnostics, so it is a "
                      "confirmation set, not a blind test."),
    }
    (MODELS / "champion_config.json").write_text(
        json.dumps(cfg, indent=2, default=str) + "\n")

    (RESULTS / "phase7_refreeze.json").write_text(json.dumps({
        "phase": "7_refreeze",
        "generated_at": cfg["generated_at"],
        "tuned": tuned,
        "metrics": {f"{n}_{r}": mets[(n, r)] for n in SPACES for r in ("A", "B")},
        "champion": champ,
        "n_scored": len(scored),
        "environment": capture_environment(),
    }, indent=2, default=str) + "\n")

    pd.concat([preds[(champ, r)].assign(model=champ, regime=r)
               for r in ("A", "B")]).to_csv(
        RESULTS / "phase7_champion_predictions.csv", index=False)

    print(f"\nwrote {MODELS / 'champion_config.json'}")
    print(f"      {MODELS / 'champion_config.phase6b.json'} (previous, kept)")
    print(f"      {RESULTS / 'phase7_refreeze.json'}")
    print(f"      {RESULTS / 'phase7_champion_predictions.csv'}")


if __name__ == "__main__":
    main()
