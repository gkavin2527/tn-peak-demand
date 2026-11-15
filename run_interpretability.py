"""
W2-2 - interpretability for the champion, in two parts.

    python run_interpretability.py

The champion is Ridge + a quantile-regression GBM on the residual, so a single
attribution number would be a category error. The two components answer
different questions and are reported separately:

  Ridge component   coefficients on STANDARDISED features, i.e. MW per 1 SD.
                    Signed, global, linear - this is where the trend and the
                    weekly/annual structure live, and it is the part that
                    extrapolates.
  Residual GBM      exact TreeSHAP via LightGBM's own pred_contrib=True. No
                    `shap` package is required, and the values are exact
                    rather than sampled. This is where the non-linear
                    temperature response lives.

A model-agnostic permutation importance is computed as a cross-check: if the
two disagree sharply about a feature, that is worth knowing before it goes in
a paper.

Deliberately NOT done: presenting TreeSHAP alone as if it explained the whole
model. On this data the Ridge component carries most of the level, so a
TreeSHAP-only chart would understate the lag block and overstate weather.
"""

from __future__ import annotations

import threadlimit  # noqa: F401

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.envinfo import capture_environment
from src.models import build_from_config
from run_phase6_benchmark import FOLDS
from run_phase7_festival import load_set

RESULTS = Path("results")
CHAMPION = Path("models/champion_config.json")
SEED = 42


def main():
    cfg = json.loads(CHAMPION.read_text())
    model, feats, _ = build_from_config(cfg)
    fset = cfg["feature_set"]

    frames = {r: load_set(fset, r) for r in ("A", "B")}
    # Fit the way the final fold does: train on all ERA5 history <= 2024.
    tr = frames["A"][frames["A"].index.year <= 2024]
    ok = tr["y"].notna() & tr[feats].notna().all(axis=1)
    X, y = tr.loc[ok, feats], tr.loc[ok, "y"]
    model.fit(X, y)

    # Attribute on the scored validation sample, not on training rows.
    val_years = {f["val_year"] for f in FOLDS}
    vb = frames["B"]
    vmask = vb[feats + ["y"]].notna().all(axis=1) & vb.index.year.isin(val_years)
    V = vb.loc[vmask, feats]

    print("=" * 74)
    print("W2-2  CHAMPION INTERPRETABILITY")
    print("=" * 74)
    print(f"{cfg['model_type']}  {fset} ({len(feats)} features)")
    print(f"fitted on {len(X):,} rows (<=2024), attributed on {len(V):,} "
          f"Regime B validation rows")

    # ---- part 1: Ridge, standardised coefficients --------------------
    r = model.ridge
    w = np.asarray(r.w_, float)
    Xs = (V.to_numpy(float) - r.mu_X_) / r.sig_X_
    ridge_contrib = Xs * w                      # per-row, per-feature, in MW
    ridge_abs = np.abs(ridge_contrib).mean(axis=0)

    # ---- part 2: residual GBM, exact TreeSHAP ------------------------
    # pred_contrib returns n_features + 1 columns; the last is the base value.
    contrib = model.gbm.predict(V, pred_contrib=True)
    contrib = np.asarray(contrib, float)
    shap_vals, shap_base = contrib[:, :-1], contrib[:, -1]
    shap_abs = np.abs(shap_vals).mean(axis=0)

    # TreeSHAP must reconstruct the GBM's own prediction exactly.
    recon = shap_vals.sum(axis=1) + shap_base
    gbm_pred = np.asarray(model.gbm.predict(V), float)
    shap_err = float(np.abs(recon - gbm_pred).max())
    print(f"\nTreeSHAP reconstruction error: {shap_err:.3e} MW "
          f"({'exact' if shap_err < 1e-6 else 'CHECK THIS'})")

    # ---- how much work does each component do? -----------------------
    ridge_pred = r.predict(V)
    total = ridge_pred + gbm_pred
    print(f"\ncomponent split on the validation sample:")
    print(f"  Ridge      mean {ridge_pred.mean():>10,.1f} MW   "
          f"sd {ridge_pred.std():>8.1f}")
    print(f"  GBM resid  mean {gbm_pred.mean():>10,.1f} MW   "
          f"sd {gbm_pred.std():>8.1f}")
    print(f"  -> the GBM moves the prediction by "
          f"{100 * gbm_pred.std() / total.std():.1f}% of total sd; "
          f"Ridge carries the level")

    # ---- part 3: permutation importance, model-agnostic --------------
    rng = np.random.default_rng(SEED)
    yv = vb.loc[vmask, "y"].to_numpy(float)
    base_mae = float(np.abs(model.predict(V) - yv).mean())
    perm = np.zeros(len(feats))
    REPEATS = 5
    for j in range(len(feats)):
        d = []
        for _ in range(REPEATS):
            Vp = V.copy()
            Vp.iloc[:, j] = rng.permutation(Vp.iloc[:, j].to_numpy())
            d.append(float(np.abs(model.predict(Vp) - yv).mean()) - base_mae)
        perm[j] = float(np.mean(d))

    # ---- table -------------------------------------------------------
    tab = pd.DataFrame({
        "feature": feats,
        "ridge_coef_mw_per_sd": w,
        "ridge_mean_abs_contrib_mw": ridge_abs,
        "gbm_mean_abs_shap_mw": shap_abs,
        "combined_mean_abs_mw": ridge_abs + shap_abs,
        "permutation_dmae_mw": perm,
    }).sort_values("combined_mean_abs_mw", ascending=False).reset_index(drop=True)

    print(f"\nbase MAE {base_mae:.2f} MW; permutation = MAE increase when "
          f"the column is shuffled ({REPEATS} repeats)\n")
    print(f"{'feature':<20}{'ridge MW/SD':>12}{'|ridge|':>9}{'|SHAP|':>9}"
          f"{'combined':>10}{'perm dMAE':>11}")
    print("-" * 71)
    for _, row in tab.iterrows():
        print(f"{row.feature:<20}{row.ridge_coef_mw_per_sd:>+12.1f}"
              f"{row.ridge_mean_abs_contrib_mw:>9.1f}"
              f"{row.gbm_mean_abs_shap_mw:>9.1f}"
              f"{row.combined_mean_abs_mw:>10.1f}"
              f"{row.permutation_dmae_mw:>+11.1f}")

    # ---- where do the two disagree? ---------------------------------
    t = tab.copy()
    for c in ("combined_mean_abs_mw", "permutation_dmae_mw"):
        t[c + "_rank"] = t[c].rank(ascending=False)
    t["rank_gap"] = (t["combined_mean_abs_mw_rank"]
                     - t["permutation_dmae_mw_rank"]).abs()
    worst = t.nlargest(3, "rank_gap")[["feature", "combined_mean_abs_mw_rank",
                                       "permutation_dmae_mw_rank", "rank_gap"]]
    print("\nlargest attribution/permutation rank disagreements:")
    for _, rw in worst.iterrows():
        print(f"  {rw.feature:<20} attribution #{int(rw.combined_mean_abs_mw_rank):<3} "
              f"permutation #{int(rw.permutation_dmae_mw_rank):<3} "
              f"(gap {int(rw.rank_gap)})")
    print("  (a large gap usually means the feature is collinear with another:")
    print("   attribution splits credit, permutation does not, because the")
    print("   model recovers the information from the correlated column)")

    # ---- weather vs lags vs calendar --------------------------------
    WEATHER = {"t_max", "t_mean", "t_min", "cdd_max", "hours_above_32",
               "d_cdd_max", "cdd_max_vs_roll7"}
    LAGS = {f for f in feats if f.startswith(("peak_lag", "energy_lag"))}
    groups = {"weather": WEATHER & set(feats), "demand_lags": LAGS,
              "calendar_trend": set(feats) - WEATHER - LAGS}
    print("\nby feature block:")
    block = {}
    for g, cols in groups.items():
        m = tab.feature.isin(cols)
        block[g] = {
            "n": int(m.sum()),
            "combined_mean_abs_mw": float(tab.loc[m, "combined_mean_abs_mw"].sum()),
            "gbm_shap_mw": float(tab.loc[m, "gbm_mean_abs_shap_mw"].sum()),
            "ridge_mw": float(tab.loc[m, "ridge_mean_abs_contrib_mw"].sum()),
            "permutation_dmae_mw": float(tab.loc[m, "permutation_dmae_mw"].sum()),
        }
        b = block[g]
        print(f"  {g:<16} n={b['n']:<3} combined {b['combined_mean_abs_mw']:>8.1f} MW"
              f"   (ridge {b['ridge_mw']:>7.1f}, SHAP {b['gbm_shap_mw']:>6.1f})"
              f"   perm {b['permutation_dmae_mw']:>+7.1f}")

    RESULTS.mkdir(exist_ok=True)
    tab.to_csv(RESULTS / "feature_importance.csv", index=False)
    (RESULTS / "feature_importance.json").write_text(json.dumps({
        "phase": "w2_2_interpretability",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "champion": cfg["model_type"], "feature_set": fset,
        "method": {
            "ridge": "coefficients on standardised features (MW per 1 SD)",
            "gbm": ("exact TreeSHAP via LightGBM pred_contrib=True - no shap "
                    "package, values are exact not sampled"),
            "cross_check": f"permutation importance, {REPEATS} repeats, seed {SEED}",
            "attributed_on": "Regime B validation rows (2023-2025)",
            "caveat": ("the two components answer different questions; "
                       "TreeSHAP alone would understate the demand-lag block "
                       "and overstate weather"),
        },
        "n_train": int(len(X)), "n_attributed": int(len(V)),
        "base_mae": base_mae,
        "treeshap_reconstruction_error_mw": shap_err,
        "component_split": {
            "ridge_mean_mw": float(ridge_pred.mean()),
            "ridge_sd_mw": float(ridge_pred.std()),
            "gbm_mean_mw": float(gbm_pred.mean()),
            "gbm_sd_mw": float(gbm_pred.std()),
            "gbm_share_of_total_sd_pct": float(100 * gbm_pred.std() / total.std()),
        },
        "by_block": block,
        "features": tab.to_dict(orient="records"),
        "environment": capture_environment(),
    }, indent=2, default=str) + "\n")
    print(f"\nwrote {RESULTS / 'feature_importance.json'}")
    print(f"      {RESULTS / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
