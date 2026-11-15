"""
W1-2 - surge-day bias: quantile residual head, swept over tau.

    python run_phase7_surge.py

THE PROBLEM. On the champion's Regime B validation residuals the top-5% surge
days carry a bias of -549.0 MW (n=56, 48 of 56 under-forecast) against an
overall bias of -69.6. The model is systematically low exactly when the
reserve margin is thinnest - the operationally expensive direction.

THE INSTRUMENT. RidgeGBMHybrid fits its residual GBM to squared error, so the
head targets the conditional MEAN residual. QuantileHybrid changes only that
objective to LightGBM's quantile loss at level tau. Everything else -
features, Ridge component, GBM size, protocol - is held fixed.

THE TRADE-OFF, STATED BEFORE THE NUMBERS. Raising tau shifts the residual
head upward on EVERY day, not only surge days, so overall MAE must be
expected to rise. tau=0.5 is the median control, isolating "mean vs median"
from "median vs upper quantile". This script therefore reports the whole
curve: a single tau would hide the shape of the cost.

ACCEPTANCE (from REMEDIATION_PLAN.md W1-2, fixed in advance):
    surge |bias| falls by >=30% (-549 -> <=-385) AND overall MAE worsens by
    <=10 MW. Both conditions, on the champion's operational cell (Regime B).

Whether any tau is worth adopting is ultimately a question about the cost
asymmetry of over- versus under-forecasting peak demand, which an MAE gate
cannot answer. The curve is the deliverable; the gate is a filter.
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
from src.evaluation import (compute_metrics, diebold_mariano,
                            moving_block_bootstrap, surge_mask)
from experiments.run_phase6_benchmark import FOLDS
from experiments.run_phase7_festival import load_set, walk_forward

RESULTS = Path("results")
CHAMPION = Path("models/champion_config.json")
FEATURE_SET = "A2_weather_deltas_anomalies"
TAUS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]

BIAS_REDUCTION_TARGET = 0.30     # surge |bias| must fall by this fraction
MAE_TOLERANCE_MW = 10.0          # overall MAE may worsen by at most this


def segment(preds: pd.DataFrame) -> dict:
    """Overall and surge-day metrics for one prediction frame."""
    y, p = preds.y_true.to_numpy(float), preds.y_pred.to_numpy(float)
    err = p - y
    sm = surge_mask(pd.DatetimeIndex(preds.date), y)
    return {
        "n": int(len(y)),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "surge_n": int(sm.sum()),
        "surge_mae": float(np.abs(err[sm]).mean()),
        "surge_bias": float(err[sm].mean()),
        "surge_under_forecast": int((err[sm] < 0).sum()),
        "nonsurge_mae": float(np.abs(err[~sm]).mean()),
        "nonsurge_bias": float(err[~sm].mean()),
    }


def main():
    cfg = json.loads(CHAMPION.read_text())
    hp = dict(cfg["hyperparameters"])
    feats = list(PHASE5_FEATURE_SETS[FEATURE_SET])

    print("=" * 78)
    print("W1-2  SURGE-DAY BIAS - quantile residual head swept over tau")
    print("=" * 78)
    print(f"base champion {cfg['model_type']}  {FEATURE_SET} ({len(feats)} features)")
    print(f"acceptance    surge |bias| -{BIAS_REDUCTION_TARGET:.0%} AND "
          f"overall MAE +{MAE_TOLERANCE_MW:.0f} MW at most, on Regime B")

    frames = {r: load_set(FEATURE_SET, r) for r in ("A", "B")}
    val_years = {f["val_year"] for f in FOLDS}
    scored = None
    for r in ("A", "B"):
        df = frames[r]
        ok = df[feats + ["y"]].notna().all(axis=1)
        d = df.index[ok & df.index.year.isin(val_years)]
        scored = d if scored is None else scored.intersection(d)
    scored = pd.DatetimeIndex(sorted(scored))
    print(f"scoring       {len(scored)} dates, folds {sorted(val_years)}")

    # ---- baseline: the frozen champion -------------------------------
    base = {}
    for r in ("A", "B"):
        base[r] = walk_forward(frames["A"], frames[r], feats,
                               "ridge_gbm_hybrid", scored, hp)
    base_seg = {r: segment(base[r]) for r in ("A", "B")}

    b = base_seg["B"]
    print(f"\nbaseline (Regime B): MAE {b['mae']:.2f}  bias {b['bias']:+.1f}   "
          f"surge MAE {b['surge_mae']:.1f}  surge bias {b['surge_bias']:+.1f} "
          f"(n={b['surge_n']}, {b['surge_under_forecast']} under)")

    target_bias = abs(b["surge_bias"]) * (1 - BIAS_REDUCTION_TARGET)
    print(f"             -> need surge |bias| <= {target_bias:.1f} MW "
          f"and MAE <= {b['mae'] + MAE_TOLERANCE_MW:.2f} MW")

    # ---- sweep -------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"{'tau':>5} {'MAE':>8} {'dMAE':>8} {'bias':>9} {'surgeMAE':>9} "
          f"{'surgeBias':>10} {'|d|bias':>9} {'under':>7}  verdict")
    print("-" * 78)

    rows = []
    for tau in TAUS:
        q = {**hp, "quantile": tau}
        preds = {r: walk_forward(frames["A"], frames[r], feats,
                                 "quantile_hybrid", scored, q)
                 for r in ("A", "B")}
        seg = {r: segment(preds[r]) for r in ("A", "B")}
        sb = seg["B"]

        d_mae = sb["mae"] - b["mae"]
        bias_cut = 1 - abs(sb["surge_bias"]) / abs(b["surge_bias"])
        ok_bias = abs(sb["surge_bias"]) <= target_bias
        ok_mae = d_mae <= MAE_TOLERANCE_MW
        verdict = "PASS" if (ok_bias and ok_mae) else (
            "fail: MAE" if ok_bias else ("fail: bias" if ok_mae else "fail: both"))

        # paired significance against the champion on the same dates
        eb = base["B"].y_pred.to_numpy() - base["B"].y_true.to_numpy()
        ec = preds["B"].y_pred.to_numpy() - preds["B"].y_true.to_numpy()
        bs = moving_block_bootstrap(np.abs(ec) - np.abs(eb))
        dm = diebold_mariano(ec, eb, loss="abs")

        print(f"{tau:>5.2f} {sb['mae']:>8.2f} {d_mae:>+8.2f} {sb['bias']:>+9.1f} "
              f"{sb['surge_mae']:>9.1f} {sb['surge_bias']:>+10.1f} "
              f"{bias_cut:>8.0%} {sb['surge_under_forecast']:>4}/{sb['surge_n']:<2}  {verdict}")

        rows.append({
            "tau": tau,
            "regime_a": seg["A"], "regime_b": seg["B"],
            "delta_mae_vs_champion_mw": d_mae,
            "surge_bias_reduction_frac": float(bias_cut),
            "meets_bias_target": bool(ok_bias),
            "meets_mae_tolerance": bool(ok_mae),
            "accepted": bool(ok_bias and ok_mae),
            "paired_vs_champion_regime_b": {
                "delta_mae_mw": float(np.mean(np.abs(ec) - np.abs(eb))),
                "ci_95_low": bs["ci_95_low"], "ci_95_high": bs["ci_95_high"],
                "dm_stat": dm["dm_stat"], "dm_p_two_sided": dm["p_value"],
            },
        })

    accepted = [r for r in rows if r["accepted"]]
    print("-" * 78)

    # ---- verdict -----------------------------------------------------
    print()
    if accepted:
        best = min(accepted, key=lambda r: r["regime_b"]["mae"])
        print(f"ACCEPTED: tau={best['tau']} meets both conditions "
              f"(surge bias {best['regime_b']['surge_bias']:+.1f} MW, "
              f"MAE {best['delta_mae_vs_champion_mw']:+.2f} MW)")
    else:
        # Report the frontier rather than nothing: the shape of the refusal is
        # the finding.
        bias_ok = [r for r in rows if r["meets_bias_target"]]
        print("NOT ADOPTED - no tau satisfies both conditions.")
        if bias_ok:
            cheap = min(bias_ok, key=lambda r: r["delta_mae_vs_champion_mw"])
            print(f"  cheapest tau that fixes the bias: {cheap['tau']} "
                  f"-> surge bias {cheap['regime_b']['surge_bias']:+.1f} MW "
                  f"but overall MAE "
                  f"{cheap['delta_mae_vs_champion_mw']:+.2f} MW "
                  f"({MAE_TOLERANCE_MW:.0f} MW allowed)")
        print("  the surge bias is not separable from overall accuracy by a")
        print("  global quantile shift - it needs a CONDITIONAL correction.")

    # ---- honest validation: pick tau on inner folds, score 2025 once ----
    #
    # The sweep above chose tau from the full 2023-2025 sample, which is one
    # more selection round against the same validation data the audit flags.
    # It also cannot be checked by run_selection_audit.py's replay, because
    # that ranks families on DEFAULT hyperparameters and quantile_hybrid's
    # default tau=0.9 scores 669 MW - screened out before tuning could ever
    # find tau~0.6. That is a genuine limitation of rank-then-tune: a family
    # with a badly chosen default never reaches the tuning round.
    #
    # So the quantile head is validated here on its own terms: tune every
    # hyperparameter INCLUDING tau on inner folds 2023/2024 only, then score
    # once on 2025 and compare against the champion scored the same way.
    print("\n" + "=" * 78)
    print("HONEST VALIDATION - tau chosen from 2023/2024 only, 2025 scored once")
    print("=" * 78)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    INNER = [{"train_end": 2022, "val_year": 2023},
             {"train_end": 2023, "val_year": 2024}]

    def inner_objective(params, model_name):
        maes = []
        for f in INNER:
            tr = frames["A"][frames["A"].index.year <= f["train_end"]]
            dates = scored[scored.year == f["val_year"]]
            if len(dates) == 0:
                continue
            ok = tr["y"].notna() & tr[feats].notna().all(axis=1)
            from src.models import build_model
            m = build_model(model_name, **params).fit(tr.loc[ok, feats],
                                                      tr.loc[ok, "y"])
            va = frames["A"].loc[frames["A"].index.isin(dates)]
            maes.append(float(np.abs(m.predict(va[feats])
                                     - va["y"].to_numpy(float)).mean()))
        return float(np.mean(maes))

    def suggest(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-6, 1e3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 3, 31),
                "quantile": trial.suggest_float("quantile", 0.50, 0.90)}

    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(lambda t: inner_objective(suggest(t), "quantile_hybrid"),
                n_trials=50, show_progress_bar=False)
    qhp = st.best_params
    print(f"inner-selected tau = {qhp['quantile']:.4f}   "
          f"(inner MAE {st.best_value:.2f} MW)")
    print(f"  full params: {qhp}")

    # score 2025 once, both models, same dates
    from src.models import build_model
    outer = scored[scored.year == 2025]
    tr = frames["A"][frames["A"].index.year <= 2024]
    ok = tr["y"].notna() & tr[feats].notna().all(axis=1)
    honest = {}
    for label, name, params in (("champion", "ridge_gbm_hybrid", hp),
                                ("quantile_head", "quantile_hybrid", qhp)):
        m = build_model(name, **params).fit(tr.loc[ok, feats], tr.loc[ok, "y"])
        va = frames["B"].loc[frames["B"].index.isin(outer)]
        yp = np.asarray(m.predict(va[feats]), float)
        yt = va["y"].to_numpy(float)
        sm = surge_mask(pd.DatetimeIndex(va.index), yt)
        e = yp - yt
        honest[label] = {
            "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
            "surge_mae": float(np.abs(e[sm]).mean()),
            "surge_bias": float(e[sm].mean()), "surge_n": int(sm.sum()),
            "n": int(len(yt)), "errors": e,
        }
        h = honest[label]
        print(f"  {label:<14} 2025 Regime B: MAE {h['mae']:7.2f}  "
              f"bias {h['bias']:+7.1f}   surge MAE {h['surge_mae']:6.1f}  "
              f"surge bias {h['surge_bias']:+7.1f} (n={h['surge_n']})")

    dq = np.abs(honest["quantile_head"]["errors"]) - np.abs(honest["champion"]["errors"])
    bsq = moving_block_bootstrap(dq)
    dmq = diebold_mariano(honest["quantile_head"]["errors"],
                          honest["champion"]["errors"], loss="abs")
    print(f"\n  paired on untouched 2025 (n={len(dq)}): "
          f"{dq.mean():+.2f} MW  95% CI "
          f"[{bsq['ci_95_low']:+.2f}, {bsq['ci_95_high']:+.2f}]  "
          f"DM p={dmq['p_value']:.3f}")
    survives = bsq["ci_95_high"] < 0
    print(f"  -> quantile head {'SURVIVES' if survives else 'does NOT survive'} "
          f"on data that never informed tau")
    bias_cut_honest = 1 - abs(honest["quantile_head"]["surge_bias"]) / \
                      abs(honest["champion"]["surge_bias"])
    print(f"  -> surge |bias| change on 2025: {bias_cut_honest:+.0%}")

    honest_out = {k: {kk: vv for kk, vv in v.items() if kk != "errors"}
                  for k, v in honest.items()}

    payload = {
        "phase": "w1_2_surge_bias",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_set": FEATURE_SET,
        "base_champion": {"model": cfg["model_type"], "hyperparameters": hp},
        "protocol": {
            "validation": "expanding walk-forward, val 2023/2024/2025",
            "training_regime": "A (ERA5); scored under both",
            "surge_definition": "top 5% of observed peak within each year",
            "n_scored": len(scored),
        },
        "acceptance_criterion": {
            "surge_bias_reduction": BIAS_REDUCTION_TARGET,
            "max_mae_increase_mw": MAE_TOLERANCE_MW,
            "regime": "B",
            "fixed_in_advance": True,
        },
        "baseline": base_seg,
        "sweep": rows,
        "accepted_tau": (min(accepted, key=lambda r: r["regime_b"]["mae"])["tau"]
                         if accepted else None),
        "conclusion": ("adopted" if accepted else
                       "not adopted - a global quantile shift trades overall "
                       "MAE for surge bias roughly one-for-one; a conditional "
                       "correction is required"),
        "honest_validation": {
            "method": ("every hyperparameter including tau tuned on inner "
                       "folds 2023/2024 only (50 TPE trials, Regime A "
                       "objective), then 2025 scored once under Regime B"),
            "why": ("the sweep above selected tau on the full 2023-2025 "
                    "sample. run_selection_audit.py cannot check it because "
                    "that replay ranks families on default hyperparameters, "
                    "and quantile_hybrid's default tau=0.9 scores 669 MW - "
                    "screened out before tuning. A badly chosen default "
                    "hides a family under rank-then-tune."),
            "inner_selected_tau": qhp["quantile"],
            "inner_selected_params": qhp,
            "inner_mae": st.best_value,
            "scored_2025": honest_out,
            "paired_vs_champion_2025": {
                "delta_mae_mw": float(dq.mean()),
                "ci_95_low": bsq["ci_95_low"], "ci_95_high": bsq["ci_95_high"],
                "dm_stat": dmq["dm_stat"], "dm_p_two_sided": dmq["p_value"],
                "n": int(len(dq)),
            },
            "survives_on_untouched_2025": bool(survives),
            "surge_bias_reduction_frac_2025": float(bias_cut_honest),
        },
        "environment": capture_environment(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase7_surge.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {RESULTS / 'phase7_surge.json'}")


if __name__ == "__main__":
    main()
