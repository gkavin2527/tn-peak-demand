"""
W1-4 - Diebold-Mariano tests on every headline comparison.

    python run_dm_tests.py

Reported ALONGSIDE the moving-block bootstrap that this project has used
throughout, never instead of it. The bootstrap makes no asymptotic-normality
assumption on ~1,000 daily observations and is arguably the stronger
instrument; DM is what forecasting referees ask for by name. Where the two
disagree, say so rather than quoting whichever is kinder.

Sign convention matches moving_block_bootstrap: negative favours the
candidate.
"""

from __future__ import annotations

import threadlimit  # noqa: F401

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.envinfo import capture_environment
from src.evaluation import diebold_mariano, moving_block_bootstrap

RESULTS = Path("results")


def pair(cand_err, ref_err, label, note=""):
    dm = diebold_mariano(cand_err, ref_err, loss="abs")
    bs = moving_block_bootstrap(np.abs(cand_err) - np.abs(ref_err))
    agree = (dm["p_value"] < 0.05) == (bs["ci_95_high"] < 0 or bs["ci_95_low"] > 0)
    print(f"  {label:<46}")
    print(f"    delta MAE {dm['mean_loss_diff']:+8.2f} MW   "
          f"DM {dm['dm_stat']:+7.3f}  p={dm['p_value']:.2e}   "
          f"boot CI [{bs['ci_95_low']:+8.2f},{bs['ci_95_high']:+8.2f}]"
          f"   {'agree' if agree else 'DISAGREE'}")
    return {"comparison": label, "note": note, "n": dm["n"],
            "delta_mae_mw": dm["mean_loss_diff"],
            "dm_stat": dm["dm_stat"], "dm_p_two_sided": dm["p_value"],
            "dm_p_one_sided": dm.get("p_value_one_sided"),
            "dm_nw_lag": dm["nw_lag"], "dm_favours": dm.get("favours"),
            "bootstrap_mean": bs["mean"],
            "bootstrap_ci_95": [bs["ci_95_low"], bs["ci_95_high"]],
            "significant_dm": bool(dm["p_value"] < 0.05),
            "significant_bootstrap": bool(bs["ci_95_high"] < 0 or bs["ci_95_low"] > 0),
            "methods_agree": bool(agree)}


def main():
    out = {}
    print("=" * 74)
    print("W1-4  DIEBOLD-MARIANO, alongside the moving-block bootstrap")
    print("=" * 74)

    # ---- 1. Phase 6 model benchmark, Regime A, every model vs Ridge ----
    print("\n1. Phase 6 benchmark (Regime A) - each model vs Ridge")
    b = pd.read_csv(RESULTS / "phase6_benchmark_predictions.csv",
                    parse_dates=["date"])
    wide = {m: g.sort_values("date").set_index("date")
            for m, g in b[b.regime == "A"].groupby("model")}
    ref = wide["ridge"]
    rows = []
    for m, g in wide.items():
        if m == "ridge":
            continue
        assert g.index.equals(ref.index), f"{m} is not paired with ridge"
        rows.append(pair(g.y_pred - g.y_true, ref.y_pred - ref.y_true,
                         f"{m} vs ridge"))
    out["phase6_vs_ridge_regime_a"] = rows

    # ---- 2. the paper's central claim: Regime A vs Regime B ------------
    print("\n2. Operational weather gap - Regime B vs Regime A (same model)")
    c = pd.read_csv(RESULTS / "champion_regime_ab_predictions.csv",
                    parse_dates=["date"])
    rows = []
    for m, g in c.groupby("model") if "model" in c.columns else []:
        a = g[g.regime == "A"].sort_values("date").set_index("date")
        bb = g[g.regime == "B"].sort_values("date").set_index("date")
        common = a.index.intersection(bb.index)
        if len(common) < 100:
            continue
        rows.append(pair(bb.loc[common].y_pred - bb.loc[common].y_true,
                         a.loc[common].y_pred - a.loc[common].y_true,
                         f"{m}: Regime B vs Regime A",
                         "positive delta = forecast weather costs accuracy"))
    out["regime_b_vs_a"] = rows

    # ---- 3. the Phase 7 festival gate --------------------------------
    gate = RESULTS / "phase7_festival.json"
    if gate.exists():
        print("\n3. Phase 7 festival gate (A4 vs A2) - DM on the same pairs")
        print("   (recomputed from the gate's own walk-forward)")
        import experiments.run_phase7_festival as F
        from src.contract import PHASE5_FEATURE_SETS
        cfg = json.loads(Path("models/champion_config.json").read_text())
        sets = {t: list(PHASE5_FEATURE_SETS[t])
                for t in (F.BASE, F.CAND)}
        frames = {t: {r: F.load_set(t, r) for r in ("A", "B")} for t in sets}
        from run_phase6_benchmark import FOLDS
        val_years = {f["val_year"] for f in FOLDS}
        scored = None
        for t, ft in sets.items():
            for r in ("A", "B"):
                df = frames[t][r]
                ok = df[ft + ["y"]].notna().all(axis=1)
                d = df.index[ok & df.index.year.isin(val_years)]
                scored = d if scored is None else scored.intersection(d)
        scored = pd.DatetimeIndex(sorted(scored))
        rows = []
        for model, hp in (("ridge_gbm_hybrid", cfg["hyperparameters"]),
                          ("ridge", None)):
            for r in ("A", "B"):
                pb = F.walk_forward(frames[F.BASE]["A"], frames[F.BASE][r],
                                    sets[F.BASE], model, scored, hp)
                pc = F.walk_forward(frames[F.CAND]["A"], frames[F.CAND][r],
                                    sets[F.CAND], model, scored, hp)
                rows.append(pair(pc.y_pred - pc.y_true, pb.y_pred - pb.y_true,
                                 f"{model}, Regime {r}: A4 vs A2"))
        out["phase7_a4_vs_a2"] = rows

    payload = {
        "phase": "w1_4_diebold_mariano",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": {
            "statistic": "Diebold-Mariano, absolute-error loss",
            "variance": "Newey-West (Bartlett), lag floor(4*(n/100)^(2/9))",
            "small_sample": "Harvey-Leybourne-Newbold correction, t(n-1)",
            "sign": "negative favours the candidate",
            "reported_with": ("moving-block bootstrap, 7-day blocks, n=2000, "
                              "seed 42 - DM supplements it, never replaces it"),
        },
        "results": out,
        "environment": capture_environment(),
    }
    (RESULTS / "dm_tests.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")

    allrows = [r for v in out.values() for r in v]
    dis = [r["comparison"] for r in allrows if not r["methods_agree"]]
    print("\n" + "=" * 74)
    print(f"{len(allrows)} comparisons; DM and bootstrap agree on "
          f"{len(allrows) - len(dis)}/{len(allrows)}")
    if dis:
        print("DISAGREEMENTS (report both, do not pick the kinder one):")
        for c in dis:
            print(f"  - {c}")
    print(f"wrote {RESULTS / 'dm_tests.json'}")


if __name__ == "__main__":
    main()
