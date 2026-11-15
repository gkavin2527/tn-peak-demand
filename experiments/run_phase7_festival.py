"""
Phase 7 - A2 vs A4 festival-feature gate.

    python run_phase7_festival.py

Tests whether per-festival Diwali features earn their place, using the SAME
protocol and the SAME bar as every prior gate in this project:

  * expanding walk-forward, val 2023 / 2024 / 2025
  * train on the full ERA5 history, score under both regimes
  * one shared scoring sample across both feature sets
  * paired moving-block bootstrap (7-day blocks, n=2000, seed 42)
  * adopt only if the 95% CI on the paired difference EXCLUDES zero

The motivation is diagnostic, not a search: the champion over-forecasts Diwali
by +945 MW (MAE 1,263 vs a 474 MW no-holiday baseline, n=21) and the three
worst days in the validation sample are consecutive Diwalis.

Reported for BOTH the champion (ridge_gbm_hybrid, tuned) and plain Ridge,
because a feature that only helps the tuned hybrid is more likely to be
interacting with that model's hyperparameters than to be real signal.

This is an additional selection round against 2023-2025. That cost is real and
is quantified by run_selection_audit.py (W0-1), which must run AFTER this.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy / sklearn / lightgbm

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import PHASE5_FEATURE_SETS, load_phase5_data, swap_weather
from src.evaluation import compute_metrics, moving_block_bootstrap
from src.models import build_model
from src.envinfo import capture_environment
from experiments.run_phase6_benchmark import FOLDS

RESULTS = Path("results")
CHAMPION = Path("models/champion_config.json")
BASE, CAND = "A2_weather_deltas_anomalies", "A4_festival"


def load_set(feature_set: str, regime: str):
    """Regime A/B frame for an arbitrary feature set.

    run_phase6_benchmark.load_regime() is pinned to the canonical A2 list and
    asserts against drift - a guard that protects the committed Phase 6
    numbers, so it is not relaxed here. This is the same call underneath, with
    the feature set as a parameter.

    Regime B goes through the gateway's forecast path, which raises rather
    than silently leaving ERA5 weather in place on an uncovered date.
    """
    source = "reanalysis" if regime == "A" else "forecast"
    df, feats = load_phase5_data(feature_set, weather_source=source,
                                 forecast_lead=1)
    want = list(PHASE5_FEATURE_SETS[feature_set])
    assert feats == want, f"{feature_set} drifted: {set(feats) ^ set(want)}"
    return df


def walk_forward(train_df, score_df, features, model_name, scored, hp=None):
    """Expanding walk-forward; train on train_df, score scored dates of score_df."""
    rows = []
    for f in FOLDS:
        tr = train_df[train_df.index.year <= f["train_end"]]
        va = score_df[(score_df.index.year == f["val_year"])
                      & score_df.index.isin(scored)]
        if va.empty:
            continue
        ok = tr["y"].notna() & tr[features].notna().all(axis=1)
        m = build_model(model_name, **(hp or {})).fit(tr.loc[ok, features],
                                                      tr.loc[ok, "y"])
        rows.append(pd.DataFrame({
            "date": va.index, "fold": f["name"], "val_year": f["val_year"],
            "y_true": va["y"].to_numpy(float),
            "y_pred": np.asarray(m.predict(va[features]), float),
            "n_train": int(ok.sum()),
        }))
    return pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)


def main():
    cfg = json.loads(CHAMPION.read_text())
    sets = {BASE: list(PHASE5_FEATURE_SETS[BASE]),
            CAND: list(PHASE5_FEATURE_SETS[CAND])}
    added = [f for f in sets[CAND] if f not in sets[BASE]]

    print("=" * 74)
    print("PHASE 7 - FESTIVAL FEATURE GATE (A2 vs A4)")
    print("=" * 74)
    print(f"base      {BASE} ({len(sets[BASE])} features)")
    print(f"candidate {CAND} ({len(sets[CAND])} features)")
    print(f"added     {added}")

    # ---- frames and ONE shared scoring sample across both sets --------
    frames = {}
    for tag in sets:
        frames[tag] = {r: load_set(tag, r) for r in ("A", "B")}

    val_years = {f["val_year"] for f in FOLDS}
    scored = None
    for tag, feats in sets.items():
        for r in ("A", "B"):
            df = frames[tag][r]
            ok = df[feats + ["y"]].notna().all(axis=1)
            d = df.index[ok & df.index.year.isin(val_years)]
            scored = d if scored is None else scored.intersection(d)
    scored = pd.DatetimeIndex(sorted(scored))
    print(f"\nshared scoring sample: {len(scored)} dates "
          f"({scored.min():%Y-%m-%d} .. {scored.max():%Y-%m-%d})")

    # Champion first, read from config rather than hardcoded: the gate was
    # first run against the Phase 6b mean-head champion, and W1-6 later found
    # the quantile head makes Diwali WORSE (+1,519 -> +1,766 MW bias). The
    # gate's answer can therefore change with the champion, so it must track
    # whatever is currently frozen.
    models = [(cfg["model_type"], cfg["hyperparameters"]), ("ridge", None)]
    out = {}

    for model_name, hp in models:
        print(f"\n{'-' * 74}\n{model_name}{' (tuned champion)' if hp else ''}\n{'-' * 74}")
        preds, mets = {}, {}
        for tag, feats in sets.items():
            for r in ("A", "B"):
                p = walk_forward(frames[tag]["A"], frames[tag][r], feats,
                                 model_name, scored, hp)
                preds[(tag, r)] = p
                mets[(tag, r)] = compute_metrics(p.y_true.to_numpy(),
                                                 p.y_pred.to_numpy())
        print(f"{'feature set':<32}{'Regime A':>12}{'Regime B':>12}{'A->B':>10}")
        for tag in sets:
            a, b = mets[(tag, "A")]["mae"], mets[(tag, "B")]["mae"]
            print(f"{tag:<32}{a:>12.2f}{b:>12.2f}{b - a:>+10.2f}")

        res = {"model": model_name, "tuned": bool(hp), "gates": {}}
        for r in ("A", "B"):
            pb = preds[(BASE, r)]; pc = preds[(CAND, r)]
            assert (pb.date.to_numpy() == pc.date.to_numpy()).all(), "unpaired"
            d = np.abs(pc.y_pred - pc.y_true) - np.abs(pb.y_pred - pb.y_true)
            bs = moving_block_bootstrap(d.to_numpy())
            adopt = bs["ci_95_high"] < 0
            res["gates"][r] = {
                "base_mae": mets[(BASE, r)]["mae"],
                "candidate_mae": mets[(CAND, r)]["mae"],
                "delta_mae": float(d.mean()),
                "ci_95_low": bs["ci_95_low"], "ci_95_high": bs["ci_95_high"],
                "ci_excludes_zero": bool(bs["ci_95_high"] < 0 or bs["ci_95_low"] > 0),
                "adopt": bool(adopt),
            }
            verdict = "ADOPT" if adopt else (
                "REJECT (CI spans 0)" if bs["ci_95_low"] <= 0 <= bs["ci_95_high"]
                else "REJECT (significantly WORSE)")
            print(f"  Regime {r}: delta {d.mean():+8.2f} MW  "
                  f"95% CI [{bs['ci_95_low']:+8.2f}, {bs['ci_95_high']:+8.2f}]  -> {verdict}")

        # festival-segment breakdown - where the feature is supposed to act
        from tn_holidays import calendar as tn_calendar
        tn = tn_calendar(range(2013, 2031))
        diw = {pd.Timestamp(k) for k, v in tn.items()
               if "diwali" in v.lower() or "deepavali" in v.lower()}
        pb, pc = preds[(BASE, "B")], preds[(CAND, "B")]
        win = pb.date.map(lambda d: any(abs((d - x).days) <= 2 for x in diw))
        seg = {}
        for label, mask in (("diwali_window", win), ("rest", ~win)):
            if mask.sum() == 0:
                continue
            eb = np.abs(pb.y_pred - pb.y_true)[mask]
            ec = np.abs(pc.y_pred - pc.y_true)[mask]
            bb = (pb.y_pred - pb.y_true)[mask]
            bc = (pc.y_pred - pc.y_true)[mask]
            seg[label] = {"n": int(mask.sum()),
                          "base_mae": float(eb.mean()), "candidate_mae": float(ec.mean()),
                          "base_bias": float(bb.mean()), "candidate_bias": float(bc.mean())}
            print(f"  [{label:<14} n={int(mask.sum()):>4}] MAE {eb.mean():7.1f} -> "
                  f"{ec.mean():7.1f}   bias {bb.mean():+8.1f} -> {bc.mean():+8.1f}")
        res["segments_regime_b"] = seg
        out[model_name] = res

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "phase": "7_festival_gate",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_feature_set": BASE, "candidate_feature_set": CAND,
        "added_features": added,
        "n_scored": len(scored),
        "protocol": {
            "validation": "expanding walk-forward, val 2023/2024/2025",
            "training_regime": "A (ERA5) for both, per Phase 6",
            "bootstrap": "paired moving-block, 7-day blocks, n=2000, seed 42",
            "adoption_bar": "95% CI on paired MAE difference excludes zero",
            "primary_metric": "MAE",
        },
        "note_selection": ("This is an additional selection round against "
                           "2023-2025. Quantified by run_selection_audit.py."),
        "results": out,
        "environment": capture_environment(),
    }
    (RESULTS / "phase7_festival.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {RESULTS / 'phase7_festival.json'}")

    champ_b = out[cfg["model_type"]]["gates"]["B"]
    print("\n" + "=" * 74)
    print(f"DECISION (champion, Regime B): "
          f"{'ADOPT A4' if champ_b['adopt'] else 'KEEP A2'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
