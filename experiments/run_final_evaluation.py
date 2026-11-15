"""
W0-5 - the single 2026 confirmation evaluation.

    python run_final_evaluation.py [--force]

THIS IS A ONE-SHOT. Run it once, on a frozen champion, after every selection
decision is closed. It refuses to overwrite its own artifact without --force
for exactly that reason.

IT IS NOT A BLIND TEST, and must never be described as one. 2026 was
inspected during Track A feature diagnostics - experiment_a1/a2/a3_results.json
each carry a diagnostic_2026_results block comparing feature sets on 2026
days. That inspection cannot be undone, so the honest label is CONFIRMATION:
evidence that the frozen model behaves on held-back data roughly as the
validation protocol predicted, not an unbiased estimate of generalisation.

The unbiased estimate this project does have comes from run_selection_audit.py,
which held 2025 out of every selection round and found the reported figures
to be conservative rather than optimistic (-5.80 MW on Regime B).

PROTOCOL
  train   everything through 2025-12-31, Regime A (ERA5), as the model is
          served - training weather is always reanalysis
  score   2026 under Regime B (archived lead-1 forecast), the operational
          regime, once
  report  overall and per-segment metrics, plus interval coverage against the
          2026-FREE conformal widths from calibrate_intervals.py - quoting
          coverage against widths fitted on 2026 would be circular
"""

from __future__ import annotations

import threadlimit  # noqa: F401

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import PHASE5_FEATURE_SETS
from src.envinfo import capture_environment
from src.evaluation import compute_metrics, moving_block_bootstrap, surge_mask
from src.models import build_from_config, build_model
from experiments.run_phase7_festival import load_set

RESULTS = Path("results")
CHAMPION = Path("models/champion_config.json")
OUT = RESULTS / "final_evaluation.json"
TRAIN_END = 2025
CONFIRM_YEAR = 2026


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing confirmation (think first)")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        prev = json.loads(OUT.read_text())
        raise SystemExit(
            f"{OUT} already exists, written {prev.get('generated_at')} for "
            f"champion {prev.get('champion', {}).get('model')}.\n"
            f"This is a one-shot confirmation set. Re-running it after seeing "
            f"the result is how a confirmation quietly becomes another "
            f"selection round.\n"
            f"If the champion genuinely changed, pass --force and say so in "
            f"the commit message.")

    cfg = json.loads(CHAMPION.read_text())
    if not cfg.get("frozen"):
        raise SystemExit("champion is not marked frozen - freeze it first")
    _, feats, param = build_from_config(cfg)
    fset = cfg["feature_set"]
    assert feats == list(PHASE5_FEATURE_SETS[fset]), "feature list drifted"

    print("=" * 78)
    print("W0-5  2026 CONFIRMATION EVALUATION  (one-shot)")
    print("=" * 78)
    print(f"champion   {cfg['model_type']}  {fset} ({len(feats)} features)")
    print(f"frozen     {cfg.get('generated_at')}")
    print(f"train      <= {TRAIN_END}-12-31, Regime A (ERA5)")
    print(f"score      {CONFIRM_YEAR}, Regime B (archived lead-1 forecast), ONCE")
    print("\nThis is a CONFIRMATION set, not a blind test: 2026 was inspected")
    print("during Track A feature diagnostics. Do not describe it otherwise.")

    frames = {r: load_set(fset, r) for r in ("A", "B")}
    tr = frames["A"][frames["A"].index.year <= TRAIN_END]
    ok = tr["y"].notna() & tr[feats].notna().all(axis=1)
    if int(ok.sum()) < 3000:
        raise SystemExit(f"only {int(ok.sum())} training rows; expected >3000")

    sc = frames["B"]
    m = sc.index.year == CONFIRM_YEAR
    m &= sc[feats + ["y"]].notna().all(axis=1)
    va = sc.loc[m]
    if len(va) < 100:
        raise SystemExit(f"only {len(va)} scorable {CONFIRM_YEAR} rows")

    model = build_model(cfg["model_type"], **cfg["hyperparameters"])
    model.fit(tr.loc[ok, feats], tr.loc[ok, "y"])
    yt = va["y"].to_numpy(float)
    yp = np.asarray(model.predict(va[feats]), float)
    err = yp - yt

    print(f"\ntrained on {int(ok.sum()):,} rows; scoring {len(va)} days "
          f"({va.index.min():%Y-%m-%d} .. {va.index.max():%Y-%m-%d})")

    mets = compute_metrics(yt, yp)
    exp_b = cfg["validation_performance"]["regime_b_mae"]
    print("\n" + "-" * 78)
    print(f"{'metric':<14}{'2026 confirmation':>20}{'2023-25 validation':>22}"
          f"{'delta':>12}")
    print("-" * 78)
    print(f"{'MAE (MW)':<14}{mets['mae']:>20.2f}{exp_b:>22.2f}"
          f"{mets['mae'] - exp_b:>+12.2f}")
    for k, lab in (("rmse", "RMSE (MW)"), ("mape", "MAPE (%)"),
                   ("bias", "bias (MW)")):
        if k in mets:
            print(f"{lab:<14}{mets[k]:>20.2f}{'':>22}{'':>12}")

    bs = moving_block_bootstrap(np.abs(err) - mets["mae"])
    print(f"\nMAE 95% CI (moving-block, 7d, n=2000): "
          f"[{mets['mae'] + bs['ci_95_low']:.1f}, "
          f"{mets['mae'] + bs['ci_95_high']:.1f}] MW")

    # ---- segments ----------------------------------------------------
    from tn_holidays import calendar as tn_calendar
    tn = {pd.Timestamp(k): v for k, v in tn_calendar(range(2013, 2031)).items()}
    idx = va.index
    sm = surge_mask(idx, yt)
    diw = {d for d, n in tn.items()
           if "diwali" in n.lower() or "deepavali" in n.lower()}
    # np.asarray throughout: DatetimeIndex.day_of_week and .month.isin()
    # already return ndarrays, so .to_numpy() is not available on them.
    dow = np.asarray(idx.day_of_week)
    is_hol = np.array([d in tn for d in idx])
    segs = {
        "overall": np.ones(len(idx), bool),
        "working_day": (dow < 5) & ~is_hol,
        "weekend": dow >= 5,
        "holiday": is_hol,
        "summer_mar_jun": np.asarray(idx.month.isin([3, 4, 5, 6])),
        "surge_top5pct": np.asarray(sm),
        "diwali_window": np.array([any(abs((d - x).days) <= 2 for x in diw)
                                   for d in idx]),
    }
    print(f"\n{'segment':<18}{'n':>5}{'MAE':>10}{'bias':>10}  flag")
    print("-" * 50)
    seg_out = {}
    for name, mk in segs.items():
        n = int(mk.sum())
        if n == 0:
            continue
        e = err[mk]
        flag = "ANECDOTAL" if n < 20 else ""
        print(f"{name:<18}{n:>5}{np.abs(e).mean():>10.1f}{e.mean():>+10.1f}  {flag}")
        seg_out[name] = {"n": n, "mae": float(np.abs(e).mean()),
                         "bias": float(e.mean()),
                         "anecdotal_n_lt_20": bool(n < 20)}

    # ---- interval coverage, against the 2026-FREE widths --------------
    cov_out = None
    iv = json.loads((RESULTS / "intervals.json").read_text())
    pc = iv.get("paper_calibration_2026_free")
    if pc:
        print("\n" + "-" * 78)
        print("interval coverage vs the 2026-FREE conformal widths "
              f"(fitted <= {pc['cutoff']})")
        print("-" * 78)
        season = np.where(np.asarray(idx.month.isin([3, 4, 5, 6])),
                          "summer", "rest")
        cov_out = {}
        for lvl in ("0.90", "0.95"):
            hw = np.array([pc["seasons"][s]["relative_halfwidth"][lvl]
                           for s in season])
            lo, hi = yp * (1 - hw), yp * (1 + hw)
            inside = (yt >= lo) & (yt <= hi)
            realised = float(inside.mean())
            gap = 100 * (realised - float(lvl))
            width = float((hi - lo).mean())
            flag = "OK" if abs(gap) <= 3 else "OFF"
            print(f"  {lvl}: realised {realised:.1%} "
                  f"(nominal {float(lvl):.0%}, {gap:+.1f} pts)  "
                  f"mean width {width:,.0f} MW  [{flag}]")
            cov_out[lvl] = {"nominal": float(lvl), "realised": realised,
                            "gap_pts": gap, "mean_width_mw": width,
                            "within_3pts": bool(abs(gap) <= 3)}
        print("  (widths fitted on residuals up to 2025-12-31, so this is an")
        print("   out-of-sample coverage claim rather than a circular one)")

    payload = {
        "phase": "w0_5_final_confirmation",
        "evaluation_type": "confirmation_not_blind_test",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "why_not_blind": (
            "2026 was inspected during Track A feature diagnostics - "
            "experiment_a1/a2/a3_results.json each contain a "
            "diagnostic_2026_results block. The unbiased estimate comes from "
            "run_selection_audit.py, which held 2025 out of every selection "
            "round."),
        "champion": {
            "model": cfg["model_type"], "feature_set": fset,
            "hyperparameters": cfg["hyperparameters"],
            "frozen_at": cfg.get("generated_at"),
            "target_parameterisation": param,
            "lag_representation": cfg.get("lag_representation"),
        },
        "protocol": {
            "train": f"<= {TRAIN_END}-12-31, Regime A (ERA5)",
            "score": f"{CONFIRM_YEAR}, Regime B (archived lead-1 forecast)",
            "n_train": int(ok.sum()), "n_scored": int(len(va)),
            "date_range": [f"{va.index.min():%Y-%m-%d}",
                           f"{va.index.max():%Y-%m-%d}"],
            "runs_allowed": 1,
        },
        "metrics_2026": mets,
        "mae_ci_95": [mets["mae"] + bs["ci_95_low"],
                      mets["mae"] + bs["ci_95_high"]],
        "validation_2023_25_regime_b_mae": exp_b,
        "delta_vs_validation_mw": mets["mae"] - exp_b,
        "segments": seg_out,
        "interval_coverage_vs_2026_free_widths": cov_out,
        "environment": capture_environment(),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    pd.DataFrame({"date": idx, "y_true": yt, "y_pred": yp,
                  "error": err}).to_csv(
        RESULTS / "final_evaluation_predictions.csv", index=False)
    print(f"\nwrote {OUT}")
    print(f"      {RESULTS / 'final_evaluation_predictions.csv'}")
    print("\nONE-SHOT CONSUMED. Re-running needs --force; doing so after "
          "seeing\nthis result turns a confirmation into another selection "
          "round.")


if __name__ == "__main__":
    main()
