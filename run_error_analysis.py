"""
W1-5 / W1-6 - segment error analysis and residual diagnostics for the champion.

    python run_error_analysis.py

W1-5: the project only ever reported surge and summer segments. Missing were
regime changes (2015 Chennai floods, 2020 COVID, post-COVID recovery), heat
waves, long weekends, per-festival breakdowns and weekday/weekend.

W1-6: the +0.41 lag-1 residual ACF quoted in the docs was measured on the OLD
weather-free Ridge, not on the champion, and has been repeated since as if
current. It is re-measured here and the old figure should be corrected or
retracted wherever it appears.

Every segment carries n, MAE, bias and a bootstrap CI on the bias. Segments
with n < 20 are flagged anecdotal and must not carry a claim - Pongal is
n=6-8 and has been quoted before.
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
from src.evaluation import moving_block_bootstrap, surge_mask
from src.models import build_from_config
from run_phase6_benchmark import FOLDS
from run_phase7_festival import load_set, walk_forward

RESULTS = Path("results")
CHAMPION = Path("models/champion_config.json")
MIN_N = 20


def acf(x, nlags=21):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    denom = float(x @ x)
    return [1.0] + [float(x[k:] @ x[:-k]) / denom for k in range(1, nlags + 1)]


def main():
    cfg = json.loads(CHAMPION.read_text())
    _, feats, _ = build_from_config(cfg)
    fset = cfg["feature_set"]
    frames = {r: load_set(fset, r) for r in ("A", "B")}
    val_years = {f["val_year"] for f in FOLDS}
    scored = None
    for r in ("A", "B"):
        ok = frames[r][feats + ["y"]].notna().all(axis=1)
        d = frames[r].index[ok & frames[r].index.year.isin(val_years)]
        scored = d if scored is None else scored.intersection(d)
    scored = pd.DatetimeIndex(sorted(scored))

    print("=" * 78)
    print("W1-5 / W1-6  CHAMPION ERROR ANALYSIS")
    print("=" * 78)
    print(f"{cfg['model_type']}  {fset}  n={len(scored)} scored dates")

    p = walk_forward(frames["A"], frames["B"], feats, cfg["model_type"],
                     scored, cfg["hyperparameters"]).set_index("date")
    err = (p.y_pred - p.y_true).rename("err")
    master = pd.read_csv("data/master.csv", parse_dates=["date"]).set_index("date")
    idx = err.index

    from tn_holidays import calendar as tn_calendar
    tn = tn_calendar(range(2013, 2031))
    hol = {pd.Timestamp(k): v for k, v in tn.items()}

    def near(names, days=2):
        tgt = {d for d, n in hol.items()
               if any(s.lower() in n.lower() for s in names)}
        return pd.Series([any(abs((d - t).days) <= days for t in tgt)
                          for d in idx], index=idx)

    sm = pd.Series(surge_mask(idx, p.y_true.to_numpy()), index=idx)
    tmax = master.loc[idx, "t_max"]
    hot = tmax >= tmax.quantile(0.90)
    dow = idx.day_of_week

    segments = {
        "overall": pd.Series(True, index=idx),
        "working_day": pd.Series(dow < 5, index=idx) & ~pd.Series(
            [d in hol for d in idx], index=idx),
        "weekend": pd.Series(dow >= 5, index=idx),
        "holiday": pd.Series([d in hol for d in idx], index=idx),
        "long_weekend": (pd.Series(dow.isin([0, 4]), index=idx)
                         & pd.Series([d in hol for d in idx], index=idx)),
        "summer_mar_jun": pd.Series(idx.month.isin([3, 4, 5, 6]), index=idx),
        "monsoon_oct_dec": pd.Series(idx.month.isin([10, 11, 12]), index=idx),
        "surge_top5pct": sm,
        "hottest_decile": hot,
        "diwali_window": near(["diwali", "deepavali"]),
        "pongal_window": near(["pongal", "uzhavar"]),
        "may_day": near(["may day"]),
        "dussehra_ayudha": near(["dussehra", "ayudha"]),
        "ramadan_eid": near(["id-ul-fitr"]),
        "post_covid_2023": pd.Series(idx.year == 2023, index=idx),
        "year_2024": pd.Series(idx.year == 2024, index=idx),
        "year_2025": pd.Series(idx.year == 2025, index=idx),
    }

    print(f"\n{'segment':<22}{'n':>5}{'MAE':>9}{'bias':>9}"
          f"{'bias 95% CI':>22}  flag")
    print("-" * 78)
    out = {}
    for name, mask in segments.items():
        m = mask.reindex(idx).fillna(False).to_numpy(bool)
        n = int(m.sum())
        if n == 0:
            continue
        e = err[m].to_numpy()
        bs = moving_block_bootstrap(e) if n >= 14 else None
        ci = (f"[{bs['ci_95_low']:+8.1f},{bs['ci_95_high']:+8.1f}]"
              if bs else "  (n too small)")
        flag = "" if n >= MIN_N else "ANECDOTAL"
        print(f"{name:<22}{n:>5}{np.abs(e).mean():>9.1f}{e.mean():>+9.1f}"
              f"{ci:>22}  {flag}")
        out[name] = {
            "n": n, "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "bias_ci_95": [bs["ci_95_low"], bs["ci_95_high"]] if bs else None,
            "anecdotal_n_lt_20": bool(n < MIN_N),
        }

    # ---- W1-6 residual diagnostics on the CHAMPION -------------------
    print("\n" + "-" * 78)
    print("W1-6  residual autocorrelation - on the CHAMPION, not the old Ridge")
    print("-" * 78)
    # contiguous daily run, so lags mean calendar days
    span = pd.date_range(idx.min(), idx.max(), freq="D")
    e_full = err.reindex(span)
    a = acf(e_full.dropna().to_numpy(), nlags=21)
    print("  lag:  " + "  ".join(f"{k:>5}" for k in range(1, 8)))
    print("  acf:  " + "  ".join(f"{a[k]:>+5.2f}" for k in range(1, 8)))
    print(f"  lag-7  {a[7]:+.3f}   lag-14 {a[14]:+.3f}   lag-21 {a[21]:+.3f}")
    ci95 = 1.96 / np.sqrt(len(e_full.dropna()))
    sig = [k for k in range(1, 22) if abs(a[k]) > ci95]
    print(f"  +/-1.96/sqrt(n) = +/-{ci95:.3f}; "
          f"significant lags: {sig if sig else 'none'}")
    print(f"\n  NOTE the docs quote +0.41 at lag 1 from the OLD weather-free")
    print(f"  Ridge. The champion's is {a[1]:+.3f} - correct or retract that.")

    # normality / shape, since intervals assume a residual shape
    e_all = err.to_numpy()
    print(f"\n  residual shape: mean {e_all.mean():+.1f}  sd {e_all.std():.1f}  "
          f"skew {pd.Series(e_all).skew():+.2f}  "
          f"excess kurtosis {pd.Series(e_all).kurtosis():+.2f}")

    payload = {
        "phase": "w1_5_w1_6_error_analysis",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "champion": cfg["model_type"], "feature_set": fset,
        "regime": "B (archived lead-1 forecast)",
        "n_scored": len(scored),
        "min_n_for_a_claim": MIN_N,
        "segments": out,
        "residual_diagnostics": {
            "acf": {str(k): a[k] for k in range(0, 22)},
            "acf_significance_band": float(ci95),
            "significant_lags": sig,
            "lag1": a[1], "lag7": a[7],
            "superseded_claim": ("docs quote lag-1 ACF +0.41 measured on the "
                                 "weather-free Ridge; the champion's value is "
                                 f"{a[1]:+.3f}"),
            "mean": float(e_all.mean()), "sd": float(e_all.std()),
            "skew": float(pd.Series(e_all).skew()),
            "excess_kurtosis": float(pd.Series(e_all).kurtosis()),
        },
        "environment": capture_environment(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "error_analysis.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    err.reset_index().to_csv(RESULTS / "champion_residuals_regime_b.csv",
                             index=False)
    print(f"\nwrote {RESULTS / 'error_analysis.json'}")


if __name__ == "__main__":
    main()
