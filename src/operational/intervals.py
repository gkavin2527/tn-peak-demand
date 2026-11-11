"""
Prediction intervals by rolling split-conformal calibration (roadmap D6).

Track A owns the full interval study in Phase 4.3, including the
quantile-LightGBM comparison and coverage tables. This module is the
*serving* half that Phase 7.1 needs: given a history of walk-forward
predictions, produce the multipliers `predict_next_day.py` applies tonight,
and persist them so inference does not refit anything.

Method
------
Split-conformal on **relative** residuals, r = |y - yhat| / yhat. Relative
rather than absolute because the error scales with the level: a 400 MW miss
at 11,000 MW and at 21,000 MW are not the same event, and a single absolute
width would be too wide in winter and too narrow in May.

Calibrated **separately for March-June and the rest of the year**, because
summer errors are materially larger - the season is in the interval width,
not averaged away.

Only the trailing `window_days` of predictions are used, so the calibration
tracks the current regime rather than 2021's.

Honest caveat, stated in the paper per D6
-----------------------------------------
Conformal prediction assumes exchangeability. A demand time series is not
exchangeable - it trends, it has seasons, and errors autocorrelate. So the
coverage guarantee is approximate, not exact, and is why `coverage_report()`
measures realised coverage instead of asserting the nominal level.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import RESULTS, SUMMER_MONTHS

WINDOW_DAYS = 365
LEVELS = (0.90, 0.95)
SEASONS = ("summer", "rest")


def _season(dates) -> np.ndarray:
    m = pd.DatetimeIndex(dates).month
    return np.where(np.isin(m, SUMMER_MONTHS), "summer", "rest")


def calibrate(preds: pd.DataFrame, window_days: int = WINDOW_DAYS,
              levels=LEVELS) -> dict:
    """Relative-residual quantiles per season and level.

    preds: date, y_true, y_pred (run_walk_forward's contract output).
    Returns {"summer": {"0.90": q, ...}, "rest": {...}, ...} where a forecast
    yhat becomes yhat * (1 +/- q).
    """
    d = preds.dropna(subset=["y_true", "y_pred"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    cutoff = d["date"].max() - pd.Timedelta(days=window_days)
    d = d[d["date"] > cutoff]
    d = d[d["y_pred"] > 0]
    d["rel"] = (d["y_true"] - d["y_pred"]).abs() / d["y_pred"]
    d["season"] = _season(d["date"])

    out = {"calibrated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
           "window_days": window_days,
           "window": [str(d["date"].min().date()), str(d["date"].max().date())],
           "method": "rolling split-conformal on relative residuals",
           "seasons": {}}

    for s in SEASONS:
        sub = d[d["season"] == s]["rel"]
        if len(sub) < 30:
            # too thin to calibrate separately; fall back to the pooled
            # quantile rather than inventing a narrow interval
            sub = d["rel"]
            note = f"pooled (only {len(d[d['season'] == s])} {s} days)"
        else:
            note = f"{len(sub)} days"
        qs = {}
        for lv in levels:
            # finite-sample conformal correction: ceil((n+1)*level)/n
            n = len(sub)
            k = min(int(np.ceil((n + 1) * lv)) / n, 1.0)
            qs[f"{lv:.2f}"] = float(np.quantile(sub, k))
        out["seasons"][s] = {"n": int(len(sub)), "note": note,
                             "relative_halfwidth": qs}
    return out


def apply(yhat: float, date, calib: dict, level: float = 0.90):
    """(lower, upper) in MW for one prediction."""
    s = _season([date])[0]
    q = calib["seasons"][s]["relative_halfwidth"][f"{level:.2f}"]
    return float(yhat * (1 - q)), float(yhat * (1 + q))


def coverage_report(preds: pd.DataFrame, calib: dict, levels=LEVELS) -> dict:
    """Realised coverage and mean width - does the calibration hold up?"""
    d = preds.dropna(subset=["y_true", "y_pred"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["season"] = _season(d["date"])
    rep = {}
    for lv in levels:
        lo, hi = [], []
        for r in d.itertuples():
            a, b = apply(r.y_pred, r.date, calib, lv)
            lo.append(a); hi.append(b)
        lo, hi = np.array(lo), np.array(hi)
        inside = (d["y_true"].to_numpy() >= lo) & (d["y_true"].to_numpy() <= hi)
        # interval score (Gneiting): width + asymmetric miss penalty
        alpha = 1 - lv
        y = d["y_true"].to_numpy()
        score = ((hi - lo)
                 + (2 / alpha) * (lo - y) * (y < lo)
                 + (2 / alpha) * (y - hi) * (y > hi))
        rep[f"{lv:.2f}"] = {
            "nominal": lv,
            "realised_coverage": round(float(inside.mean()), 4),
            "mean_width_mw": round(float((hi - lo).mean()), 1),
            "interval_score": round(float(score.mean()), 1),
            "by_season": {
                s: round(float(inside[(d["season"] == s).to_numpy()].mean()), 4)
                for s in SEASONS if (d["season"] == s).any()},
        }
    return rep


def save(calib: dict, coverage: dict | None = None, path=None):
    path = path or (RESULTS / "intervals.json")
    payload = dict(calib)
    if coverage:
        payload["coverage"] = coverage
    payload["caveat"] = (
        "Time series are not exchangeable, so conformal coverage is "
        "approximate. Realised coverage is reported rather than assumed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load(path=None) -> dict:
    path = path or (RESULTS / "intervals.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run 'python calibrate_intervals.py' first")
    return json.loads(path.read_text())
