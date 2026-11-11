"""
Error metrics for the operational experiment, including the surge subsets.

Track A owns `src/evaluation.py` and this is NOT it. Track B needs surge-subset
bias and confidence intervals to report the Phase 5 gap, and cannot wait for
Track A to ship (roadmap C.3). At G2 the two must be reconciled - the roadmap
is explicit that "A's top-5% bias and B's gap numbers use the same
definitions" - so the surge rules here follow D4 exactly and any divergence
should be resolved by deleting this file in favour of Track A's.

Two deliberate choices, both from D4:

* The surge threshold is a **per-calendar-year quantile**, not a fixed MW
  value. Tamil Nadu's peak grew ~3.5%/yr and is accelerating; a fixed
  19,000 MW line selects two days in 2015 and a third of 2026, so it measures
  demand growth rather than model behaviour. The legacy fixed threshold is
  still computed, for continuity only.

* **Bias is reported next to error everywhere.** On a 15-20 day subset, MAE
  alone hides the thing that matters operationally: a model that is 400 MW
  low on every record day is far more dangerous than one that is 400 MW off
  in alternating directions, and MAE cannot tell them apart.

Confidence intervals use a **moving-block bootstrap**. Daily peak errors are
strongly autocorrelated - a heatwave misses for a week straight - so an iid
bootstrap would report intervals roughly sqrt(block) times too narrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RANDOM_SEED, SUMMER_MONTHS, SURGE_LEGACY_MW, SURGE_QUANTILE

BLOCK_DAYS = 7          # one week: the scale on which forecast errors persist
N_BOOT = 2000


def _core(y_true, y_pred):
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    y = np.asarray(y_true, float)
    ok = np.isfinite(e) & np.isfinite(y) & (y != 0)
    e, y = e[ok], y[ok]
    if not len(e):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan,
                "bias": np.nan, "r2": np.nan}
    return {
        "n": int(len(e)),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "mape": float(np.mean(np.abs(e / y)) * 100),
        "bias": float(e.mean()),
        "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
    }


def moving_block_ci(y_true, y_pred, stat="mae", alpha=0.05,
                    block=BLOCK_DAYS, n_boot=N_BOOT, seed=RANDOM_SEED):
    """Percentile CI for one statistic, resampling contiguous blocks."""
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    y = np.asarray(y_true, float)
    ok = np.isfinite(e) & np.isfinite(y) & (y != 0)
    e, y = e[ok], y[ok]
    n = len(e)
    if n < 2 * block:
        return (np.nan, np.nan)

    def val(ei, yi):
        if stat == "mae":
            return np.abs(ei).mean()
        if stat == "bias":
            return ei.mean()
        if stat == "mape":
            return np.mean(np.abs(ei / yi)) * 100
        if stat == "rmse":
            return np.sqrt((ei ** 2).mean())
        raise ValueError(stat)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in starts[i]])[:n]
        draws[i] = val(e[idx], y[idx])
    return (float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))


def surge_mask(df, kind="per_year", quantile=SURGE_QUANTILE):
    """Boolean mask selecting peak-surge days. df needs `date` and `y_true`.

    per_year  (D4 primary)   top `quantile` of each calendar year
    summer    (D4 secondary) top `quantile` of March-June days, per year
    legacy    (D4 legacy)    y_true > SURGE_LEGACY_MW, drifts with growth
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if kind == "legacy":
        return d["y_true"] > SURGE_LEGACY_MW
    if kind == "summer":
        d = d[d["date"].dt.month.isin(SUMMER_MONTHS)]
        thr = d.groupby(d["date"].dt.year)["y_true"].transform(
            lambda s: s.quantile(1 - quantile))
        hot = pd.Series(False, index=df.index)
        hot.loc[d.index] = d["y_true"] >= thr
        return hot
    if kind == "per_year":
        thr = d.groupby(d["date"].dt.year)["y_true"].transform(
            lambda s: s.quantile(1 - quantile))
        return d["y_true"] >= thr
    raise ValueError(kind)


def evaluate(preds: pd.DataFrame, with_ci=True, label="") -> dict:
    """Full metric bundle for one prediction frame.

    preds: date, y_true, y_pred (the run_walk_forward contract output).
    Returns overall, March-June, and the three surge subsets, each with bias.
    """
    d = preds.dropna(subset=["y_true", "y_pred"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    out = {"label": label, "span": [str(d["date"].min().date()),
                                    str(d["date"].max().date())]}

    subsets = {
        "overall": pd.Series(True, index=d.index),
        "summer_mar_jun": d["date"].dt.month.isin(SUMMER_MONTHS),
        "surge_per_year": surge_mask(d, "per_year"),
        "surge_summer": surge_mask(d, "summer"),
        "surge_legacy_19000mw": surge_mask(d, "legacy"),
    }

    for name, m in subsets.items():
        s = d[m]
        block = _core(s["y_true"], s["y_pred"])
        if with_ci and len(s) >= 2 * BLOCK_DAYS:
            for stat in ("mae", "bias"):
                lo, hi = moving_block_ci(s["y_true"], s["y_pred"], stat=stat)
                block[f"{stat}_ci95"] = [round(lo, 1), round(hi, 1)]
        out[name] = block
    return out


def compare(regime_a: dict, regime_b: dict) -> dict:
    """Degradation from A to B on each subset: the Phase 5 headline."""
    delta = {}
    for key in regime_a:
        if not isinstance(regime_a.get(key), dict):
            continue
        a, b = regime_a[key], regime_b.get(key, {})
        if not b:
            continue
        delta[key] = {
            "n": a.get("n"),
            "mae_a": round(a["mae"], 1), "mae_b": round(b["mae"], 1),
            "d_mae": round(b["mae"] - a["mae"], 1),
            "d_mae_pct": (round((b["mae"] / a["mae"] - 1) * 100, 1)
                          if a["mae"] else None),
            "mape_a": round(a["mape"], 3), "mape_b": round(b["mape"], 3),
            "d_mape": round(b["mape"] - a["mape"], 3),
            "d_rmse": round(b["rmse"] - a["rmse"], 1),
            "bias_a": round(a["bias"], 1), "bias_b": round(b["bias"], 1),
            "d_bias": round(b["bias"] - a["bias"], 1),
        }
    return delta


def table(rows: list[tuple[str, dict]], subset="overall") -> str:
    """Small fixed-width table for terminal output."""
    head = (f"  {'model / regime':<34}{'n':>6}{'MAE':>8}{'MAPE':>8}"
            f"{'RMSE':>9}{'bias':>9}")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for label, res in rows:
        m = res.get(subset, {})
        if not m or not m.get("n"):
            continue
        lines.append(f"  {label:<34}{m['n']:>6}{m['mae']:>8.1f}"
                     f"{m['mape']:>7.2f}%{m['rmse']:>9.1f}{m['bias']:>+9.1f}")
    return "\n".join(lines)
