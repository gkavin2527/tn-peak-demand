"""
Shared evaluation primitives for Track A (Phase 6 onward).

Roadmap C.1 assigns `src/evaluation.py` to Track A. Until Phase 6 it did not
exist, and `run_phase5_regime_ab.py` carried private copies of the metric and
bootstrap functions. Those copies are reproduced here **byte-for-byte in
behaviour** so Phase 6 numbers remain directly comparable with the committed
Phase 5 decision gate and Regime A/B results. Do not "improve" them: a changed
metric silently invalidates every prior comparison.

Two conventions worth stating because they are easy to get backwards:

* `moving_block_bootstrap` resamples **contiguous 7-day blocks**, not
  individual days. Daily peak-demand errors autocorrelate - a heatwave or a
  festival misses for a week straight - so an iid bootstrap would report
  intervals roughly sqrt(block) too narrow.

* The paired difference convention is `d_t = |e_model| - |e_reference|`, so a
  **negative** mean means the candidate model beats the reference. This matches
  the Phase 5 decision gate's `direction_convention` field exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_DAYS = 7
N_BOOT = 2000
BOOT_SEED = 42

__all__ = ["BLOCK_DAYS", "N_BOOT", "BOOT_SEED", "compute_metrics",
           "moving_block_bootstrap", "surge_mask", "subset_metrics"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / MAPE / bias. Identical to the Phase 5 definition."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    pct_err = np.abs((y_true - y_pred) / y_true) * 100.0
    bias = y_pred - y_true
    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "mape": float(np.mean(pct_err)),
        "bias": float(np.mean(bias)),
        "n_samples": int(len(y_true)),
    }


def moving_block_bootstrap(diffs: np.ndarray, block_size: int = BLOCK_DAYS,
                           n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Percentile CI for the mean paired difference, resampling 7-day blocks.

    `diffs` is the per-day paired series |e_candidate| - |e_reference|.
    The one-sided p-value tests H0: mean >= 0 (candidate no better) against
    H1: mean < 0 (candidate better), i.e. p = P(draw >= 0).

    Note the sign flip relative to the Phase 5 Regime A/B script: there, the
    alternative was "Regime B is WORSE", so it used p = mean(draws <= 0). Here
    the question is "is this model BETTER than Ridge", so the tail is mirrored.
    The CI and the mean are computed identically.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2 * block_size:
        return {"mean": float(np.mean(diffs)) if n else float("nan"),
                "ci_95_low": float("nan"), "ci_95_high": float("nan"),
                "p_value_one_sided": float("nan"), "std_err": float("nan"),
                "n": int(n)}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=(n_boot, n_blocks))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts[i]])[:n]
        draws[i] = np.mean(diffs[idx])

    return {
        "mean": float(np.mean(diffs)),
        "ci_95_low": float(np.percentile(draws, 2.5)),
        "ci_95_high": float(np.percentile(draws, 97.5)),
        "p_value_one_sided": float(np.mean(draws >= 0)),
        "std_err": float(np.std(draws)),
        "n": int(n),
    }


def surge_mask(dates, y_true, quantile: float = 0.05) -> np.ndarray:
    """Top-`quantile` demand days, thresholded **within each calendar year**.

    Per-year rather than a fixed MW line because Tamil Nadu's peak grew ~3.5%/yr
    and is accelerating: a fixed 19,000 MW threshold selects almost no days in
    2015 and a third of 2026, so it would measure demand growth rather than
    model behaviour. This matches decision D4.
    """
    s = pd.DataFrame({"date": pd.to_datetime(dates), "y": np.asarray(y_true, float)})
    thr = s.groupby(s["date"].dt.year)["y"].transform(lambda v: v.quantile(1 - quantile))
    return (s["y"] >= thr).to_numpy()


def subset_metrics(dates, y_true, y_pred, mask, label: str) -> dict:
    """Metrics on a boolean subset, with the mask size recorded.

    Bias matters more than MAE on small subsets like surge days: a model that
    is several hundred MW low on every record day fails in one direction,
    exactly when reserve margin is thinnest, and MAE cannot express that.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return {"label": label, "n_samples": 0}
    out = compute_metrics(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])
    out["label"] = label
    return out
