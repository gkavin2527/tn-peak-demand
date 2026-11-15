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


def newey_west_lrv(x: np.ndarray, lag: int | None = None) -> float:
    """Long-run variance of a mean, Newey-West (Bartlett kernel).

    Daily peak-demand loss differentials are autocorrelated, so the naive
    variance of the mean understates uncertainty - the same reason this module
    uses a moving-block bootstrap rather than an iid one. `lag` defaults to
    the standard floor(4 * (n/100)^(2/9)) truncation.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")
    if lag is None:
        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(int(lag), n - 2))

    d = x - x.mean()
    gamma0 = float(d @ d) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        gk = float(d[k:] @ d[:-k]) / n
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gk   # Bartlett weight
    # A Bartlett-weighted sum can still go non-positive in small samples.
    if lrv <= 0:
        lrv = gamma0
    return lrv


def diebold_mariano(e_candidate: np.ndarray, e_reference: np.ndarray,
                    loss: str = "abs", lag: int | None = None) -> dict:
    """Diebold-Mariano test of equal predictive accuracy.

    Reported ALONGSIDE moving_block_bootstrap, never instead of it. The
    bootstrap is arguably the stronger instrument here - it makes no
    asymptotic-normality assumption on a 356-day sample - but DM is the
    forecasting literature's lingua franca and referees ask for it by name.

    H0: equal expected loss. The statistic is

        DM = mean(d) / sqrt(LRV(d) / n),   d_t = L(e_cand) - L(e_ref)

    so **negative DM favours the candidate**, matching the sign convention of
    moving_block_bootstrap's `diffs`. The long-run variance uses Newey-West
    because d_t is serially correlated.

    Uses the Harvey-Leybourne-Newbold small-sample correction and refers the
    statistic to t(n-1) rather than the normal, both of which matter at these
    sample sizes.

    loss: "abs" (|e|, pairs with MAE) or "sq" (e^2, pairs with RMSE).
    """
    ec = np.asarray(e_candidate, dtype=float).ravel()
    er = np.asarray(e_reference, dtype=float).ravel()
    if ec.shape != er.shape:
        raise ValueError(f"unpaired errors: {ec.shape} vs {er.shape}")
    finite = np.isfinite(ec) & np.isfinite(er)
    ec, er = ec[finite], er[finite]
    n = ec.size
    if n < 3:
        raise ValueError(f"need at least 3 paired observations, got {n}")

    if loss == "abs":
        d = np.abs(ec) - np.abs(er)
    elif loss == "sq":
        d = ec ** 2 - er ** 2
    else:
        raise ValueError(f"unknown loss {loss!r}; use 'abs' or 'sq'")

    dbar = float(d.mean())
    lrv = newey_west_lrv(d, lag=lag)
    se = float(np.sqrt(lrv / n))
    if se == 0:
        # Identical forecasts - OLS and Ridge at alpha=0.001 do this exactly.
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_loss_diff": dbar,
                "n": n, "loss": loss, "se": 0.0, "nw_lag": lag,
                "hln_corrected": True, "note": "identical loss series"}

    dm = dbar / se
    # Harvey-Leybourne-Newbold correction for h=1 (one-step-ahead losses).
    dm *= float(np.sqrt((n - 1.0) / n))

    from scipy import stats
    p_two = float(2.0 * stats.t.sf(abs(dm), df=n - 1))
    p_one = float(stats.t.cdf(dm, df=n - 1))   # H1: candidate better (dm < 0)
    return {"dm_stat": float(dm), "p_value": p_two, "p_value_one_sided": p_one,
            "mean_loss_diff": dbar, "se": se, "n": n, "loss": loss,
            "nw_lag": int(lag) if lag is not None
                      else int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))),
            "hln_corrected": True,
            "favours": "candidate" if dm < 0 else "reference"}
