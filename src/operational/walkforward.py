"""
Stand-in walk-forward engine, so Track B's Phase 5 is not blocked.

Roadmap C.3 anticipates exactly this: "Track B's Phase 5 needs a model. Until
A's champion exists, B runs the experiment end to end with a stand-in. The
code is identical; only champion_config.json changes at hand-over."

`run_walk_forward` is **Track A's deliverable**, in src/walkforward.py. This
file implements the frozen G0 signature so the operational experiment can run
today. src.contract.run_walk_forward prefers Track A's implementation the
moment it appears, and this becomes dead code - delete it then rather than
letting two engines drift apart.

Deliberately plain: expanding window, periodic refit, optional exponential
recency weights, no tuning. If this ever beats Track A's champion, something
is wrong with the champion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "y"


def exponential_weights(dates: pd.DatetimeIndex, as_of: pd.Timestamp,
                        half_life_years: float | None) -> np.ndarray | None:
    """Recency weights halving every `half_life_years` (D1's recency scheme)."""
    if not half_life_years:
        return None
    age_years = (as_of - dates).days / 365.25
    return np.power(0.5, age_years / float(half_life_years))


def run_walk_forward(model_factory, df: pd.DataFrame, start, end,
                     refit_days: int = 7,
                     half_life_years: float | None = None,
                     features: list[str] | None = None,
                     differenced: bool = False,
                     anchor: str | None = None) -> pd.DataFrame:
    """Expanding-window walk-forward with periodic refit.

    model_factory   callable returning a fresh unfitted estimator
    df              load_master() output: indexed by target date, target in `y`
    start, end      inclusive test window
    refit_days      refit cadence; 7 = weekly (config.REFIT_DAYS)
    half_life_years recency weighting; None = uniform
    features        columns to use; defaults to every numeric non-target column
    differenced     fit (y - df[anchor]) and add the anchor back afterwards
    anchor          column carrying the level when differenced=True

    Returns columns: date, y_true, y_pred, refit_date.

    Never refits on anything dated >= the prediction date, so the expanding
    window is honest. Rows whose target or anchor is NaN are dropped from
    training but still predicted, so the output covers the full window.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if differenced and not anchor:
        raise ValueError("differenced=True requires an anchor column")

    if features is None:
        features = [c for c in df.columns
                    if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]

    test_idx = df.index[(df.index >= start) & (df.index <= end)]
    if len(test_idx) == 0:
        raise ValueError(f"no rows in {start:%Y-%m-%d}..{end:%Y-%m-%d}")

    # Refit boundaries: every `refit_days` across the window. Each block is
    # predicted by a model that saw only data strictly before the block opens.
    blocks = []
    cursor = test_idx[0]
    while cursor <= test_idx[-1]:
        stop = cursor + pd.Timedelta(days=refit_days - 1)
        blocks.append((cursor, min(stop, test_idx[-1])))
        cursor = stop + pd.Timedelta(days=1)

    rows = []
    for block_start, block_stop in blocks:
        train = df[df.index < block_start]
        test = df[(df.index >= block_start) & (df.index <= block_stop)]
        if test.empty:
            continue

        y = train[TARGET] - train[anchor] if differenced else train[TARGET]
        ok = y.notna() & np.isfinite(y)
        if ok.sum() < 100:
            continue

        model = model_factory()
        w = exponential_weights(train.index[ok], block_start, half_life_years)
        try:
            model.fit(train.loc[ok, features], y[ok], sample_weight=w)
        except TypeError:          # estimator without sample_weight support
            model.fit(train.loc[ok, features], y[ok])

        pred = model.predict(test[features])
        if differenced:
            pred = pred + test[anchor].to_numpy()

        rows.append(pd.DataFrame({
            "date": test.index,
            "y_true": test[TARGET].to_numpy(),
            "y_pred": pred,
            "refit_date": block_start,
        }))

    if not rows:
        raise RuntimeError("walk-forward produced no predictions")
    return pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)
