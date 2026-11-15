"""
Reference baseline for next-day peak demand.

    python baseline.py

Not a tuned model - a correct one, so there is a number to beat that is not
accidentally inflated. Three decisions do almost all the work, and the first
one matters more than any hyperparameter.

1. PREDICT THE CHANGE, NOT THE LEVEL
   Tamil Nadu's peak demand grew ~3.5%/yr over 2014-2025 and is running ~10%
   above 2025 in 2026. A gradient-boosted tree cannot extrapolate: its
   prediction is a leaf average, so it can never output a value above the
   largest it saw in training. Fit it to the LEVEL and it under-forecasts
   every record-breaking day - the 2026 holdout comes out ~421 MW low, on 75%
   of days.

   Fitting y - peak_lag_1 removes the trend from the target entirely. The
   model only has to learn "how different is tomorrow from today", which is
   stationary, and the growing level is reinstated by adding peak_lag_1 back.
   Same features, same model, MAE 547 -> 388 and the bias collapses to ~-90.

   This is also why no yoy_growth_* column ships in master.csv: differencing
   already adapts to the growth regime, and adding explicit growth ratios on
   top measured no better (392 vs 391).

2. DROP THE WEATHER LEVELS
   Redundant with the autoregressive block - see config.WEATHER_LEVEL_COLS.

3. DROP `trend`
   A monotone, unbounded feature is the worst possible input to a tree in an
   extrapolating regime: every future row falls in the top bin, so `trend`
   contributes a constant and crowds out splits that generalise. Removing it
   improves the 2026 bias from -131 to -52 MW.

Evaluated with rolling-origin CV: train on everything up to a cutoff, test the
following year, step forward. No shuffling - that would leak the future.
"""

import threadlimit  # noqa: F401  - must precede numpy/sklearn

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import MASTER, PEAK_TARGET, WEATHER_LEVEL_COLS

TARGET = PEAK_TARGET
ANCHOR = "peak_lag_1"          # known before day T; carries the level
FOLDS = [("2022-11-09", "2023-11-09"), ("2023-11-09", "2024-11-09"),
         ("2024-11-09", "2025-11-09"), ("2025-11-09", "2026-11-09")]
SEEDS = 3
DROP = ("trend",)


def features(df):
    exclude = {"date", TARGET, "holiday_name", *WEATHER_LEVEL_COLS, *DROP}
    return [c for c in df.columns if c not in exclude]


def fit_predict(train, test, cols, differenced=True):
    """Fit on `train`, return predictions for `test` in MW."""
    y = train[TARGET] - train[ANCHOR] if differenced else train[TARGET]
    ok = y.notna()
    preds = [
        HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                      random_state=s)
        .fit(train.loc[ok, cols], y[ok]).predict(test[cols])
        for s in range(SEEDS)
    ]
    p = np.mean(preds, axis=0)
    return p + test[ANCHOR].values if differenced else p


def score(name, df, cols, differenced=True):
    rows = []
    for cutoff, end in FOLDS:
        train = df[df["date"] <= cutoff]
        test = df[(df["date"] > cutoff) & (df["date"] <= end)]
        p = fit_predict(train, test, cols, differenced)
        y = test[TARGET].values
        good = np.isfinite(p)
        e = p[good] - y[good]
        rows.append({"fold": end[:4], "n": int(good.sum()),
                     "mae": np.abs(e).mean(),
                     "mape": np.mean(np.abs(e / y[good])) * 100,
                     "bias": e.mean()})
    r = pd.DataFrame(rows)
    print(f"  {name:<34} " + " | ".join(f"{v:5.0f}" for v in r.mae)
          + f"   MAE {r.mae.mean():6.1f}   MAPE {r.mape.mean():4.2f}%"
          + f"   bias {r.bias.iloc[-1]:+7.1f}")
    return r


def main():
    df = pd.read_csv(MASTER, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cols = features(df)
    print(f"{len(df):,} rows, {len(cols)} features, "
          f"folds ending {', '.join(e[:4] for _, e in FOLDS)}\n")
    print("  model                              " +
          " | ".join(f"{e[:4]:>5}" for _, e in FOLDS) + "       mean")

    # naive references
    for name, pred in (("persistence (peak_lag_1)", df[ANCHOR]),
                       ("7-day rolling mean", df["peak_roll_mean_7"])):
        maes, last = [], None
        for cutoff, end in FOLDS:
            t = df[(df["date"] > cutoff) & (df["date"] <= end)]
            e = (pred.loc[t.index] - t[TARGET]).dropna()
            maes.append(e.abs().mean())
            last = e.mean()
        print(f"  {name:<34} " + " | ".join(f"{v:5.0f}" for v in maes)
              + f"   MAE {np.mean(maes):6.1f}" + " " * 16 + f"bias {last:+7.1f}")

    score("GBM, level target (the trap)", df, features(df) + list(DROP), differenced=False)
    score("GBM, level target, no trend", df, cols, differenced=False)
    best = score("GBM, differenced (recommended)", df, cols)

    print(f"\nrecommended baseline: {best.mae.mean():.0f} MW MAE "
          f"({best.mape.mean():.2f}% MAPE) on unseen years.")
    print("Reproduce with: python baseline.py")


if __name__ == "__main__":
    main()
