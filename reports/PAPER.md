# Quantifying the Operational Cost of Weather-Forecast Uncertainty in Next-Day Peak Electricity Demand Forecasting for Tamil Nadu

**Draft — generated 2026-11-15 by `make_paper.py`.** Every
number below is read from committed artifacts at build time. Re-run before
each submission. Convert to the venue template last.

> **Author checklist before submission**
> - [ ] Zenodo DOI for the historical backfill (3,852 of 4,905 target rows) — **not recorded anywhere in the repository**
> - [ ] Resolve the seven `[FILL]` citations in `RELATED_WORK.md`, especially India/TN-specific STLF
> - [ ] Verify every citation against the publisher record
> - [ ] Author list, affiliations, funding, conflicts

---

## Abstract

Short-term load forecasting is routinely evaluated using observed or
reanalysis weather for the target day — information no operational forecaster
possesses. We quantify what that assumption is worth. Using 4,905 days of
Tamil Nadu daily peak demand (2013–2026) together with a 29,535-row archive of
genuine lead-1 numerical weather forecasts spanning 5.4 years, we evaluate a
single leakage-controlled, horizon-aware pipeline under three conditions: no
weather, reanalysis weather, and operationally available forecast weather.

The forecasting horizon is itself a measured constraint. Grid-India publishes
day *D*'s peak on *D+1*, so an evening forecast for the next day has demand
only through *D−1*; we denote this H = 2 and show the most predictive recent
lag is therefore unusable.

Weather reduces mean absolute error by 17.9% under reanalysis
(696 → 572 MW). When only archived lead-1 forecasts are used,
error rises to 612 MW — **giving back 32% of the weather gain**
(penalty 40.3 MW, 95% CI [25.9, 54.1],
Diebold–Mariano p = 1.4e-08). This penalty is invisible to
reanalysis-based evaluation.

Across nine model families, regularised linear regression outperformed four
tuned gradient-boosting implementations, which we attribute to the inability of
tree ensembles to extrapolate a trending target. Our final model pairs a linear
trend component with a quantile-regression residual learner, reaching
497.90 MW under operational weather in validation; a held-out
confirmation year gives 503.49 MW (2.74% MAPE). We further
quantify our own selection optimism by replaying the entire selection pipeline
with one year held out, and find it negative — the reported figures are
conservative rather than inflated.

**Keywords:** short-term load forecasting; numerical weather prediction;
forecast-weather uncertainty; leakage control; conformal prediction; Tamil Nadu

---

## 1. Introduction

*(Draft the motivation here. The structure that works:)*

1. Peak demand sets reserve commitment; errors are asymmetrically costly.
2. Weather — principally temperature — is the dominant exogenous driver, and
   the STLF literature has established this thoroughly.
3. **But** evaluation in that literature overwhelmingly conditions on
   target-day weather that is observed or reanalysed, i.e. unavailable at
   forecast time. Reported errors are therefore not operational errors.
4. We measure the difference, under one fixed harness, with a real forecast
   archive.

**Contributions.**

1. A three-condition evaluation ladder (no weather → reanalysis → genuine
   lead-1 forecast) under a single leakage-controlled protocol, isolating the
   operational weather penalty as a measured quantity with a paired confidence
   interval.
2. A measured forecasting horizon (H = 2) derived from publication-timing
   evidence rather than assumed, with its consequences enforced in code.
3. A negative result on model families: four tuned GBM implementations lose to
   regularised linear regression on a trending target, with paired CIs.
4. A quantification of our own selection optimism by nested replay of the full
   selection pipeline — reported whichever way it came out.

---

## 2. Data

| | |
|---|---|
| Target | Tamil Nadu daily peak demand (MW) |
| Period | 2013-01-01 → 2026 (4,905 days) |
| Source | Grid-India (formerly POSOCO) Daily PSP reports |
| Weather | Open-Meteo: ERA5 reanalysis + Previous Runs forecast archive |
| Forecast archive | 29,535 rows, lead-resolved, from mid-2021 |
| Missing calendar days | 157 (left as NaN, never interpolated) |

State-level weather is a population-weighted composite over major load
centres. Temperature-derived features include cooling-degree days, hours above
32 °C, day-on-day deltas and 7-day anomalies.

**Calendar.** The Tamil Nadu holiday calendar is resolved from a curated
source rather than a package default, because package coverage of lunisolar
festivals is incomplete and would make `is_holiday` mean "a holiday the
package happened to know about" in a year-dependent way.

---

## 3. Problem formulation

### 3.1 The horizon is a measurement, not a choice

Grid-India publishes day *D*'s peak demand report on *D+1*, around
09:00–11:00 IST. Across 46 observed pipeline runs — including one at 18:07 —
same-day data was **never** available. An evening forecast issued on day *D*
for target *D+1* therefore has demand observations only through *D−1*.

We write H = 2. The practical consequence is severe: `peak_lag_1`, the
second-most-correlated predictor in the dataset (r = 0.927), **cannot be
used**. Training on it yields a model that cannot be run. This is enforced in
the data gateway and asserted mechanically in the test suite.

### 3.2 Two weather regimes

- **Regime A (reanalysis).** Target-day weather from ERA5. A retrospective
  best estimate assimilating observations that postdate the forecast moment —
  a benchmark, not a servable input.
- **Regime B (operational).** Target-day weather from archived **lead-1**
  forecasts: what a forecaster would actually have held.

Both regimes share every non-weather feature. Only target-day weather columns
are exchanged, verified by an identity test that reproduces Regime A to
0.000e+00 MW when reanalysis is swapped back in.

Lead-awareness matters for derived features. Predicting day *T*, the forecast
supplies *T*, but *T−1* has already happened and is observed, so deltas are
computed as forecast(*T*) − observed(*T−1*).

> **Disclosed limitation.** The archive records forecast *lead*, not *issue
> hour*. Our claim is "archived lead-1 aligned to valid date", never
> "available at 20:00 IST".

### 3.3 Protocol

Expanding-window walk-forward, validation years 2023 / 2024 / 2025; no random
splitting anywhere. Models train on the full ERA5 history in both regimes;
only the scored rows differ, so the A–B difference is attributable to
target-day weather and not to training volume. One shared scoring sample
(1,066 dates) across all models and both regimes.

Primary metric: MAE. Paired comparisons use a moving-block bootstrap (7-day
blocks, n = 2000, seed 42), justified by measured residual autocorrelation;
Diebold–Mariano with Newey–West variance and the Harvey–Leybourne–Newbold
correction is reported alongside.

---

## 4. Results

### 4.1 The operational weather penalty *(headline)*

**Figure 2.** All three conditions, Ridge, same 1,066 shared dates:

| condition | MAE (MW) |
|---|---|
| no weather (A0) | 696.27 |
| reanalysis (Regime A) | 571.61 |
| **archived lead-1 forecast (Regime B)** | **611.95** |

- Weather gain: **124.66 MW** (17.9% of the weather-free error)
- Operational penalty: **40.34 MW**, 95% CI [25.9, 54.1], DM p = 1.4e-08
- **32.4% of the weather gain is given back**

The penalty is stable across folds (39.5 / 36.8 / 44.6 MW) and
across seasons (summer 43.4, non-summer 38.7 MW).

> Report the penalty as a fraction of the **gain** (32.4%), not of the
> Regime A MAE (7.1%). Both are correct arithmetic; only the first
> answers "how much of the benefit survives contact with operations?"

### 4.2 Model families: linear beats boosted trees

**Figure 3.** Nine candidates, Regime A, paired against Ridge:

| model | ΔMAE vs Ridge | 95% CI | DM p |
|---|---|---|---|
| ridge_gbm_hybrid | -105.15 | [-120.80, -88.03] | 3.1e-33 |
| ols | -0.00 | [-0.00, +0.00] | 1.4e-01 |
| lightgbm_linear | +55.98 | [+9.27, +101.83] | 1.6e-02 |
| xgboost | +59.24 | [+14.26, +107.19] | 1.3e-02 |
| lightgbm | +69.09 | [+24.73, +118.68] | 3.8e-03 |
| catboost | +80.71 | [+32.69, +134.72] | 2.0e-03 |
| seasonal_naive | +426.85 | [+355.25, +499.33] | 6.5e-31 |
| naive_persistence | +472.77 | [+435.92, +509.56] | 2.7e-114 |

**Every pure gradient-boosting implementation lost to Ridge**, with CIs
excluding zero. We attribute this to extrapolation: tree ensembles predict
leaf averages and cannot exceed the training range, while the target grew
from 17,563 MW (2022) to 21,724 MW (2026). OLS and Ridge are numerically
identical here, confirming the L2 penalty is inert at this scale.

### 4.3 Final model

`quantile_hybrid` — a linear trend component plus a **quantile-regression**
gradient-boosted residual learner (τ = 0.5906) on 26 features.

| | Regime A | Regime B | A→B |
|---|---|---|---|
| final model | 443.62 | **497.90** | +54.28 |
| mean-residual variant | 455.96 | 520.37 | +64.42 |
| Ridge | 571.61 | 611.95 | +40.34 |

Paired vs the mean-residual variant: -22.47 MW,
CI [-34.49, -9.46], DM p = 5.6e-04.

The reason is elementary and worth stating: **MAE is minimised by the
conditional median, not the conditional mean**, so a squared-error residual
learner was never the right estimator for the reported metric. The effect
survives honest validation — with τ selected on 2023–24 alone, the variant was
-29.45 MW on untouched 2025
(CI [-55.11, -3.09], p = 0.025).

The operational penalty persists for the stronger model (+54.28 MW), though
it is smaller than for Ridge — a median-targeting model is somewhat less
sensitive to forecast weather error.

### 4.4 Validity

**Selection optimism.** One validation set drove four sequential selection
rounds. We replayed the entire pipeline — feature set, model ranking, 50
hyperparameter trials, final selection — using only 2023–24, then scored 2025
once:

| | reported | honest | optimism |
|---|---|---|---|
| Regime A | 438.19 | 421.50 | **-16.69 MW** |
| Regime B | 520.55 | 514.75 | **-5.80 MW** |

Optimism is *honest − reported*; negative means reported figures are
conservative. This is a **lower bound** — it replays reproducible selection
rounds, not informal inspection that shaped the search spaces.

**Confirmation evaluation.** Trained through 2025, evaluated once on 314 days
of 2026 under Regime B: MAE **503.49** vs 497.90 in validation
(+5.59 MW), MAPE 2.74%, RMSE 689.84.

> We call this a **confirmation**, not a blind test. 2026 was inspected during
> exploratory feature work. We do not claim it as an unbiased generalisation
> estimate; §4.4's held-out replay serves that role.

**Prediction intervals.** Split-conformal on relative residuals, fitted on
data through 2025 only. Realised 2026 coverage: 91.1% (nominal 90%) and
93.9% (nominal 95%) — genuinely out-of-sample. Residuals are
non-Gaussian (skew +1.18, excess kurtosis +4.48), supporting a
distribution-free method over normal theory.

### 4.5 What the model uses

**Figure 5.** Attribution is reported in two parts (linear coefficients;
exact TreeSHAP on the residual learner), with permutation importance as a
model-agnostic cross-check:

| block | mean \|attribution\| | permutation ΔMAE |
|---|---|---|
| weather | 2,428 MW | **+1,739 MW** |
| demand lags | 4,304 MW | +1,093 MW |
| calendar / trend | 1,484 MW | +451 MW |

**The two instruments disagree, and the disagreement is informative.**
Attribution ranks the demand-lag block roughly twice as important as weather;
permutation reverses the ordering. Both are correct: the five demand lags are
mutually collinear, so permuting one leaves the model able to recover from the
others, whereas weather supplies information no other feature carries. Mean
temperature alone has the largest single permutation importance (+991 MW).

This is stronger evidence that weather is load-bearing than an importance
ranking alone would provide.

---

## 5. Error structure and limitations

**Figure 4.**

| segment | n | MAE | bias |
|---|---|---|---|
| overall | 1066 | 497.9 | +17.0 |
| working day | 702 | 474.9 | -61.2 |
| weekend | 308 | 463.6 | +107.6 |
| summer (Mar–Jun) | 366 | 529.8 | +58.6 |
| **surge (top 5%)** | 56 | 548.2 | **-496.0** |
| holiday | 70 | 888.3 | +400.6 |
| **Diwali window** | 15 | 1801.7 | **+1766.4** |

Segments with n < 20 are anecdotal and no claim rests on them alone.

**Limitations, stated plainly.**

1. **Surge days are under-forecast by 496 MW** — the operationally
   expensive direction. A global quantile shift reduces this in-sample but
   transfers poorly (+7% on untouched data); a conditional
   correction is required and is left to future work.
2. **Festival days are badly over-forecast** (Diwali +1766 MW). Dedicated
   festival features remove roughly two thirds of this but do not reach
   significance on aggregate MAE (-3.37 MW, CI [-14.94, 6.91]) because
   the affected days are 15 of 1,066.
3. **Residual autocorrelation of +0.481 at lag 1** indicates remaining
   exploitable structure, consistent with multi-day weather shocks that H = 2
   cannot observe.
4. **2026 is a confirmation set, not a blind test.**
5. **Forecast issue hour is unverified.**
6. **Single region, single target** — generality is not demonstrated.
7. **Live operation is unproven.** Deployment rests on backtests, including a
   1016-prediction weekly-refit walk-forward (MAE 479.7 MW), which is a
   simulation.

---

## 6. Conclusion

We measured what the standard evaluation assumption costs. Weather improves
next-day peak demand forecasting for Tamil Nadu by 17.9% under reanalysis,
but **32% of that improvement does not survive** the transition to
operationally available forecasts. Studies that evaluate on reanalysis
therefore report an error that is systematically optimistic relative to what a
control room would experience.

We do not claim state-of-the-art accuracy. We claim the measurement, and we
provide the archive, protocol and code to reproduce it.

---

## Reproducibility

All results regenerate via `./reproduce.sh`. Dependencies are pinned to the
versions that produced them and recorded in the model configuration
(3.14.7, LightGBM 4.7.0, scikit-learn 1.9.1). Every headline
figure is recomputed from saved per-row predictions rather than transcribed.
