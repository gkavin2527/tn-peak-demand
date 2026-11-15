# Methodology — how this project actually works

Detailed reference for anyone modifying the pipeline. For results see
[RESULTS.md](RESULTS.md); for an assessment see [ANALYSIS.md](ANALYSIS.md).

---

## 1. The forecasting problem

Predict **tomorrow's daily peak demand** for Tamil Nadu, in MW, issued the
evening before.

### 1.1 H = 2 — the horizon, measured

Grid-India publishes day *D*'s peak demand on *D+1*, around 09:00–11:00 IST.
So a forecast issued on the evening of *D* for target *D+1* has demand
observations only through *D−1*.

**This was measured, not assumed.** `measure_publish_time.py` logged 46
pipeline runs — earliest 06:00, median 07:23, latest 18:07 — and same-day data
was available on **zero** of them. Recorded in `results/publish_time.json`.

**Consequence.** `peak_lag_1` (yesterday's peak) is unavailable. It correlates
0.927 with the target — the second-strongest predictor in the dataset. Using
it produces a model that cannot be served.

### 1.2 Per-group horizons

Different sources publish on different schedules, so `config.H_BY_GROUP`
assigns each its own:

| group | H | newest available |
|---|---|---|
| demand | 2 | target − 2 |
| energy | 2 | target − 2 |
| grid (`sr_*`) | 3 | target − 3 |

Before this existed, all four `sr_*_lag1` columns were NaN on the predicted
row. Fixing it was worth 204 MW.

> As of 2026-11, grid data has actually been arriving level with demand (0-day
> lag). `test_contract.py` reports this but does **not** fail, because arriving
> *earlier* than declared is conservative and leak-free. Only the reverse is
> dangerous.

---

## 2. Two lag conventions — the most dangerous thing in this repo

There are two representations of the same data. They are **not**
interchangeable, and mixing them fails silently.

| | `load_master(H=2)` | `load_phase5_data` |
|---|---|---|
| How H is enforced | **shifts** the demand block by H−1 | **excludes** `peak_lag_1`, `energy_lag_1` |
| `peak_lag_2` holds | T−3 | **T−2** (literal) |
| `peak_lag_7` lands on | the wrong weekday | the **same** weekday |
| Used by | the differenced stand-in model | **every Phase 5/6/7 model, incl. the champion** |

Both are correct for their own consumer. Serving a literal-lag model from a
shifted frame raises no error and returns a plausible number — while every
demand lag is a day stale and the weekly lags have lost weekday alignment.

**Measured cost of getting this wrong:** 2025 MAE goes from 438.2 → 683.7 MW
(**+56%**), roughly 2.3× the champion's entire gain over Ridge.

`models/champion_config.json` therefore records `"lag_representation":
"literal"`, `predict_next_day.py` reads it rather than inferring, and
`test_serving.py` asserts the served feature row is identical to
`load_phase5_data` for all 26 features.

---

## 3. Weather: two regimes

| | Regime A | Regime B |
|---|---|---|
| Source | ERA5 reanalysis | archived **lead-1** forecast |
| Servable? | **No** — assimilates observations postdating the forecast | **Yes** |
| Role | benchmark ceiling | operational reality |

Both regimes share every non-weather feature. Only target-day weather columns
are swapped, by `swap_weather()`. A **G0 identity test** confirms that swapping
reanalysis back in reproduces Regime A to 0.000e+00 MW.

### 3.1 Lead-awareness in derived features

Predicting day *T*, the forecast supplies *T*, but *T−1* has already happened:

```
d_t_max        = forecast(T) − observed(T−1)
t_max_vs_roll7 = forecast(T) − mean(observed[T−7 … T−1])
```

Treating both sides as forecast would overstate the degradation.

### 3.2 Documented residual optimism

The four T−1-dependent features use complete ERA5 for T−1 in **both** regimes,
because the archive has no lead-0. Both regimes are equally optimistic there,
so the A−B *difference* is unaffected — but the absolute Regime B number is
slightly optimistic.

### 3.3 Issue hour is unverified

The archive records forecast *lead*, not *issue hour*. The claim is **"archived
lead-1 aligned to valid date"**, never "available at 20:00 IST". Keep this
wording.

---

## 4. Features

**A2, 26 features**, the frozen specification:

| block | features |
|---|---|
| calendar (6) | `dow`, `is_weekend`, `is_holiday`, `day_before_holiday`, `day_after_holiday`, `pongal_window` |
| Fourier + trend (5) | `sin1`, `cos1`, `sin2`, `cos2`, `trend` |
| demand lags (5) | `peak_lag_2/3/7/14/364` |
| energy lags (3) | `energy_lag_2/7/14` |
| weather levels (5) | `t_max`, `t_mean`, `t_min`, `cdd_max`, `hours_above_32` |
| weather derived (2) | `d_cdd_max`, `cdd_max_vs_roll7` |

Other registered sets: **A0** (19, weather-free), **A1** (24, weather levels),
**A4** (28, A2 + Diwali features — tested, not adopted).

> **A2 must never be edited in place.** Every committed Phase 5/6/7 number is
> measured against that exact list. Add a new key instead.

### 4.1 Rules that prevent leakage

- Rolling windows use `.shift(1)` — current day excluded
- Lags computed on a **calendar-continuous** index, never positionally (157
  missing days would make a positional shift silently borrow from 2–3 days back)
- Nothing forward-filled; a NaN says "missing" and trees handle it
- 12 `LEAKY_SAMEDAY` columns dropped; `health_check.py` fails if any reappears
- Scaling fit inside `fit()`, reused in `predict()`
- No GBM early stopping on validation data

---

## 5. Evaluation protocol

**Expanding walk-forward**, validation 2023 / 2024 / 2025:

| fold | train | validate |
|---|---|---|
| 1 | ≤2022 | 2023 |
| 2 | ≤2023 | 2024 |
| 3 | ≤2024 | 2025 |

### 5.1 Train on A, score on B

Both regimes **train on the full ERA5 history**; only the scored rows differ.
This makes the A–B difference attributable to target-day weather rather than
to training volume.

> An earlier version restricted training to the scoring mask. Because the
> Regime B frame only spans the forecast archive, that silently cut fold 1
> from ~2,900 training rows to 364 and moved every number in the table.
> `test_phase6_benchmark.py` now asserts `n_train > 2500`.

### 5.2 One shared scoring sample

1,066 dates where A2 features and the target are finite in **both** regimes.
Trees tolerate NaN, linear models do not — without one mask the comparison
measures sample composition, not model skill.

### 5.3 Statistics

- **Primary metric: MAE.** Changing it invalidates comparison with committed gates.
- **Moving-block bootstrap**, 7-day blocks, n=2000, seed 42. Daily errors are
  autocorrelated (lag-1 ACF +0.481), so an iid bootstrap would give intervals
  roughly √7 too narrow.
- **Diebold–Mariano** with Newey–West variance and the HLN small-sample
  correction, reported *alongside* the bootstrap. They agree on 14/14 headline
  comparisons.
- **Adoption bar:** a candidate is adopted only if the 95% CI on the paired MAE
  difference **excludes zero**.

---

## 6. The champion

`quantile_hybrid` — Ridge for the trend, LightGBM **quantile** regression
(τ = 0.5906) on the residual.

```
prediction = Ridge(X) + QuantileGBM(X)
```

**Why a hybrid:** trees cannot extrapolate. A tree predicts leaf averages and
cannot exceed its training range, but the target grew 17,563 → 21,724 MW. Ridge
supplies the trend; the GBM captures non-linear temperature response.

**Why quantile:** MAE is minimised by the conditional **median**, not the mean,
so a squared-error residual head was the wrong estimator for the reported
metric. Worth 22.47 MW.

| | Regime A | Regime B |
|---|---|---|
| champion | 443.62 | **497.90** |

### 6.1 Serving contract

`predict_next_day.py --champion models/champion_config.json` reads three
things from the config rather than inferring them:

| field | value | if wrong |
|---|---|---|
| `target_parameterisation` | `level` | adding an anchor → **95% overshoot** |
| `lag_representation` | `literal` | **+56% MAE**, silently |
| `features` | the 26 | wrong input space |

Serving **fails closed** on: a shifted frame, any NaN feature on the target
row, a non-level parameterisation, or a forecast outside a 7,000 MW sanity
band (calibrated from the measured max 2-day swing of 5,124 MW).

---

## 7. Uncertainty

**Split-conformal** on relative residuals, per season, from a weekly-refit
walk-forward that trains on Regime A and scores on Regime B.

Two calibrations are maintained:

| | window | used for |
|---|---|---|
| operational | trailing 365 days | **serving** — widths should reflect recent residuals |
| 2026-free | ≤2025-12-31 | **reporting 2026 coverage** — avoids circularity |

2026 coverage against the 2026-free widths: **91.1%** (nominal 90%) and
**93.9%** (nominal 95%).

Conformal is used rather than normal theory because residuals are clearly
non-Gaussian (skew +1.18, excess kurtosis +4.48).

---

## 8. Validity machinery

### 8.1 Selection-optimism replay

`run_selection_audit.py` holds 2025 out and re-runs **every** selection step
on 2023–24 only: feature set → 9-model ranking → 50 TPE trials → champion
pick. Then scores 2025 once.

```
optimism = honest_2025_MAE − reported_2025_fold_MAE
```

Negative means the reported figure is conservative. Result: **−5.80 MW**
(Regime B).

**Known limitation:** this replay ranks families on *default* hyperparameters,
so a family with a badly chosen default never reaches tuning. `quantile_hybrid`
at its default τ=0.9 scores 669 MW and gets screened out — which is why the
quantile head was validated separately on inner folds.

### 8.2 The 2026 confirmation

One shot, consumed. Trained ≤2025, scored once on 2026 under Regime B.
`run_final_evaluation.py` refuses to overwrite its artifact without `--force`.

**It is a confirmation, not a blind test** — 2026 was inspected during Track A
feature diagnostics.

### 8.3 Test suites — 147 checks

| suite | checks | guards |
|---|---|---|
| `test_contract.py` | 22 | the Track A↔B contract, source freshness |
| `test_phase5_gateway.py` | 43 | gateway, anti-leakage, reproduction |
| `test_phase6_benchmark.py` | 31 | model zoo, train/score separation, shuffle control |
| `test_serving.py` | 28 | the serving contract |
| `test_phase7_festival.py` | 23 | festival features, A2 immutability |

Every check exists because of a specific failure, not hypothetically.

---

## 9. Operations

| job | schedule | installs | does |
|---|---|---|---|
| `daily.yml` | 07:00 IST | `requirements-data.txt` (195 MB) | log forecast, fetch demand, rebuild, health check |
| `evening.yml` | 20:00 IST | `requirements-model.txt` (355 MB) | log forecast, score matured, predict |

Both reclaim disk before a free-space gate, reuse the venv unless its tier
file's SHA changes, and never leave a pip cache. The runner is self-hosted and
persistent — it filled its disk on 2026-11-12 installing 855 MB of packages
the jobs never imported, losing four days of unrecoverable forecast-log rows.

---

## 10. Reproducing

```bash
./reproduce.sh            # everything: tests, experiments, figures, paper, deck
./reproduce.sh --quick    # skips tuning and calibration
./reproduce.sh --tests    # the 147 checks only, ~36s
```

Deliberately **not** run: `run_final_evaluation.py` (consumed one-shot) and
`build_dataset.py` (needs network).

```bash
git diff --stat results/ figures/ models/
```

Any change means a dependency moved or a seed drifted. Investigate before
publishing — `requirements.txt` pins the versions that produced the committed
numbers.
