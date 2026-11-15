# Phase 6 — Model Benchmark on the Frozen A2 Specification

> ### ⚠️ Superseded by Phase 7
>
> The champion is now **`quantile_hybrid`** (Regime B **497.90 MW**), not
> `ridge_gbm_hybrid` (520.37). The benchmark below is still the valid record
> of how the nine model FAMILIES ranked, and its conclusion holds — every
> pure GBM lost to Ridge. But any champion number here is one generation old.
> See **[`PHASE7_RESULTS.md`](RESULTS.md)**.

**Branch:** `phase-6-modeling` (from `phase-5-regime-ab` @ `1607acd`)
**Question:** Phase 5 fixed *what information* the model gets. Phase 6 asks *which model class best exploits it.*

---

## Headline result

**The Ridge + GBM residual hybrid wins decisively, in both weather regimes.**

Regime A (reanalysis), 1,085 scored validation days, expanding walk-forward 2023/24/25:

| rank | model | MAE | RMSE | MAPE | bias | surge bias | vs Ridge (95% CI) |
|---|---|---|---|---|---|---|---|
| 1 | **ridge_gbm_hybrid** | **465.88** | 634.77 | **2.89%** | −39.1 | **−473.9** | **−105.15 [−120.80, −88.03]** ✅ |
| 2 | ols | 571.03 | 752.58 | 3.54% | −71.6 | −735.2 | −0.00 [−0.00, +0.00] — |
| 3 | ridge *(incumbent)* | 571.03 | 752.58 | 3.54% | −71.6 | −735.2 | reference |
| 4 | lightgbm_linear | 627.02 | 797.73 | 3.83% | −366.7 | −874.4 | +55.98 [+9.27, +101.83] ❌ |
| 5 | xgboost | 630.28 | 808.80 | 3.79% | −367.5 | −1348.3 | +59.24 [+14.26, +107.19] ❌ |
| 6 | lightgbm | 640.13 | 818.63 | 3.85% | −355.8 | −1347.5 | +69.09 [+24.73, +118.68] ❌ |
| 7 | catboost | 651.74 | 840.13 | 3.91% | −394.5 | −1437.9 | +80.71 [+32.69, +134.72] ❌ |
| 8 | seasonal_naive | 997.88 | 1329.74 | 6.19% | −20.7 | −984.2 | +426.85 ❌ |
| 9 | naive_persistence | 1043.80 | 1411.98 | 6.54% | −0.8 | −735.6 | +472.77 ❌ |

**−105 MW (−18.4%) over the incumbent**, CI excluding zero, stable across all three folds (470.5 / 471.0 / 455.9).

---

## Four findings

### 1. Pure gradient boosting loses to a linear model — decisively

All four GBMs are **worse** than Ridge, with CIs excluding zero. The reason is visible in the bias column: GBM bias runs −356 to −395 MW against Ridge's −72, and surge-day bias reaches **−1,438 MW**.

This is the extrapolation failure. A tree predicts a leaf average, so it can never output a value above its training maximum. Tamil Nadu's peak grew ~3.5%/yr and 2026 ran ~10% above 2025, so every validation year contains demand levels the training set never saw. Ridge extrapolates along its fitted trend; trees cannot.

`lightgbm_linear` (linear models in leaves) is the best GBM at 627.02 and roughly halves the surge-day bias (−874 vs −1,348), which supports the diagnosis — but it still loses to plain Ridge.

### 2. The hybrid resolves the trade-off

Ridge handles the trend and extrapolates; the GBM models the non-linear residual. The result beats both parents by a wide margin and cuts surge-day bias from −735 to −474 MW.

**The gain was verified not to be leakage.** A shuffle control that destroys the feature→residual relationship collapses the improvement from −105.4 MW to **−1.0 MW**. The GBM is learning genuine structure. The gain is also broad — on the 2025 fold it improves ordinary days by −136 MW and festival-adjacent days by −115 MW, so it is not merely a festival patch.

### 3. Ridge's L2 penalty is doing nothing

OLS and Ridge are **identical to two decimal places** (571.03 both; paired difference −0.00 MW, CI [−0.00, +0.00]). At α = 0.001 on 26 standardised features, the penalty is numerically irrelevant.

This is a legitimate finding, not a gap: the incumbent is effectively unregularised OLS. It also means **α is an untested hyperparameter**, and Phase 6b should tune it rather than inherit 0.001.

### 4. Forecast-weather degradation is model-dependent

On the 1,066 common dates where both regimes are valid:

| model | Regime A | Regime B | degradation |
|---|---|---|---|
| ridge | 571.61 | 611.95 | **+40.34 (+7.06%)** |
| **ridge_gbm_hybrid** | **465.63** | **526.05** | **+60.42 (+12.98%)** |
| lightgbm_linear | 629.91 | 654.31 | +24.40 (+3.87%) |

The hybrid degrades **more** under forecast weather than Ridge does — it extracts more from weather, so it loses more when weather is imperfect. **This is why finalists are re-run under Regime B rather than only the champion.**

It still wins comfortably: **526.05 vs 611.95 MW, −85.9 MW** in the operational regime.

---

## Validation: Phase 5 reproduced exactly

The Phase 6 harness reproduces the committed Phase 5 Regime A/B result to the decimal:

| | Phase 5 committed | Phase 6 harness |
|---|---|---|
| Common scored dates | 1,066 | **1,066** ✓ |
| Ridge Regime A | 571.61 MW | **571.61 MW** ✓ |
| Ridge Regime B | 611.95 MW | **611.95 MW** ✓ |
| Degradation | +40.34 MW | **+40.34 MW** ✓ |

Asserted in `tests/test_phase6_benchmark.py`. If a future refactor moves these, Phase 6 stops being comparable with Phase 5 and the test fails loudly.

---

## A bug found and fixed during development

The first version of `run_walk_forward` restricted **training** to the common scoring mask. Because the Regime B frame spans only the forecast archive (2021-06 onward, 1,920 rows), this silently cut fold 1's training set from 2,897 rows to **364** and changed every number in the table. Nothing warned.

Training and scoring are now separated explicitly: both regimes train on the identical full ERA5 history, and the mask constrains only which days are *scored*. `tests/test_phase6_benchmark.py` asserts `n_train > 2500` for Regime B fold 1.

---

## Protocol

| | |
|---|---|
| Feature set | `A2_weather_deltas_anomalies` — canonical 26, asserted against `PHASE5_FEATURE_SETS` |
| Horizon | H = 2 (`peak_lag_1` / `energy_lag_1` excluded) |
| Validation | Expanding walk-forward, validate 2023 / 2024 / 2025 |
| Training | Full ERA5 history ≤ fold's `train_end`, identical across regimes |
| Scoring | One shared date mask for every model and regime |
| Primary metric | **MAE** (RMSE / MAPE / bias reported, not selected on) |
| Comparison | Paired moving-block bootstrap, 7-day blocks, n=2000, seed 42 |
| Convention | `d = |e_candidate| − |e_ridge|`; **negative = candidate better** |
| Tuning | **None.** All models use conservative untuned defaults |
| 2026 | **Not touched.** Confirmation set only |

---

## Reproduce

```bash
python tests/test_phase5_gateway.py            # Phase 5 pipeline still valid
python tests/test_phase6_benchmark.py          # 22 Phase 6 acceptance checks
python experiments/run_phase6_benchmark.py                                   # Regime A, all 9
python experiments/run_phase6_benchmark.py --regime-b \
    --models ridge,ridge_gbm_hybrid,lightgbm_linear \
    --out phase6_regime_b_finalists.json                         # finalists, both regimes
```

**Artifacts** (Phase 5 results untouched):

- `results/phase6_benchmark.json` + `_predictions.csv`
- `results/phase6_regime_b_finalists.json` + `_predictions.csv`

---

## Limitations

1. **Untuned.** GBMs may improve with tuning; the benchmark ranks families, not tuned ceilings. Phase 6b tunes finalists only.
2. **No champion declared yet.** `ridge_gbm_hybrid` leads in both regimes, but tuning could change the order.
3. **Regime A is not achievable in production.** Quote the Regime B figure (526.05 MW) for deployable accuracy.
4. **Forecast issue time is unverified** — archived lead-1 forecasts aligned to valid date. No claim of availability at a specific 20:00 IST wall-clock time.
5. **2026 is a confirmation set, not a blind test.** It was inspected during Track A diagnostics.

---

---

# Phase 6b — Tuning and Champion Selection

**Done.** `python experiments/run_phase6_tuning.py --trials 50`

## Procedure (documented per handover §8.5)

| | |
|---|---|
| Candidates | **Top 2 only** by benchmark MAE: `ridge_gbm_hybrid`, `ridge` |
| Sampler | Optuna TPE, seed 42 |
| Budget | **50 trials per model** (roadmap cap) |
| Objective | **Mean validation MAE across folds 2023 / 2024 / 2025** |
| Tuning regime | A (reanalysis) — weather realism settled separately |
| Scored dates | 1,066, shared with Regime B so tuned/untuned stay comparable |
| 2026 | **Never touched.** Not a tuning signal |
| Champion rule | Lowest **Regime B** MAE — the deployable regime |

Search spaces: `ridge` → α over 1e-6…1e3 (log). `ridge_gbm_hybrid` → α, `n_estimators` 100–800, `learning_rate` 0.01–0.15 (log), `num_leaves` 3–31. The residual GBM is bounded deliberately small: it corrects a residual, and a strong learner there would overwrite Ridge's extrapolation with leaf averages, reintroducing the failure the hybrid exists to avoid.

## Results

| model | Regime A | Regime B | degradation | surge bias (B) |
|---|---|---|---|---|
| **ridge_gbm_hybrid** 🏆 | **455.96** | **520.37** | +64.42 (14.13%) | **−549.0** |
| ridge | 571.61 | 611.95 | +40.34 (7.06%) | −823.0 |

**Champion: `ridge_gbm_hybrid`**, selected on Regime B.
Paired vs Ridge in Regime B: **−91.58 MW, 95% CI [−115.99, −68.30], p = 0.0000.**

Best hyperparameters: `alpha=1.08e-05, n_estimators=700, learning_rate=0.0258, num_leaves=10`.

## Two findings from tuning

**1. Ridge's L2 penalty is confirmed inert.** Optuna searched α across **nine orders of magnitude** and chose 1.02e-06 — the smallest end. Of 50 trials, the 41 with α < 0.01 span a total MAE range of **0.01 MW** (571.42–571.43). The benchmark's OLS≡Ridge result was not an artifact of α=0.001; regularisation simply does not help here, because with 26 standardised features on ~3,000 rows the problem is not over-parameterised. Report Ridge honestly as *effectively unregularised least squares*.

**2. Tuning widened the operational gap.** The hybrid's A→B degradation rose from 12.98% (untuned) to **14.13%** — it extracts more from weather after tuning, so it loses more when weather is imperfect. It still wins the operational regime by 91.6 MW. This is the concrete reason the champion is selected on Regime B: a Regime-A-only ranking would have overstated the margin.

## Artifacts

- `models/champion_config.json` — frozen champion, hyperparameters, feature list, validation performance *(Gate G2 hand-off to Track B)*
- `results/champion_regime_ab.json` — tuning trials, both regimes, paired comparison
- `results/champion_regime_ab_predictions.csv` — row-level predictions

---

## Next

Festival features, heat persistence and extreme-day work remain **conditional** on post-tuning diagnostics, per the agreed order. The champion's Regime B surge bias is still **−549 MW**, so extreme-day work has a live diagnostic trigger if the team chooses to pursue it.

Final step: a **single** 2026 confirmation run with the frozen champion — described as a confirmation evaluation, never a blind test.
