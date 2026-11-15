# Independent Audit — TN Next-Day Peak Demand Forecasting

**Date:** 2026-11-15 · **Branch:** `phase-6-modeling` @ `871f6dc` · **Target venue:** ICAIEA 2026

> ### ⚠️ Conflict of interest — read this first
>
> I built a substantial part of what I am auditing: `src/contract.py`'s Phase 5
> gateway, `src/evaluation.py`, `src/models.py`, `run_phase6_benchmark.py`,
> `run_phase6_tuning.py`, `test_contract.py`, `test_phase6_benchmark.py`, the
> Track B operational pipeline, and most of the documentation. **A referee
> cannot neutrally audit their own work.** I have tried to compensate by
> re-deriving every number from raw artifacts rather than reading it from a
> doc, and by weighting criticism toward my own contributions. You should still
> obtain a second opinion on Sections 4 and 7 specifically.

**Method:** all three test suites executed; `data/master.csv` profiled directly;
every headline metric recomputed from saved per-row predictions; leakage checks
run against source via `inspect.getsource`. Claims below are marked
**[VERIFIED]** (I ran it) or **[ASSUMED]** (I did not).

---

## 1. Project Status

| Phase | Status | Evidence |
|---|---|---|
| Data pipeline + automation | **DONE** | `build_dataset.py`, `fetch_weather.py`, `.github/workflows/daily.yml`+`evening.yml`, `health_check.py` |
| EDA | **DONE** | `notebooks/01_exploratory_data_analysis.ipynb` (39 cells) |
| Leakage-safe feature prep | **DONE** | `notebooks/02`, `src/contract.py:load_phase5_data` |
| Evaluation protocol + baselines | **DONE** | `src/evaluation.py`, naive + seasonal-naive + OLS + Ridge in `src/models.py` |
| Weather feature selection (A0–A3) | **DONE** | `run_experiment_a0..a3.py`, `results/experiment_a*_results.json` |
| Forecast-gap experiment (Regime A/B) | **DONE** | `run_phase5_regime_ab.py`, verified below |
| GBDT benchmark + tuning | **DONE** | `run_phase6_benchmark.py`, `run_phase6_tuning.py`, `models/champion_config.json` |
| Daily inference + dashboard | **DONE (barely exercised)** | `predict_next_day.py`, `dashboard/app.py`; `data/predictions_log.csv` holds **1** forecast |
| SHAP / interpretability | **NOT STARTED** | no SHAP anywhere in repo |
| Error analysis by segment | **PARTIAL** | surge/summer in `src/operational/metrics.py`; no festival/regime/post-COVID study |
| Uncertainty quantification | **PARTIAL** | `results/intervals.json` is Track B's stand-in, **not** fitted on the champion |
| Final 2026 confirmation | **NOT STARTED** | no `results/final_evaluation.json` |
| Paper + deck | **NOT STARTED** | no `reports/`, no `presentation/` |

**Overall: ~75% complete for engineering, ~45% for paper-readiness.**

Engineering is genuinely finished. What is missing is almost entirely the
*research-writing* half: interpretability, segment-level error analysis, a
confirmation run, figures that match the final model, and the manuscript.

---

## 2. Pros — what is genuinely strong

**[VERIFIED] Every headline number reproduces exactly from saved predictions.**
I recomputed from `results/*_predictions.csv`, not from the JSONs:

| Claim | Doc | Recomputed | Match |
|---|---|---|---|
| Regime A MAE | 571.61 | **571.61** | ✅ |
| Regime B MAE | 611.95 | **611.95** | ✅ |
| Degradation | +40.34 | **+40.34** | ✅ |
| Regime A RMSE | 753.83 | **753.83** | ✅ |
| Regime A MAPE | 3.532 | **3.532** | ✅ |
| All 9 Phase 6 models | — | **9/9 exact** | ✅ |
| Champion A / B | 455.96 / 520.37 | **455.96 / 520.37** | ✅ |

This is better reproducibility than most submitted papers. It should be stated
in the manuscript.

**[VERIFIED] The operational-gap experiment is the real contribution.** The
three-condition ladder — no weather (695.58) → reanalysis (571.61) → genuine
archived lead-1 forecast (611.95) — under one fixed leak-free harness, with a
bootstrapped CI, is rare in the load-forecasting literature, which overwhelmingly
reports only the middle condition. `data/weather_forecast_archive.csv` carries
29,535 real archived forecast rows.

**[VERIFIED] The leakage machinery is unusually serious.** 22 contract checks +
22 Phase 6 checks + the Phase 5 gateway suite, all passing. `verify_no_leakage()`
resolves each lag column against the raw series rather than trusting its name —
that caught a genuine weekday-misalignment bug. The `swap_weather` coverage guard
and fail-closed masking are regression-tested against a planted missing date.

**[VERIFIED] H=2 is measured, not assumed.** `measure_publish_time.py` →
`results/publish_time.json`: Grid-India's day-D report was never in hand on day D
across 46 observed runs. The consequence — `peak_lag_1` is unusable — is
enforced in code and asserted in tests.

**[VERIFIED] Data quality is high.** 4,905 rows × 64 cols, 0 duplicate dates,
monotonic, 0 target NaN, all 8 checked variables within physical range
(`tn_peak_demand_mw` ∈ [7,359, 21,724] MW; `t_max` ∈ [24.0, 40.9] °C;
`hours_above_32` ∈ [0, 12.7] h). 1,801 missing cells, all lag/rolling warm-up.

---

## 3. Cons / Risks — blunt

1. **No clean holdout exists.** See §4.1. This is the single biggest threat.
2. **`figures/` are stale.** [VERIFIED] All five were committed in `01dfc7e`,
   which *predates* all weather work. They depict the **weather-free A0-era Ridge
   model**, not the champion. `fig5_feature_coefficients.png` shows coefficients
   for a feature set that is no longer used. **Do not put these in the paper.**
3. **No interpretability at all.** For a paper claiming weather drives demand,
   having zero SHAP/PDP is a predictable referee complaint.
4. **The champion has no prediction intervals.** `results/intervals.json` was
   calibrated on Track B's stand-in GBM, not `ridge_gbm_hybrid`.
5. **Live validation is ~nil.** `data/predictions_log.csv` has one row. The
   deployment claim rests on backtests.
6. **Dependencies effectively unpinned.** [VERIFIED] Only `holidays==0.108` is
   pinned; `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `optuna`, `numpy`,
   `pandas` float. The champion is reconstructed by re-fitting, so a LightGBM
   minor-version change can silently move the headline number.
7. **No data licence or citation block.** [VERIFIED] `README.md` contains no
   licence/DOI/citation text. Grid-India PSP terms and the Zenodo record both
   need stating before submission.
8. **`champion_config.json` records no library versions.** [VERIFIED].
9. **Single-region, single-target study.** Limits generality claims.
10. **Pongal/holiday segment sample sizes are tiny** (Pongal n=8). Any claim
    there is anecdotal.

---

## 4. Correctness and Leakage Audit

### 4.1 🔴 **CRITICAL — validation-set reuse (selection optimism)**

**[VERIFIED]** The *same* 2023–2025 validation set has driven **four sequential
selection decisions**:

| # | Decision | Artifact |
|---|---|---|
| 1 | A1 → A2 weather feature gate | `experiment_a2_results.json` (`3-fold walk-forward 2023/24/25`) |
| 2 | A2 → A3.4 gate | `phase5_decision_gate_results.json` |
| 3 | 9-model ranking | `phase6_benchmark.json` |
| 4 | 100 Optuna trials (2 models × 50) | `champion_regime_ab.json` |

**And 2026 is contaminated too.** [VERIFIED] `experiment_a1/a2/a3_results.json`
each contain a `diagnostic_2026_results` block — 2026 was inspected *during*
feature selection (e.g. A2 vs A1 on 313 2026 days: 604.59 vs 622.11 MW).

**Consequence: there is no untouched data on which to report generalisation.**
Every number in the project is a quantity that model/feature/hyperparameter
selection was optimised against. The champion's 520.37 MW is therefore an
**optimistically biased** estimate of operational error, and the project
currently has no way to say by how much.

This is the one finding that could sink the paper at review. It is **fixable**
— see §8 items 1–2.

### 4.2 ✅ Feature-level leakage — clean

| Check | Result |
|---|---|
| Scaling fit on train only | **[VERIFIED]** `TrackARidge.fit` computes `mu_X_`/`sig_X_` inside `fit`; `predict` reuses them |
| Rolling windows exclude current day | **[VERIFIED]** `add_lags` uses `y.shift(1).rolling(...)`; `weather_derivatives` uses `.shift(1).rolling(7, ...)` |
| GBM early stopping on validation | **[VERIFIED] none** — no `early_stopping`/`eval_set`/`callbacks` in `src/models.py`; fixed `n_estimators` |
| Same-day target info | **[VERIFIED]** 12 `LEAKY_SAMEDAY` cols dropped; `health_check.py` fails if any reappears |
| Holiday features use future knowledge | **No.** `tn_holidays.py` is a calendar known years ahead — legitimate |
| `trend` feature | Days since series start — deterministic, no leakage |

### 4.3 ✅ Actual vs forecast weather — handled honestly, better than most

Models **train on ERA5** and the champion is **selected on Regime B** (archived
lead-1 forecast). `swap_weather` replaces only target-day weather; the G0 identity
test confirms swapping reanalysis back reproduces Regime A to **0.000e+00 MW**.
Deltas are `forecast(T) − observed(T−1)`, which is the correct information set.

**One honest caveat already documented:** forecast *issue hour* is not recorded,
so the claim is "archived lead-1 aligned to valid date", never "available at
20:00 IST". That framing is correct and should be preserved verbatim.

**[VERIFIED] residual optimism:** the four T−1-dependent features use complete
ERA5 for day T−1 in *both* regimes, because the archive has no lead-0. Both
regimes are equally optimistic there, so the A−B difference is unaffected, but
the absolute Regime B number is slightly optimistic.

### 4.4 ✅ Chronological splits and walk-forward — correct

**[VERIFIED]** Expanding folds (train ≤2022 → val 2023; ≤2023 → 2024; ≤2024 →
2025). No random splitting anywhere. Training and scoring are explicitly
separated in `run_phase6_benchmark.py:117-148`; `test_phase6_benchmark.py`
asserts `n_train > 2500` for Regime B — this check exists because an earlier
version silently trained fold 1 on 364 rows instead of 2,897.

**[VERIFIED] shared evaluation sample:** one date mask across all models and both
regimes (1,085 Regime-A-only; 1,066 when Regime B coverage is required), so no
model is scored on a different sample. The 19 excluded dates (1 in 2023, 18 in
Jan 2024) are documented, not imputed.

### 4.5 ⚠️ Baselines fair; statistics sound but one test missing

Naive persistence correctly echoes `peak_lag_2` (T−2, honouring H=2) — **not**
the more flattering `peak_lag_1`. Seasonal naive uses `peak_lag_7`.

Paired **moving-block bootstrap** (7-day blocks, n=2000, seed 42) is used
throughout — correct for autocorrelated daily errors, where an iid bootstrap
would give intervals ~√7 too narrow.

**Missing:** no **Diebold–Mariano** test. Referees in forecasting venues expect
DM specifically. The bootstrap is arguably stronger, but DM is cheap to add and
is the field's lingua franca.

---

## 5. What Is Missing

**Analyses (none of these exist):**
- SHAP / permutation importance / PDP for the champion
- Error analysis by: regime change (2015 Chennai floods, 2020 COVID, post-COVID recovery), heat waves, long weekends
- Diebold–Mariano tests
- Learning curves (does more history help, or does the 2020 regime hurt?)
- Ablation of the demand-lag block (how much is weather *really* worth vs lags?)
- Residual autocorrelation on the **champion** (the +0.41 lag-1 ACF was measured on the old weather-free Ridge)

**Artifacts:**
- Conformal/quantile intervals fitted on the champion
- `results/final_evaluation.json` (confirmation run)
- Figures regenerated from the champion
- `reports/`, `presentation/`

**Reproducibility:**
- Pinned versions; env capture in `champion_config.json`
- Data licence/citation block; Zenodo DOI; Grid-India PSP terms
- One-command reproduction script

---

## 6. How to Improve Performance (classical ML only)

Grounded in the **champion's actual Regime B residuals** [VERIFIED, n=1,066]:

| segment | n | MAE | bias |
|---|---|---|---|
| overall | 1,066 | 520.4 | −69.6 |
| working day | 702 | 507.4 | −130.8 |
| weekend | 308 | 469.4 | +13.6 |
| summer Mar–Jun | 366 | 544.8 | −2.5 |
| **holiday** | **70** | **899.9** | **+128.7** |
| **festival-adjacent** | **196** | **725.4** | +14.6 |
| **Pongal window** | **8** | **931.4** | **−470.7** |
| **top-5% surge** | **56** | **592.5** | **−549.0** |

**Worst 6 days are 3 consecutive Diwalis + May Day** — over-forecast by
2,574–4,171 MW:
`2024-10-31 (+4171)`, `2025-10-20 (+3492)`, `2024-11-01 (+3338)`,
`2025-10-21 (+2827)`, `2023-11-13 (+2675)`, `2025-05-01 (+2574)`.

| # | Action | Impact | Effort | File |
|---|---|---|---|---|
| 1 | **Diwali window feature** (±2 days), like `pongal_window`. Holiday MAE is 77% above working-day and the three worst days are Diwali. `tn_holidays.csv` already resolves Diwali to 2030 | **High** | 2 h | `build_dataset.py:calendar_features` |
| 2 | **Generic multi-day festival window** + `days_to_nearest_holiday` (signed). Generalises #1 and fixes `day_before/after` being too narrow | **High** | 3 h | same |
| 3 | **Surge-day bias correction**: quantile regression (τ≈0.9) as a second head, or asymmetric loss. Bias −549 MW on top-5% is the operational risk | **High** | 4 h | `src/models.py` |
| 4 | **Recency weighting** (exponential half-life 2–4 yr) in `TrackARidge.fit` — already supports `sample_weight`. 2026 runs ~10% above 2025 | **Med** | 2 h | `src/models.py`, `run_phase6_tuning.py` |
| 5 | **Long-weekend / bridge-day flags** (holiday adjacent to weekend) | **Med** | 2 h | `build_dataset.py` |
| 6 | **Stack Ridge + LightGBM + XGBoost residuals** rather than LightGBM alone | **Med** | 3 h | `src/models.py` |
| 7 | **Heat-persistence**: consecutive days > 32 °C, 3-day rolling `cdd_max`. Hottest-decile bias is −99 MW | **Med** | 2 h | `build_dataset.py` |
| 8 | **Tune the residual GBM's `min_child_samples`/`reg_lambda`** — not in the current search space | **Low–Med** | 1 h | `run_phase6_tuning.py:SPACES` |
| 9 | **Regime dummy for 2020–21 COVID**, or drop/downweight | **Low–Med** | 2 h | `build_dataset.py` |
| 10 | **Target transform**: model `log(y)` or `y − peak_lag_7` as an ablation | **Low** | 2 h | `src/models.py` |

Avoid: more model families (the benchmark already shows the family question is
settled), and deep learning (out of scope, and ~4,900 rows won't support it).

---

## 7. Paper Readiness

**Would it survive review today? No — but the gap is writing and two experiments, not science.**

| Dimension | State | Gap |
|---|---|---|
| Novelty | **Strong** | The 3-condition weather ladder with a real forecast archive is genuinely uncommon |
| Methodology | **Strong** | Leakage controls exceed the norm |
| **Validity of reported numbers** | 🔴 **Weak** | §4.1 — no clean holdout |
| Related work | **Absent** | No literature review exists |
| Results presentation | **Weak** | Figures are stale/wrong model |
| Interpretability | **Absent** | No SHAP |
| Reproducibility | **Good→Excellent** | Needs pins + licence |

**Strongest honest contribution claim:**

> *We quantify the operational cost of weather-forecast uncertainty in next-day
> peak electricity demand forecasting for Tamil Nadu, using a 5.4-year archive of
> genuine lead-1 numerical weather forecasts under a leakage-controlled,
> horizon-aware (H=2) walk-forward protocol. Weather reduces MAE by 17.8% under
> **⚠️ CORRECTION (2026-11-15).** The sentence below is wrong and is
> preserved only as the record of what this document originally said.
> "7.1%" is 40.34/571.61 — the penalty as a fraction of the **Regime A
> MAE**, not of the gain. Against the actual gain of 124.66 MW the
> penalty is **32.4%**. See `reports/PAPER.md` §4.1, which now reports
> both denominators explicitly.

> reanalysis, but 7.1% of that gain (95% CI [25.9, 54.2] MW) is lost when only
> operationally available forecasts are used — a penalty that is invisible to
> the reanalysis-based evaluations standard in the literature.*

Do **not** claim "we beat SOTA." Claim the **measurement**, which is defensible
and genuinely novel.

**Also disclose explicitly:** H=2 (so `peak_lag_1` is unavailable), forecast
issue hour unverified, 2026 inspected during development, single region.

---

## 8. Prioritized Action Plan

### Critical blockers (do before writing)

| # | Action | Hours | File |
|---|---|---|---|
| 1 | **Nested/forward validation to quantify selection optimism.** Re-select features+model+hyperparameters using only ≤2024, then evaluate once on 2025. The gap vs the reported 2025 fold *is* the optimism estimate | 6 | new `run_selection_audit.py` |
| 2 | **Freeze and run the single 2026 confirmation**, labelled a *confirmation set*, never a blind test | 2 | new `run_final_evaluation.py` |
| 3 | **Regenerate all 5 figures from the champion.** Current ones depict the weather-free model | 3 | new `make_figures.py` |
| 4 | **Pin all dependencies**; record versions in `champion_config.json` | 1 | `requirements.txt`, `run_phase6_tuning.py` |
| 5 | **Add licence + citation block** (Grid-India PSP terms, Zenodo DOI, Open-Meteo attribution) | 1 | `README.md` |

### High impact

| # | Action | Hours | File |
|---|---|---|---|
| 6 | Diwali + generic festival window; re-run benchmark | 4 | `build_dataset.py` |
| 7 | SHAP for the champion (linear coefficients + TreeSHAP on the residual GBM) | 4 | new `run_interpretability.py` |
| 8 | Conformal intervals fitted on the champion | 3 | `calibrate_intervals.py` |
| 9 | Surge-day bias work (quantile head) | 4 | `src/models.py` |
| 10 | Diebold–Mariano alongside the bootstrap | 2 | `src/evaluation.py` |
| 11 | Segment error analysis incl. COVID/floods regimes | 3 | new `run_error_analysis.py` |

### Polish

| # | Action | Hours |
|---|---|---|
| 12 | Related-work section (15+ STLF papers) | 8 |
| 13 | One-command reproduction script | 2 |
| 14 | Run the evening cron ≥2 weeks for live evidence | 0 (wall-clock) |
| 15 | Deck (12 slides) | 6 |

---

### 🎯 If I only had 1 week, do these 5 things

1. **Quantify selection optimism** (§8.1) — 6 h. Without it, every number is contestable.
2. **Run the 2026 confirmation once** (§8.2) — 2 h.
3. **Regenerate figures from the champion** (§8.3) — 3 h. Current figures show the wrong model and would be caught.
4. **Add the Diwali/festival window and re-run** (§8.6) — 4 h. Biggest single accuracy win; 3 of the worst 6 days are Diwali.
5. **Pin dependencies + licence/citation** (§8.4–5) — 2 h. Cheap, and reviewers check.

**~17 hours.** That converts a strong engineering project into a defensible submission.

---

## Appendix — Verification Log

| Check | Result |
|---|---|
| `test_contract.py` | **PASS** 22/22, exit 0 |
| `test_phase5_gateway.py` | **PASS** all, exit 0 |
| `test_phase6_benchmark.py` | **PASS** 22/22, exit 0 |
| `master.csv` shape | 4,905 × 64 |
| Duplicate dates / target NaN | 0 / 0 |
| Missing calendar days / cells | 157 / 1,801 |
| Unit range violations (8 vars) | 0 |
| Headline metrics recomputed | **14/14 exact** |
| Scaling fit on train only | confirmed via `inspect.getsource` |
| GBM early stopping | none present |
| Rolling `shift(1)` | confirmed both modules |
| 2026 in selection artifacts | **present in a1/a2/a3** |

**Not verified / assumed:** notebook cell *outputs* (code read, not re-executed);
upstream Grid-India and Open-Meteo correctness; that `figures/` were generated by
the notebooks in the same commit (inferred from commit contents).
