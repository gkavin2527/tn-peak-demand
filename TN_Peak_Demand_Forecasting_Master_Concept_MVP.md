# Tamil Nadu Peak Demand Forecasting — Master Concept MVP

## 1. Project Title

**Next-Day Peak Electricity Demand Forecasting for Tamil Nadu Using Machine Learning**

---

## 2. Problem Statement

Electricity demand in Tamil Nadu exhibits substantial non-linear variability driven by meteorological extremes, industrial work cycles, calendar anomalies, agricultural pumping schedules, and long-term economic expansion. While predicting total daily energy consumption (GWh) provides volumetric energy balancing, it fails to address the primary operational vulnerability of grid operators: **managing the daily peak demand (MW)**.

Peak demand dictates generation capacity commitment, spinning reserves, interstate grid drawal limits, and transmission line congestion management. Underestimating peak demand risks localized load-shedding, grid frequency drops, or exorbitant purchases from the day-ahead and real-time power exchange markets.

This project develops an operational, machine-learning-based forecasting framework to predict **Tamil Nadu's next-day peak electricity demand (MW)** using historical demand telemetry, multi-city load-weighted weather indicators, calendar dynamics, and Southern Region grid context, strictly relying on information available prior to the target operating day ($T+1$).

---

## 3. Main Objective

To develop, validate, and operationalize a machine learning model that forecasts **Tamil Nadu's peak electricity demand for the next day (MW)** using data available before 00:00 IST of the target day.

### Specific Objectives
1. **Production Pipeline & Automated Ingestion:** Build and maintain an automated, zero-leakage data engineering pipeline that continuously ingests, cleans, and joins peak demand, energy consumption, multi-city weather observations, and day-ahead weather forecasts.
2. **Multi-City Meteorological Aggregation:** Aggregate historical reanalysis and operational forecast weather across Tamil Nadu's 5 key load centers (Chennai, Coimbatore, Madurai, Trichy, Salem) using load-share weights into unified state-level cooling and humidity stress features.
3. **Empirical Pattern & Driver Identification:** Quantify the predictive power of heat stress duration (`hours_above_32`), Cooling Degree Days ($CDD$), Temperature-Humidity Index ($THI$), thermal inertia lags, and industrial festival anomalies (e.g., Pongal holidays).
4. **Leak-Free Time-Series Modeling:** Build, optimize, and benchmark competitive machine learning models (Regularized Linear Baselines, Random Forest, LightGBM, XGBoost, CatBoost) under three treatments of structural demand growth (raw-MW target with recency weighting, Ridge + tree residual hybrid, linear-leaf trees) plus a predict-the-change target. Deep sequence models are out of scope (see Section 19).
5. **Operational vs. Reanalysis Evaluation:** **COMPLETE (Phase 5).** Both regimes measured end-to-end with a stand-in model, weekly refit, leads 1–3, at both $H=2$ and $H=1$:
   - *Retrospective* using ERA5 reanalysis actuals (Regime A).
   - *Operational* using archived Open-Meteo Previous Runs forecasts, 2021-06 onward, 5.4 years (Regime B), validated against our live log to 0.36 °C MAE on `t_max`.
   Results in `results/operational_gap.json` and §17. Re-run at Gate G2 with Track A's champion.
6. **Interpretability & Error Diagnostics:** Apply feature attribution methods (e.g., TreeSHAP) to explain model decisions during extreme heatwaves, sudden weather shifts, and seasonal transition periods.

---

## 4. Project Goal & Core Research Hypotheses

The overarching goal is to deliver an **end-to-end, production-grade, and research-oriented next-day peak demand forecasting system for Tamil Nadu**.

The project investigates five core empirical questions:
1. **Predictive Accuracy:** Can machine learning algorithms reliably forecast Tamil Nadu's next-day peak demand within operational industry standards ($\text{MAPE} < 2.5\%-3.5\%$)? *An informal stand-in model (not the final champion) already reaches 2.38% MAPE on a differenced target — see §17.*
2. **Weather vs. Autoregression:** How much incremental predictive gain does multi-city weather data provide over demand-history-only baselines across different seasons?
3. **Thermal Inertia & Heat Accumulation:** Does multi-day heat accumulation (lagged CDD and consecutive hours above 32°C) significantly improve peak prediction during summer heatwaves (March–June)?
4. **Operational Forecast Degradation:** What is the performance gap between models trained on retrospective weather actuals versus models evaluated on operational day-ahead numerical weather forecasts? **Answered: +8.6% MAE** (483.5 → 525.2 MW) at 1-day lead, with a stand-in model — see §17. Notably, *data latency* ($H$) costs about twice as much as forecast-weather uncertainty, and the largest individual errors are driven by neither: they are festival demand collapses (§13.5).
5. **Generalizability & Structural Breaks:** How resilient are non-linear tree models and linear/hybrid models to regime shifts such as the Chennai floods (2015), COVID-19 industrial shutdowns (2020), and rapid solar penetration in the Southern Region (post-2017)?

---

## 5. Why Peak Demand? (Operational Context)

| Metric | Dimension | Grid Operational Significance | Market & Planning Impact |
|---|---|---|---|
| **Energy Consumption** | GWh (Volume over 24h) | Fuel procurement, thermal coal scheduling, dam reservoir dispatch | Medium-term bilateral power purchasing contracts |
| **Peak Demand** | MW (Instantaneous Maximum) | **Transmission line thermal limits, substation capacity, spinning reserves, frequency stability (49.90–50.05 Hz)** | **Spot market purchases (IEX / DAM / RTM), avoiding overdrawal penalties under Grid Code regulations** |

Peak demand in Tamil Nadu routinely exceeds 20,000 MW (reaching all-time records of 21,724 MW). Failing to forecast the peak by even 5% (1,000 MW) requires sudden dispatches of expensive gas peaker plants or emergency power procurement at peak exchange tariffs. Predicting next-day peak demand in MW provides direct operational utility to state utilities (TANGEDCO/TANTRANSPO) and system operators (SLDC).

---

## 6. Project Scope

### Included
- **Target Geography:** State of Tamil Nadu, India.
- **Forecasting Horizon:** Day-ahead / Next-day ($T+1$) peak electricity demand in MW. **$H$ is now measured: $H = 2$.** Grid-India publishes day $D$'s report on $D+1$, so an evening-of-$D$ forecast for $D+1$ holds demand only through $D-1$. The honest name for the result is *"day-ahead forecast with one-day demand-data latency"*. See §7 and roadmap v4 Section B.
- **Dataset Span:** ~14 years of daily telemetry (January 2013 to the present), totaling **4,904 days, 157 of them missing** (see §10 — this is a correction from the "4,878+ days, 163 missing" figure in earlier revisions of this document).
- **Automated Data Pipeline:** Daily 07:00 IST GitHub Actions pipeline with zero-leakage data ingestion, feature generation, and automated validation.
- **Operational Forecast Logging:** Daily automated logging of day-ahead numerical weather forecasts across all load centers, plus a 5.4-year archived forecast history (`weather_forecast_archive.csv`, new since the last revision — see §17).
- **Feature Engineering:** Calendar harmonics, holiday shoulder windows, autoregressive demand lags, rolling volatility statistics, and non-linear weather indices.
- **Time-Series Aware Validation:** Expanding-window walk-forward validation preserving temporal causality.
- **Model Explainability:** SHAP feature attribution and residual analysis during extreme load days.

### Outside the Scope
- Real-time intra-day 15-minute dispatch and AGC (Automatic Generation Control).
- Substation-level spatial power flow and transmission line contingency analysis ($N-1$).
- Electricity spot price forecasting on power exchanges.
- Individual consumer / Smart meter load profiling.

---

## 7. Target Variable & Core Context Columns

### Primary Target
- **`tn_peak_demand_mw`**: Tamil Nadu daily maximum peak electricity demand met in Megawatts (MW).
  - **Distribution:** Min: **7,359 MW** | Mean: **14,329 MW** | Max: **21,724 MW** | Std: **~2,470 MW**
  - **Temporal Growth:** Grown structurally from ~10,911 MW (2013 average) to over 18,987 MW (2026 year-to-date average) — **running about 10% above the same period in 2025**, well above the ~3.5%/yr historical CAGR. This acceleration is the reason D1 (trend-handling) matters in practice, not just in theory: a tree model fit to the raw level under-forecasts the 2026 holdout by ~420 MW on 75% of days, because it cannot output a value above the largest it saw in training. See §17.

### Supporting Context Columns — **correction: these are not in `master.csv`**

Earlier revisions of this document listed `tn_shortage_at_peak_mw` and `tn_energy_gwh` here as supporting context columns present in the modelling table, with a quoted correlation of $r = +0.972$ for energy. **That was a leakage bug and both columns have been removed from `master.csv`.** Both are Grid-India PSP values published the *morning after* the day they describe — they are not knowable before the target day, so including them as same-day features let the model see tomorrow's answer sheet. `tn_energy_gwh`'s 0.971 correlation with the target is exactly why it mattered: a backtest that includes it reads about 24% better than one that doesn't (452 MW MAE vs 591 MW, on the pre-fix feature set), and that number cannot be reproduced in production because the PSP report for day T+1 does not exist yet when you are trying to forecast day T+1.

What remains, correctly lagged:
- **`energy_lag_1`, `energy_lag_7`, …, `energy_roll_mean_7`, …**: yesterday's and earlier energy met, and rolling statistics of it. These are legitimate — the information is same-day energy's *predictive* content without the leak, and `energy_lag_1` still correlates **r = 0.910** with the target.
- `tn_shortage_at_peak_mw` is dropped entirely, not just lagged. Besides being same-day, the raw PSP column only exists from 2024 onward; 79% of history had no value and was previously being filled with 0.0, which asserted zero deficit through the 2013–2015 load-shedding years — false, not merely missing. `sr_peak_shortage_mw_lag1` (Southern Region, lagged) is the shortage signal that survives, and remains usable (r ≈ −0.48).
- The raw, same-day values for both are still available for analysis in `data/peak_demand.csv` and `data/demand_daily.csv` — just not in the leak-checked modelling table.

### Forecast Formulation (Zero Leakage)
To forecast peak demand on target day $T+1$:
$$\widehat{Y}_{T+1} = f\left(\mathbf{X}_{\text{weather}, T+1}^{\text{forecast}}, \mathbf{X}_{\text{calendar}, T+1}, \mathbf{X}_{\text{demand}, \le T}^{\text{observed}}, \mathbf{X}_{\text{weather}, \le T}^{\text{observed}}, \mathbf{X}_{\text{grid}, \le T}^{\text{observed}}\right)$$
*Information available strictly up to day $T$ is used to forecast day $T+1$.*

> **Horizon note (v2, still current):** "Observed up to day $T$" holds only if day $T$'s Grid-India PSP report is published before the forecast is issued. A day's report cannot be final until that day has ended, so for an evening issue time the last available demand day is probably $T-1$. The modelling table is therefore built around a configurable data-latency horizon $H$. **$H$ has not yet been measured** — see roadmap v3, Phase 0.5.2. Until it is, `master.csv` implicitly assumes whatever latency the 07:00 daily job happens to have, which has not been formally characterized as $H=1$ or $H=2$.

---

## 8. Dataset Architecture & Feature Set

The production modelling dataset (**`data/master.csv`**) contains **64 columns** (62 features plus `date` and the target) engineered to guarantee zero temporal data leakage — **corrected from the "54 strictly aligned columns" in earlier revisions; the count audit (roadmap v3, §0.5.1) is resolved.**

> **Count audit — resolved (was "pending" in earlier revisions).** Row and column counts no longer differ across documents: **4,904 rows, 64 columns**, 2013-01-01 to the latest pulled day. `peak_demand.csv` is 3,852 Zenodo rows + 1,052 Grid-India PSP rows (not 1,033 — that was a stale count). See roadmap v3's Count Audit box for the full reconciliation, including why "missing days" (157) and "missing values" (2,419 NaN cells, all legitimate lag warm-up) are different numbers that earlier revisions conflated.

```
                        ┌────────────────────────────────────────────────┐
                        │              RAW DATA SOURCES                  │
                        └────────────────────────────────────────────────┘
                               │                │               │
        Grid-India PSP Daily   │   CICERO /     │  Open-Meteo   │  Python Holidays
        Reports (API & CDN)    │   POSOCO Slice │  Reanalysis,  │  (patched — see
        [PDF & XLS Scrapers]   │   [Daily GWh]  │  Forecast &   │  §10, tn_holidays.py)
                               │                │  Previous Runs│
                               │                │  (archive)    │
                               ▼                ▼               ▼
                        ┌────────────────────────────────────────────────┐
                        │      INGESTION & CACHED DAILY SLICES           │
                        ├────────────────────────────────────────────────┤
                        │ • peak_demand.csv (4,904 rows: 3,852 + 1,052) │
                        │ • demand_daily.csv (4,947 rows)               │
                        │ • weather_daily.csv (25,310 city-days,        │
                        │   provenance-tagged: archive / update)        │
                        │ • weather_forecast_log.csv (98 issue dates,   │
                        │   1,470 rows, since 2026-08-04)               │
                        │ • weather_forecast_archive.csv (NEW: 29,535   │
                        │   city-day-leads, 2021-06 → latest, leads 1-3)│
                        │ • tn_holidays.csv (NEW: 440 rows, 2013-2030)  │
                        └────────────────────────────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │         FEATURE ENGINEERING PIPELINE           │
                        │             (build_dataset.py)                 │
                        ├────────────────────────────────────────────────┤
                        │ 1. Load-share weighted 5-city weather blend    │
                        │ 2. Continuous calendar reindexing (no jumps,   │
                        │    no forward-fill across gaps)                │
                        │ 3. Autoregressive lags, rolling stats,         │
                        │    day-over-day weather deltas (NEW)           │
                        │ 4. Drop ALL same-day columns (config           │
                        │    .LEAKY_SAMEDAY — 12 cols, not just          │
                        │    "regional"; tn_energy_gwh and               │
                        │    tn_shortage_at_peak_mw included — NEW fix)  │
                        └────────────────────────────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │       PRODUCTION MODELLING DATASET             │
                        │             (data/master.csv)                  │
                        │    4,904 Rows × 64 Columns (2013 - latest)    │
                        └────────────────────────────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │        AUTOMATED DATA QUALITY SUITE            │
                        │             (health_check.py)                  │
                        │ Freshness, plausibility, bounds, correlation,  │
                        │ same-day leak guard (NEW), holiday-calendar    │
                        │ drift (NEW), weather provenance (NEW)          │
                        └────────────────────────────────────────────────┘
```

### Complete Feature Taxonomy (64 Columns) — corrected

| Category | Columns | Mathematical / Operational Description | Empirical Target Correlation ($r$), current feature set |
|---|---|---|---|
| **Target** | `tn_peak_demand_mw` | Primary forecasting target (MW). **No same-day context columns ship alongside it** — see §7 correction. | Target: $1.000$ |
| **Meteorological (State-Weighted)** | `t_mean`, `t_max`, `t_min`<br>`rh_mean`, `app_t_mean`, `app_t_max`<br>`hours_above_32`<br>`cdd`, `cdd_max`, `cdd_apparent`, `thi`<br>**NEW:** `d_t_mean`, `d_t_max`, `d_rh_mean`, `d_app_t_max`, `d_cdd`, `d_cdd_max`, `d_thi` (day-over-day change)<br>**NEW:** `t_max_vs_roll7`, `cdd_max_vs_roll7`, `thi_vs_roll7` (anomaly vs trailing week) | Weighted across 5 cities: Chennai (0.36), Coimbatore (0.22), Madurai (0.16), Trichy (0.14), Salem (0.12). $CDD = \max(0, T_{\text{mean}} - 25)$. $THI = T_{\text{mean}} - (0.55 - 0.0055 \cdot RH) \cdot (T_{\text{mean}} - 14.5)$. **Finding (new):** the level columns are near-redundant once autoregressive lags are present (dropping all of them improves backtest MAE); the delta/anomaly columns were added because they carry signal the levels don't (corr of $\Delta$peak with $\Delta t_{\text{mean}}$ is +0.15, versus ~0 for the level once lags are controlled for). | $t_{\text{max}}: +0.40$ (level, redundant with lags in practice — see `config.WEATHER_LEVEL_COLS`) |
| **Calendar & Holidays** | `dow`, `is_weekend`<br>**NEW:** `dow_sin`, `dow_cos` (day-of-week cycle)<br>`is_holiday`, `holiday_name`<br>`day_before_holiday`, `day_after_holiday`<br>`pongal_window`<br>`sin1`, `cos1`, `sin2`, `cos2` | Day of week (0=Mon), weekend flag, **plus a day-of-week cycle (new)** because Saturday (−364 MW vs a working day) and Sunday (−2,207 MW) are not the same day type and `is_weekend` alone averages them into one that matches neither. Indian/TN statutory holidays — **now resolved by `tn_holidays.py`, not the `holidays` package directly; see §10.** Pongal window (Jan 13–17): multi-day major industrial shutdown. Fourier annual seasonal harmonics ($k=1, 2$). | Working Day: **14,628 MW**<br>Weekend: **13,733 MW**<br>Holiday: **13,608 MW**<br>Pongal: **12,316 MW** |
| **Long-Term Trend** | `trend` | Days elapsed since 2013-01-01. **Finding (new): actively harmful to tree models in the current growth regime** — it is monotone and unbounded, so every future row falls in the top training bin, contributing a near-constant while crowding out splits that generalize. Dropping it improved a stand-in model's 2026 holdout bias from −420 to −282 MW (before differencing fixed it further — see §17). Track A's locked decision (D1) to exclude `trend` from the main model is independently confirmed by this. | $+0.80$ (correlated with the target, but not useful to a tree the way the lags are) |
| **Peak Autoregressive Lags** | `peak_lag_1`, `peak_lag_2`, `peak_lag_3`, `peak_lag_7`, `peak_lag_14`, `peak_lag_364` | Historical peak demand, on a calendar-continuous reindexed series. **No forward-fill across gaps (changed):** a lag that crosses one of the 157 missing days is `NaN`, not a stale value from 2–3 days earlier under the wrong label, which is what an earlier pipeline version did (`ffill(limit=2)`). | $\text{lag}_1: +0.93$<br>$\text{lag}_7: +0.91$ |
| **Peak Rolling Statistics** | `peak_roll_mean_7`, `peak_roll_std_7`, `peak_roll_mean_30`, `peak_roll_std_30` | 7-day and 30-day backward rolling averages and volatility, `shift(1)`'d to avoid leakage. **Std is left `NaN`, not `0.0`, when the window is short (changed)** — a zero would have claimed the series was perfectly flat rather than simply unknown. | $\text{roll\_mean}_7: +0.93$ |
| **Demand Momentum** | `peak_lag_1_diff`, `peak_lag_1_vs_week` | Day-over-day acceleration and week-over-week deviation. | small, as before |
| **Energy Consumption Lags** | `energy_lag_1`, `energy_lag_2`, `energy_lag_7`, `energy_lag_14`, `energy_roll_mean_7`, `energy_roll_std_7`, `energy_roll_mean_30`, `energy_roll_std_30` | Lags and rolling trends of **yesterday's and earlier** energy met (GWh) — **same-day `tn_energy_gwh` itself is not a column; see §7 correction.** | $\text{energy\_lag}_1: +0.91$ |
| **Thermal Inertia Lags** | `cdd_lag1`, `cdd_max_lag1`, `thi_lag1` | Yesterday's heat exposure. | similar to before |
| **Regional Grid Context Lags** | `sr_max_demand_mw_lag1`, `sr_wind_gwh_lag1`, `sr_solar_gwh_lag1`, `sr_peak_shortage_mw_lag1` | Southern Region grid conditions, lagged. **All same-day regional columns are purged, not a subset** — see §7 and `config.LEAKY_SAMEDAY` (12 columns total, not just the ones previously called "leaky regional metrics"). | $\text{sr\_max\_lag1}: +0.86$ |

---

## 9. Production Data Pipeline & Quality Infrastructure

The data engineering pipeline is fully implemented, verified, and automated. It executes daily at **07:00 IST** via GitHub Actions on a dedicated runner.

```
 Daily 07:00 IST Schedule
           │
           ▼
 ┌────────────────────┐
 │ fetch_weather.py   │  Logs day-ahead forecast across 5 cities (now to lead 3)
 │ (forecast mode)    │  ──> Appends to data/weather_forecast_log.csv
 └────────────────────┘
           │
           ▼
 ┌────────────────────┐
 │ fetch_demand.py    │  Downloads & slices latest Grid-India demand CSV
 └────────────────────┘  ──> Updates data/demand_daily.csv
           │
           ▼
 ┌────────────────────┐
 │update_peak_demand  │  Fetches yesterday's Grid-India Daily PSP report via API
 └────────────────────┘  ──> Upserts data/peak_demand.csv (XLS/PDF parsing)
           │
           ▼
 ┌────────────────────┐
 │ fetch_weather.py   │  Pulls past 7 days of observed weather (provisional —
 │ (update mode)      │  tagged source="update" until ERA5 lands)
 └────────────────────┘  ──> Appends actuals to data/weather_daily.csv
           │
           ▼
 ┌────────────────────┐
 │ fetch_weather.py   │  NEW STEP: replaces provisional rows with ERA5
 │ (reconcile mode)   │  reanalysis once it lands (~5-day delay)
 └────────────────────┘
           │
           ▼
 ┌────────────────────┐
 │ build_dataset.py   │  Re-indexes, merges, engineers 64 columns,
 │                    │  drops all 12 same-day leaky columns
 └────────────────────┘  ──> Emits fresh data/master.csv
           │
           ▼
 ┌────────────────────┐
 │ health_check.py   │  Validates freshness, completeness, bounds, correlation,
 │                    │  same-day leak guard, holiday drift, weather provenance
 └────────────────────┘  ──> Exit 0 on success; Exit 1 turns CI red & opens issue
           │
           ▼
 ┌────────────────────┐
 │ Git Auto-Commit   │  Commits & pushes updated data/ folder automatically
 └────────────────────┘
```

### Automated Health & Integrity Rules (`health_check.py`)
1. **Freshness Verification:** weather lag ≤ 3 days, peak demand lag ≤ 5 days, master rebuilt within ≤ 1 day of latest inputs, forecast log written within ≤ 2 days.
2. **Upstream Latency Decoupling:** compares local slice against fresh upstream downloads.
3. **Physical Bound Verification:** `tn_peak_demand_mw` within 3,000–35,000 MW.
4. **Target Freeze Detection:** fails if the target is identical for 30 consecutive days.
5. **Join Alignment Sentinel:** fails if $|corr(\text{target}, cdd)| < 0.20$ (currently 0.374).
6. **NEW — Same-day leak guard:** fails if any of the 12 `config.LEAKY_SAMEDAY` columns reappears in `master.csv`. This is the direct fix for the leakage bug described in §7 — it exists so the same bug cannot be silently reintroduced by a future feature-engineering change.
7. **NEW — Holiday calendar drift warning:** counts `is_holiday` per *calendar year* (not per `master.csv` row, which conflates true calendar drift with the 157 missing-day gaps) and warns if any year falls more than 5 below the median. Currently 23–26/year, median 25.
8. **NEW — Weather provenance warning:** warns if more than 10 days of `weather_daily.csv` are still provisional (`source="update"`, awaiting ERA5 reconciliation).
9. **NEW — Forecast-log horizon warning:** warns if the forecast log isn't yet reaching the lead configured in `FORECAST_DAYS`.
10. **NEW (v4) — Source-staleness drift:** warns if any upstream publication's observed lag stops matching the horizons declared in `config.H_BY_GROUP`, so a change in Grid-India's or CICERO's cadence surfaces as a CI warning instead of as silently-NaN features at inference.
11. **Automated Issue Tracking:** opens a GitHub issue tagged `automation` on failure, auto-closes on the next green run.

Separately, `python test_contract.py` is a **22-check** acceptance gate (Gate G0) covering `feature_groups.json` exhaustiveness, the $H$ shift proven against the raw series rather than trusted from column names, the identity test that `swap_weather` perturbs nothing outside target-day weather, and **train/serve parity**. Run it after any feature-engineering change.

---

## 10. Known Data Realities & Historical Anomalies — corrected

0. **Counts (v4).** `master.csv` is **4,904 rows x 64 columns**, with **1,801** missing cells — down from 2,419 after a bug fix (item 9 below) and from an apparent 6,996 before the `holiday_name` sentinel fix. The target is never missing.

1. **157 missing days, not 163.** Over ~14 years, 157 calendar days had unreadable or unreleased upstream PSP reports — concentrated almost entirely before 2022 (an 82-day block in Q1 2013 alone accounts for more than half). Only 2 missing days fall in 2022–2026. **The pipeline no longer forward-fills lags across these gaps** (an earlier version used `ffill(limit=2)`, which mislabeled ~60 rows per lag column as a precise lag when they actually carried a value from 2–3 days earlier; this has been removed — a lag crossing a gap is now honestly `NaN`). Separately, **2,419 individual cells** are `NaN` inside otherwise-present rows — all of them legitimate lag/rolling warm-up, not a data quality problem. See roadmap v3's Count Audit box for the full breakdown.
2. **Chennai Floods (Nov–Dec 2015):** demand dropped anomalously low (~224 GWh vs ~314 GWh mean) due to extreme flooding. Genuine climate extreme event, unchanged from earlier revisions.
3. **COVID-19 National Lockdown (March–May 2020):** dramatic drop in peak demand (mean ~13,025 MW in 2020). Unchanged.
4. **Pongal Festival Demand Collapse (Jan 13–17):** structural collapse of ~2,300 MW (current figure: 2,312 MW below the working-day mean). Captured via `pongal_window`, which — unlike `is_holiday` — was never affected by the calendar bug below, because it's a fixed solar date range rather than a lunisolar lookup.
5. **NEW — Tamil Nadu holiday calendar was inconsistent across years, now fixed.** The `holidays` package resolves some lunisolar festivals in only some years: Maha Shivaratri in 6 of 15 years, Ram Navami in 12, Janmashtami in 14. The same festival was a public holiday in one year and an ordinary day in another, purely as an artifact of the dependency's coverage — `is_holiday` ranged 16–24 per year as a result. `tn_holidays.py` patches this with a curated override table, cross-validated against the package on every year both have data for (exact agreement on all checked years), and raises if a future package release contradicts it. Per-year count is now stable at 23–26, median 25.
6. **NEW — Same-day leakage was present and is now fixed.** `tn_energy_gwh` and `tn_shortage_at_peak_mw` were same-day Grid-India values, published the morning after the day they describe, present as same-day features in `master.csv`. See §7 for the full correction and the ~24% backtest-accuracy distortion this caused.
7. **NEW — Forecast weather bias is not stationary.** Measured against a 5.4-year archive of Open-Meteo's actual historical day-ahead forecasts (not reanalysis): lead-1 `t_mean` bias has walked from +0.49°C (2021) to −0.55°C (2026), reversing sign and growing. A bias correction fitted on the full archive average would be roughly 1°C wrong for current data. Relevant to objective 5 (operational degradation) and to Phase 5.2/0.5.3 in the roadmap.
8. **Regional Metric Pre-History:** utility-scale solar in the Southern Region was near-zero prior to September 2016; peak demand was reported as "Demand Met" prior to April 2017. Unchanged, cleanly backfilled.
9. **NEW (v4) — weather anomaly columns were inheriting demand's gaps.** `d_*` and `*_vs_roll7` were derived on the merged modelling frame, which is reindexed over `master.csv`'s span and therefore NaN on the 157 missing-**demand** days — even though the **weather** for those days is perfectly well known. `d_t_max` went NaN the day after every gap, and the trailing mean behind `t_max_vs_roll7` silently averaged fewer than 7 days (`min_periods=3` concealed it). Found by Gate G0's identity test, which compared the same columns computed two ways and flagged a 0.995 °C disagreement. Fixed by deriving them in `weather_derivatives()` on the gap-free weather series; **recovered 618 cells**, and the affected columns went from 61 NaN to 1.
10. **NEW (v4) — `H = 2`, and `peak_lag_1` is unavailable in production.** See §7. This is a property of Grid-India's publishing schedule, not of the data, but it changes which columns a servable model may use.
11. **NEW (v4) — the sources do not all arrive on the same day.** `demand` and `energy` come from the Grid-India PSP report ($H=2$); the Southern Region `sr_*` columns come from the POSOCO/CICERO slice, which runs a *further* day behind ($H=3$). `demand_daily.csv` held data through 2026-11-08 while `peak_demand.csv` held 2026-11-09, and 60 of the last 61 days show the same one-day offset. Shifting all three groups uniformly left **all four `sr_*_lag1` columns NaN on the row being predicted** — present in training, absent at inference, with nothing to warn you. Found only by running `predict_next_day.py` end to end; fixing it moved a test forecast by 204 MW. `config.H_BY_GROUP` now carries per-source horizons.

---

## 11. Proposed Machine Learning Workflow

*(unchanged from earlier revisions — this is Track A's plan, not yet started; reproduced for reference)*

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                       PHASE 1: BENCHMARKING                        │
   │  • Naive Persistence (T-1)  • Seasonal Persistence (T-7)           │
   │  • 7-Day & 30-Day Moving Average  • Ridge / Lasso / ElasticNet     │
   └────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                  PHASE 2: TREE-BASED ENSEMBLES                     │
   │  • Random Forest  • LightGBM  • XGBoost  • CatBoost                │
   │  • Ridge + tree hybrid, linear-leaf trees, predict-the-change      │
   └────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │           PHASE 3: EXTREME-DAY & REGIME-SHIFT ANALYSIS             │
   └────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                PHASE 4: TIME-SERIES EVALUATION                     │
   └────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │          PHASE 5: OPERATIONAL FORECAST GAP EXPERIMENT              │
   │  Status: dry run complete with a stand-in model — see §17          │
   └────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │               PHASE 6: EXPLAINABILITY & SHAP                       │
   └────────────────────────────────────────────────────────────────────┘
```

### Evaluation Protocol
Unchanged — expanding-window walk-forward, tuning on three expanding folds (validation 2022/23/24), test on 2025-onward evaluated once. Surge subset: top 5% of days per calendar year (primary), top 5% of March–June days (secondary), legacy `Y > 19,000 MW`.

---

## 12. Expected Deliverables & Outcomes

1. **Production-Ready Modelling Table:** verified **64-column** `data/master.csv` covering 2013–present with zero temporal leakage — **now actually leak-free, following the §7/§10 fixes**, and with a regression test (`test_contract.py`) plus a CI guard (`health_check.py`) so it stays that way.
2. **Automated CI/CD ETL Service:** unchanged, plus the leak guard, holiday-drift and provenance checks in §9.
3. **Operational Weather Forecast Repository:** `weather_forecast_log.csv` (98 days, growing) **plus** `weather_forecast_archive.csv` (NEW — 5.4 years of real historical day-ahead forecasts, enabling the operational backtest that objective 5 and Phase 5 need without waiting years for the log to grow).
4. **Comprehensive Model Comparison Suite:** not yet started (Track A).
5. **Research Paper / Technical Report:** not yet started.
6. **NEW (v4) — Modelling contract:** `src/contract.py` (`load_master(H)`, `swap_weather`, `verify_no_leakage`), `feature_groups.json`, and `test_contract.py` as Gate G0's acceptance check. Lets Track A and Track B work independently against a frozen, tested interface.
7. **NEW (v4) — Operational forecast experiment:** `run_phase5.py` plus `results/operational_gap.json`, `results/archive_vs_log_overlap.csv`, and `notebooks/05_operational_forecast_experiment.ipynb`. Regime A vs B across both $H$ values, leads 1–3, on overall / March–June / three surge subsets, with moving-block bootstrap CIs.
8. **NEW (v4) — Production forecasting service:** `predict_next_day.py` (evening inference with conformal intervals), `track_predictions.py` (live scoring once actuals land), `calibrate_intervals.py`, `dashboard/app.py` (Streamlit), and `.github/workflows/evening.yml`.

---

## 13. High-Value Research Directions

*(unchanged in substance; status notes added)*

1. **Quantifying the Weather Forecast Degradation Gap:** **MEASURED — see §17.** Full dual-regime experiment with weekly refit, leads 1–3, both $H$ values, surge subsets and bootstrap CIs. One finding worth the paper on its own: **the forecast bias is non-stationary**, walking from $+0.49$ °C (2021) to $-0.55$ °C (2026) as Open-Meteo changed models. A bias correction fitted on the full archive would be ~1 °C wrong today, so none is applied. Awaiting only Track A's champion for the final numbers.
2. **Non-Linear Heat Duration Sensitivity:** not yet evaluated (Track A, Phase 1/3).
3. **Extreme Value Behavior During Summer Records:** not yet evaluated (Track A, Phase 4).
4. **Structural Regime Shift Adaptability:** not yet evaluated (Track A, Phase 4). *Note: the 2026 growth acceleration described in §7 is itself a mild structural shift worth treating alongside 2015/2020 — current demand is accelerating away from its own 13-year trend, not just recovering from a shock.*
5. **NEW (v4) — Festival demand collapse is the dominant error source, and it is a feature gap.** Phase 5.4 found that weather forecast error is a minor driver (+179 MW/°C; days missing by >1 °C average 541 MW error against 522 MW on calm days) while **9 of the 10 worst days are festival-adjacent**, over-forecast by 2,300–3,400 MW. Three are the Diwali cluster (day before / day of / day after); the others are Pongal, Uzhavar Thirunal, Ayudha Puja and May Day. Across the full test window, festival-adjacent days ($n=124$) average **782.9 MW** absolute error against **467.3 MW** elsewhere ($n=552$) — 67% worse — and the bias flips sign, $+161.8$ MW versus $-100.1$ MW. `is_holiday` is a single binary, but Diwali is a multi-day industrial shutdown exactly like Pongal — and only Pongal has a `pongal_window` feature. A `diwali_window` (or a general multi-day-festival window) is the obvious candidate and belongs in Phase 3.4's ablation, where it can be measured. Not added in v4 because `master.csv`'s schema is frozen at Gate G0 and roadmap C.1 requires both tracks to agree. Per-day predictions: `results/phase5_predictions.csv`.

---

## 14. Suggested Presentation & Slide Deck Structure

*(unchanged — not yet built)*

---

## 15. Team Topic Assignment Guide

### Person 1 — Data Pipeline, Automation & Quality Engineering (Track B)
- **Current Status:** **Complete.** Beyond the pipeline itself: the modelling contract (`src/contract.py`, `feature_groups.json`, `test_contract.py`), the `H` measurement, the Phase 5 operational experiment, and the Phase 7 forecasting service (evening inference, conformal intervals, live scoring, dashboard).
- **Ongoing:** monitor both workflows; re-run `measure_publish_time.py` if Grid-India's publishing schedule shifts (it would change `H`, which changes every demand feature); keep the evening cron running — `predictions_log.csv` cannot be backfilled.

### Person 2 — Literature Review & Domain Benchmarking
- Unchanged — independent of both tracks, not yet started.

### Person 3 — Exploratory Data Analysis & Machine Learning Modeling
- Unchanged — **not yet started.** Should read §7, §8 and §10 of this revision before opening `master.csv`: the column list, correlations and row/column counts in earlier revisions are stale, and the energy/shortage columns referenced there no longer exist in the file.

---

## 16. Key Technical Terms

*(unchanged)*

---

## 17. Current Project Status & Achievements — rewritten for v3

### Established & Completed
- **Target pivot fully executed:** next-day peak demand (MW) is the target, unchanged.
- **Backfill complete:** Zenodo archive (3,852 rows) + Grid-India PSP reports (**1,052** rows, not 1,033) = **4,904 days**, 2013-01-01 to the latest pulled day.
- **5-City load-weighted weather dataset built**, with provenance tracking (NEW): rows are tagged `archive` (ERA5) or `update` (provisional, forecast-endpoint analysis, which runs ~0.9°C off ERA5 until reconciled).
- **Live operational forecast logger**: 98 issue dates since August 2026, now logging to lead 3.
- **NEW — Historical operational forecast archive built**: `weather_forecast_archive.csv`, 29,535 city-day-leads, **2021-06-01 to the present** (5.4 years — more than double the "January 2024 onward" originally planned), built from Open-Meteo's Previous Runs API using the identical aggregation code as the reanalysis features. Cross-validated against the live log on their 97-day overlap: 0.36°C MAE on `t_max` at lead 1 — the archive is a valid stand-in for forecasts that would actually have been held.
- **NEW — Operational gap measured.** Weekly refit, test window 2025-01-01 → 2026-11-10, n=676, both $H$ values, leads 1–3:

  | weather at scoring time | MAE | MAPE | bias |
  |---|---|---|---|
  | H=2 reanalysis (Regime A) | 483.5 | 2.81% | −3 |
  | **H=2 forecast lead 1 (Regime B)** | **525.2** | **3.03%** | −52 |
  | H=2 forecast lead 3 | 531.3 | 3.06% | −54 |
  | H=1 reanalysis *(idealised)* | 399.6 | 2.30% | +3 |
  | H=1 forecast lead 1 *(idealised)* | 422.8 | 2.42% | −57 |

  Not knowing tomorrow's weather costs **+42 MW (+8.6%)** at lead 1 and degrades gently with horizon. **Data latency costs more than weather uncertainty** — the $H{=}1 \rightarrow H{=}2$ step is ~84 MW, twice the forecast-weather penalty, which is precisely why $H$ was worth measuring. On the top-5% surge days (n=35) the gap widens to 439 → 551 MW with bias −335 → −418 MW. Uses a stand-in model, not Track A's champion; re-run at G2.
- **NEW — Demand-growth handling previewed (informal, not the formal D1 comparison):** fitting the raw MW level, a stand-in tree model under-forecasts the 2026 holdout by ~420 MW on 75% of days — trees cannot extrapolate past their training maximum, and 2026 demand is running above anything in training. Fitting `Y − peak_lag_1` instead (same features, same model) and adding the lag back: MAE 547→392 MW, bias −420→−52 MW, **MAPE 2.38%** — inside the ≤2.5–3.5% target in objective 1, though on a stand-in model, not the frozen champion.
- **NEW — Same-day leakage found and fixed:** see §7, §10.
- **NEW — Tamil Nadu holiday calendar found inconsistent and fixed:** see §10.
- **Feature engineering pipeline operational:** **64-column** leak-free modelling table, including new day-over-day weather deltas and a day-of-week cycle.
- **Production CI/CD active:** daily GitHub Actions workflow, now with a reconcile step and three new health checks.
- **Count audit resolved:** see §8 — no more disagreement between documents on rows/columns.

### Immediate Priority
- **`H` (data-latency horizon) is unmeasured** and is now the single largest open item — Phase 0.5.2 in the roadmap. It blocks the evening-inference work (Phase 7) and the final version of the Phase 5 operational-gap experiment.
- **Track A has not started.** Phases 1–4 and 6 (EDA, baselines, tree models, extreme-day analysis, SHAP) are all open. The informal previews above (D1, operational gap) are Track B side effects of building the Phase 5 pipeline and should not be mistaken for Track A's formal results — they used a plain stand-in model, not Optuna, weekly refit, or recency weighting.
- **Contract files (`config.py` additions, `feature_groups.json`, `src/contract.py`) do not exist yet.** Gate G0 (contract freeze) has not been reached.

---

## 18. One-Line Project Summary

> **An automated, production-grade machine learning system that forecasts Tamil Nadu's next-day peak electricity demand (MW) using multi-city weather indices, calendar dynamics, and historical demand telemetry, supported by daily CI/CD pipelines and operational forecast logging.**

---

## 19. Revision Notes

### v2 (prior revision)
1. Demand growth: main model uses the raw-MW target without the `trend` feature, weekly walk-forward refit and recency weighting. A Ridge + tree hybrid, linear-leaf trees and a predict-the-change target are compared against it.
2. Operational experiment: the long window comes from archived day-ahead forecasts (Open-Meteo Previous Runs API, January 2024 onward); the real forecast log (about 80 days from August 2026, no heat season) is a small gold-standard check.
3. Horizon: a configurable data-latency horizon $H$, set by measuring when each day's PSP report is published; inference runs in the evening.
4. Surge definition: top 5% of days per calendar year, with bias reported alongside error.
5. Tuning and refit: Optuna on three expanding folds (validation 2022, 2023, 2024), then weekly refit; test period evaluated once.
6. Intervals: rolling conformal on relative residuals plus a quantile-LightGBM comparison. The "peak hour window" idea is dropped because there is no hourly demand data.
7. Deep learning: removed from scope.

### v3
1. **Count audit resolved.** 4,904 rows, 64 columns, 157 missing days.
2. **Same-day leakage found and fixed.** `tn_energy_gwh` (r=0.971) and `tn_shortage_at_peak_mw` removed from `master.csv`.
3. **Holiday calendar fixed.** `tn_holidays.py` patches the `holidays` package's lunisolar coverage gaps.
4. **Phase 5.1/5.2 done**, ahead of plan — forecast archive reaches 2021-06, not January 2024.
5. **Phase 5.3 and D1 previewed informally** with a stand-in model.
6. **Forecast bias found to be non-stationary** (+0.49 °C → −0.55 °C, 2021→2026).
7. **`H` remained unmeasured** and was flagged as the top open item.

### v4 (this revision) — Track B built

1. **`H = 2`, measured** (`results/publish_time.json`). Day *D*'s report was never in hand on day *D* across 46 observed runs. The circular evidence (`report_published`, which the extractor derives from the data date) was identified and rejected. **Consequence: `peak_lag_1` is unavailable for a day-ahead forecast**, so `baseline.py`'s 392 MW is an $H=1$ number.
2. **Sources arrive on different days.** `sr_*` are a further day behind the PSP report, so `config.H_BY_GROUP` gives each group its own horizon. Found by end-to-end smoke test, worth 204 MW on a test forecast. §10 item 11.
3. **Gate G0 MET.** `src/contract.py` (`load_master`, `swap_weather`, `verify_no_leakage`), `feature_groups.json`, and `test_contract.py` — **22 checks**, prediction-level identity exact to 0.000e+00 MW.
4. **Phase 5 complete** with a stand-in model: dual-regime walk-forward at both $H$ values, leads 1–3, overall / March–June / three surge subsets, moving-block bootstrap CIs, sensitivity analysis and a real-log check. Artefacts in `results/`.
5. **Phase 7 complete:** evening inference with split-conformal intervals, `issue_slot`-aware forecast logging, live scoring, a Streamlit dashboard, and `evening.yml`.
6. **0.5.3 resolved favourably:** training-time and inference-time lagged weather are the *same* Open-Meteo product (bit-identical for recent data), and archive values are never revised. No second train/serve skew.
7. **Four latent bugs fixed** (§10 items 9 and 11, the `swap_weather` trailing-mean index, and the `holiday_name` `"None"` sentinel). Missing values 6,996 → 2,419 → **1,801**.
7a. **Train/serve parity is now tested, not hoped for.** Three of those bugs shipped features present in training and missing at serving — a failure a model cannot report, since it just stops using the column. Two were caught only by running inference by hand. `test_contract.py` now builds the real inference row and requires all 46 features populated, checks each source's observed freshness against `config.H_BY_GROUP`, and verifies the serving path reproduces `build_dataset.py` on a shared date. Reintroducing any bug turns the suite red.
7b. **A 10x performance bug.** `HistGradientBoostingRegressor` on ~4,900 rows is ten times slower on all cores than on one (9.68 s vs 0.95 s per fit) — thread synchronisation dominates at this data size. `threadlimit.py` pins threads to 1; interval calibration went from 15+ minutes to under a minute, and the full Phase 5 run from hours to minutes. This is what makes G3's "one command rebuilds everything" realistic.
8. **Track A is now the entire critical path** and has no remaining Track B dependency. See `SYNC.md`.
