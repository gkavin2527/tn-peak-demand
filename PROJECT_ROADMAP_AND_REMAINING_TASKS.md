# Tamil Nadu Peak Electricity Demand Forecasting — Project Completion Roadmap (v4)

This document lists everything left to take the **Tamil Nadu Next-Day Peak Electricity Demand Forecasting** project from its finished data foundation to modelling, the operational-forecast experiment, deployment and the paper.

**v4 changes (this revision) — Track B is built.** `H` is **measured: H = 2**, and the consequence is sharper than anyone had assumed — `peak_lag_1`, the second-most-predictive column in the dataset, is **not available for a genuine day-ahead forecast**. Gate **G0 is MET** (22/22 checks, prediction-level identity exact to 0.000e+00 MW); the contract files exist and are frozen. Phase 5 is complete end-to-end with a stand-in model and all artefacts exported. Phase 7 is built: evening inference with conformal intervals, an evening workflow, live scoring, and a dashboard. Four latent bugs were found and fixed, three of them the same failure - a feature present in training and missing at serving - and the test suite now asserts that property instead of relying on someone noticing. **Track A remains entirely unstarted** and is the only thing between here and the paper.

See the "v4 status" box in each section for specifics. Earlier revision notes below.

**v3 changes:** the Phase 0.5 count audit is resolved — real figures below, not provisional ones. A same-day leakage bug was found and fixed in `master.csv` (two columns that are published the morning *after* the day they describe were being used as same-day features). The Tamil Nadu holiday calendar's lunisolar coverage gap is fixed. **Track B's Phase 5.1 and 5.2 are done**, ahead of the plan in two ways: the forecast archive reaches **mid-2021**, not January 2024, and it has already been cross-validated against the real log. A stand-in model (`baseline.py`) has run the growth-handling question (D1) and the operational-gap question (5.3) end-to-end as a dry run, which is informative but is **not** Track A's frozen champion and is **not** yet built to the `src/contract.py` shape this document specifies — see the status notes inline. **`H` (the data-latency horizon, Section B) is still unmeasured** — this is now the single largest open item blocking Phase 7, and whoever is free should pick it up next.

Everything in Section A, B and C.1–C.2 (the locked decisions, the `H` definition, the track split and the integration contract) is unchanged from v2. This revision only updates status, counts and what Track B has actually shipped.

---

## Executive Status Overview

| Phase | Description | Track | Status |
|---|---|:---:|:---:|
| **Phase 0** | Data engineering, scrapers, pipelines, CI/CD | — | **COMPLETE** |
| **Phase 0.5** | Pre-flight audit and contract freeze | Shared | **COMPLETE** — counts audited, `H=2` measured, contract frozen, **G0 MET** |
| **Phase 1** | EDA and statistical profiling | A | **Not started** (teammate) |
| **Phase 2** | Evaluation protocol and baseline models | A | **Not started** (teammate) |
| **Phase 3** | Tree models, growth-handling comparison, Optuna | A | **Not started** (teammate) |
| **Phase 4** | Extreme-day, regime-shift and uncertainty analysis | A | **Not started** (teammate) |
| **Phase 5** | Operational forecast degradation experiment | B | **COMPLETE** with a stand-in model; re-run at G2 with the champion |
| **Phase 6** | Interpretability (SHAP) | A | **Not started** (teammate) |
| **Phase 7** | Evening inference, intervals, dashboard | B | **COMPLETE** |
| **Phase 8** | Paper, deck, packaging | Shared | Not started |

**The critical path is now entirely Track A.** Every Track B dependency it had
is satisfied: `load_master(H)` and `swap_weather` exist and are tested,
`feature_groups.json` is frozen, and the operational experiment runs on a
`--champion` flag. Track A can start Phase 1 today against a stable contract.

## What Has Already Been Completed (Phase 0)

- [x] **Grid-India ingestion**: automated scraping of daily PSP reports (PDF and XLS) (`fetch_peak_demand.py`, `update_peak_demand.py`).
- [x] **Historical target backfill**: Zenodo archive (**3,852** rows) combined with Grid-India observations (**1,052** rows, not 1,033) into `data/peak_demand.csv` — **4,904** rows, 2013-01-01 to the latest pulled day.
- [x] **State-level weather**: load-weighted blend of Chennai (36%), Coimbatore (22%), Madurai (16%), Trichy (14%), Salem (12%) (`fetch_weather.py`), **25,310** city-days.
- [x] **Zero-leakage feature engineering**: continuous calendar reindexing, lags, rolling statistics, Fourier seasonality, Pongal window, cooling indices (`data/master.csv`).
- [x] **Operational forecast logging**: day-ahead numerical weather forecasts for all 5 load centres logged to `data/weather_forecast_log.csv` since 2026-08-04 — **98 issue dates, 1,470 rows**, now logging to lead 3 (`FORECAST_DAYS=4` in `config.py`; the log currently reaches lead 2 and will reach lead 3 on the next run).
- [x] **CI/CD and health suite**: `daily.yml` at 07:00 IST with `health_check.py` and auto-commit.
- [x] **(New since v2) Same-day leakage removed.** `tn_energy_gwh` and `tn_shortage_at_peak_mw` are Grid-India PSP values published the *morning after* the day they describe. Both were present as same-day features in `master.csv`. `tn_energy_gwh` correlates **r = 0.971** with the target; leaving it in overstated backtest accuracy by roughly 24% (452 MW MAE with it in, 591 MW without, on the pre-fix feature set). Fixed: all 12 same-day columns are named in `config.LEAKY_SAMEDAY`, `build_dataset.py` drops them after lagging, and `health_check.py` now **fails the run** if any of them reappear in `master.csv`. `tn_shortage_at_peak_mw` also had a second, independent defect — 79% of rows (everything before 2024) had no raw value and were being `fillna(0.0)`'d, which asserted zero deficit through the 2013–2015 load-shedding years. It is dropped from `master.csv` entirely now; the raw, un-filled column is still in `data/peak_demand.csv`.
- [x] **(New) Tamil Nadu holiday calendar fixed.** The `holidays` package resolves Maha Shivaratri in only 6 of 15 years, Ram Navami in 12, Janmashtami in 14 — the same festival was a holiday one year and an ordinary day the next. `tn_holidays.py` wraps the package with a curated, cross-validated override table (every filled date reproduces the package exactly in every year both agree) and raises if a future package version contradicts it. `is_holiday` now carries **23–26 holidays/year, median 25** (was 16–24). Resolved calendar committed to `data/tn_holidays.csv`; `holidays==0.108` pinned in `requirements.txt`.
- [x] **(New) Weather-level redundancy found and documented.** `t_mean`, `t_max`, `cdd`, `thi`, etc. add nothing once the autoregressive lags are present (`config.WEATHER_LEVEL_COLS`). Day-over-day deltas (`d_t_mean`, `d_cdd`, …) and 7-day anomalies (`t_max_vs_roll7`, …) carry the signal the levels don't; dropping levels and keeping deltas improves backtest MAE from 591 to 545 MW.
- [x] **(New) Weather provenance tracking.** `weather_daily.csv` now carries a `source` column (`archive` = ERA5 reanalysis, `update` = the forecast endpoint's own provisional analysis, which runs ~0.9°C off ERA5). `fetch_weather.py reconcile` backfills provisional rows to ERA5 once it lands (~5-day delay); wired into `daily.yml` and `health_check.py` warns if provisional rows pile up past 10 days.

> **Count audit — resolved.** The real numbers, from the files on disk:
>
> | | v2 said | **actual** |
> |---|---|---|
> | `master.csv` rows | 4,878 / 4,885 | **4,904** |
> | `master.csv` columns | 54 / 52 | **64** (62 features + `date` + target) |
> | Missing **days** | 163 | **157** |
> | Missing **values** (NaN cells) | not tracked | **2,419** — see note below |
> | `peak_demand.csv` split | 3,852 + 1,033 | 3,852 + **1,052** |
> | `weather_daily.csv` | 25,210 city-days | **25,310** |
> | `weather_forecast_log.csv` | ~79–80 issue dates | **98** issue dates, **1,470** rows |
>
> **"Missing days" and "missing values" are different numbers and both matter for different reasons.** 157 is the count of calendar dates with no row at all — concentrated almost entirely before 2022 (an 82-day block in Q1 2013 accounts for more than half of it; only 2 missing days fall in 2022–2026). 2,419 is the count of individual `NaN` **cells** inside the 4,904 rows that do exist — every one of them is a lag or rolling-window column during its warm-up period or at the edge of a gap (`peak_lag_364` alone accounts for 423, simply because there is no day-minus-364 in the series' first year). The target, `tn_peak_demand_mw`, is **never** `NaN`. Nothing is imputed and nothing is forward-filled across a gap — an earlier pipeline version used `ffill(limit=2)`, which left ~60 rows per lag column holding a value from 2–3 days back under a label that no longer described it; that has been removed. A prior build also showed **6,996** apparent missing values because the `holiday_name` "no holiday" sentinel was the literal string `"None"`, which collides with `pandas.read_csv`'s default NA-string list — a plain `pd.read_csv` silently read 4,577 real values as missing. The sentinel is now `"No Holiday"`, which round-trips correctly.

---

## Section A — Locked Design Decisions

*(unchanged from v2 — see that revision for full rationale; this section lists what Track A will still decide formally, with a note on what Track B's exploratory testing has previewed)*

| # | Issue | Decision |
|---|---|---|
| **D1** | Trees cannot extrapolate beyond the training range (2026 mean is about 19,095 MW; the record is 21,724 MW). | **Main model (Option A):** raw-MW target, **no `trend` feature**, weekly walk-forward refit, recent years weighted higher (exponential half-life tuned in Optuna). **Comparison rows:** (B) Ridge on trend and lags plus LightGBM on the Ridge residuals, (C) LightGBM with `linear_tree=True`, (D) predict the change, `Y − peak_roll_mean_7`, and add the level back. Champion = best mean validation MAPE across the three folds; if two models are within one standard error, prefer the one with the smaller top-5% bias, then the simpler one. The comparison itself becomes a paper paragraph ("three ways of handling demand growth"). <br><br>*Status note: this is still Track A's decision to make formally, with Optuna and the real walk-forward engine. A quick, informal preview exists — a plain `HistGradientBoostingRegressor` fit on the raw level, no trend, rolling-origin CV, mean MAE 547, 2026 bias −420 MW on 75% of days (the extrapolation failure D1 describes, reproduced exactly). The same features fit to `Y − peak_lag_1` (the sibling of Option D) instead: mean MAE 392, bias −52. This doesn't replace the formal comparison — it has no Optuna, no weekly refit, no recency weighting, and uses `peak_lag_1` rather than `peak_roll_mean_7` as the anchor — but it is one independent data point suggesting the champion will land in the "predict-the-change" family. Reproducible with `python baseline.py`.* |
| **D2** | The real forecast log has only about 80 days (August 2026 onward), so it cannot support a 2025–2026 operational test. | **Main result:** archived day-ahead forecasts from Open-Meteo's **Previous Runs API** (`*_previous_day1`), which was assumed to be available from January 2024. That covers the whole 2025-to-2026 test period, including three March–June summers. **Small gold-standard check:** the real log (about 80 days, no heat season), labelled as a small sample. Compare archive and real log on the overlap and report any difference. If the archive disagrees badly with the log, call the result preliminary. Keep logging daily and re-run the experiment on real forecasts after summer 2027. <br><br>*Status note: done, and better than assumed — see Phase 5.1/5.2 below. The archive actually reaches back to 2021-06, not Jan 2024, and the cross-check against the real log passed (0.36°C MAE on `t_max` at lead 1).* |
| **D3** | A 07:00 run may not have yesterday's demand, and an evening run does not fix that by itself. | The model is built around a configurable data-latency horizon **`H`** (Section B). Inference moves to the evening, and the forecast logger gets a matching evening run. <br><br>*Status note: **unmeasured**. This is 0.5.2 below and is now the single biggest open item.* |
| **D4** | The 19,000 MW "peak surge" threshold drifts with demand growth. | **Primary:** top 5% of days within each calendar year. **Secondary:** top 5% of March–June days. **Legacy:** `Y > 19,000 MW`, kept for continuity only. Always report **bias** (mean of prediction minus actual) next to error on these subsets, with block-bootstrap confidence intervals, because each subset has only about 15–20 days per year. |
| **D5** | Tuning on 2024 alone is fragile; daily refits are slow. | Optuna over **3 expanding folds** (validation years 2022, 2023, 2024). Hyperparameters are then frozen. Evaluation uses a **weekly refit** walk-forward. Inside Optuna a monthly refit is allowed for speed. The test period (2025-01-01 onward) is evaluated once, at the end. |
| **D6** | Row/column inconsistencies; promised intervals and "peak hour window" have no method or data. | Audit counts first (Phase 0.5) — **done, see above.** Intervals: **rolling split-conformal on relative residuals**, plus a **quantile-LightGBM** comparison. **The "peak hour window" claim is dropped** (no hourly demand data). |
| **D7** | Deep learning (old Phase 4). | **Removed.** About 4,500 daily rows (now 4,904) favour gradient-boosted trees, and the project is positioned as classical ML. Replaced by Phase 4 (extreme-day, regime-shift and uncertainty analysis, Master Concept §13). The paper states plainly that deep sequence models are future work. |

---

## Section B — Forecast Horizon `H` (read before building any features)

> **v4 status: RESOLVED. `H = 2`.** Measured, not assumed — and the consequence
> is bigger than this section anticipated. Everything in Section B below is now
> implemented in `src/contract.py`.

### The measurement

Three sources of evidence were considered. Only one is trustworthy, and saying
why matters because the obvious one is circular:

| Evidence | Verdict |
|---|---|
| `report_published` in `peak_demand.csv` | **Rejected — circular.** It is the report's own internal "Date of Reporting", and `extract_peak_demand.py` derives the observation date as `report_date − 1 day`. Comparing the two just recovers that subtraction. An earlier draft of this measurement used it and got a meaningless 100%. |
| Grid-India API metadata (`CreatedOn`, `Field2`) | **Supporting only.** Genuinely observed upload timestamps (next-day uploads land 08:59–20:52 IST, median 09:57), but `Field2` is not consistently the data date upstream, so it cannot set `H`. |
| **What the pipeline actually held on each run** | **Primary.** For all 46 observed daily runs, the newest data obtained was always `run_day − 1`. Minimum gap ever observed: **1 day**, including a run at 18:07. No interpretation of upstream metadata required. |

So day *D*'s report is never in hand on day *D*. An evening-of-*D* forecast for
target *D+1* holds demand through *D−1* → **`H = 2`**.

```bash
python measure_publish_time.py --probe   # -> results/publish_time.json
```

### The consequence nobody had flagged

**`peak_lag_1` is not available for a genuine day-ahead forecast.** It
correlates 0.927 with the target and is the natural anchor for a differenced
model — and it cannot be used. The newest usable demand observation is
`peak_lag_2`. `baseline.py`'s 392 MW figure anchors on `peak_lag_1` and is
therefore an **H=1 number**: real as a feature-selection benchmark, not
achievable in production.

### Sources do not all arrive together — a second finding

`H` is not one number. The groups in `feature_groups.json` exist partly for
this: they publish on different schedules.

| Group | Source | Horizon |
|---|---|---|
| `demand` | Grid-India PSP report, published *D+1* | **H = 2** |
| `energy` | same report (`build_dataset` backfills `tn_energy_gwh` from `tn_energy_met_mu`, which agree to 0.02 GWh over 1,050 days) | **H = 2** |
| `grid` (`sr_*`) | POSOCO/CICERO slice, a *further* day behind | **H = 3** |

Shifting them uniformly looked correct and was not. Smoke-testing
`predict_next_day.py` end to end returned **all four `sr_*_lag1` columns NaN on
the row being predicted** — the grid slice simply had not published
`target − 2` yet (`demand_daily.csv` held data through 11-08 while
`peak_demand.csv` held 11-09; 60 of the last 61 days versus 61). In training
those columns are present; at inference they were absent. Nothing would have
warned about it, and the model would have quietly leaned on four features it
never actually has. Fixing it moved a test forecast by **204 MW**.

`config.H_BY_GROUP` now carries the per-source horizons and
`src.contract.group_offsets()` is shared by `load_master()` and
`predict_next_day.py`, because duplicating that logic is what caused the bug
in the first place.

`load_master(H)` shifts each group by its own `H−1` on a calendar-continuous index. Column names are deliberately *not*
renamed (shifting `{1,2,3,7,14,364}` by one collides with itself at 2 and 3), so
the convention is: **lag numbers are relative to the newest available demand
day, `target − H`**. Rather than trusting that convention,
`verify_no_leakage()` resolves each lag column against the raw series and
proves no column carries data newer than `target − H`; `test_contract.py`
runs it.

*(Original Section B text follows, unchanged, for reference.)*

**The problem.** The Master Concept formula uses demand observed "up to day T" to forecast day T+1. Grid-India's PSP report for day T cannot be final until T has ended, so it is normally published on T+1. A forecast issued on the evening of day D therefore probably has demand only up to **D−1**. Moving the run to the evening fixes the *weather forecast* timing but does **not** by itself make `peak_lag_1` available.

**Definition.**
- Issue time: evening of day D. Target: day D+1.
- `H = 1` if the PSP report for day D is published before the issue time (demand features end at D).
- `H = 2` if it is not (demand features end at D−1).
- All demand, energy and Southern Region grid features are built with the extra shift, so nothing later than `target_day − H` is used. Rolling statistics and the "predict the change" anchor are shifted too.
- Target-day weather features (`t_max`, `cdd_max`, `thi`, `hours_above_32`, …) come from the forecast issued at the same time. Lagged weather features (`cdd_lag1`, …) use observed weather.

**What to do.**
1. ~~Measure the real publish time of the day-D report.~~ **DONE — `H=2`**, `results/publish_time.json`.
2. ~~Build `load_master(H)`.~~ **DONE** — `src/contract.py`. Both `H=2` (headline) and `H=1` (idealised) are reported in `results/operational_gap.json`.
3. Name the result honestly in the paper: **"day-ahead forecast with one-day demand-data latency"**. Carried into the dashboard caption and `predict_next_day.py` output.
4. ~~Match the logger to the inference time.~~ **DONE** — `fetch_weather.py forecast --slot=evening`, scheduled 20:00 IST in `evening.yml`. An `issue_slot` column was added because the original dedup key `(issued_on, city, date)` meant an evening run would silently *overwrite* the morning's rows — losing half the data and making the forecast actually used at inference indistinguishable from one issued twelve hours earlier. Inference prefers the evening slot.
5. ~~Check weather-lag skew.~~ **DONE — not material**, `results/weather_lag_source_check.json`. The result is cleaner than expected: Open-Meteo's archive endpoint and the forecast endpoint's `past_days` window return **bit-identical** values for the recent window (100% exact agreement for data 0–40 days old). Training-time and inference-time lagged weather are therefore the *same product*, not two that happen to agree — so there is no second train/serve skew hiding under the obvious one. Separately verified via git that archive values are **not revised** as ERA5 finalises (0 of 25,175 city-days changed between a 2026-10-15 snapshot and now), so `master.csv`'s recent rows are reproducible.

---

## Section C — Two-Track Parallel Plan

*(unchanged from v2 — the split, ownership, paper sections and branch names below are the agreed contract. This revision only annotates what each side has actually shipped.)*

### C.1 Split

| | **Track A — Modelling and Insight** | **Track B — Operations and Data** |
|---|---|---|
| **Phases** | 1, 2, 3, 4, 6 | 5, 7 (plus data side of 0.5) |
| **Owns** | `src/evaluation.py`, `src/walkforward.py`, `src/models/`, `notebooks/01–04, 06`, `figures/`, `results/` model outputs, `models/` | `src/data/`, `src/operational/`, `predict_next_day.py`, `dashboard/`, `notebooks/05`, `.github/workflows/`, `data/weather_forecast_archive.csv`, `data/predictions_log.csv` |
| **Paper sections** | §4 Methodology (models, validation), §5.1–5.3 Results, §6 Interpretability | §3 Data and pipeline, §5.4 Operational gap, §7 Deployment |
| **Branch** | `track-a-modelling` | `track-b-operations` |
| **Current owner** | Teammate | This session / assistant, going forward |

Rules: each track edits only what it owns; shared files (`config.py`, `src/contract.py`, `feature_groups.json`) change only by agreement of both; merge through pull requests; keep a short `SYNC.md` listing decisions and any contract change. The literature review (Master Concept, Person 2) is independent of both tracks and can go to whoever has slack.

### C.2 Integration contract (frozen at Gate 0)

**Status: FROZEN. Gate G0 is MET — 22 of 22 checks pass** (`python test_contract.py`), including the identity test the roadmap specifies literally: swapping ERA5 weather through `swap_weather` reproduces the Regime A predictions to **0.000e+00 MW** over 313 days.

**Shared files**
- `config.py`: **DONE** — `H = 2`, `H_IDEALISED`, `TUNING_FOLDS`, `TEST_START`, `REFIT_DAYS`, `SURGE_QUANTILE`, `SURGE_LEGACY_MW`, `SUMMER_MONTHS`, `RANDOM_SEED`, plus `RESULTS`/`MODELS`/`FEATURE_GROUPS` paths.
- `feature_groups.json`: **DONE** — all 62 columns classified into the six agreed groups, verified exhaustive and non-overlapping by G0. Also carries `weather_swap_rules` (how each weather column kind is rebuilt) and `recommended_drops` (advisory, not contract).
- `src/contract.py`: **DONE**
  - `load_master(H) -> DataFrame` — target-date index, target renamed `y`, demand/energy/grid shifted by `H−1`, `.attrs` carrying `H` / `lag_offset` / `newest_demand`.
  - `swap_weather(df, weather_df, observed=None) -> DataFrame` — replaces only `weather_target_day`. Lead-aware: deltas are `forecast(T) − observed(T−1)` and anomalies are `forecast(T) − mean(observed[T−7..T−1])`, because yesterday is already known at issue time. `forecast_skew.py`'s private `substitute()` has been deleted in favour of this.
  - `verify_no_leakage(df)` — **added beyond the contract.** Resolves each lag column against the raw series and asserts nothing is newer than `target − H`, so the H shift is proven rather than named.
  - `model_features(df)` — convenience feature list (advisory).
  - `run_walk_forward(...)` — **Track A's to write.** Re-exported here, preferring `src/walkforward.py` the moment it exists and otherwise falling back to Track B's stand-in at `src/operational/walkforward.py`, which implements the identical frozen signature. Delete the stand-in at G2.

**Track A hands over** *(none of this exists yet — Track A has not started)*
- `models/champion_config.json`, `models/champion.pkl`, `results/walkforward_test_preds.csv`, `results/leaderboard.json`
- `results/intervals.json` — **note:** Track B has already written this file for Phase 7.1 serving (`calibrate_intervals.py`, split-conformal per season). Track A's Phase 4.3 study supersedes it; keep the same schema or update `src/operational/intervals.py:load()`.

**Track B hands over — ALL DELIVERED**
- `data/weather_forecast_archive.csv` — 29,535 rows, 2021-06-01 → latest (5.4y), leads 1–3, state-weighted through the same `state_weather()` the reanalysis features use. Carries `lead`; add a `model` column only if more than one Open-Meteo model is ever mixed in.
- `results/archive_vs_log_overlap.csv` — **exported.**
- `results/operational_gap.json` — **exported.** Regime A vs B for both `H=2` and `H=1`, leads 1–3, on overall / March–June / three surge subsets, with moving-block bootstrap CIs.
- `predict_next_day.py` and `data/predictions_log.csv` — **built.** Plus `track_predictions.py` (live scoring), `results/intervals.json`, `results/live_performance.json`, `dashboard/app.py`, and `.github/workflows/evening.yml`.
- Also delivered beyond the contract: `measure_publish_time.py`, `check_weather_lag_source.py`, `test_contract.py`, `src/operational/{metrics,intervals,walkforward}.py`, `notebooks/05_operational_forecast_experiment.ipynb`.

### C.3 Dependencies and how each track avoids waiting

*(unchanged — this is still the right plan)*

- Track A needs only `load_master(H)` to start. Until Track B ships it, A builds on `master.csv` with `H=2` as the conservative default.
- Track B's Phase 5 needs a model. Until A's champion exists, B runs the experiment end to end with a **stand-in** (Ridge, then LightGBM with default parameters on the same feature groups). The code is identical; only `champion_config.json` changes at hand-over. *Status: Track B has already done this, informally — `baseline.py`'s differenced `HistGradientBoostingRegressor` served as the stand-in for the Phase 5 dry run. It should be treated as disposable, exactly as this clause anticipates: swap it for the real champion at G2 without re-deriving the experiment design.*
- Track B's dashboard reads `results/leaderboard.json`; build it first against a small fake file.

### C.4 Integration gates

| Gate | When | Acceptance check | Status |
|---|---|---|---|
| **G0 Contract freeze** | End of Phase 0.5 | `H` decided. `load_master`, `swap_weather`, `run_walk_forward` signatures agreed. **Identity test:** swapping ERA5 weather into `swap_weather` reproduces the Regime A predictions exactly. | **MET.** `H=2` measured. 22/22 checks; identity exact to 0.000e+00 MW, plus train/serve parity. `python test_contract.py` |
| **G1 Dry run** | A has Phase 2 baselines; B has archive and overlap check (5.1–5.2) | Ridge baseline run through Regime A and Regime B on the real archive, end to end. Row counts and date alignment verified. Fix contract issues now, not later. | **Track B side complete.** The full dual-regime experiment runs end to end on the real archive with weekly refit, both H values, leads 1–3 (`python run_phase5.py`). Row counts and date alignment verified by G0. Formally passes once Track A runs *its* Phase 2 baseline through the same path — which is now a one-line model swap. |
| **G2 Champion hand-off** | A has finished Phase 3 and the intervals | `champion_config.json` loads in B's scripts. Final Phase 5 and the production inference script run on the champion. A's top-5% bias and B's gap numbers use the same definitions. | **Waiting on Track A only.** Both consumers already take `--champion`; the loader is the one stub left (it needs `champion_config.json`'s schema, which Track A defines). Surge definitions live in `config.py` + `src/operational/metrics.py` and follow D4 exactly — reconcile against Track A's `src/evaluation.py` here, preferring theirs. |
| **G3 Release** | Phases 4, 6, 7 done | Clean-clone reproduction: one command rebuilds the leaderboard, operational gap table and figures. Paper tables are generated from `results/` files, not typed by hand. | Pending. |

---

## Phase 0.5 — Pre-flight Audit and Contract Freeze (Shared)

**COMPLETE. Gate G0 is MET.**

- [x] **0.5.1 Count audit.** Resolved: **4,904 rows / 64 columns**, 157 missing days, 1,801 missing values (all legitimate lag warm-up), 0 rows with a missing target. See the Count Audit box above. *(Figures changed again in v4 — see "bugs found" below.)*
- [x] **0.5.2 Publish-time check.** **`H = 2`.** `python measure_publish_time.py --probe` → `results/publish_time.json`. Evening run time chosen: **20:00 IST** (`evening.yml`). Full method and the rejected circular evidence are in Section B.
- [x] **0.5.3 Weather-lag source check.** **Not material** → `results/weather_lag_source_check.json`. Inference-time and training-time lagged weather are bit-identical (same Open-Meteo product), and archive values are never revised. Details in Section B item 5.
- [x] **0.5.4 Contract files.** `config.py` additions, `feature_groups.json`, `src/contract.py` all written; `python test_contract.py` passes 22/22, including the G0 identity test and train/serve parity.

### Two latent bugs the new tests found

Worth recording, because both were invisible to the health check and would have
quietly corrupted Phase 5's headline number.

1. **Weather anomalies inherited demand's gaps.** `build_dataset.py` derived
   `d_*` and `*_vs_roll7` on the merged frame, which is reindexed over
   `master.csv`'s span and so carries NaN on the 157 missing-demand days — even
   though the weather for those days is perfectly well known. `d_t_max` went
   NaN the day after every gap, and the trailing mean behind `t_max_vs_roll7`
   silently averaged fewer than 7 days (`min_periods=3` hid it). Found because
   `swap_weather` computed the same columns from the gap-free weather series
   and G0's identity test caught the disagreement (max diff 0.995 °C).
   Fixed by moving them into `weather_derivatives()`, on the weather series.
   **Recovered 618 cells**: missing values 2,419 → **1,801**, and the weather
   derived columns went from 61 NaN to 1.

2. **`holiday_name` read back as NaN.** The non-holiday sentinel was the literal
   string `"None"`, which is in `pandas.read_csv`'s default NA list — so a plain
   read turned 4,577 real values into missing. Sentinel is now `"No Holiday"`.
   This is why the "missing values" count fell from 6,996 to 2,419 before the
   fix above took it to 1,801.

Two more, both the same shape — **a feature present in training and absent at
serving**, which a model cannot report because it simply stops using the column:

3. **Sources do not publish on the same day.** All three demand-side groups
   were shifted uniformly by `H-1`, but the `sr_*` grid columns come from a
   slice landing a day after the PSP report. Result: **all four `sr_*_lag1`
   columns NaN on the row being predicted.** Worth 204 MW on a test forecast.
   Fixed with `config.H_BY_GROUP` and a shared
   `src.contract.group_offsets()` - duplicating that logic between
   `load_master` and `predict_next_day` is what caused it.
4. **`swap_weather`'s trailing mean excluded the target date.** The 7-day mean
   behind `*_vs_roll7` was computed on the observed index, which at inference
   does not contain tomorrow, so reindexing onto it returned NaN - silently
   dropping three more features.

And `predict_next_day.py` originally hand-rolled the target-day lag shifts,
producing NaN energy and grid lags because there is no row at `target - 1` to
shift from. Fixed by extracting `build_dataset.merged_frame(extend_to=...)` so
inference runs the *real* `add_lags()`. Verified behaviour-preserving -
`master.csv` rebuilds byte-identical.

### The actual fix: close the class, not the instances

Bug 1 was caught by a test. Bugs 3 and 4 were caught **by luck** - someone ran
`predict_next_day.py` by hand and read a warning line. That is not a control,
and the next one would ship. So `test_contract.py` gained two sections that
assert the property directly:

| Check | What it would have caught |
|---|---|
| `grid lag behind demand matches H_BY_GROUP` - compares each source's observed freshness on disk against the declared offsets | bug 3, and any future change in upstream publishing cadence |
| `no feature is NaN on the row being predicted` - builds the real inference row, requires all 46 features populated | bugs 3 and 4 |
| `serving frame exposes the same features as training` | a column appearing or vanishing between the two paths |
| `serving path reproduces training features on a shared date` | inference drifting from `build_dataset` on a historical row |

Verified by reintroducing each bug and confirming the suite goes red: reverting
`H_BY_GROUP` to a uniform shift fails 2 checks; reverting the trailing-mean fix
fails 1. The suite is now **22 checks**.

The freshness half is also wired into `health_check.py`, so the daily CI catches
upstream cadence drift without anyone running the contract test.

## Phase 1 — Exploratory Data Analysis (Track A)

*Not started. Teammate's track — see Master Concept §8 for the (now-corrected) feature taxonomy to work from.*

- [ ] 1.1 Target distribution and outliers — [ ] 1.2 Seasonal and temporal decomposition — [ ] 1.3 Demand–weather response curves — [ ] 1.4 Feature redundancy — [ ] 1.5 Growth diagnostics

**Deliverables:** `notebooks/01_exploratory_data_analysis.ipynb`, `figures/` (300 DPI).

---

## Phase 2 — Evaluation Setup and Baselines (Track A)

*Not started. Teammate's track.*

- [ ] 2.1 Split design — [ ] 2.2 Walk-forward engine (`src/walkforward.py`) — [ ] 2.3 Metrics module (`src/evaluation.py`) — [ ] 2.4 Baselines

**Deliverables:** `src/evaluation.py`, `src/walkforward.py`, `notebooks/02_baseline_models.ipynb`.

---

## Phase 3 — Tree Models and Demand-Growth Handling (Track A)

*Not started formally. Teammate's track — see the D1 status note above for an informal preview Track B produced as a side effect of building the Phase 5 stand-in model.*

- [ ] 3.1 Models — [ ] 3.2 Optuna (D5) — [ ] 3.3 Growth-handling comparison (D1) — [ ] 3.4 Feature ablation — [ ] 3.5 Fixed final protocol

**Deliverables:** `src/models/train_trees.py`, `notebooks/03_gradient_boosted_trees.ipynb`, `models/champion_config.json`, `models/champion.pkl`, `results/leaderboard.json`, `results/walkforward_test_preds.csv`.

---

## Phase 4 — Extreme-Day, Regime-Shift and Uncertainty Analysis (Track A)

*Not started. Teammate's track.*

- [ ] 4.1 Extreme days — [ ] 4.2 Regime shifts — [ ] 4.3 Prediction intervals (D6)

**Deliverables:** `notebooks/04_extreme_days_regimes_uncertainty.ipynb`, `results/intervals.json`.

---

## Phase 5 — Operational Forecast Degradation Experiment (Track B)

**Objective:** measure the penalty when moving from retrospective weather (ERA5) to real day-ahead forecasts.

- [x] **5.1 Build the forecast archive (D2):** for five cities, pull hourly `*_previous_day1..3` variables from the Previous Runs API; aggregate with the **same code** that builds ERA5 features (`build_dataset.state_weather`, load weights, `cdd`, `cdd_max`, `thi`, `hours_above_32`, apparent temperature). *Done — and the archive reaches back to **2021-06-01**, not "January 2024 onward" as originally planned, giving 5.4 years rather than ~2.8. Output: `data/weather_forecast_archive.csv`, 29,535 rows, leads 1/2/3, 5 cities. Not yet carrying a `model` column.*
- [x] **5.2 Validate the archive against the real log** (August 2026 onward): bias, MAE and correlation for `t_max`, `t_mean`, `rh_mean`, `cdd_max`. *Done. Cross-check on the overlap (97 days, lead 1): `t_max` bias +0.15°C, MAE 0.36°C — "consistent" by the pre-agreed rule (MAE under 0.5°C). Lead 2: bias +0.05°C, MAE 0.29°C. The archive is a valid stand-in for the real log. Not yet exported to `results/archive_vs_log_overlap.csv` as its own file — the numbers exist in `forecast_skew.py`'s output, not in that path.* <br><br>**Important finding not anticipated by the plan:** the forecast bias is **not stationary**. Lead-1 `t_mean` bias walked +0.49°C (2021) → +0.53 (2022) → +0.07 (2023) → +0.29 (2024) → −0.26 (2025) → **−0.55 (2026, every month negative)**. Averaged over the whole archive the bias looks nearly zero (+0.08°C); that average would be the wrong number to correct by today. `forecast_skew.py` reports a trailing-90-day estimate specifically because of this. Whoever uses the archive for anything beyond this dry run should read that section before assuming a fixed bias correction is safe.
- [~] **5.3 Dual-regime test on 2025-01-01 onward** (same frozen model, same weekly refit, only the target-day weather swapped through `swap_weather`): **Regime A:** ERA5 actuals. **Regime B:** archived day-ahead forecasts. *Dry-run complete with the `baseline.py` stand-in (differenced GBM, not weekly-refit, not the shared `swap_weather` contract function — `forecast_skew.py` has a private lead-aware substitution instead). Numbers, rolling-origin CV 2023–2026:*

  | weather at scoring time | mean MAE | 2026 bias |
  |---|---|---|
  | reanalysis (Regime A) | 392 MW | −52 MW |
  | forecast, lead 1 (Regime B) | 428 MW | −199 MW |
  | forecast, lead 2 | 437 MW | −211 MW |
  | forecast, lead 3 | 445 MW | −217 MW |

  *Not knowing tomorrow's weather costs ~9% MAE at lead 1, degrading gently with horizon. This is a real, reproducible number (`python forecast_skew.py`) but it is not the final Phase 5.3 result — it uses a stand-in model, not weekly refit, and doesn't yet break out March–June or the top-5% days as the roadmap specifies. Re-run once the champion exists (G2).* Optional Regime A′ and Regime C not attempted.
- [x] **5.4 Sensitivity and failure modes.** Numbers in
  `results/operational_gap.json` → `phase_5_4_sensitivity`, per-day predictions
  in `results/phase5_predictions.csv`.

  Weather error contributes **+179 MW per °C** of day-ahead `t_max` error
  (r=0.43), and the dashboard's weather sandbox is driven by that slope. But
  the headline finding is that **weather is not what goes wrong**: days where
  the temperature forecast missed by more than 1 °C average 541 MW absolute
  error against 522 MW on calm days — a 19 MW difference.

  **9 of the 10 worst days are festival-adjacent**, over-forecast by
  2,300–3,400 MW with `t_max` errors under 1 °C. Three are the Diwali cluster
  (day before / day of / day after); the rest are Pongal, Uzhavar Thirunal,
  Ayudha Puja and May Day. Across the full window, festival-adjacent days
  (n=124) average **782.9 MW** absolute error against **467.3 MW** elsewhere
  (n=552) — 67% worse — and the bias flips sign, **+161.8 MW** versus
  −100.1 MW. The likely cause is a feature gap: `is_holiday` is a single
  binary, but Diwali is a multi-day industrial shutdown exactly like Pongal,
  and Pongal has `pongal_window` while Diwali has no equivalent.

  **Deliberately not fixed in Track B.** Adding a `diwali_window` column
  changes `master.csv`'s schema, and the contract was just frozen at G0 —
  roadmap C.1 says shared files change only by agreement. It belongs in Track
  A's Phase 3.4 ablation where it can be measured rather than assumed.
  `tn_holidays.csv` already resolves Diwali for 2013–2030, so it is a ~10-line
  change plus a rebuild whenever both tracks agree. Flagged in `SYNC.md`.
- [x] **5.5 Real-log gold-standard check.** Runs Regime B on the window our
  live log covers, against the same model and the same Regime A baseline.
  **Labelled a small sample in the artefact itself** — one monsoon / NE-monsoon
  season, no March–June heat — so it validates forecast *fidelity*, not the
  heat-driven gap. Re-run after summer 2027.

**Deliverables — all delivered:** `notebooks/05_operational_forecast_experiment.ipynb` ✅ (reads the committed artefact rather than recomputing, so the notebook and the paper cannot disagree), `data/weather_forecast_archive.csv` ✅, `results/archive_vs_log_overlap.csv` ✅, `results/operational_gap.json` ✅. Runner: `python run_phase5.py` (`--quick` for a fast monthly-refit sanity pass).

**Runtime — and a 10x performance bug found along the way.** The full run is
~690 weekly refits (down from ~1,200: walks are now memoised, so 5.4 and 5.5
reuse 5.3's fits rather than recomputing them, and the idealised `H=1` row runs
at the headline lead only).

More importantly, `HistGradientBoostingRegressor` on this dataset is **ten
times slower on all cores than on one**. Measured, one fit of the production
configuration:

| `OMP_NUM_THREADS` | time per fit |
|---|---|
| 1 | **0.95 s** |
| 2 | 1.66 s |
| 4 | 3.21 s |
| 10 (the default) | 9.68 s |

At ~4,900 rows x 46 features the per-split thread synchronisation costs far
more than the work it distributes. The default turned a ~12-minute job into a
two-hour one, and running the calibration concurrently made both worse.
`threadlimit.py` now pins the thread count to 1 and is imported before numpy in
every compute script; interval calibration went from 15+ minutes to under a
minute. Set `TN_THREADS` to override.

This matters for G3's "one command rebuilds everything": the full Phase 5 run
is now minutes rather than hours, so regenerating the artefact on a clean clone
is realistic. `--quick` (monthly refit, lead 1) remains as a fast sanity pass.

**Still open:** the optional Regime A′ (day-0 analysis, to split "ERA5 vs model analysis" from "lead-time error") and Regime C (retrain on forecasts) variants were not attempted. Neither changes the headline, and both are marked optional in the original plan.

---

## Phase 6 — Interpretability (Track A)

*Not started. Teammate's track.*

- [ ] 6.1 Global TreeSHAP — [ ] 6.2 Local cases — [ ] 6.3 Interactions

**Deliverables:** `notebooks/06_explainability_and_shap.ipynb`, `figures/shap_summary.png`, `figures/shap_heatwave_waterfall.png`.

---

## Phase 7 — Evening Inference, Intervals and Dashboard (Track B)

**COMPLETE.** Both blockers cleared: `H` is measured (0.5.2) and the champion
dependency is handled by the `--champion` flag rather than by waiting.

- [x] **7.1 `predict_next_day.py`.** Builds features through
  `build_dataset.merged_frame(extend_to=target)` and the real `add_lags()` — not
  a reimplementation, so tonight's feature row is constructed identically to the
  rows the model trained on. Applies the `H−1` shift, swaps in the logged
  forecast weather via `swap_weather`, predicts the differenced target and adds
  the anchor back, attaches 90% and 95% intervals from `results/intervals.json`,
  and appends to `data/predictions_log.csv` with issue time, `H`, weather
  provenance and model version. Refuses to issue a forecast if the anchor is
  NaN rather than silently emitting a wrong number.
- [x] **7.2 Evening workflow.** `.github/workflows/evening.yml` at 20:00 IST:
  log evening forecast → score matured forecasts → issue tomorrow's → commit.
  Weekly interval recalibration runs on Sundays in a second job. The 07:00 data
  job is unchanged. **The forecast logger gained an `issue_slot` dimension** —
  the original dedup key `(issued_on, city, date)` meant an evening run would
  overwrite the morning's rows, losing half the data and making the forecast
  actually used at inference indistinguishable from one issued twelve hours
  earlier. Both slots are now kept; inference prefers the evening one.
- [x] **7.3 Dashboard.** `dashboard/app.py` (Streamlit): tomorrow's forecast
  card with intervals and the horizon label, live vs backtested performance as
  separate tabs, the Regime A/B table, and a weather sandbox driven by the
  measured MW-per-°C sensitivity. Reads only committed artefacts and fits
  nothing — if a panel is empty, an upstream job has not run. **No "peak hour
  window" panel**, per D6: there is no hourly demand data, so it was dropped
  rather than faked from a daily series.
- [x] **7.4 Matured-forecast tracking.** `track_predictions.py` fills in
  actuals as Grid-India publishes them, computes live MAE / MAPE / bias and
  realised interval coverage, and writes `results/live_performance.json`. It
  labels itself provisional until at least 20 March–June days are in the
  sample, because the heat season is where the model is hardest pressed.

**Why 7.4 matters more than its size suggests.** Every other number in this
project is a backtest — a reconstruction of what *would* have been known,
assembled by someone who already knows the answer. `predictions_log.csv`
accumulates forecasts committed to disk before the outcome existed. When the
paper quotes an operational MAPE, this is the only table that cannot be
accused of hindsight. It starts empty and becomes the most valuable artefact
here over time, which is an argument for starting the evening cron now rather
than at G2.

**Deliverables:** `predict_next_day.py` ✅, `dashboard/app.py` ✅,
`data/predictions_log.csv` (created on first run), `track_predictions.py` ✅,
`calibrate_intervals.py` ✅, `results/intervals.json` ✅,
`results/live_performance.json` ✅, `.github/workflows/evening.yml` ✅.

**Open at G2:** load Track A's `champion_config.json` in `predict_next_day.py`
and `run_phase5.py` (both already accept `--champion`; only the loader is a
stub, because the schema is Track A's to define).

---

## Phase 8 — Paper, Deck, Packaging (Shared)

*Not started.*

- [ ] 8.1 Paper — [ ] 8.2 Slide deck — [ ] 8.3 Packaging

**Deliverables:** `reports/Tamil_Nadu_Peak_Demand_Forecasting_Paper.pdf`, `presentation/Tamil_Nadu_Peak_Demand_Forecasting_Deck.pptx`, clean repository.

---

## Dependency Overview

```
                    Phase 0 (done, + leakage fix + holiday fix)
                         │
                 Phase 0.5 (shared)  ──►  G0 contract freeze — NOT REACHED
                    0.5.1 done │ 0.5.2/0.5.3/0.5.4 open
         ┌───────────────────┴───────────────────┐
         ▼                                        ▼
   TRACK A (not started)                    TRACK B (this session)
   Ph.1 EDA                                 0.5.2 publish-time check, H  ← next
   Ph.2 baselines + engine                  5.1 forecast archive  — DONE
   Ph.3 models, Optuna, D1 table            5.2 archive vs real log — DONE
     (informal D1 preview: baseline.py)      5.3 dry run with stand-in — DONE (informal)
         │                                        │
         └──────────► G1 dry run ─────────────────┘
                     mechanically proven,
                     not formally passed
   Ph.4 extremes, regimes, intervals         7.x inference + dashboard — blocked on H, G2
         │
         ├──────────► G2 champion hand-off ───────► final Ph.5, final Ph.7
   Ph.6 SHAP
         │
         └──────────────────► G3 release ───── Phase 8 (shared)
```

---

## Milestone Tracker

- [x] **M0:** data pipeline operational, leakage-free, holiday-calendar-correct
- [x] **M0.5 / G0:** audit done ✅ · `H` measured (**H=2**) ✅ · contract frozen ✅ · identity test passes (0.000e+00 MW) ✅
- [ ] **M1:** EDA complete (Track A) — **not started**
- [ ] **M2:** walk-forward engine and baselines built (Track A) — **not started.** A stand-in engine exists at `src/operational/walkforward.py` with the frozen signature; delete it when Track A's lands.
- [ ] **M3:** growth-handling comparison finished, champion chosen (Track A) — **not started.** Informal preview suggests the predict-the-change family; note it must anchor on `peak_lag_2` at H=2, not `peak_lag_1`.
- [x] **M4:** forecast archive built and validated against the real log (Track B) — 5.4 years, cross-check 0.36 °C
- [x] **G1 (Track B half):** full dual-regime experiment runs end to end on the real archive. Formally passes once Track A runs its own baseline through the same path — a one-line model swap.
- [ ] **M5:** extreme-day, regime and interval analysis (Track A) — **not started.** `results/intervals.json` already exists for serving; Track A's Phase 4.3 study supersedes it.
- [ ] **G2:** champion handed over — **waiting on Track A only.** Both consumers take `--champion`; only the loader is a stub.
- [ ] **M6:** SHAP diagnostics (Track A) — **not started**
- [x] **M7:** evening inference and dashboard live (Track B)
- [ ] **G3 / M8:** paper, deck and reproducible repository (Shared) — not started

---

## Where this leaves the project

**Track B is done.** Phases 0.5, 5 and 7 are complete, Gate G0 is met, and the
two artefacts Track A needs (`load_master`, `swap_weather`) are frozen and
tested.

**The critical path is now entirely Track A**, and it has no remaining
dependency on Track B. Phase 1 can start immediately.

Three things Track A should read before opening `master.csv`:

1. **`H = 2`, so `peak_lag_1` is unavailable.** Use
   `load_master()` — not `pd.read_csv("master.csv")`. Anchor differenced
   targets on `peak_lag_2`. The 392 MW baseline figure is an H=1 number.
2. **Quote Regime B, not Regime A.** The reanalysis MAE is not reproducible on
   any real evening; it belongs in feature-selection comparisons where both
   sides use it.
3. **Run `python test_contract.py` after touching feature engineering.** It
   caught two latent bugs that `health_check.py` could not see.

### Verification — the serving path has now actually run

"The code exists" is not "it has run", so each piece was exercised for real:

- **A genuine day-ahead forecast is logged.** `data/predictions_log.csv` holds
  **17,860 MW for 2026-11-12**, issued the evening of 11-11 with 90% PI
  16,677–19,043, anchored on `peak_lag_1` = 16,819 MW (the 11-10 observation).
  Its actual publishes 11-13, so the row is genuinely hindsight-free — which is
  the only reason that file is worth keeping, and why no back-dated prediction
  was ever written into it.
- **The tracker's scoring path is verified** against a synthetic log in `/tmp`
  with known offsets (+100, −200, +300 → MAE 200, bias +66.7): actual backfill,
  MAE/MAPE/bias, interval coverage, `live_performance.json` export and the
  `is_provisional` flag all asserted correct. The real log stays clean; on the
  real data the tracker correctly reports 1 issued / 0 scored / 1 awaiting.
- **The dashboard renders**, verified with `streamlit.testing.AppTest`:
  0 exceptions, 6 metrics, both tabs, the sandbox slider. Two bugs fixed in the
  process — a deprecated `use_container_width` (already past its 2025-12-31
  removal date) and a false "Features missing at inference: nan" warning caused
  by `str(NaN) == "nan"` being truthy, the same trap as the `holiday_name`
  sentinel.
- **`evening.yml` has been executed**, both jobs, in a throwaway clone so the
  real history was untouched: logger → score → issue → commit in order, the
  commit step verified idempotent (no empty commits on a clean tree), and the
  Sunday recalibration gate tested on both branches including forcing the
  `calibrate_intervals.py` path. Cron `30 14 * * *` confirmed to be exactly
  20:00 IST.

**A fourth train/serve bug was caught by the new guard, not by luck.** With the
data moved forward, every `d_*` delta came back NaN on the predicted row: the
deltas need `observed(target − 1)`, which at an evening issue time is *today*,
and `fetch_weather.py update` excludes the current day as partial.
`test_contract.py` failed on it immediately.

`best_known_observed()` fills the tail from the forecast log's **lead-0** row.
That substitute is measured, not assumed: against the archive over 99 lead-0
rows from the 07:00 job (the pessimistic case), `t_max` MAE is **0.31 °C**,
`t_mean` 0.39 °C, `rh_mean` 2.00 points — roughly 55 MW of induced noise at the
measured 180 MW/°C, against a 525 MW MAE.

**The lead cap is the part that matters.** The first version took the newest
`issued_on`, which silently breaks when the logger has been down: a row for
today issued three days earlier is a lead-3 forecast labelled "today's
analysis", and 0.31 °C would not apply to it. Now each day is filled from its
lowest available lead, `MAX_FILL_LEAD = 1` bounds it, the lead used is printed,
and `test_contract.py` asserts it — **22 checks**. Verified by planting a
lead-3 row with identical values: rejected on provenance alone, and the
resulting NaN fails the parity check loudly. A stale fill is worse than a
visible gap. The forecast moved 17,955 → 17,860 MW.

**Still genuinely open on the Track B side**, and honest about it:

- The **champion loader** (`--champion`) is a stub pending Track A's schema.
- Phase 5's **Regime A′ and Regime C** variants were not attempted — both are
  optional in the roadmap, and neither changes the headline.
- `predictions_log.csv` is **empty until the evening cron runs**, so
  `results/live_performance.json` has nothing in it yet. Worth starting the
  cron now: live, pre-committed forecasts are the one artefact that cannot be
  backfilled, and the paper is stronger the longer that log has been running.
- Phase 5.5's real-log check covers **one monsoon season with no heat in it**.
  It validates forecast fidelity, not the heat-driven gap. Re-run after summer
  2027.
