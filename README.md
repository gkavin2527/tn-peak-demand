# Tamil Nadu Daily Peak Electricity Demand Dataset

> ### Where things are
>
> | | |
> |---|---|
> | **[docs/ANALYSIS.md](docs/ANALYSIS.md)** | **Start here** — complete project analysis: what's done, strong, weak, missing, and what would improve the model |
> | [docs/](docs/README.md) | All project documentation, with an index |
> | [paper/](paper/) | Manuscript, deck and their generators. Self-contained — deletable as a unit |
> | this file | The dataset and pipeline reference |


A daily-updating dataset of Tamil Nadu's peak electricity demand (MW), joined with
energy consumption, weather features, and calendar indicators. Built for next-day peak demand forecasting.

Every column is knowable before the target day begins, so a backtest on this
file is a backtest of a forecast you could actually have made.

Updates automatically every morning at 07:00 IST via GitHub Actions.

---

## The dataset

**`data/master.csv`** — one row per day, ready to model.

| | |
|---|---|
| Rows | 4,904+ |
| Coverage | 2013-01-01 → yesterday, minus 157 missing days (see Gaps) |
| Columns | 64 (62 features) |
| Primary Target | `tn_peak_demand_mw` (Tamil Nadu Peak Demand in MW) |
| Freshness | ~1–2 days behind real time |
| Baseline to beat | 1,044 MW persistence → **520 MW** with the frozen champion at H=2 under real forecast weather (`models/champion_config.json`; see [PHASE6_MODEL_BENCHMARK.md](docs/BENCHMARK.md)) |

```python
import pandas as pd
df = pd.read_csv("data/master.csv", parse_dates=["date"])
```

---

## Columns

### Primary Target

| Column | Description |
|---|---|
| `tn_peak_demand_mw` | **Primary Target:** Tamil Nadu daily peak electricity demand, in MW |

Same-day energy met and shortage-at-peak are **not** columns here. Grid-India
publishes them the morning *after* the day they describe, so they are not
inputs to a next-day forecast — they enter only as `energy_lag_*`. The raw
same-day values are in [`data/demand_daily.csv`](data/demand_daily.csv) and
[`data/peak_demand.csv`](data/peak_demand.csv) if you want them for analysis.

### Weather

Five cities (Chennai, Coimbatore, Madurai, Trichy, Salem) collapsed into one
state series using load-share weights, representing both the interior and coastal regions.

| Column | Description |
|---|---|
| `t_mean`, `t_max`, `t_min` | Temperature, °C |
| `rh_mean` | Relative humidity, % |
| `app_t_mean`, `app_t_max` | Apparent ("feels like") temperature, °C |
| `hours_above_32` | Hours above 32 °C — duration of extreme heat driving daytime and evening cooling demand |
| `cdd` | Cooling degree days, `max(0, t_mean − 25)` |
| `cdd_max` | Cooling degree days on daily max temperature |
| `cdd_apparent` | Cooling degree days on apparent temperature |
| `thi` | Temperature-humidity index |

Weather **levels** turn out to be near-redundant once the autoregressive lags
are present — `peak_roll_mean_7` already encodes how hot this part of the
season is. What carries signal is the **change**, which is what moves peak off
yesterday's value:

| Column | Description |
|---|---|
| `d_t_mean`, `d_t_max`, `d_rh_mean`, `d_app_t_max` | Day-over-day change in temperature and humidity |
| `d_cdd`, `d_cdd_max`, `d_thi` | Day-over-day change in cooling load |
| `t_max_vs_roll7`, `cdd_max_vs_roll7`, `thi_vs_roll7` | Today against the mean of the previous 7 days — heat anomaly, not heat level |

`corr(Δpeak, Δt_mean)` is +0.15, against roughly zero for `t_mean` once the
lags are in. See **Choosing features** for what to do with this.

### Calendar

| Column | Description |
|---|---|
| `dow`, `is_weekend` | Day of week (0 = Monday), weekend flag |
| `dow_sin`, `dow_cos` | Day-of-week cycle. Saturday still runs industry (−364 MW vs a working day); Sunday does not (−2,207 MW), so `is_weekend` alone averages the two into a day type that exists on neither |
| `is_holiday`, `holiday_name` | Tamil Nadu holidays, including Pongal, Uzhavar Thirunal, Puthandu and Maha Shivaratri. Resolved by [`tn_holidays.py`](tn_holidays.py), not by the `holidays` package directly — see **Holiday calendar** |
| `day_before_holiday`, `day_after_holiday` | Shoulder days behaving differently from regular days |
| `pongal_window` | 13–17 January — multi-day industrial shutdown during festival |
| `sin1`, `cos1`, `sin2`, `cos2` | Fourier terms for smooth annual seasonality |
| `trend` | Days since series start, capturing long-term structural demand growth |

### Lagged & Rolling Features

All features are backward-looking and **date-aware**, computed on a continuous daily index:

| Column | Description |
|---|---|
| `peak_lag_1` … `peak_lag_364` | Tamil Nadu peak demand 1, 2, 3, 7, 14, and 364 days back (MW) |
| `energy_roll_std_7`, `peak_roll_std_30`, … | Rolling volatility. `NaN` where the window is short — not `0.0`, which would claim the series was flat rather than unknown |
| `peak_roll_mean_7`, `peak_roll_mean_30` | 7-day and 30-day moving average of peak demand |
| `peak_roll_std_7`, `peak_roll_std_30` | 7-day and 30-day moving standard deviation (volatility) of peak demand |
| `peak_lag_1_diff` | Day-over-day peak change (`peak_lag_1 − peak_lag_2`) |
| `peak_lag_1_vs_week` | Week-over-week peak change (`peak_lag_1 − peak_lag_7`) |
| `energy_lag_1`, `energy_lag_7`, etc. | Daily energy consumption lags (GWh) |
| `energy_roll_mean_7`, `energy_roll_mean_30` | Rolling averages of daily energy consumption (GWh) |
| `cdd_lag1`, `cdd_max_lag1`, `thi_lag1` | Yesterday's heat load |
| `sr_*_lag1` | Southern Region grid context (max demand, wind generation, solar generation, peak shortage) |

---

## No leakage

Every feature on day $T$ is computable **before the target day begins**:

- Peak and energy lags start at $T-1$ (`peak_lag_1`, `energy_lag_1`).
- Rolling windows use `shift(1)` before computing rolling statistics.
- **Every same-day quantity from the Grid-India PSP report is dropped**, not
  just the regional ones — including Tamil Nadu's own energy met and shortage
  at peak. The report lands the morning after the day it covers.
- Lags are computed on a continuous calendar index, so `peak_lag_7` is always
  the day exactly 7 days earlier and never "7 rows earlier".
- Nothing is forward-filled across a gap. A missing day is `NaN`, not the last
  value carried forward under a label that no longer describes it.

Weather is the one same-day input that stays, because at prediction time you
substitute a forecast for it. That substitution is not free — see
**Forecast-time weather** below.

`health_check.py` enforces this: if any column in `config.LEAKY_SAMEDAY`
reappears in `master.csv`, the run fails.

### What this costs, and why it is worth it

An earlier version of this dataset shipped `tn_energy_gwh` — same-day total
energy met — as a feature. It correlates **0.971** with the target, and it is
published after the fact. Rolling-origin backtest, 4 folds, HistGradientBoosting:

| | fold MAE (MW) | mean |
|---|---|---|
| with same-day energy (leaky) | 470 / 394 / 292 / 654 | **452** |
| without it (current) | 564 / 577 / 456 / 765 | **591** |

The leaky number was **24% better than anything achievable in production**. If
you have seen ~450 MW MAE quoted for this dataset, that is the figure being
read off tomorrow's energy report.

---

## Choosing features

All 62 columns are leak-free, but they are not all useful. Dropping the weather
*levels* and keeping only the deltas measurably helps:

```python
import pandas as pd
from config import WEATHER_LEVEL_COLS

df = pd.read_csv("data/master.csv", parse_dates=["date"])
X = df.drop(columns=["date", "tn_peak_demand_mw", "holiday_name",
                     *WEATHER_LEVEL_COLS])
```

Rolling-origin CV, 4 one-year folds ending 2026-11-09, 3 seeds averaged:

| feature set | fold MAE (MW) | mean MAE | MAPE |
|---|---|---|---|
| persistence (`peak_lag_1`) | 775 / 839 / 940 / 889 | 861 | 5.2% |
| all 62 columns | 534 / 568 / 450 / 761 | 578 | 3.34% |
| **minus `WEATHER_LEVEL_COLS`** | 544 / 575 / 438 / 630 | **547** | **3.19%** |

Levels are not harmful so much as redundant, and they give a boosted model
4,900 rows of opportunity to overfit a seasonal shape the Fourier terms
already describe.

---

## Holiday calendar

`is_holiday` comes from [`tn_holidays.py`](tn_holidays.py), which wraps
`holidays.India(subdiv="TN")` and repairs its coverage gaps. The gaps were not
random — lunisolar festivals were resolved in some years and silently absent in
others:

| festival | years the package resolves | filled by curated table |
|---|---|---|
| Maha Shivaratri | 6 of 15 | 9 |
| Ram Navami | 12 of 15 | 3 |
| Janmashtami | 14 of 15 | 1 |

That is worse than a missing row. The same festival was a holiday in 2017 and
an ordinary Tuesday in 2016, so `is_holiday` meant "a holiday the package
happened to know about" and a model could learn the package's coverage pattern
instead of Tamil Nadu's calendar.

**Why the fills are trustworthy.** Every curated date is cross-checked on
import against the package for the years where both have the festival — all 6
package-known Maha Shivaratri dates, all 12 Ram Navami dates and all 14
Janmashtami dates reproduce **exactly**. If a future `holidays` release ever
contradicts the table, `validate()` raises rather than quietly picking a
winner, and `health_check.py` turns the run red.

Result: **23–26 holidays per year, median 25**, against 16–24 before.

```bash
python tn_holidays.py             # validate + per-year coverage report
python tn_holidays.py --export    # rewrite data/tn_holidays.csv
```

The resolved calendar is committed to
[`data/tn_holidays.csv`](data/tn_holidays.csv) and `holidays` is pinned in
`requirements.txt`, so the dataset does not change shape when a dependency
floats. Islamic holidays were never affected — Id-ul-Fitr, Bakrid, Muharram and
Milad-un-Nabi are complete across 2013–2026.

Two caveats worth knowing. `holiday_name` can carry two festivals on one date
(2017: Janmashtami fell on Independence Day), so split on `"; "` if you parse
it. And a curated table has a horizon: it runs to 2027, after which the
package's own coverage takes over again.

---

## Forecast-time weather

`master.csv` is built on Open-Meteo **reanalysis**. In production you have a
**forecast**. Measured over the full 5.4-year archive, state-weighted:

| horizon | `t_mean` bias / MAE | `t_max` bias / MAE | `rh_mean` bias / MAE |
|---|---|---|---|
| h=1 | +0.08 / 0.60 °C | +0.18 / 0.73 °C | +0.5 / 2.8 pts |
| h=2 | +0.07 / 0.66 °C | +0.24 / 0.82 °C | +0.4 / 3.3 pts |
| h=3 | +0.10 / 0.71 °C | +0.32 / 0.94 °C | +0.5 / 3.7 pts |

### The bias is not stationary — this is the part that matters

Averaged over the archive the forecast looks almost unbiased. It isn't. The
bias has **walked about 1 °C** as Open-Meteo upgraded its models and as the
reanalysis it is scored against moved from ERA5 to ERA5T for recent dates:

| year | lead-1 `t_mean` bias |
|---|---|
| 2021 | +0.49 °C |
| 2022 | +0.53 °C |
| 2023 | +0.07 °C |
| 2024 | +0.29 °C |
| 2025 | −0.26 °C |
| **2026** | **−0.55 °C** — every month negative |

Through 2021–2025 the bias was also strongly seasonal (+0.5 to +0.6 in
April–October, −0.4 in December–March). In 2026 that pattern is gone and every
month runs cold.

**So do not hard-code a bias correction.** A constant fitted on the whole
archive would be roughly 1 °C wrong for current data, in the wrong direction.
`tools/forecast_skew.py` reports a trailing-90-day estimate alongside the
full-archive figure for exactly this reason; re-estimate it on a rolling window
or skip the correction entirely and let the model see raw forecasts.

An earlier version of this README reported a flat "−0.74 °C cold, +3.9 pts wet"
from the 98-day live log. That was the current regime measured on one season,
not a stable property of the forecast — the long archive is what shows the
drift.

Our own live log starts 2026-08-04 — ground truth about this pipeline, but one
season, far too short to backtest. So the repo also carries a real historical
forecast archive.

### The forecast archive

[`data/weather_forecast_archive.csv`](data/weather_forecast_archive.csv) holds
Open-Meteo's **archived past model runs**, resolved by lead time, back to
mid-2021 — roughly 5.5 years against the log's 3 months. Requesting
`temperature_2m_previous_day2` for date *D* returns what the model run two days
before *D* predicted for *D*. These are genuine historical forecasts, not
reanalysis and not synthetic noise added to actuals.

```bash
python fetch_weather.py archive-forecast   # one-time, ~30 requests
python tools/forecast_skew.py                    # bias tables + operational backtest
```

`tools/forecast_skew.py` does three things:

1. Reports bias and MAE per variable, per lead, **per season** — a cold bias
   that only appears in the monsoon matters differently from a constant one.
2. **Cross-checks the archive against our own live log** where they overlap. If
   Open-Meteo's archived runs disagreed with what we logged live, the archive
   would not be a valid stand-in and every number below it would be void. This
   check is the reason to trust the backtest.
3. Runs the **operational backtest**: train on reanalysis, score with forecast
   weather substituted in, as a real morning would. The gap between that and
   `tools/baseline.py` is the honest cost of not knowing tomorrow's weather.

Substitution is lead-aware, which is easy to get wrong. Predicting day *T* at
lead 1 you hold the forecast for *T*, but **yesterday's weather is already
observed** — so `d_t_max = forecast(T) − actual(T−1)`, not forecast minus
forecast. Treating both sides as forecast would overstate the degradation.

### Results

Run `python experiments/run_phase5.py` (~13 min). The cross-check passes — our live log and
Open-Meteo's archived runs agree to **0.36 °C MAE** on `t_max`, under the
0.5 °C threshold fixed before any demand result was looked at, so the archive
is a valid stand-in for forecasts we would actually have held.

Weekly refit, test window 2025-01-01 → 2026-11-10, n=676:

| weather at scoring time | MAE | MAPE | RMSE | bias |
|---|---|---|---|---|
| **H=2** reanalysis (Regime A) | 483.5 | 2.81% | 684 | −3 |
| **H=2** forecast lead 1 (Regime B) | **525.2** | **3.03%** | 713 | −52 |
| H=2 forecast lead 2 | 530.8 | 3.06% | 731 | −51 |
| H=2 forecast lead 3 | 531.3 | 3.06% | 726 | −54 |
| H=1 reanalysis *(idealised)* | 399.6 | 2.30% | 563 | +3 |
| H=1 forecast lead 1 *(idealised)* | 422.8 | 2.42% | 579 | −57 |

**525 MW is the number you can hit tomorrow evening.** Not knowing tomorrow's
weather costs **+42 MW (+8.6%)**, and degrades gently with horizon — lead 3 is
only 6 MW worse than lead 1. Data latency costs more than weather uncertainty:
the H=1→H=2 step is ~84 MW, twice the forecast-weather penalty, which is the
case for caring about `H` at all.

On the top-5% surge days (n=35) the gap widens and the bias is what matters:
**439 → 551 MW MAE, bias −335 → −418 MW.** A model several hundred MW low on
every record-breaking day is operationally dangerous in a way MAE cannot
express.

Quote the H=2 Regime B figure for production. Quote reanalysis only when
comparing feature sets, where both sides use it and the comparison is fair.

### What actually goes wrong — and it is not the weather

Weather forecast error contributes **+179 MW per °C** of day-ahead `t_max`
error (r=0.43). But days where the temperature forecast missed by more than
1 °C average **541 MW** absolute error against **522 MW** on calm days. A 19 MW
difference. Weather is a real but minor driver.

**9 of the 10 worst days are festival-adjacent**, over-forecast by
2,300–3,400 MW with `t_max` errors under 1 °C — three of them the Diwali
cluster (day before, day of, day after), plus Pongal, Uzhavar Thirunal,
Ayudha Puja and May Day. Across the whole test window:

| | n | mean abs error | bias |
|---|---|---|---|
| festival-adjacent | 124 | **782.9 MW** | **+161.8** |
| all other days | 552 | 467.3 MW | −100.1 |

**67% higher error on festival days, and the bias flips sign** — the model
over-forecasts them while slightly under-forecasting everything else.
`is_holiday` is a single binary, but Diwali is a multi-day industrial shutdown
exactly like Pongal — and Pongal has `pongal_window` while Diwali has no
equivalent.

That is a feature gap, not a weather problem, and it is left for Track A's
Phase 3.4 ablation to measure rather than assumed here, because `master.csv`'s
schema is frozen at Gate G0. Per-day predictions are in
`results/phase5_predictions.csv` if you want to redo the diagnosis.

**Caveat:** the previous-runs archive carries `relative_humidity_2m` only from
2024 onward, so `rh_mean` and `thi` bias figures before then are unavailable
(shown as `nan`). Temperature coverage is complete from 2021-06.

The live log still cannot be reconstructed from anything, which is why
`fetch_weather.py forecast` runs first in the daily job and `health_check.py`
fails if it misses a day. The archive is a one-time backfill; the log is
irreplaceable.

---

## Accelerating growth, and why it breaks tree models

2026 peak demand averages **18,987 MW** YTD against 17,256 MW for the same
window in 2025 — **+10.0%**, against a 3.5%/yr CAGR over 2014–2025. All-time
maximum is 21,724 MW on 2026-07-14.

The naive worry is that `trend` is linear and will under-shoot. The real
problem is sharper: **a gradient-boosted tree cannot extrapolate at all.** Its
prediction is a leaf average, so it can never output a value above the largest
it saw in training. Fit one to the level of a series hitting new records and it
under-forecasts every record-breaking day:

| | 2026 holdout MAE | 2026 bias | under-forecast |
|---|---|---|---|
| GBM on the level | 627 MW | **−420 MW** | 75% of days |
| GBM on the level, no `trend` | 583 MW | −282 MW | — |
| **GBM on `y − peak_lag_1`** | **407 MW** | **−52 MW** | — |

Fitting the **difference** removes the trend from the target entirely. The
model only learns "how different is tomorrow from today", which is stationary,
and the growing level is reinstated by adding `peak_lag_1` back. Same features,
same hyperparameters: mean MAE **547 → 392**, a 29% improvement, and the bias
collapses by a factor of 8.

Dropping `trend` helps on top of that. A monotone unbounded feature is the
worst possible input to a tree in an extrapolating regime: every future row
lands in the top bin, so it contributes a constant while crowding out splits
that generalise.

**This is also why no `yoy_growth_*` column ships here.** Differencing already
adapts to the growth regime. Adding explicit growth ratios
(`peak_roll_mean_364`, `yoy_ratio_30`, `peak_lag_364_scaled`) on top measured
*no better* — 392 against 391 MW — and columns that do not earn their place are
how the weather block got bloated in the first place.

If you need the level rather than next-day change — capacity planning, annual
peak — use a model class that extrapolates (linear, or GBM on a linear
residual). Trees are the wrong tool for that question.

---

## Baseline

[`tools/baseline.py`](tools/baseline.py) is a reference model, not a tuned one, so there is
a number to beat that is not accidentally inflated:

```bash
python tools/baseline.py
```

| model | 2023 | 2024 | 2025 | 2026 | mean MAE | MAPE |
|---|---|---|---|---|---|---|
| persistence (`peak_lag_1`) | 775 | 839 | 941 | 889 | 861 | 5.2% |
| 7-day rolling mean | 804 | 894 | 1004 | 873 | 894 | — |
| GBM, level target *(the trap)* | 547 | 573 | 439 | 627 | 547 | 3.19% |
| **GBM, differenced** | **364** | **405** | **391** | **407** | **392** | **2.38%** |

Rolling-origin CV: train on everything up to a cutoff, test the following year,
step forward. No shuffling — that would leak the future.

---

## Sources

- **Peak Demand:** Grid Controller of India (Grid-India, formerly POSOCO) Daily PSP Reports.
- **Daily Energy:** Grid-India daily PSP reports via Robbie Andrew (CICERO).
- **Weather:** Open-Meteo historical reanalysis and forecast APIs.
- **Historical forecasts:** Open-Meteo Previous Runs API — archived past model
  runs resolved by lead time, back to mid-2021.
- **Holidays:** Python `holidays` package (`subdiv="TN"`, pinned), with curated
  gap fills for Maha Shivaratri, Ram Navami and Janmashtami — see
  [`tn_holidays.py`](tn_holidays.py).

---

## Licence, data terms and citation

**Code** is MIT — see [`LICENSE`](LICENSE). **Data is not.** The two are
separate, and the MIT grant does not extend to anything in `data/`.

### Per-source terms

| Source | Covers | Terms | Attribution required |
|---|---|---|---|
| Grid-India (formerly POSOCO) Daily PSP Reports | `tn_peak_demand_mw`, grid columns | Published by the Grid Controller of India for public information. **Redistribution terms are not explicitly stated** on the PSP portal — confirm before republishing the derived files | Yes — cite Grid-India as the source |
| Robbie Andrew / CICERO daily compilation | `tn_energy_mu` and the historical PSP compilation | Research compilation of Grid-India reports; credit the compiler as well as the primary source | Yes |
| Open-Meteo (Historical Weather, Previous Runs) | all weather columns, `data/weather_forecast_archive.csv` | API data offered under **CC-BY-4.0** | Yes — "Weather data by Open-Meteo.com" |
| Copernicus / ECMWF **ERA5** | upstream of Open-Meteo's reanalysis | Copernicus Climate Change Service licence; contains modified Copernicus Climate Change Service information | Yes |
| Python `holidays` | `is_holiday` and the calendar derivatives | MIT | No |

Two consequences worth stating plainly:

- **The weather archive is redistributable, the demand series needs a check.**
  Open-Meteo's CC-BY-4.0 covers the 29,535-row forecast archive. Grid-India's
  PSP terms are not published in a form that clearly permits republishing a
  derived daily series, so `data/peak_demand.csv` and `data/master.csv` should
  be treated as *reproducible from source* rather than as freely
  redistributable. [`tools/backfill_peak_demand.py`](tools/backfill_peak_demand.py) and
  [`fetch_weather.py`](fetch_weather.py) rebuild both from the primaries.
- **Neither source endorses this work.** Any error in the derived dataset is
  this repository's, not theirs.

### Historical backfill provenance

3,852 of the target rows (2013-01-01 → 2024-04-28) come from a published
Zenodo dataset; the remaining 1,057 (2015-01-27 → present) are fetched
directly from Grid-India PSP reports. The split is recorded per row in
`data/peak_demand.csv` under `source`.

> **Zenodo dataset — CC-BY-4.0, attribution required**
>
> Hunt, K. & Bloomfield, H. (2025). *Daily electricity demand data for Indian
> states (2014–2024)*, v1.0. Zenodo.
> DOI: [10.5281/zenodo.14983362](https://doi.org/10.5281/zenodo.14983362)
>
> Funded by NERC (MITRE project NE/W007924/1). Derived in turn from
> Grid-India (POSOCO) reports. The companion paper is Hunt & Bloomfield
> (2025), *Building and explaining data-driven energy demand models for Indian
> states*, Environmental Research: Energy,
> DOI [10.1088/2753-3751/adc7bc](https://doi.org/10.1088/2753-3751/adc7bc).

**Because this source is CC-BY-4.0, those rows are redistributable with
attribution** — which resolves the ambiguity noted in the table above for the
majority of the series. Rows fetched directly from Grid-India remain subject
to whatever terms the PSP portal implies.

> ✅ **Resolved 2026-11-15 by downloading the deposit.** The Zenodo files
> cover **2013-01-01 → 2024-04-28**, exactly matching our backfill range — the
> title's "(2014–2024)" understates the actual coverage. All 269 of our 2013
> rows are present in the source and match it exactly. The attribution was
> correct.

**Which file, and a data-quality note.** Our backfill matches
`corrected_peak_met_MW.csv` (not `peak_met_MW.csv`): 3,852 of 3,852 rows agree
exactly once source duplicates are accounted for.

> ⚠️ **The upstream file contains 55 duplicated dates, 7 of which carry
> conflicting values** (e.g. 2016-05-26 appears as both 14,431 and 11,981 MW).
> `data/peak_demand.csv` is de-duplicated and holds one of the two source
> copies for each. This is a property of the published dataset, not of our
> pipeline, but it means **anyone re-deriving our series from the deposit must
> apply the same de-duplication or they will not reproduce it.**

### Citing this work

```bibtex
@misc{tn_peak_demand_2026,
  author       = {gkavin2527},
  title        = {Next-Day Peak Electricity Demand Forecasting for
                  Tamil Nadu Using Machine Learning},
  year         = {2026},
  howpublished = {\url{https://github.com/gkavin2527/tn-peak-demand}},
  note         = {Dataset: 4,905 daily observations, 2013-01-01 onward,
                  derived from Grid-India PSP reports and Open-Meteo}
}
```

Replace the author field with the full author list and add the venue once the
ICAIEA 2026 submission has an identifier. **Cite the backfill dataset
alongside this work** — CC-BY-4.0 requires it:

```bibtex
@dataset{hunt_bloomfield_2025_data,
  author    = {Hunt, Kieran and Bloomfield, Hannah},
  title     = {Daily electricity demand data for Indian states (2014--2024)},
  year      = {2025}, version = {1.0}, publisher = {Zenodo},
  doi       = {10.5281/zenodo.14983362}
}

@article{hunt_bloomfield_2025_paper,
  author  = {Hunt, Kieran and Bloomfield, Hannah},
  title   = {Building and explaining data-driven energy demand models for
             Indian states},
  journal = {Environmental Research: Energy}, year = {2025},
  doi     = {10.1088/2753-3751/adc7bc}
}
```

---

## Forecasting in production

The dataset is indexed by the day demand was **measured**. A forecaster is
indexed by the day being **predicted**, and the gap between them is `H`.

### H = 2, and why it is the single most important number here

Grid-India publishes day *D*'s peak demand report on *D+1*, around
09:00–11:00 IST. Across 46 observed daily runs — including one at 18:07 — the
pipeline **never once** held same-day data. So a forecast issued on the evening
of day *D* for target *D+1* is built on demand through *D−1*.

**`peak_lag_1` is therefore not available for a genuine day-ahead forecast.**
It is the second-most-predictive column in the file (r = 0.927), and training
on it produces a model that cannot be run. `src/contract.py` shifts the whole
demand/energy/grid block by `H−1` so this cannot happen by accident, and
`tests/test_contract.py` proves it mechanically rather than trusting column names.

```bash
python tools/measure_publish_time.py --probe   # re-measure H from pipeline evidence
```

Measured, not assumed — `results/publish_time.json`. Note the lag names: under
`load_master(H)`, **lag numbers are relative to the newest available demand
day** (`target − H`), not to the target.

#### Two lag conventions exist. They are not interchangeable.

This is the single easiest way to break a forecast silently, so it is worth
being explicit about.

| | `load_master(H=2)` | `load_phase5_data` |
|---|---|---|
| How `H` is enforced | **shifts** the demand block by `H−1` | **excludes** `peak_lag_1`, `energy_lag_1` |
| `peak_lag_2` holds | T−3 | **T−2** (literal) |
| `peak_lag_7` lands on | the wrong weekday | the **same** weekday |
| Used by | `predict_next_day.py`'s differenced stand-in, whose anchor *is* the shifted `peak_lag_1` | every Phase 5 / Phase 6 model, **including the champion** |

Both are correct for their own consumer. The failure mode is mixing them:
serving a literal-lag model from a shifted frame raises no error and returns
an entirely plausible number, while every demand lag is a day stale and the
weekly lags have lost weekday alignment. Backtested on 2025 (n = 352), that
mistake moves the champion's MAE from **438.2 → 683.7 MW (+56%)** — roughly
2.3× the champion's entire measured gain over Ridge.

`predict_next_day.py` therefore takes the representation from
`models/champion_config.json` (`"lag_representation": "literal"`) rather than
inferring it, and refuses to serve a champion from the wrong frame.
[`tests/test_serving.py`](tests/test_serving.py) asserts the served feature row is
identical to `load_phase5_data` for all 26 features and that `peak_lag_N`
resolves against the **raw series** at T−N rather than being trusted by name.

### The contract (Track A ↔ Track B)

```python
from src.contract import load_master, swap_weather, model_features

df = load_master()                      # indexed by target date, target -> `y`
X  = model_features(df)                 # drops weather levels and `trend`
```

| Function | Owner | Purpose |
|---|---|---|
| `load_master(H)` | Track B | target-date index, demand shifted by `H−1` |
| `swap_weather(df, weather)` | Track B | replaces **only** target-day weather |
| `run_walk_forward(...)` | Track A | expanding window, weekly refit |

`feature_groups.json` is the shared column taxonomy. `python tests/test_contract.py`
is Gate G0's acceptance check — **22 checks**, including the identity test that
swapping reanalysis back in reproduces predictions to `0.000e+00 MW`. Run it
after any change to feature engineering.

Five of those checks exist because four separate bugs shipped features that
were present in training and **missing at serving** — the failure mode a model
cannot report, because it just stops using the column. Two were found only
because someone ran inference by hand and read a warning line. The suite now
asserts the property instead: it builds the real inference row and requires all
46 features populated, compares each upstream source's observed freshness
against the horizons declared in `config.H_BY_GROUP`, and checks the serving
path reproduces `build_dataset` on a shared historical date. Reintroducing any
of the bugs turns the suite red.

### Sources don't arrive together

`H` is not one number. `config.H_BY_GROUP` carries a horizon per source:

| Group | Source | Horizon |
|---|---|---|
| `demand`, `energy` | Grid-India PSP report, published *D+1* | **H = 2** |
| `grid` (`sr_*`) | POSOCO/CICERO slice, a *further* day behind | **H = 3** |

Shifting them uniformly left all four `sr_*_lag1` columns NaN on the predicted
row and moved a test forecast by 204 MW. `health_check.py` warns if the
observed lag ever stops matching the declared one.

### Issuing tomorrow's forecast

```bash
python fetch_weather.py forecast --slot=evening   # must run FIRST
python predict_next_day.py                        # -> data/predictions_log.csv
python track_predictions.py                       # score matured forecasts
streamlit run dashboard/app.py
```

Ordering is load-bearing: inference reads the forecast the logger writes. The
`--slot` tag keeps the evening run from overwriting the morning's, and
inference prefers the evening forecast because that is what is actually in hand
at issue time.

Prediction intervals are rolling split-conformal on **relative** residuals,
calibrated separately for March–June and the rest of the year (summer errors
are larger, and a single absolute width would be too wide in winter and too
narrow in May):

```bash
python calibrate_intervals.py     # -> results/intervals.json, weekly
```

Coverage is approximate by construction — a demand series is not exchangeable,
so the conformal guarantee does not transfer exactly. Realised coverage is
measured and reported rather than assumed.

---

## Rebuilding & Automation

Morning job — collect data (`07:00 IST`, `daily.yml`):

```bash
python fetch_weather.py forecast     # irreplaceable - runs first
python fetch_demand.py
python update_peak_demand.py
python fetch_weather.py update
python fetch_weather.py reconcile    # provisional rows -> ERA5
python build_dataset.py
python health_check.py
```

Evening job — issue the forecast (`20:00 IST`, `evening.yml`):

```bash
python fetch_weather.py forecast --slot=evening
python track_predictions.py
python predict_next_day.py
```

One-time setup:

```bash
python fetch_weather.py backfill           # reanalysis history
python fetch_weather.py archive-forecast   # 5.4y historical forecast archive
python tn_holidays.py --export             # resolved holiday calendar
python tools/measure_publish_time.py             # H
python calibrate_intervals.py              # interval multipliers
```

### Files

| File | Purpose |
|---|---|
| `config.py` | Cities, weights, paths, feature groups, **`H` and the modelling contract constants** |
| `feature_groups.json` | Shared column taxonomy (Track A ↔ Track B) |
| `fetch_demand.py` | Downloads and slices the demand data |
| `fetch_weather.py` | Weather history, daily updates, forecast log, forecast archive, ERA5 reconcile |
| `tn_holidays.py` | Tamil Nadu holiday calendar, gap-filled and self-validating |
| `build_dataset.py` | Merge and feature engineering |
| `health_check.py` | Validation, exits non-zero on failure |
| `tools/measure_publish_time.py` | Measures `H` from pipeline evidence |
| `tools/check_weather_lag_source.py` | Train/serve skew on the lagged weather columns |
| `src/contract.py` | `load_master`, `swap_weather` — the frozen contract |
| `src/operational/walkforward.py` | Stand-in walk-forward until Track A ships theirs |
| `src/operational/metrics.py` | Surge subsets, bias, block-bootstrap CIs |
| `src/operational/intervals.py` | Split-conformal calibration |
| `tests/test_contract.py` | **Gate G0 acceptance (22 checks) — run after touching features** |
| `threadlimit.py` | Pins BLAS/OpenMP threads to 1; import before numpy |
| `tools/baseline.py` | Reference model — the number to beat |
| `tools/forecast_skew.py` | Forecast bias tables and drift |
| `experiments/run_phase5.py` | The operational degradation experiment |
| `calibrate_intervals.py` | Writes `results/intervals.json` |
| `predict_next_day.py` | Issues tomorrow's forecast |
| `track_predictions.py` | Scores forecasts once actuals land |
| `dashboard/app.py` | Streamlit dashboard |
| `.github/workflows/daily.yml` | Morning data pipeline |
| `.github/workflows/evening.yml` | Evening forecast + weekly recalibration |

### Generated artefacts

`results/` is committed so the paper's tables come from files rather than
retyped terminal output.

| File | Contents |
|---|---|
| `publish_time.json` | The `H` measurement and its evidence |
| `weather_lag_source_check.json` | Lagged-weather train/serve skew |
| `archive_vs_log_overlap.csv` | Archive vs live log (Phase 5.2 gate) |
| `operational_gap.json` | Regime A vs B, all subsets (Phase 5.3–5.5) |
| `intervals.json` | Conformal multipliers + realised coverage |
| `live_performance.json` | Pre-committed forecast performance |

---

## Automation

Runs at 07:00 IST daily. Fetches new data, rebuilds `master.csv`, runs the
health check, commits. Pull with `git pull`.

`health_check.py` fails the run if the weather falls more than 3 days behind,
`master.csv` drops below 3,000 rows or lags the demand slice it is built from,
the target goes constant or leaves a plausible range, the weather–demand
correlation collapses below 0.2 — the signature of a misaligned join — or **any
same-day column reappears in `master.csv`**, which is how the leak described
above gets reintroduced by accident, or the curated holiday table contradicts
the installed `holidays` package.

Demand staleness is judged against the source rather than the calendar. Past 5
days behind, the check compares our slice to the raw upstream download from the
same run:

| | |
|---|---|
| Upstream has rows we don't | **fail** — published data isn't landing, so the slice or the commit is broken |
| Upstream is level with us | **warn** — Grid India slips over weekends and national holidays, and the rows arrive when it catches up |
| Level with us for over 14 days | **fail** — too long to still be "running late"; assume the source has moved or died |

A third party's publishing schedule shouldn't turn the run red every morning —
that only teaches you to ignore a red run. What should turn it red is data we
could have collected and didn't.

Run it locally any time:

```bash
python health_check.py
```

GitHub disables scheduled workflows after 60 days of repository inactivity. If
updates stop, check that first.

---


### Dependency tiers, and why the runner ran out of disk

The CI runner is **self-hosted** and persistent, so disk is a finite shared
resource rather than a fresh allocation per run. Both jobs used to recreate
`.venv` from scratch nightly and install the **entire** `requirements.txt`.

On **2026-11-12** that filled the disk — `[Errno 28] No space left on device`,
0 MB free — and the pipeline stayed down for four days. The morning job's
weather-forecast log is the one artifact that **cannot be backfilled**, so
those rows are permanently lost.

The daily pipeline never imported most of what it was installing. Measured by
tracing transitive imports from each job's entry points:

| Tier | File | Packages | Installed size | Used by |
|---|---|---|---|---|
| data | `requirements-data.txt` | 7 direct / 21 total | **195 MB** | `daily.yml` |
| model | `requirements-model.txt` | +3 (scipy, scikit-learn, lightgbm) | **355 MB** | `evening.yml` |
| full | `requirements.txt` | +5 (xgboost, catboost, optuna, matplotlib, streamlit) | **855 MB** | benchmark, tuning, figures, dashboard |

So the morning job was installing 855 MB nightly to do 195 MB of work, and the
evening job 855 MB to do 355 MB. Three changes, all in the workflows:

1. **Install the narrowest tier the job actually imports.** `daily.yml` →
   data, `evening.yml` → model. The evening job can skip XGBoost, CatBoost and
   Optuna because `src/models.py` imports those *inside* their factory
   functions, so serving `ridge_gbm_hybrid` never touches them.
2. **Reuse the venv** unless its tier file's SHA-256 changes, instead of
   rebuilding nightly. A missing `.venv/bin/python` also forces a rebuild, so
   a half-finished install self-heals.
3. **`--no-cache-dir`, plus `pip cache purge` on exit even when the run
   fails** — a failed pip is exactly the state that leaves a cache behind and
   the next run needing the space.

A preflight step now refuses to start below `MIN_FREE_MB` (2,000) with an
actionable message, rather than dying half way through pip and leaving a
corrupt venv behind.

> Two caveats worth keeping in mind. `xlrd` and `openpyxl` stay in the data
> tier even though nothing imports them by name: `pandas.read_excel` loads
> them dynamically by engine, so static import analysis does not see them.
> And if a job ever gains an import, the narrow tier will fail loudly at
> import time rather than silently — which is the intended trade.
## Known quirks

### Gaps

**157 missing days**, not the 13 an earlier version of this README claimed.
They are not evenly spread:

| | |
|---|---|
| 2013-01-08 → 2013-03-30 | **82 consecutive days** — no upstream peak report. Q1 2013 is effectively absent, so "14 continuous years" is not accurate |
| 2013-12-31 → 2014-01-07 | 8 consecutive days |
| everything else | 75 scattered single days, thinning out after 2021 |

Nothing is imputed and nothing is forward-filled, so a lag that crosses a gap
is `NaN` rather than a stale value wearing the wrong label. If you need
complete cases, `df.dropna(subset=["peak_lag_1", "peak_lag_7"])` costs ~90 rows;
if you train with gradient boosting, pass the `NaN`s through.

A row exists because the **target** exists. Six days where peak demand had been
published but the energy series lagged used to be dropped; they are now kept,
with `NaN` energy lags. **The target itself is never `NaN`** — all 4,904 rows
have a real `tn_peak_demand_mw`.

### Missing days vs. missing values — these are not the same number

`df.isna().sum().sum()` on `master.csv` reports **2,419** missing cells, not
157 — that count is cells, not rows, and most rows that touch a gap are
missing in more than one lag column at once. All 2,419 are the same thing: a
lag or rolling-window column during its warm-up period, or crossing one of the
157 gap days above. `peak_lag_364` alone accounts for 423 of them, simply
because there's no day-minus-364 for the first year of the series. None of
this is imputed and none of it is a defect — a tree model passes `NaN` through
natively, and `df.dropna(subset=["peak_lag_1", "peak_lag_7"])` costs ~90 rows
if you need complete cases.

**A previous version of this dataset also showed 4,577 *extra* false
positives here** — `holiday_name` was filled with the literal string `"None"`
on non-holiday rows, which collides with `pandas.read_csv`'s default
NA-string list (`""`, `"None"`, `"NA"`, `"null"`, …). A plain
`pd.read_csv("master.csv")` therefore read 4,577 real values as missing,
inflating the apparent total to 6,996. The sentinel is now `"No Holiday"`,
which round-trips through a default `pd.read_csv` as a string, not a NaN.

### `tn_shortage_at_peak_mw` was removed

Besides being same-day, it was mostly fiction. The PSP reports only carry it
from 2024 onward — 1,052 of 4,904 rows — and the remaining 79% was being
`fillna(0.0)`'d to zero. That stamped "no deficit at peak" across the
2013–2015 rolling-blackout years, which is wrong rather than merely missing.
Even within the populated range only **6 rows are non-zero**. The raw column
is still in `data/peak_demand.csv`, un-filled, if you want to work with the
2024+ slice honestly.

`sr_peak_shortage_mw_lag1` is the usable shortage signal (r = −0.48 with the
target) and is retained.

### `is_holiday` — fixed, but know the horizon

This used to vary by year because the `holidays` package resolved some
lunisolar festivals inconsistently. [`tn_holidays.py`](tn_holidays.py) now
repairs it and the per-year count is stable at 23–26. See **Holiday calendar**.

Two residual limits: the curated table runs to **2027**, after which the
package's own coverage takes over; and `holiday_name` can carry two festivals
on one date, so split on `"; "` if you parse it. `pongal_window` is a fixed
solar date range and was never affected — it remains the most reliable single
indicator of the January industrial shutdown (−4,213 MW).

**November–December 2015 is anomalously low** (~224 GWh against a ~314 series
mean). This is the Chennai floods, a genuine event rather than bad data.

**April–May 2020 is depressed** by the COVID-19 lockdown (~243 GWh mean in
April 2020). Also real. Both periods are regime breaks worth handling
explicitly if you model across them.

**`sr_*_lag1` columns are sparse before 2022.** Drop them if you want complete
cases reaching further back.

**2026 is growing far faster than trend** (+10.0% YoY against a 3.5% CAGR).
This breaks tree models in a specific and fixable way — see **Accelerating
growth** above. Short version: fit `y − peak_lag_1`, not `y`, and drop `trend`.
