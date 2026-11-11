# SYNC — Track A ↔ Track B

Required by roadmap C.1: *"keep a short `SYNC.md` listing decisions and any
contract change."* Append at the top. Anything that changes a shared file
(`config.py`, `feature_groups.json`, `src/contract.py`) belongs here.

---

## 2026-11-11 — Track B complete, contract frozen, G0 met

**Owner:** Track B (operations/data). **Track A has not started.**

### Read these three things before opening `master.csv`

1. **`H = 2`. `peak_lag_1` is not available for a day-ahead forecast.**
   Grid-India publishes day *D*'s peak on *D+1* (never in hand on day *D*
   across 46 observed runs). So use `load_master()`, **not**
   `pd.read_csv("data/master.csv")`:

   ```python
   from src.contract import load_master, model_features
   df = load_master()              # target-date index, target -> `y`
   X  = model_features(df)         # 46 columns; drops weather levels + trend
   ```

   Under `load_master(H)`, **lag numbers are relative to the newest available
   demand day (`target − H`)**, not the target. At H=2, `peak_lag_1` holds the
   observation from `target − 2`. Names were not changed because shifting
   `{1,2,3,7,14,364}` by one collides with itself at 2 and 3;
   `verify_no_leakage()` proves the shift instead of trusting the names.

   **Consequence for D1:** a differenced target must anchor on `peak_lag_2`
   at H=2. `baseline.py`'s 392 MW / 2.38% MAPE anchors on `peak_lag_1` and is
   therefore an **H=1 number** — fine for comparing feature sets, not
   achievable in production.

2. **Run `python test_contract.py` after any feature-engineering change.**
   22 checks, including the G0 identity test and train/serve parity. It
   caught four latent bugs that `health_check.py` could not see (below), and
   three of them were the same failure: a feature present in training and
   missing at serving.

3. **Quote Regime B, not Regime A, for anything operational.** The reanalysis
   MAE is not reproducible on a real evening. Regime A belongs in
   feature-selection comparisons where both sides use it.

### Shared files — now frozen

| File | State |
|---|---|
| `config.py` | `H=2`, `H_IDEALISED=1`, `TUNING_FOLDS`, `TEST_START`, `REFIT_DAYS=7`, `SURGE_QUANTILE=0.05`, `SURGE_LEGACY_MW`, `SUMMER_MONTHS`, `RANDOM_SEED=42`, `RESULTS`/`MODELS`/`FEATURE_GROUPS` paths |
| `feature_groups.json` | all 62 columns in the six agreed groups; verified exhaustive and non-overlapping by G0 |
| `src/contract.py` | `load_master(H)`, `swap_weather(df, weather, observed)`, `verify_no_leakage(df)`, `model_features(df)`, `run_walk_forward(...)` re-export |

### What Track B still owes, and what it is waiting on

- **Owes nothing blocking.** Phases 0.5, 5 and 7 are complete.
- **Waiting on Track A for:** `models/champion_config.json`'s schema. Both
  `predict_next_day.py` and `run_phase5.py` already take `--champion`; only the
  loader is a stub. That is Gate G2.
- `run_walk_forward` is **yours** (`src/walkforward.py`). Track B wrote a
  stand-in at `src/operational/walkforward.py` with the identical frozen
  signature so Phase 5 was not blocked (roadmap C.3). `src.contract` prefers
  yours automatically the moment the module exists — **delete the stand-in then**
  rather than letting two engines drift.
- `src/operational/metrics.py` is **not** `src/evaluation.py`. Its surge rules
  follow D4 exactly so the two agree, but at G2 reconcile them and prefer
  yours — the roadmap requires that "A's top-5% bias and B's gap numbers use
  the same definitions."
- `results/intervals.json` exists because Phase 7.1 needs multipliers to serve
  tonight. Your Phase 4.3 study supersedes it; keep the schema or update
  `src/operational/intervals.py:load()`.

### Contract-relevant bugs found and fixed

1. **Weather anomalies inherited demand's gaps.** `d_*` and `*_vs_roll7` were
   derived on the merged frame, which is NaN on the 157 missing-*demand* days
   even though the *weather* is known. `d_t_max` went NaN after every gap and
   the trailing mean behind `t_max_vs_roll7` silently averaged fewer than 7
   days. Found by G0's identity test (0.995 °C disagreement between
   `build_dataset` and `swap_weather`). Moved into
   `build_dataset.weather_derivatives()`. **Recovered 618 cells**; missing
   values 2,419 → 1,801.
2. **`holiday_name` read back as NaN.** Sentinel was the literal `"None"`,
   which is in `pandas.read_csv`'s default NA list, so a plain read turned
   4,577 real values into missing. Now `"No Holiday"`.
3. **Sources do not publish on the same day.** All three demand-side groups
   were shifted uniformly, but `sr_*` comes from a slice landing a day after
   the PSP report — so **all four `sr_*_lag1` columns were NaN on the row being
   predicted**. Worth 204 MW on a test forecast. Fixed via
   `config.H_BY_GROUP` + a shared `src.contract.group_offsets()`.
4. **`swap_weather`'s trailing mean excluded the target date**, silently
   dropping `t_max_vs_roll7`, `cdd_max_vs_roll7` and `thi_vs_roll7` at
   inference.
5. `predict_next_day.py` hand-rolled the target-day lag shifts and produced
   NaN energy/grid lags. Fixed by extracting
   `build_dataset.merged_frame(extend_to=...)` so inference runs the real
   `add_lags()`. Verified behaviour-preserving: `master.csv` rebuilds
   byte-identical.

**What changed as a result.** Bugs 3 and 4 were found by running inference by
hand, not by a test — so `test_contract.py` now builds the real inference row
and requires all 46 features populated, checks each source's observed freshness
against `H_BY_GROUP`, and verifies the serving path reproduces `build_dataset`
on a shared date. Reintroducing bug 3 fails 2 checks; bug 4 fails 1.
`health_check.py` carries the freshness half so the daily CI catches upstream
cadence drift.

**Bug 6, caught by the guard rather than by luck.** Every `d_*` delta came back
NaN on the predicted row once the data moved forward: the deltas need
`observed(target − 1)`, which at an evening issue time is *today*, and
`fetch_weather.py update` deliberately excludes the current day as partial.
`test_contract.py` failed immediately. Fourth instance of this failure mode,
first one a test found before a human did.

`predict_next_day.best_known_observed()` fills the tail of the observed series
from the forecast log's **lead-0** row — at 20:00 IST today's peak has already
happened, so lead-0 is mostly observation. **Measured** against the archive
over 99 lead-0 rows from the 07:00 job (the pessimistic case, far less of the
day observed at issue time):

| | bias | MAE |
|---|---|---|
| `t_max` | −0.08 °C | **0.31 °C** |
| `t_mean` | −0.38 °C | 0.39 °C |
| `rh_mean` | +1.95 pts | 2.00 pts |

At the measured ~180 MW/°C that is ~55 MW of induced noise on a 525 MW MAE.

**The lead is capped at `MAX_FILL_LEAD = 1`, and that cap is the real fix.** An
earlier version took simply the newest `issued_on`, which breaks when the
logger has been down: a row for today issued three days ago is a lead-3
*forecast* wearing the label "today's analysis", and the accuracy above would
not apply to it at all. Each missing day is now filled from its lowest
available lead, nothing above the cap is used, and the lead used is printed and
asserted in `test_contract.py`. Verified by planting a lead-3 row with
*identical values* — it is rejected on provenance alone, and the resulting
visible NaN fails the parity check loudly, which is the correct outcome. A
stale fill is worse than a visible gap.

**Also: a 10x performance bug.** `HistGradientBoostingRegressor` on ~4,900 rows
is ten times slower on all cores than on one (9.68 s vs 0.95 s per fit) —
thread sync dominates at this size. `threadlimit.py` pins threads to 1 and must
be imported before numpy. Import it in any new compute script.

### Dataset facts that changed since the v2 docs

`master.csv` is **4,904 rows × 64 columns** (not 4,878 × 54), 157 missing days
(not 163), 1,801 missing cells — all legitimate lag warm-up; the target is
never NaN. `tn_energy_gwh` and `tn_shortage_at_peak_mw` are **gone** from
`master.csv` (same-day leak, r=0.971); they survive as `energy_lag_*`. Any
Phase 1 notebook that references them by name will `KeyError`.

### Finding for Track A: the dominant error source is festivals, not weather

Phase 5.4 asked what the forecast gets wrong. The answer is not what the
sensitivity number suggests.

Weather error contributes **+179 MW per °C** of day-ahead `t_max` error
(r=0.43), and days with |t_max error| > 1 °C average **541 MW** absolute error
against **522 MW** on calm days — a 19 MW difference. Weather is a real but
minor driver.

The worst errors are festivals. **9 of the 10 largest misses are
festival-adjacent**, with `t_max` errors mostly under 1 °C, and the model
**over-forecasts** them by 2,300–3,400 MW. Across the full test window,
festival-adjacent days (n=124) average **782.9 MW** absolute error against
**467.3 MW** elsewhere (n=552) — 67% worse — and the bias flips sign,
**+161.8 MW** on festivals versus −100.1 MW on other days:

| date | actual | predicted | error | context |
|---|---|---|---|---|
| 2025-10-20 | 10,756 | 14,020 | +3,264 | **Diwali** |
| 2025-10-21 | 11,627 | 15,066 | +3,439 | day after Diwali |
| 2026-11-08 | 12,550 | 15,084 | +2,534 | **Diwali** |
| 2026-11-07 | 14,837 | 17,561 | +2,724 | day before Diwali |
| 2026-11-09 | 13,897 | 16,646 | +2,749 | day after Diwali |
| 2025-01-14 | 12,629 | 15,916 | +3,287 | **Pongal** |
| 2026-01-16 | 12,475 | 14,744 | +2,269 | Uzhavar Thirunal |
| 2025-10-01 | 14,546 | 16,924 | +2,378 | Ayudha Puja |

**The likely cause is a feature gap.** `is_holiday` is a single binary, but
Diwali is a multi-day industrial shutdown exactly like Pongal — and Pongal has
`pongal_window` while Diwali has nothing equivalent. The Diwali cluster spans
at least three days (before / day / after), all badly over-forecast.

**Deliberately NOT fixed here.** Adding a `diwali_window` column changes
`master.csv`'s schema, and the contract was just frozen at G0; roadmap C.1 says
shared files change only by agreement. This belongs in your Phase 3.4 feature
ablation, where it can be measured rather than assumed. If you want it, say so
and Track B will add it to `build_dataset.calendar_features()` and re-freeze —
it is a ~10-line change plus a rebuild, and `tn_holidays.csv` already resolves
Diwali's date for every year 2013–2030.

Full per-day predictions are in `results/phase5_predictions.csv` so you can
redo this analysis without re-running the experiment.

### Non-stationary forecast bias — do not hard-code a correction

Lead-1 `t_mean` bias walked **+0.49 °C (2021) → −0.55 °C (2026)** as Open-Meteo
changed models. A correction fitted on the full archive would be ~1 °C wrong
today, in the wrong direction. `forecast_skew.py` reports a trailing-90-day
estimate for this reason. None is applied anywhere.

### Artefacts Track A can read

| File | Contents |
|---|---|
| `results/publish_time.json` | the `H` measurement and its evidence |
| `results/weather_lag_source_check.json` | lagged-weather train/serve skew (not material) |
| `results/archive_vs_log_overlap.csv` | archive vs live log (Phase 5.2 gate) |
| `results/operational_gap.json` | Regime A vs B, both `H`, leads 1–3, all subsets, bootstrap CIs |
| `results/intervals.json` | conformal multipliers + realised coverage |
| `results/live_performance.json` | pre-committed forecast performance (empty until the evening cron runs) |
