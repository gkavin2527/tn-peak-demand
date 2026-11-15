# Remediation Plan — fixing every finding in `AUDIT_REPORT.md`

**Created:** 2026-11-15 · **Against:** `phase-6-modeling` @ `871f6dc` · **Audit:** [`AUDIT_REPORT.md`](AUDIT_REPORT.md)

Every item traces to a specific audit finding. Each has an exact file, the change,
effort, an **acceptance criterion** (how you know it's done), and a **verify**
command. Nothing here is speculative — all file paths and line numbers were
checked against the repo on 2026-11-15.

---

## How to use this

Work in **wave order**. Waves 0→1 are dependency-ordered: W1 items change the
feature set or the model, which invalidates figures and intervals, so W0's
measurement items must come first or you will redo them.

**Branch:** create `phase-7-remediation` from `phase-6-modeling`. Do not work on
`main`.

**Standing guardrails** (violating these re-creates audited problems):

| Rule | Why |
|---|---|
| Run `test_contract.py`, `test_phase5_gateway.py`, `test_phase6_benchmark.py` before every commit | 3 suites, 44+ checks; they caught 6 real bugs |
| Never change `PHASE5_FEATURE_SETS["A2_weather_deltas_anomalies"]` in place | Phase 5's gate selected it. **Add `A4_*` as a new key** |
| Assert feature-list **identity**, never length | A count check already let a swapped pair through once |
| 2026 is touched **once**, after the champion is frozen | Already partly contaminated; don't make it worse |
| Primary metric stays **MAE** | Changing it invalidates comparison with committed gates |
| Keep the "issue hour unverified" wording | It is the honest claim |

---

## Dependency graph

```
W0-1 selection optimism ─┐
W0-2 serving hazard      ├─► W1 (features/model change)
W0-3 pin deps            │          │
W0-4 licence             │          ▼
                         │   W2 (figures, SHAP, intervals) ─► W3 (paper, deck)
W0-5 2026 confirmation ◄─┴──────────┘   (AFTER champion re-frozen)
```

**W0-5 moves to the end if you do Wave 1.** Any feature or model change
re-opens selection, so the confirmation run must come after the *final* freeze.
Doing it twice spends a one-shot you do not get back.

---

# Wave 0 — Critical blockers

## W0-1 🔴 Quantify selection optimism

**Finding:** §4.1. The 2023–2025 validation set drove **four** selection rounds
(A2 gate → A3.4 gate → 9-model ranking → 100 Optuna trials), and 2026 is
contaminated via `diagnostic_2026_results` in `experiment_a1/a2/a3_results.json`.
The champion's 520.37 MW is optimistically biased by an **unknown** amount.

**Why it matters:** this is the finding most likely to sink the paper. A referee
will ask "how much of your improvement is selection noise?" and right now there
is no answer.

**File:** new `run_selection_audit.py`

**Change:** replay the *entire* selection pipeline inside an outer fold that
never sees 2025.

```python
# Outer holdout = 2025. Inner selection uses ONLY <= 2024.
# 1. re-run the A1->A2 feature gate on inner folds (val 2023, 2024)
# 2. re-rank the 9 models on inner folds
# 3. re-tune the top-2 with the same 50-trial budget
# 4. score the resulting champion ONCE on 2025
#
# optimism = reported_2025_fold_MAE - honest_2025_MAE
```

Report the gap, not just the number. If it is small (<15 MW) the project's
numbers are robust and you say so with evidence. If large, you report the
honest figure and the paper is *stronger* for having measured it.

**Effort:** 6 h · **Acceptance:** `results/selection_audit.json` contains
`reported_2025_mae`, `honest_2025_mae`, `optimism_mw`, and the inner-selected
feature set + model (which may differ from A2/hybrid — that itself is a result).

**Verify:** `python run_selection_audit.py && python -c "import json;d=json.load(open('results/selection_audit.json'));print(d['optimism_mw'])"`

---

## W0-2 🔴 Fix the champion serving hazard (Gate G2)

**Finding:** §3.5 / `SYNC.md`. **Verified:** `predict_next_day.py:212` trains on
`y − peak_lag_1` and line 243 returns `model.predict(...) + anchor_val`. The
champion (`ridge_gbm_hybrid`) is a **direct-level** model. Serving it through
that path unchanged would **roughly double the forecast** (~16,000 MW added to
a ~16,000 MW prediction).

Second hazard: the champion uses the **26-feature A2 list**, not
`model_features()`'s 46 columns.

**File:** `predict_next_day.py` (lines 55–56, 205–244, 263–278)

**Change:**

```python
cfg = json.loads(Path(args.champion).read_text())
feats = cfg["features"]                       # 26, NOT model_features()
model = build_model(cfg["model_type"])        # src.models
_apply(model, cfg["hyperparameters"])
DIFFERENCED = cfg.get("target_parameterisation", "level") == "differenced"
...
yhat = model.predict(row)[0] + (anchor_val if DIFFERENCED else 0.0)
```

Also add `"target_parameterisation": "level"` to `models/champion_config.json`
so the serving path reads it rather than inferring.

**Effort:** 3 h · **Acceptance:** `python predict_next_day.py --champion
models/champion_config.json --dry-run` emits a forecast within ±3,000 MW of
`peak_lag_1`, and a regression test asserts the level/differenced branch.

**Verify:** add to `test_phase6_benchmark.py`: assert the served prediction for a
known date is within 5% of the benchmark's prediction for that same date.

---

## W0-3 🟡 Pin dependencies and capture environment

**Finding:** §3.6. Only `holidays==0.108` is pinned. `lightgbm`, `xgboost`,
`catboost`, `scikit-learn`, `numpy`, `pandas`, `optuna` all float. The champion
is reconstructed by **re-fitting**, so a LightGBM minor bump can silently move
520.37 MW.

**Files:** `requirements.txt`; `run_phase6_tuning.py` (champion_config writer)

**Change:** `pip freeze` the five modelling libraries into exact pins. Add to
`champion_config.json`:

```json
"environment": {"python": "3.14.7", "numpy": "...", "pandas": "...",
                "scikit-learn": "...", "lightgbm": "..."}
```

**Effort:** 1 h · **Acceptance:** fresh venv from `requirements.txt` reproduces
520.37 MW to 0.01 MW.

**Verify:** `python -m venv /tmp/v && /tmp/v/bin/pip install -r requirements.txt && /tmp/v/bin/python run_phase6_tuning.py --trials 5 --models ridge`

---

## W0-4 🟡 Licence, citation, data terms

**Finding:** §3.7. **Verified:** `README.md` has no licence/DOI/citation text.

**File:** `README.md` (new section before `## Sources`)

**Change:** add code licence (MIT or similar), Zenodo DOI for the historical
peak-demand archive, Grid-India PSP redistribution terms, Open-Meteo attribution
(CC-BY-4.0) and the CICERO/Robbie Andrew credit the config already names.
Add a BibTeX block.

**Effort:** 1 h · **Acceptance:** a reader can determine what they may reuse.

---

## W0-5 🔴 Final 2026 confirmation — **DO LAST**

**Finding:** §1, §5. No `results/final_evaluation.json`.

**File:** new `run_final_evaluation.py`

**Change:** load the frozen champion, train on ≤2025, predict 2026 **once**,
under **Regime B**. Report MAE/RMSE/MAPE/bias overall and on the segments in
W1-5. Call it a **confirmation evaluation** — never a blind test, because 2026
was inspected during Track A diagnostics.

**Effort:** 2 h · **Acceptance:** the artifact exists, carries
`"evaluation_type": "confirmation_not_blind_test"`, and the script refuses to
run twice without `--force`.

> ⚠️ **Only run after W1 is finished and the champion is re-frozen.**

---

# Wave 1 — Accuracy and required analyses

## W1-1 🟢 Diwali + generic festival window — *biggest single win*

**Finding:** §6. **Verified:** holiday MAE **899.9** vs working-day **507.4**
(+77%); three of the worst six days are consecutive Diwalis, over-forecast by
**+4,171 / +3,492 / +3,338 MW**. `pongal_window` exists; Diwali has no
equivalent. **Verified:** `data/tn_holidays.csv` resolves Diwali for **2013–2030
(18 rows)**.

**File:** `build_dataset.py:113` (beside `cal["pongal_window"]`)

```python
from tn_holidays import calendar as tn_calendar
diwali = {d for d, n in tn_calendar(years).items() if "Diwali" in n or "Deepavali" in n}
cal["diwali_window"] = d.dt.date.map(
    lambda x: int(any(abs((x - w).days) <= 2 for w in diwali)))
cal["days_to_holiday"] = ...   # signed distance, clipped to +/-3
```

Then: add `A4_festival` to `PHASE5_FEATURE_SETS` (**new key — do not edit A2**),
re-run the benchmark, and gate A2→A4 with the *same* paired bootstrap used by
the Phase 5 decision gate.

**Effort:** 4 h · **Acceptance:** `results/phase7_festival.json` with a paired
CI. Adopt **only if the CI excludes zero** — same bar as every prior gate.

**Caution:** adding features re-opens selection. Re-run W0-1 after, or report
the A4 gate as an additional selection round.

---

## W1-2 🟢 Surge-day bias

**Finding:** §6. Top-5% surge bias is **−549.0 MW** on n=56 (Regime B) — the
model is systematically low exactly when reserve margin is thinnest.

**File:** `src/models.py` — add `QuantileHybrid` (Ridge + LightGBM
`objective="quantile", alpha=0.9` on the residual), or asymmetric loss.

**Effort:** 4 h · **Acceptance:** surge bias magnitude drops ≥30% (−549 → ≤−385)
**without** overall MAE worsening by >10 MW. State that trade-off in advance.

---

## W1-3 🟢 Recency weighting

**Finding:** §6 item 4. 2026 runs ~10% above 2025; `TrackARidge.fit` already
accepts `sample_weight` (verified, line ~30).

**File:** `src/models.py`, `run_phase6_tuning.py:SPACES`

**Change:** exponential half-life as a tuned hyperparameter (1 / 2 / 3 / 5 yr / none).

**Effort:** 2 h · **Acceptance:** half-life appears in `champion_config.json`.

---

## W1-4 🟡 Diebold–Mariano

**Finding:** §4.5. Moving-block bootstrap is used throughout; **no DM test**.
Forecasting referees expect DM by name.

**File:** `src/evaluation.py` — add `diebold_mariano(e1, e2, h=1, loss="abs")`
with HAC/Newey–West variance.

**Effort:** 2 h · **Acceptance:** DM reported **alongside** (not replacing) the
bootstrap in every comparison table.

---

## W1-5 🟢 Segment error analysis

**Finding:** §5. Only surge/summer exist. Missing: regime changes (2015 Chennai
floods, 2020 COVID, post-COVID), heat waves, long weekends, weekday/weekend,
per-festival.

**File:** new `run_error_analysis.py` → `results/error_analysis.json` + figures.

**Effort:** 3 h · **Acceptance:** a table a referee can read: n, MAE, bias, CI
per segment. Flag any segment with n<20 as anecdotal (Pongal is n=8).

---

## W1-6 🟡 Residual diagnostics on the champion

**Finding:** §5. The +0.41 lag-1 residual ACF was measured on the **old
weather-free Ridge**, not the champion. It is quoted in docs as if current.

**File:** `run_error_analysis.py`

**Effort:** 1 h · **Acceptance:** champion ACF reported; correct or retract the
old figure wherever it appears.

---

# Wave 2 — Paper assets

## W2-1 🔴 Regenerate all figures

**Finding:** §3.2. **Verified:** all five `figures/*.png` were committed in
`01dfc7e`, which predates every weather experiment. They depict the
**weather-free model**; `fig5_feature_coefficients.png` shows coefficients for a
feature set no longer in use. **These cannot go in the paper.**

**File:** new `make_figures.py`, writing to `figures/` (overwrite deliberately)

Minimum set:

| Fig | Content |
|---|---|
| 1 | Actual vs predicted, champion, Regime B, 2025 |
| 2 | **The weather ladder**: A0 695.58 → Regime A 571.61 → Regime B 611.95, with CIs — *the paper's headline* |
| 3 | Model benchmark bar chart with paired CIs vs Ridge |
| 4 | Error by month/segment |
| 5 | SHAP summary (from W2-2) |
| 6 | Residual distribution + ACF |

**Effort:** 3 h · **Acceptance:** every figure regenerable by one command; each
caption names the model and regime.

---

## W2-2 🟢 Interpretability (SHAP)

**Finding:** §3.3, §5. No SHAP anywhere. For a paper claiming weather matters,
this is a predictable referee complaint.

**File:** new `run_interpretability.py`

**Note the structure:** the champion is Ridge + GBM-residual, so attribution has
**two parts** — standardised linear coefficients for the Ridge component, and
TreeSHAP on the LightGBM residual. Report both; do not present TreeSHAP alone as
if it explains the whole model.

**Effort:** 4 h · **Acceptance:** `figures/shap_summary.png` +
`results/feature_importance.json`.

---

## W2-3 🟡 Conformal intervals on the champion

**Finding:** §3.4. **Verified:** `calibrate_intervals.py:17,24,36-39` uses
`HistGradientBoostingRegressor` with `differenced=True, anchor="peak_lag_1"` via
`load_master` — i.e. Track B's **stand-in**, not the champion.

**File:** `calibrate_intervals.py` — switch to `load_phase5_data` + the champion.

**Effort:** 3 h · **Acceptance:** `results/intervals.json` names the champion;
realised coverage within ±3 pts of nominal at 90% and 95%.

---

# Wave 3 — Polish

| ID | Item | File | Effort |
|---|---|---|---|
| W3-1 | Related work, 15+ STLF papers, position the forecast-gap contribution | `reports/` | 8 h |
| W3-2 | One-command reproduction (`make all` / `run_all.sh`) | new | 2 h |
| W3-3 | Start the evening cron; accumulate live forecasts (`predictions_log.csv` has **1** row) | `.github/workflows/evening.yml` | 0 h (wall-clock) |
| W3-4 | 12-slide deck | `presentation/` | 6 h |
| W3-5 | Update `PHASE6_MODEL_BENCHMARK.md`, `SYNC.md`, roadmap, Master Concept with W0–W2 outcomes | `*.md` | 2 h |

---

## Tracking checklist

```
WAVE 0 — blockers
[ ] W0-1  selection optimism quantified      6h  results/selection_audit.json
[ ] W0-2  champion serving hazard fixed      3h  predict_next_day.py
[ ] W0-3  dependencies pinned + env captured 1h  requirements.txt
[ ] W0-4  licence / citation / data terms    1h  README.md
[ ] W0-5  2026 confirmation (LAST)           2h  results/final_evaluation.json

WAVE 1 — accuracy + analyses
[ ] W1-1  Diwali/festival window + A4 gate   4h  build_dataset.py
[ ] W1-2  surge-day bias (quantile head)     4h  src/models.py
[ ] W1-3  recency weighting                  2h  src/models.py
[ ] W1-4  Diebold-Mariano                    2h  src/evaluation.py
[ ] W1-5  segment error analysis             3h  run_error_analysis.py
[ ] W1-6  champion residual ACF              1h  run_error_analysis.py

WAVE 2 — paper assets
[ ] W2-1  regenerate all figures             3h  make_figures.py
[ ] W2-2  SHAP (two-part attribution)        4h  run_interpretability.py
[ ] W2-3  conformal intervals on champion    3h  calibrate_intervals.py

WAVE 3 — polish
[ ] W3-1  related work                       8h
[ ] W3-2  one-command repro                  2h
[ ] W3-3  start evening cron                 0h
[ ] W3-4  deck                               6h
[ ] W3-5  docs refresh                       2h
```

**Totals:** W0 13 h · W1 16 h · W2 10 h · W3 18 h → **~57 h**

---

## Two timelines

### Option A — 1 week (~17 h) · *submission-defensible*

The five from the audit's closing section:

1. **W0-1** selection optimism — 6 h
2. **W0-5** 2026 confirmation — 2 h
3. **W2-1** regenerate figures — 3 h
4. **W1-1** Diwali window — 4 h
5. **W0-3 + W0-4** pins + licence — 2 h

Skips SHAP and intervals. Honest framing: *"interpretability and calibrated
intervals are future work."* Referees accept that; they do not accept an
unquantified selection bias or figures of the wrong model.

> **Ordering caution:** doing W1-1 (features) *after* W0-1 re-opens selection.
> Either run W1-1 first and W0-1 once at the end, or report the A4 gate as an
> additional selection round in the paper. Do not silently ignore it.

### Option B — 3 weeks (~57 h) · *strong paper*

Week 1: W0 (minus W0-5) + W1-1, W1-2, W1-3
Week 2: W1-4, W1-5, W1-6, W2-1, W2-2, W2-3 → re-freeze champion → **W0-5**
Week 3: W3

---

## What NOT to do

| Temptation | Why not |
|---|---|
| Add more model families | The benchmark settled it: 4 GBMs lost to Ridge with CIs excluding zero. More models = more selection noise |
| Deep learning | Out of scope, and ~4,900 rows won't support it. The benchmark *strengthens* this call |
| Edit A2 in place to add festival features | Phase 5's gate selected A2. Add `A4_festival` as a new key |
| Re-run 2026 after each change | It is a one-shot. Freeze first |
| Chase a lower MAE for its own sake | The contribution is the **measured forecast-weather penalty**, not a leaderboard number |
| Drop the "issue hour unverified" caveat | It is correct, and removing it is the kind of overclaim referees punish |

---

## The one-line framing to protect

> **⚠️ CORRECTION (2026-11-15).** The sentence below is wrong and is
> preserved only as the record of what this document originally said.
> "7.1%" is 40.34/571.61 — the penalty as a fraction of the **Regime A
> MAE**, not of the gain. Against the actual gain of 124.66 MW the
> penalty is **32.4%**. See `reports/PAPER.md` §4.1, which now reports
> both denominators explicitly.

> Weather reduces MAE by 17.8% under reanalysis, but **7.1%** of that gain
> (95% CI [25.9, 54.2] MW) is lost when only operationally available lead-1
> forecasts are used — a penalty invisible to the reanalysis-based evaluations
> standard in the literature.

Every item above exists to make that sentence survive a referee. W0-1 is the one
that decides whether it does.
