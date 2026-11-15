# Phase 7 — Remediation Results

**Champion frozen:** 2026-11-15T18:32:08 · **Supersedes:** [`PHASE6_MODEL_BENCHMARK.md`](PHASE6_MODEL_BENCHMARK.md)

Authoritative record of Phase 7. Every number is regenerated from
`results/*.json` by the script named beside it; nothing is transcribed by hand.

---

## 1. Champion

**`quantile_hybrid`** — Ridge trend + LightGBM **quantile** regression
(τ = 0.5906) on the residual, on the frozen 26-feature **A2** set.

| | Regime A | Regime B | A→B |
|---|---|---|---|
| **quantile_hybrid** | 443.62 | **497.90** | +54.28 (12.24%) |
| `ridge_gbm_hybrid` (Phase 6b) | 455.96 | 520.37 | +64.42 |
| `ridge` (tuned) | 571.61 | 611.95 | +40.34 |

Paired vs the model it displaces, Regime B: **-22.47 MW**,
95% CI [-34.49, -9.46], DM p = 5.6e-04.

**Why a quantile head wins** is mundane and worth stating plainly: MAE is
minimised by the conditional **median**, not the mean, so a squared-error
residual head was never the right estimator for the metric being reported —
and the Phase 6b champion carried a −69.6 MW bias that a modest upward shift
corrects at the same time.

*`run_phase7_refreeze.py` → `results/phase7_refreeze.json`*

---

## 2. Validity — the audit's two critical findings, answered

### 2.1 Selection optimism: none detected

The same 2023–25 validation set drove four selection rounds. Replaying the
**entire** pipeline with 2025 held out (feature set → 9-model ranking → 50 TPE
trials → champion pick, all on 2023/2024 only), then scoring 2025 once:

| | reported 2025 | honest 2025 | optimism |
|---|---|---|---|
| Regime A | 438.19 | 421.50 | **-16.69 MW** |
| Regime B | 520.55 | 514.75 | **-5.80 MW** |

Optimism is `honest − reported`, so **negative means the reported figures are
conservative**, not inflated. Paired on 2025: -5.80 MW,
CI [-28.23, 11.37] — spans zero.

Two documented limits: this is a **lower bound** (it replays only rounds
reproducible from committed code, not the informal inspection that shaped the
search spaces), and the A2→A3.4 gate is not replayed because the A3 variants
were never registered.

*`run_selection_audit.py` → `results/selection_audit.json`*

### 2.2 The 2026 confirmation — one-shot, consumed

Trained on 3,982 rows ≤2025-12-31 (Regime A), scored on
314 days of 2026 under Regime B, **once**.

| | 2026 | validation 2023–25 | delta |
|---|---|---|---|
| MAE | **503.49** | 497.90 | +5.59 MW (+1.1%) |
| RMSE | 689.84 | | |
| MAPE | 2.74% | | |
| bias | -135.12 | | |

95% CI on MAE [433.8, 562.6].

> **This is a confirmation set, not a blind test, and must never be described
> as one.** 2026 was inspected during Track A feature diagnostics
> (`diagnostic_2026_results` in `experiment_a1/a2/a3_results.json`). The
> unbiased estimate is §2.1. `run_final_evaluation.py` refuses to overwrite its
> own artifact without `--force`.

**Interval coverage is genuinely out-of-sample.** The operational conformal
widths use a trailing window that now reaches into 2026, so coverage is
reported against a parallel **2026-free** calibration fitted ≤2025-12-31:

| level | realised | gap | mean width |
|---|---|---|---|
| 90% | 91.1% | +1.1 pts | 2,174 MW |
| 95% | 93.9% | -1.1 pts | 2,761 MW |

*`run_final_evaluation.py` → `results/final_evaluation.json`*

---

## 3. Interpretability — the strongest single result

Two-part attribution, because a Ridge + GBM-residual model has no single
number: standardised Ridge coefficients, plus **exact** TreeSHAP via
LightGBM's `pred_contrib` (reconstruction error 3.6e-12 MW — no `shap`
dependency added). Permutation importance as a model-agnostic cross-check.

| block | attribution | permutation ΔMAE |
|---|---|---|
| weather | 2,428 MW | **+1,739** |
| demand lags | 4,304 MW | +1,093 |
| calendar/trend | 1,484 MW | +451 |

**The instruments disagree, and the disagreement is the finding.** Attribution
ranks the demand lags roughly twice as important as weather; permutation
reverses it. Both are right: the five demand lags are mutually collinear, so
shuffling one lets the model recover from the others, while weather carries
information nothing else supplies. `t_mean` alone has the largest single
permutation importance (+991 MW).

This is stronger evidence for the paper's thesis than an importance ranking
alone, and it is the right way to answer *"are you sure weather is doing
anything?"*

Ridge carries the level (sd 1,398 MW); the residual GBM supplies
27% of total sd. A TreeSHAP-only chart would have badly
misrepresented the model.

*`run_interpretability.py` → `results/feature_importance.json`*

---

## 4. What did NOT work — reported, not buried

### 4.1 Recency weighting — rejected

2026 runs +7.8% above 2025, so weighting recent years more is reasonable to
try. Every half-life is worse than uniform, **monotonically**:

| half-life | none | 8 yr | 5 yr | 3 yr | 2 yr | 1 yr |
|---|---|---|---|---|---|---|
| Regime B MAE | 520.37 | +5.65 | +11.65 | +13.07 | +19.47 | +32.48 |

Inner-fold selection agrees. The monotone shape *is* the explanation: that is
what losing effective sample size without gaining information looks like. The
trend is already carried by the `trend` feature and Ridge's linear
extrapolation — the property the hybrid exists to preserve — so down-weighting
history removes data and adds nothing.

*`run_phase7_recency.py` → `results/phase7_recency.json`*

### 4.2 Surge-day bias — reduced, not solved

The quantile head cuts surge |bias| by **31% in-sample** (τ=0.70) but only
**+7%** on untouched 2025 with the inner-selected τ. It is a genuine
**accuracy** win (-29.45 MW on 2025, CI
[-55.11, -3.09], DM p=0.025) but **not** a surge-bias fix.

Current surge bias: **-496.0 MW** on n=56 (validation),
-442.7 on 2026 (n=16, anecdotal).

**A global quantile shift cannot fix a conditional problem.** A real fix needs
a conditional correction — heat-persistence features, or a surge-probability
head. Both would re-open selection and invalidate §2.2, so this is stated as a
limitation and left to future work.

*`run_phase7_surge.py` → `results/phase7_surge.json`*

### 4.3 A4 festival features — rejected at the bar, large segment effect

Re-run against the **current** champion rather than inherited:
**-3.37 MW**, CI [-14.94, 6.91] — spans zero, so A2 is retained.

But on the segment it targets, A4 removes two thirds of the error: Diwali bias
**+1,766 → +605 MW**, MAE 1,802 → 1,275 (n=15).

**An interaction worth naming.** The quantile head made Diwali *worse*
(+1,519 → +1,766 MW) because it shifts predictions up — the wrong direction on
the one festival already over-forecast. The two Phase 7 results pull against
each other. The honest reading: A4 is a targeted festival remedy whose value
cannot show up in an aggregate MAE gate over 15 of 1,066 days.

*`run_phase7_festival.py` → `results/phase7_festival.json`*

---

## 5. Error structure

| segment | n | MAE | bias |
|---|---|---|---|
| overall | 1066 | 497.9 | +17.0 |
| working day | 702 | 474.9 | -61.2 |
| weekend | 308 | 463.6 | +107.6 |
| summer Mar–Jun | 366 | 529.8 | +58.6 |
| **surge top-5%** | 56 | 548.2 | **-496.0** |
| holiday | 70 | 888.3 | +400.6 |
| long weekend | 26 | 998.8 | +551.0 |
| **Diwali window** ⚠ | 15 | 1801.7 | **+1766.4** |

⚠ n < 20 — **anecdotal**, no claim should rest on it alone. Pongal is n=13.

### Residual diagnostics — a correction

> **The lag-1 residual ACF is +0.481, not +0.41.**
> The +0.4102 figure in `notebooks/05_model_diagnostics.ipynb` was measured on
> the **weather-free Ridge** and has been repeated since as if current. The
> champion's autocorrelation is **worse**, and significant at lags
> 1, 2, 3, 4, 5, 6 against a ±0.060 band. That notebook now carries a
> superseded banner. Correct this wherever it appears.

Residuals are clearly non-Gaussian (skew +1.18, excess kurtosis
+4.48), which retrospectively justifies split-conformal intervals over
normal theory, and the moving-block bootstrap over an iid one.

*`run_error_analysis.py` → `results/error_analysis.json`*

---

## 6. Figures

All five previously committed figures depicted the **weather-free A0-era
Ridge** and have been deleted (recoverable at `01dfc7e`). Six replacements,
each caption naming the model **and** the regime — the thing the old ones
lacked that made them actively misleading rather than merely stale.

`make_figures.py` reads only committed `results/*.json` and `*.csv`; it never
refits, so a figure cannot silently disagree with the experiment that produced
it.

---

## 7. Known limitations — state these in the paper

1. **Surge days are under-forecast** (-496 MW on the top 5%), the
   operationally expensive direction. Not solved; needs a conditional correction.
2. **Diwali is badly over-forecast** (+1,766 MW, n=15). A4 fixes two thirds of
   it but does not clear the aggregate gate.
3. **Residual autocorrelation +0.481 at lag 1** — the model leaves
   exploitable structure.
4. **2026 is a confirmation set, not a blind test.**
5. **Forecast issue hour is unverified.** The claim is "archived lead-1 aligned
   to valid date", never "available at 20:00 IST". Keep this wording.
6. **Single region, single target.**
7. **Live operation is unproven**: `data/predictions_log.csv` holds one row.
   Deployment rests on backtests, including a 1016-prediction weekly-refit
   walk-forward (MAE 479.7 MW) — a simulation, not live evidence.
8. **The Zenodo DOI for 3,852 of 4,905 target rows is not recorded anywhere in
   this repository.** This blocks submission and only the authors can resolve it.
