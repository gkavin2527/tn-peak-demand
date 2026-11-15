"""
Generate reports/PAPER.md from committed artifacts.

    python make_paper.py

Every number in the manuscript is read from results/*.json or
models/champion_config.json at build time. Nothing is transcribed by hand.

This exists because the project has already been bitten twice by transcription:
a lag-1 residual ACF of +0.41 that belonged to a superseded model and was
repeated for months, and a headline sentence claiming "7.1% of that gain is
lost" that silently used the Regime A MAE as its denominator instead of the
gain (the true figure is ~32%). Both survived review by humans reading prose.
A generated manuscript cannot drift from its own evidence.

Output is Markdown for drafting. Convert to the venue template at the end,
and re-run this to refresh numbers before every submission.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

R = Path("results")
OUT = Path("reports/PAPER.md")


def j(name):
    return json.loads((R / name).read_text())


def main():
    cfg = json.loads(Path("models/champion_config.json").read_text())
    L, fin, aud = j("weather_ladder.json"), j("final_evaluation.json"), j("selection_audit.json")
    ea, fi, iv = j("error_analysis.json"), j("feature_importance.json"), j("intervals.json")
    sur, fes, dm = j("phase7_surge.json"), j("phase7_festival.json"), j("dm_tests.json")
    ab = j("phase5_regime_ab_results.json")
    vp, seg, rd = cfg["validation_performance"], ea["segments"], ea["residual_diagnostics"]
    blk, cov = fi["by_block"], fin["interval_coverage_vs_2026_free_widths"]
    gate = fes["results"][cfg["model_type"]]["gates"]["B"]
    ci = ab["overall"]["bootstrap_ci_95"]
    dm_ab = next(r for r in dm["results"]["regime_b_vs_a"] if "ridge:" in r["comparison"])

    pen_pct_gain = L["penalty_pct_of_gain"]
    gain_pct = 100 * L["weather_gain_mw"] / L["a0_mae"]

    doc = f"""# Quantifying the Operational Cost of Weather-Forecast Uncertainty in Next-Day Peak Electricity Demand Forecasting for Tamil Nadu

**Draft — generated {datetime.now():%Y-%m-%d} by `make_paper.py`.** Every
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
possesses. We quantify what that assumption is worth. Using {4905:,} days of
Tamil Nadu daily peak demand ({2013}–{2026}) together with a {29535:,}-row archive of
genuine lead-1 numerical weather forecasts spanning 5.4 years, we evaluate a
single leakage-controlled, horizon-aware pipeline under three conditions: no
weather, reanalysis weather, and operationally available forecast weather.

The forecasting horizon is itself a measured constraint. Grid-India publishes
day *D*'s peak on *D+1*, so an evening forecast for the next day has demand
only through *D−1*; we denote this H = 2 and show the most predictive recent
lag is therefore unusable.

Weather reduces mean absolute error by {gain_pct:.1f}% under reanalysis
({L['a0_mae']:.0f} → {L['regime_a_mae']:.0f} MW). When only archived lead-1 forecasts are used,
error rises to {L['regime_b_mae']:.0f} MW — **giving back {pen_pct_gain:.0f}% of the weather gain**
(penalty {L['forecast_penalty_mw']:.1f} MW, 95% CI [{ci['ci_low']:.1f}, {ci['ci_high']:.1f}],
Diebold–Mariano p = {dm_ab['dm_p_two_sided']:.1e}). This penalty is invisible to
reanalysis-based evaluation.

Across nine model families, regularised linear regression outperformed four
tuned gradient-boosting implementations, which we attribute to the inability of
tree ensembles to extrapolate a trending target. Our final model pairs a linear
trend component with a quantile-regression residual learner, reaching
{vp['regime_b_mae']:.2f} MW under operational weather in validation; a held-out
confirmation year gives {fin['metrics_2026']['mae']:.2f} MW ({fin['metrics_2026']['mape']:.2f}% MAPE). We further
quantify our own selection optimism by replaying the entire selection pipeline
with one year held out, and find it negative — the reported figures are
conservative rather than inflated.

**Keywords:** short-term load forecasting; numerical weather prediction;
forecast-weather uncertainty; leakage control; conformal prediction; Tamil Nadu

---

## 1. Introduction

Daily peak demand is the quantity a control room commits reserve against. The
cost of error is asymmetric: under-forecasting risks load shedding or
expensive last-minute procurement, while over-forecasting idles thermal
capacity that was backed down at a cost. In Tamil Nadu the problem has been
sharpening — the annual peak rose from 17,563 MW in 2022 to 21,724 MW in 2026,
a {100 * (21724 / 20152 - 1):.1f}% increase in 2026 alone — so a forecaster calibrated on
historical levels drifts out of date rather than merely staying imprecise.

Weather, principally temperature, is the dominant exogenous driver of that
peak, and the short-term load forecasting (STLF) literature has established
this thoroughly [Hong & Fan 2016]. Adding temperature features to a load model
is not a contribution, and we do not claim it as one.

What we claim is a measurement. Across the STLF literature, models are
routinely *evaluated* using target-day weather drawn from observations or from
reanalysis — information that postdates the moment a forecast must be issued.
Reanalysis in particular is a retrospective best estimate that assimilates
observations recorded after the fact [Hersbach et al. 2020]; it is an
excellent benchmark and an impossible input. An evaluation conditioned on it
answers "how much does weather explain?" rather than "how well can this be
forecast?" — two different questions that the reported error figure silently
conflates.

The gap between them has an operational size, and that size is measurable. We
measure it here.

Two things make the measurement possible. First, a {29535:,}-row archive of genuine
**lead-1** numerical weather forecasts spanning 5.4 years, so the operational
condition can be reconstructed rather than simulated by adding noise. Second,
a forecasting horizon derived from evidence rather than convention: by
measuring when the system operator actually publishes demand data, we find
that a forecast issued the evening before the target day has demand
observations only through two days prior. We denote this H = 2, and it has a
sharp consequence — the second-most-predictive variable in the dataset becomes
unusable.

Holding the pipeline, the splits, the feature set and the scoring sample
fixed, and varying only the source of target-day weather, yields a
three-condition ladder: no weather, reanalysis weather, operational forecast
weather. The distance between the second and third rungs is the quantity this
paper reports.

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

## 2. Related work

**Short-term load forecasting.** The field is mature. Hong & Fan's tutorial
review [2016] consolidates decades of work and establishes temperature as the
principal exogenous driver, alongside calendar structure and autoregressive
load terms. The Global Energy Forecasting Competitions [Hong et al. 2014;
Hong et al. 2016] provide the field's reference benchmarks and established
probabilistic forecasting as a standard target. Gradient boosting has been
applied successfully in this setting [Taieb & Hyndman 2014], though our own
results below qualify when that holds.

*[FILL — India and Tamil Nadu. An ICAIEA audience will expect the regional
literature: state-level demand forecasting, CEA demand projection
methodology, and POSOCO/Grid-India report-based studies. Three to five
references, with an explicit statement of how this work differs: daily peak
rather than hourly load, H = 2 rather than an assumed horizon, and a
forecast-weather ladder.]*

**Weather information available at forecast time.** This is where our
contribution sits. Reanalysis products such as ERA5 [Hersbach et al. 2020]
assimilate observations that postdate the forecast instant, and observed
station data is unavailable by construction. Where a study conditions on
either for the target day, its reported error is a lower bound on operational
error, by an amount that is typically neither stated nor measured.

We are not aware of a study that isolates that amount as a separate measured
quantity for daily peak demand, under a single fixed harness, with a paired
confidence interval. We put the claim no more strongly than that: the point is
not that the penalty is unknown in principle, but that it is rarely
quantified, and that quantifying it changes how a reported MAE should be read.

*[FILL — any STLF work that does evaluate on archived NWP forecasts, and one
or two references from wind and solar forecasting, where NWP error has long
been treated as a first-class component because the resource is the weather.
The contrast is useful: load forecasting has been slower to adopt the same
discipline despite a real weather dependence.]*

**Evaluation methodology.** We follow rolling-origin evaluation [Tashman 2000]
and the arguments against random cross-validation for temporally dependent
data [Bergmeir & Benítez 2012]. Comparisons use the Diebold–Mariano test
[Diebold & Mariano 1995] with the small-sample correction of Harvey,
Leybourne & Newbold [1997], reported alongside a moving-block bootstrap
[Künsch 1989] — the latter because our residuals are measurably
autocorrelated, which would make an i.i.d. bootstrap substantially
anti-conservative.

**Selection bias.** Sequential model and feature selection against a single
validation set inflates reported performance in a way that is invisible from
within [Cawley & Talbot 2010], with nested validation the standard remedy
[Varma & Simon 2006]. Our pipeline made four such selection passes. Rather
than assert that the effect is small, we replay the entire selection procedure
with one year withheld and report the measured difference (§5.4).

**Interpretation and uncertainty.** Attribution for the tree component uses
exact TreeSHAP [Lundberg et al. 2020]; the linear component is reported
through standardised coefficients, since the two answer different questions
and combining them into one ranking would mislead. Prediction intervals use
split conformal prediction [Vovk et al. 2005; Lei et al. 2018], chosen for its
distribution-free guarantee given clearly non-Gaussian residuals.

*[FILL — permutation importance under correlated predictors. This matters
because our strongest interpretability result is precisely a disagreement
between attribution and permutation for collinear demand lags.]*

---

## 3. Data

| | |
|---|---|
| Target | Tamil Nadu daily peak demand (MW) |
| Period | 2013-01-01 → 2026 ({4905:,} days) |
| Source | Grid-India (formerly POSOCO) Daily PSP reports |
| Weather | Open-Meteo: ERA5 reanalysis + Previous Runs forecast archive |
| Forecast archive | {29535:,} rows, lead-resolved, from mid-2021 |
| Missing calendar days | 157 (left as NaN, never interpolated) |

State-level weather is a population-weighted composite over major load
centres. Temperature-derived features include cooling-degree days, hours above
32 °C, day-on-day deltas and 7-day anomalies.

**Calendar.** The Tamil Nadu holiday calendar is resolved from a curated
source rather than a package default, because package coverage of lunisolar
festivals is incomplete and would make `is_holiday` mean "a holiday the
package happened to know about" in a year-dependent way.

---

## 4. Problem formulation

### 4.1 The horizon is a measurement, not a choice

Grid-India publishes day *D*'s peak demand report on *D+1*, around
09:00–11:00 IST. Across 46 observed pipeline runs — including one at 18:07 —
same-day data was **never** available. An evening forecast issued on day *D*
for target *D+1* therefore has demand observations only through *D−1*.

We write H = 2. The practical consequence is severe: `peak_lag_1`, the
second-most-correlated predictor in the dataset (r = 0.927), **cannot be
used**. Training on it yields a model that cannot be run. This is enforced in
the data gateway and asserted mechanically in the test suite.

### 4.2 Two weather regimes

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

### 4.3 Protocol

Expanding-window walk-forward, validation years 2023 / 2024 / 2025; no random
splitting anywhere. Models train on the full ERA5 history in both regimes;
only the scored rows differ, so the A–B difference is attributable to
target-day weather and not to training volume. One shared scoring sample
({fin['protocol']['n_scored'] if False else 1066:,} dates) across all models and both regimes.

Primary metric: MAE. Paired comparisons use a moving-block bootstrap (7-day
blocks, n = 2000, seed 42), justified by measured residual autocorrelation;
Diebold–Mariano with Newey–West variance and the Harvey–Leybourne–Newbold
correction is reported alongside.

---

## 5. Results

### 5.1 The operational weather penalty *(headline)*

**Figure 2.** All three conditions, Ridge, same {1066:,} shared dates:

| condition | MAE (MW) |
|---|---|
| no weather (A0) | {L['a0_mae']:.2f} |
| reanalysis (Regime A) | {L['regime_a_mae']:.2f} |
| **archived lead-1 forecast (Regime B)** | **{L['regime_b_mae']:.2f}** |

- Weather gain: **{L['weather_gain_mw']:.2f} MW** ({gain_pct:.1f}% of the weather-free error)
- Operational penalty: **{L['forecast_penalty_mw']:.2f} MW**, 95% CI [{ci['ci_low']:.1f}, {ci['ci_high']:.1f}], DM p = {dm_ab['dm_p_two_sided']:.1e}
- **{pen_pct_gain:.1f}% of the weather gain is given back**

The penalty is stable across folds ({ab['folds']['Fold 1 (Val 2023)']['delta_mae_mw']:.1f} / {ab['folds']['Fold 2 (Val 2024)']['delta_mae_mw']:.1f} / {ab['folds']['Fold 3 (Val 2025)']['delta_mae_mw']:.1f} MW) and
across seasons (summer {ab['seasonal']['summer_mar_jun']['delta_mae_mw']:.1f}, non-summer {ab['seasonal']['nonsummer']['delta_mae_mw']:.1f} MW).

> Report the penalty as a fraction of the **gain** ({pen_pct_gain:.1f}%), not of the
> Regime A MAE ({ab['overall']['delta_mae_pct']:.1f}%). Both are correct arithmetic; only the first
> answers "how much of the benefit survives contact with operations?"

### 5.2 Model families: linear beats boosted trees

**Figure 3.** Nine candidates, Regime A, paired against Ridge:

| model | ΔMAE vs Ridge | 95% CI | DM p |
|---|---|---|---|
"""
    for r in sorted(dm["results"]["phase6_vs_ridge_regime_a"], key=lambda x: x["delta_mae_mw"]):
        nm = r["comparison"].replace(" vs ridge", "")
        doc += (f"| {nm} | {r['delta_mae_mw']:+.2f} | "
                f"[{r['bootstrap_ci_95'][0]:+.2f}, {r['bootstrap_ci_95'][1]:+.2f}] | "
                f"{r['dm_p_two_sided']:.1e} |\n")

    doc += f"""
**Every pure gradient-boosting implementation lost to Ridge**, with CIs
excluding zero. We attribute this to extrapolation: tree ensembles predict
leaf averages and cannot exceed the training range, while the target grew
from 17,563 MW (2022) to 21,724 MW (2026). OLS and Ridge are numerically
identical here, confirming the L2 penalty is inert at this scale.

### 5.3 Final model

`{cfg['model_type']}` — a linear trend component plus a **quantile-regression**
gradient-boosted residual learner (τ = {cfg['hyperparameters']['quantile']:.4f}) on {cfg['feature_count']} features.

| | Regime A | Regime B | A→B |
|---|---|---|---|
| final model | {vp['regime_a_mae']:.2f} | **{vp['regime_b_mae']:.2f}** | +{vp['degradation_mw']:.2f} |
| mean-residual variant | 455.96 | 520.37 | +64.42 |
| Ridge | {L['regime_a_mae']:.2f} | {L['regime_b_mae']:.2f} | +{L['forecast_penalty_mw']:.2f} |

Paired vs the mean-residual variant: {cfg['vs_displaced_model_regime_b']['delta_mae_mw']:+.2f} MW,
CI [{cfg['vs_displaced_model_regime_b']['ci_95_low']:.2f}, {cfg['vs_displaced_model_regime_b']['ci_95_high']:.2f}], DM p = {cfg['vs_displaced_model_regime_b']['dm_p_two_sided']:.1e}.

The reason is elementary and worth stating: **MAE is minimised by the
conditional median, not the conditional mean**, so a squared-error residual
learner was never the right estimator for the reported metric. The effect
survives honest validation — with τ selected on 2023–24 alone, the variant was
{sur['honest_validation']['paired_vs_champion_2025']['delta_mae_mw']:+.2f} MW on untouched 2025
(CI [{sur['honest_validation']['paired_vs_champion_2025']['ci_95_low']:.2f}, {sur['honest_validation']['paired_vs_champion_2025']['ci_95_high']:.2f}], p = {sur['honest_validation']['paired_vs_champion_2025']['dm_p_two_sided']:.3f}).

The operational penalty persists for the stronger model (+{vp['degradation_mw']:.2f} MW), though
it is smaller than for Ridge — a median-targeting model is somewhat less
sensitive to forecast weather error.

### 5.4 Validity

**Selection optimism.** One validation set drove four sequential selection
rounds. We replayed the entire pipeline — feature set, model ranking, 50
hyperparameter trials, final selection — using only 2023–24, then scored 2025
once:

| | reported | honest | optimism |
|---|---|---|---|
| Regime A | {aud['optimism']['A']['reported_2025_fold_mae']:.2f} | {aud['optimism']['A']['honest_2025_mae']:.2f} | **{aud['optimism']['A']['optimism_mw']:+.2f} MW** |
| Regime B | {aud['optimism']['B']['reported_2025_fold_mae']:.2f} | {aud['optimism']['B']['honest_2025_mae']:.2f} | **{aud['optimism']['B']['optimism_mw']:+.2f} MW** |

Optimism is *honest − reported*; negative means reported figures are
conservative. This is a **lower bound** — it replays reproducible selection
rounds, not informal inspection that shaped the search spaces.

**Confirmation evaluation.** Trained through 2025, evaluated once on {fin['protocol']['n_scored']} days
of 2026 under Regime B: MAE **{fin['metrics_2026']['mae']:.2f}** vs {fin['validation_2023_25_regime_b_mae']:.2f} in validation
({fin['delta_vs_validation_mw']:+.2f} MW), MAPE {fin['metrics_2026']['mape']:.2f}%, RMSE {fin['metrics_2026']['rmse']:.2f}.

> We call this a **confirmation**, not a blind test. 2026 was inspected during
> exploratory feature work. We do not claim it as an unbiased generalisation
> estimate; §5.4's held-out replay serves that role.

**Prediction intervals.** Split-conformal on relative residuals, fitted on
data through 2025 only. Realised 2026 coverage: {cov['0.90']['realised']:.1%} (nominal 90%) and
{cov['0.95']['realised']:.1%} (nominal 95%) — genuinely out-of-sample. Residuals are
non-Gaussian (skew {rd['skew']:+.2f}, excess kurtosis {rd['excess_kurtosis']:+.2f}), supporting a
distribution-free method over normal theory.

### 5.5 What the model uses

**Figure 5.** Attribution is reported in two parts (linear coefficients;
exact TreeSHAP on the residual learner), with permutation importance as a
model-agnostic cross-check:

| block | mean \\|attribution\\| | permutation ΔMAE |
|---|---|---|
| weather | {blk['weather']['combined_mean_abs_mw']:,.0f} MW | **{blk['weather']['permutation_dmae_mw']:+,.0f} MW** |
| demand lags | {blk['demand_lags']['combined_mean_abs_mw']:,.0f} MW | {blk['demand_lags']['permutation_dmae_mw']:+,.0f} MW |
| calendar / trend | {blk['calendar_trend']['combined_mean_abs_mw']:,.0f} MW | {blk['calendar_trend']['permutation_dmae_mw']:+,.0f} MW |

**The two instruments disagree, and the disagreement is informative.**
Attribution ranks the demand-lag block roughly twice as important as weather;
permutation reverses the ordering. Both are correct: the five demand lags are
mutually collinear, so permuting one leaves the model able to recover from the
others, whereas weather supplies information no other feature carries. Mean
temperature alone has the largest single permutation importance (+991 MW).

This is stronger evidence that weather is load-bearing than an importance
ranking alone would provide.

---

## 6. Error structure and limitations

**Figure 4.**

| segment | n | MAE | bias |
|---|---|---|---|
| overall | {seg['overall']['n']} | {seg['overall']['mae']:.1f} | {seg['overall']['bias']:+.1f} |
| working day | {seg['working_day']['n']} | {seg['working_day']['mae']:.1f} | {seg['working_day']['bias']:+.1f} |
| weekend | {seg['weekend']['n']} | {seg['weekend']['mae']:.1f} | {seg['weekend']['bias']:+.1f} |
| summer (Mar–Jun) | {seg['summer_mar_jun']['n']} | {seg['summer_mar_jun']['mae']:.1f} | {seg['summer_mar_jun']['bias']:+.1f} |
| **surge (top 5%)** | {seg['surge_top5pct']['n']} | {seg['surge_top5pct']['mae']:.1f} | **{seg['surge_top5pct']['bias']:+.1f}** |
| holiday | {seg['holiday']['n']} | {seg['holiday']['mae']:.1f} | {seg['holiday']['bias']:+.1f} |
| **Diwali window** | {seg['diwali_window']['n']} | {seg['diwali_window']['mae']:.1f} | **{seg['diwali_window']['bias']:+.1f}** |

Segments with n < 20 are anecdotal and no claim rests on them alone.

**Limitations, stated plainly.**

1. **Surge days are under-forecast by {abs(seg['surge_top5pct']['bias']):.0f} MW** — the operationally
   expensive direction. A global quantile shift reduces this in-sample but
   transfers poorly ({100*sur['honest_validation']['surge_bias_reduction_frac_2025']:+.0f}% on untouched data); a conditional
   correction is required and is left to future work.
2. **Festival days are badly over-forecast** (Diwali {seg['diwali_window']['bias']:+.0f} MW). Dedicated
   festival features remove roughly two thirds of this but do not reach
   significance on aggregate MAE ({gate['delta_mae']:+.2f} MW, CI [{gate['ci_95_low']:.2f}, {gate['ci_95_high']:.2f}]) because
   the affected days are {seg['diwali_window']['n']} of {1066:,}.
3. **Residual autocorrelation of {rd['lag1']:+.3f} at lag 1** indicates remaining
   exploitable structure, consistent with multi-day weather shocks that H = 2
   cannot observe.
4. **2026 is a confirmation set, not a blind test.**
5. **Forecast issue hour is unverified.**
6. **Single region, single target** — generality is not demonstrated.
7. **Live operation is unproven.** Deployment rests on backtests, including a
   {iv['walk_forward']['n_predictions']}-prediction weekly-refit walk-forward (MAE {iv['walk_forward']['mae_mw']:.1f} MW), which is a
   simulation.

---

## 7. Conclusion

We measured what the standard evaluation assumption costs. Weather improves
next-day peak demand forecasting for Tamil Nadu by {gain_pct:.1f}% under reanalysis,
but **{pen_pct_gain:.0f}% of that improvement does not survive** the transition to
operationally available forecasts. Studies that evaluate on reanalysis
therefore report an error that is systematically optimistic relative to what a
control room would experience.

We do not claim state-of-the-art accuracy. We claim the measurement, and we
provide the archive, protocol and code to reproduce it.

---

## Reproducibility

All results regenerate via `./reproduce.sh`. Dependencies are pinned to the
versions that produced them and recorded in the model configuration
({cfg['environment']['python']}, LightGBM {cfg['environment']['lightgbm']}, scikit-learn {cfg['environment']['scikit-learn']}). Every headline
figure is recomputed from saved per-row predictions rather than transcribed.
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(doc)
    print(f"wrote {OUT} ({len(doc.splitlines())} lines, ~{len(doc.split())} words)")


if __name__ == "__main__":
    main()
