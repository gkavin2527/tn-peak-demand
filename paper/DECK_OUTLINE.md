# 12-slide deck — Next-Day Peak Demand Forecasting for Tamil Nadu

**Audience:** faculty review / ICAIEA 2026 · **Target:** 12–15 min + questions
**Figures:** all from `figures/`, regenerated from the current champion.

> **The one thing to land.** This is not "we built a model that predicts
> electricity demand." It is **"we measured something the literature usually
> can't see."** If the audience remembers one slide, it should be slide 5.

---

## 1 · Title

Next-Day Peak Electricity Demand Forecasting for Tamil Nadu Using Machine Learning
· author · institution · ICAIEA 2026

**Say:** one sentence — *"We forecast tomorrow's peak load for Tamil Nadu, and
we measured what it costs to use weather forecasts instead of the weather
data most papers evaluate on."*

---

## 2 · Why peak demand, why Tamil Nadu

- Peak demand sets reserve commitment; being wrong is asymmetrically expensive
- TN peak: 17,563 MW (2022) → **21,724 MW (2026)**, +7.8% in 2026 alone
- Strong cooling-driven weather dependence; distinctive festival calendar

**Say:** the target is non-stationary and growing — which later explains why
tree models fail and why a linear trend component matters.

---

## 3 · The constraint nobody budgets for: **H = 2**

- Grid-India publishes day *D*'s peak on *D+1*, ~09:00–11:00 IST
- Across **46 observed runs**, the pipeline *never once* held same-day data
- So an evening-of-*D* forecast for *D+1* has demand only through ***D−1***
- ⇒ **`peak_lag_1` is unusable** — and it is the second-most-predictive column (r = 0.927)

**Say:** *"Training on it produces a model that cannot be run."* Measured, not
assumed — `results/publish_time.json`. This is the slide that shows the work
is operational, not academic.

---

## 4 · Data and leakage control

- 4,905 days × 66 columns, 2013→2026; 0 duplicates, 0 target NaN
- Weather: ERA5 reanalysis **+ a 29,535-row archive of genuine lead-1 forecasts**
- Expanding walk-forward (val 2023/24/25) — never random splits
- 100+ mechanical acceptance checks across 5 suites

**Say:** mention one bug the tests caught — lag columns resolved against the
*raw series* rather than trusted by name, which caught a real weekday
misalignment. Credibility comes from a specific catch, not from a count.

---

## 5 · ⭐ THE RESULT — the three-condition ladder

**Figure:** `fig2_weather_ladder.png`

| condition | MAE |
|---|---|
| no weather | 696.3 |
| reanalysis (Regime A) | 571.6 |
| **genuine lead-1 forecast (Regime B)** | **612.0** |

- Weather buys **125 MW**… but **40 MW is given back** (95% CI [25.9, 54.1])
- **32% of the weather gain is lost** to forecast error — nearly a third,
  and invisible to reanalysis-only evaluation

**Say, slowly:** *"Most papers report the middle bar. We can report the third,
because we archived real forecasts for 5.4 years."* Pause here.

---

## 6 · Does the penalty survive a better model?

**Figure:** `fig3_model_benchmark.png`

- 9 models; **every pure GBM lost to Ridge** (CIs exclude zero)
- Trees can't extrapolate a trending target — leaf averages cannot exceed the training max
- Winner: **Ridge + quantile-GBM residual**, Regime B **497.90 MW**
- Penalty persists: **+54.28 MW** A→B

**Say:** the negative GBM result is a finding, not a failure — and it is *why*
the hybrid is shaped the way it is.

---

## 7 · One honest surprise

- A **quantile** residual head beat the mean-targeting one: 520.37 → **497.90 MW**
- Reason is mundane: **MAE is minimised by the conditional median, not the mean**
- Validated honestly: τ chosen on 2023/24 only → **−29.45 MW on untouched 2025**, CI [−55.1, −3.1]

**Say:** *"We were optimising the wrong functional the whole time."* Good
slide — it shows you understood the metric, not just the model.

---

## 8 · Is any of this real? Two independent checks

**(a) Selection optimism.** One validation set drove 4 selection rounds. We
replayed the *entire* pipeline with 2025 held out:

| | reported | honest | optimism |
|---|---|---|---|
| Regime B | 520.55 | 514.75 | **−5.80 MW** |

Negative ⇒ reported numbers are **conservative**.

**(b) 2026 confirmation**, run **once**: MAE **503.49** vs 497.90 validation (+1.1%).

**Say:** *"We call 2026 a confirmation set, not a blind test, because it was
inspected during development. We won't claim otherwise."* Faculty notice this.

---

## 9 · What drives it

**Figure:** `fig5_feature_importance.png`

| block | attribution | permutation ΔMAE |
|---|---|---|
| weather | 2,428 MW | **+1,739** |
| demand lags | 4,304 MW | +1,093 |

- Attribution says lags dominate; **permutation says weather does**
- Both right: lags are mutually collinear, so shuffling one lets the model recover
- `t_mean` alone: largest single permutation importance (**+991 MW**)

**Say:** this is the slide that answers *"are you sure weather is doing
anything?"* — and it answers it better than a bar chart would.

---

## 10 · Where it fails — say this before you're asked

**Figure:** `fig4_error_by_segment.png`

- **Surge days under-forecast −496 MW** (top 5%) — the expensive direction
- **Diwali over-forecast +1,766 MW** (n=15 — anecdotal, but 3 of the worst 6 days)
- Residual ACF **+0.481** at lag 1 — structure remains
- Festival features fix two thirds of Diwali but don't clear the aggregate gate

**Say:** *"A global correction can't fix a conditional problem."* Owning the
failures here is worth more than one more decimal place.

---

## 11 · Operational use

- Evening cron → forecast + **conformal intervals**
- 2026 coverage **91.1% / 93.9%** vs nominal 90/95 — and out-of-sample, against widths fitted ≤2025
- Serving reads the champion contract from config; 28 checks guard it
- **Honest caveat:** `predictions_log.csv` has 1 row — deployment rests on backtests

**Say:** mention the trap you avoided — serving the champion through the old
path would have silently cost **+245 MW MAE (+56%)** from a lag-convention
mismatch, with no error raised.

---

## 12 · Contribution & next steps

**Contribution:** we quantify the **operational cost of weather-forecast
uncertainty** in next-day peak demand forecasting, under a leakage-controlled,
horizon-aware (H=2) protocol with a 5.4-year archive of genuine lead-1
forecasts.

**Next:** conditional surge correction · festival-specific handling · live
accumulation · multi-state generalisation.

**Do not say:** "we beat state of the art." Claim the **measurement**.

---

## Likely questions — have these ready

| Question | Answer |
|---|---|
| "Why not deep learning?" | ~4,900 rows. And the benchmark already shows 4 tuned GBMs losing to Ridge — more capacity is not the binding constraint. |
| "Is 500 MW good?" | ~2.7% MAPE on a ~18,000 MW peak. Compare to the naive baselines on slide 6, not to other papers' numbers on other grids. |
| "Isn't 2026 a test set?" | No, and we don't call it one — it was inspected during development. The unbiased estimate is the held-out 2025 replay. |
| "Why is the ACF still high?" | Honestly: unmodelled multi-day weather shocks. At H=2 we cannot observe day T−1. It is a known limitation, and it is why every CI uses a moving-block bootstrap. |
| "Could you forecast at 20:00 IST?" | We never verified forecast *issue hour*. The claim is "archived lead-1 aligned to valid date." Don't overclaim this. |
