# Next-Day Peak Electricity Demand Forecasting for Tamil Nadu

**Generated 2026-11-15 by `make_deck.py` — every number read from
committed artifacts. Re-run before presenting; do not edit figures by hand.**

12 slides · ~13 min + questions · figures in `figures/`

> **Status**
> - ✅ Citations resolved and Crossref-verified — see `paper/RELATED_WORK.md`
> - ✅ Backfill DOI: 10.5281/zenodo.14983362 (Hunt & Bloomfield 2025, CC-BY-4.0 — **must be attributed**)
> - ⚠️ **Open:** 269 rows dated 2013 attributed to a dataset whose coverage starts 2014
> - ⚠️ **Open:** no independent review by anyone who did not build this

---

## Slide 1 — Title

# Next-Day Peak Electricity Demand Forecasting for Tamil Nadu Using Machine Learning

*Author · Institution · ICAIEA 2026*

**Script.**
Good morning. This project forecasts tomorrow's peak electricity demand for
Tamil Nadu. But the part I actually want to talk about is not the model — it's
something we measured along the way. Almost every paper in this area tests its
model using weather data that a real forecaster would never have. We built the
archive that let us check what that assumption is worth, and the answer is
bigger than I expected.

---

## Slide 2 — Why peak demand, and why this is getting harder

- Peak demand decides how much **reserve capacity** gets committed
- Getting it wrong is expensive **in both directions**
- Tamil Nadu's annual peak: **17,563 MW (2022) → 21,724 MW (2026)**
- Up **7.8%** in 2026 alone — the target is moving

**Script.**
Peak demand is what a control room commits reserve against. If you forecast
too low, you're scrambling for expensive power or shedding load. If you
forecast too high, you've paid to keep plants running that nobody needed. So
both directions cost money, just in different ways.

And this is getting harder, not easier. The annual peak has gone from about
17,563 megawatts in 2022 to 21,724 this year — up 7.8% in 2026 alone. That
matters later, because it turns out some very popular model families simply
cannot handle a target that keeps climbing.

---

## Slide 3 — The constraint nobody budgets for: **H = 2**

- Grid-India publishes day *D*'s peak on **D+1**, around 09:00–11:00 IST
- Across **46 observed runs** — including one at **18:07** — we *never* had same-day data
- So an evening forecast for tomorrow has demand only through **two days ago**
- ⇒ `peak_lag_1` is **unusable** — and it's our second-best predictor (r = 0.927)

**Script.**
Here's the first thing we measured rather than assumed. The system operator
publishes yesterday's peak demand this morning. So if I'm issuing a forecast
tonight for tomorrow, the newest demand number I actually hold is from the day
before yesterday. We call that H equals 2.

We didn't guess this. We logged 46 actual pipeline runs, including one that ran
at 18:07 in the evening, and not once did same-day data exist.

The consequence is painful. Yesterday's peak is the second most predictive
column in our whole dataset — correlation of 0.927. And we cannot use it.
Training on it gives you a model that looks wonderful and cannot be run.

---

## Slide 4 — Data and leakage control

- **4,909 days**, 2013 → 2026 · Grid-India PSP reports
- Weather: ERA5 reanalysis **+ 29,535 archived lead-1 forecasts** (5.4 years)
- Expanding walk-forward, validation 2023 / 2024 / 2025 — **never** random splits
- 100+ mechanical checks across 5 test suites

**Script.**
About 4,909 days of demand data going back to 2013. For weather we have two
things: the reanalysis everyone uses, and — this is the unusual part — an
archive of 29,535 genuine forecasts, stored by how far ahead they were issued.

On validation, we always train on the past and test on the future. Never
random splits. And we wrote a lot of tests. One example of why: our tests
check that a column called "lag 7" actually contains data from seven days
earlier, by going back to the raw series instead of trusting the name. That
caught a real bug where a weekly lag had drifted onto the wrong day of the
week.

---

## Slide 5 — ⭐ The result: what operational weather costs

**Figure:** `fig2_weather_ladder.png`

| condition | MAE |
|---|---|
| no weather | 696.3 MW |
| reanalysis | 571.6 MW |
| **real lead-1 forecast** | **612.0 MW** |

- Weather buys **125 MW** — a **17.9%** improvement
- Operational forecasts **give back 40 MW**
- ⇒ **32% of the weather gain disappears** (95% CI [25.9, 54.1], p = 1e-08)

**Script.**
This is the slide I care about. Same model, same days, same everything —
we only change where the weather comes from.

With no weather at all, error is about 696 megawatts. Add reanalysis weather
and it drops to 572. That's a 17.9% improvement, and that's the number
most papers would report.

Now switch to the weather a forecaster would really have — an actual forecast
issued the day before — and error goes back up to 612.

So weather bought us about 125 megawatts of accuracy, and using real
forecasts hands back 40 of it. That's roughly **32% of the benefit —
nearly a third — gone**, purely because the weather input is a forecast
instead of a perfect retrospective record.

If you only evaluate on reanalysis, you never see this. *(pause here)*

---

## Slide 6 — Does the penalty survive a better model?

**Figure:** `fig3_model_benchmark.png`

- Nine model families, paired comparisons against Ridge
- **Every pure gradient-boosting model lost** — best of them still **+56 MW** worse
- Trees predict leaf averages → **cannot extrapolate** a rising target
- Winner: linear trend **+ boosted residual** (-105 MW vs Ridge)

**Script.**
A fair question at this point is whether a better model makes the problem go
away. It doesn't.

We tried nine model families. Every single pure gradient boosting model —
LightGBM, XGBoost, CatBoost — lost to plain old ridge regression. The best of
them was still about 56 megawatts worse.

The reason is simple once you see it. A tree predicts by averaging training
examples in a leaf. It literally cannot output a number higher than it has
seen before. Our target has been climbing for years. So every summer, the tree
models were structurally incapable of predicting a new record.

What won was a combination: a linear model to carry the trend, with boosting
on top to clean up what's left. That's 105 megawatts better than ridge alone.

---

## Slide 7 — One honest surprise

- Changed only the residual learner's **objective**: mean → **quantile**
- Validation MAE **520.37 → 497.90 MW**
- Why: **MAE is minimised by the median, not the mean** — we were optimising the wrong thing
- Held-out check: τ picked on 2023–24 only → **-29.5 MW on untouched 2025** (CI [-55.1, -3.1])

**Script.**
Here's something I didn't expect, and it's a bit embarrassing in hindsight.

Our model's second stage was trained to predict the *average* leftover error.
But we report our results using mean absolute error — and mean absolute error
is minimised by the **median**, not the mean. We'd been optimising for the
wrong statistic the entire time.

Changing just that one setting took us from 520 megawatts down to 498.

And we checked this honestly. We re-picked the setting using only 2023 and
2024, then tested once on 2025, which had played no part in choosing it. It
still improved things by about 29 megawatts. So it's real, not us fooling
ourselves.

---

## Slide 8 — Is any of this real? Two checks

**(a) Did we fool ourselves by tuning?** Replayed the *entire* selection
pipeline with 2025 held out:

| | reported | honest | difference |
|---|---|---|---|
| Regime B | 520.55 | 514.75 | **-5.80 MW** |

**(b) Held-back year**, run **once** on 2026: **503.49 MW** vs 497.90 expected (2.74% MAPE)

- That gap is **0.12 standard errors** — effective n is 110, not 314, because errors autocorrelate
- Reads as **"not inconsistent"**, *not* "confirmed"

**Script.**
Two ways we tried to catch ourselves cheating.

First — we made four rounds of choices against the same validation years. That
inflates results, and usually nobody measures by how much. So we re-ran the
whole selection process pretending 2025 didn't exist, then tested on 2025
once. It came out **5.8 megawatts better**, so there's no sign we inflated
anything — though I'd put that as *small or negative under our limitations*
rather than *zero*, because the replay still ranks models on default settings
and reuses feature sets that were chosen with 2025 in view.

Second — we froze the model and ran it once on 2026: 503 megawatts against
498 expected, about 2.7% error.

Now, I want to be careful here, because it would be easy to oversell this.
Our errors are correlated from one day to the next, so 314 days is really
worth about 110 independent ones. That gap is **0.12 of a standard error** —
statistically it's nothing. The right way to say it is that the held-back year
is **not inconsistent** with what we expected. It does not independently
confirm anything.

And the honest headline: **we have no genuinely blind evaluation.** We'd
looked at 2026 during development, and our one live forecast used an older
model. I'd rather say that than have someone ask.

---

## Slide 9 — What's actually driving it

**Figure:** `fig5_feature_importance.png`

| block | attribution | **permutation** |
|---|---|---|
| weather | 2,428 MW | **+1,739 MW** |
| demand lags | 4,304 MW | +1,093 MW |

- The two methods **disagree** — and that's the finding
- Lags are **collinear**: shuffle one, the model recovers from the others
- Weather carries information **nothing else supplies**

**Script.**
If someone asks "are you sure weather is doing anything?" — this is the answer.

We measured importance two ways. The first says the demand history matters
about twice as much as weather. The second — where you scramble one column and
see how much worse the model gets — says the opposite.

Both are correct, and the disagreement is the interesting bit. Our demand lags
all say similar things, so if you destroy one, the model just leans on the
others and barely notices. Weather has no backup. Scramble the temperature and
there's nothing to fall back on.

Mean temperature on its own is the single most load-bearing column in the
model. That's a stronger argument for weather mattering than any ranking chart.

---

## Slide 10 — Where it fails

**Figure:** `fig4_error_by_segment.png`

- **Surge days under-forecast by 496 MW** (top 5%) — the expensive direction, and **average MAE hides it**
- **Diwali over-forecast by 1,766 MW** (n=15 — anecdotal, but 3 of the worst 6 days)
- Residual autocorrelation **+0.481** — structure remains
- Festival features fix ~⅔ of Diwali but don't clear the bar (-3.4 MW, CI [-14.9, 6.9])

**Script.**
I want to show you where this breaks before anyone has to ask.

On the hottest, highest-demand days — exactly when you most need it right — we
forecast about 496 megawatts **too low**. That's the dangerous direction,
because that's when reserve margin is thinnest.

At Diwali we're wrong the other way, by a lot. Industrial demand collapses
during the festival far more than our model expects.

We tried to fix both. For Diwali we built festival-specific features — they
removed about two thirds of the error on those days, but Diwali is fifteen
days out of a thousand, so it doesn't move the overall average enough to pass
our own statistical test. We've reported that honestly rather than adopting it
quietly.

And our errors are still correlated day to day, which means there's structure
we haven't captured. We're not claiming this is finished.

---

## Slide 11 — Running it for real

- Evening job → forecast + **conformal prediction intervals**
- 2026 interval coverage: **91.1%** (target 90%) · **93.9%** (target 95%) — out-of-sample
- Backtest: **1,016 forecasts**, weekly refit, MAE 479.7 MW
- ⚠️ Honest caveat: only **1 live forecast logged** — this rests on backtests

**Script.**
This does run automatically. An evening job pulls the latest weather forecast
and issues tomorrow's number with an uncertainty range.

Those ranges are honest. We said 90% and got 1%, we said 95% and got 1% —
measured on 2026 using interval widths calculated only from earlier data, so
it's a genuine out-of-sample check.

One thing I'll be upfront about: we have exactly one forecast logged from live
operation. Everything else is backtesting. It's a simulation of deployment,
not proof of deployment.

I'll also mention one trap we nearly walked into. Our two data pipelines
number their lag columns differently. Serving the model through the wrong one
raised **no error at all** and quietly made it 56% worse. We only found it
because we tested the serving path against the training data directly.

---

## Slide 12 — Contribution and what's next

- We **quantify the operational cost** of weather-forecast uncertainty
- Leakage-controlled, **H = 2**, 29,535 real lead-1 forecasts, paired CIs
- **32% of the weather benefit does not survive** contact with operations
- Next: conditional surge correction · festival handling · live accumulation · other states

**Script.**
So, to close.

We're not claiming the best model in the world. What we're claiming is a
measurement nobody usually makes: when you move from the weather data that
research uses to the weather data an operator actually has, about **32% of
the benefit evaporates**.

That matters because it means published error figures in this field are
systematically optimistic compared to what a control room would see — and the
gap is measurable, not hand-waving.

Everything here reproduces from one command, the dependencies are pinned, and
every number on these slides was generated from our results files rather than
typed in. Thank you — happy to take questions.

---

## Questions to expect

| Question | Answer |
|---|---|
| "Why not deep learning?" | About 4,909 rows. And four tuned gradient boosting models already lost to ridge regression — capacity isn't the binding constraint, extrapolation is. |
| "Is 498 MW good?" | Roughly 2.7% error on a ~18,973 MW average 2026 peak. Compare against the no-weather baseline of 696 MW, not against other papers on other grids. |
| "Isn't 2026 your test set?" | It's a confirmation set, not a blind test — we'd seen some 2026 data during development. Our **least biased** estimate is the 2025 replay on slide 8, though even that isn't fully nested: it ranks models on default settings and reuses feature sets chosen with 2025 in view. We have no genuinely blind evaluation. |
| "Why is error still autocorrelated?" | Multi-day weather shocks we can't see at H = 2. It's a stated limitation, and it's why all our confidence intervals use a block bootstrap. |
| "Could you run this at 8pm?" | We never verified what time the forecasts were actually issued. Our claim is "lead-1 forecast matched to the target date" — we deliberately don't claim more. |
| "Where's the data from?" | Grid-India PSP reports and Open-Meteo forecasts. Historical backfill from Hunt & Bloomfield (2025), Zenodo DOI 10.5281/zenodo.14983362, CC-BY-4.0. |
| "How does this differ from Hunt & Bloomfield?" | They model total *energy* with contemporaneous reanalysis and report r². We forecast *peak* at H=2 and measure what operational forecast weather costs. Their paper is a clean example of the gap we measure. |
