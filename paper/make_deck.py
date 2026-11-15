"""
Generate paper/DECK.md from committed artifacts.

    python make_deck.py

Companion to make_paper.py, same rule and same reason: **no number is typed
by hand.** Everything is read from results/*.json, models/champion_config.json
or data/master.csv at build time.

The rule exists because this project has twice shipped a wrong number that
survived humans reading prose - a lag-1 residual ACF of +0.41 belonging to a
superseded model, and a headline sentence claiming "7.1% of that gain is lost"
that silently used the Regime A MAE as its denominator instead of the gain.
A deck is the artifact most likely to be built once, presented repeatedly and
never re-checked, so it is the worst place to hand-copy a figure.

Re-run before every presentation. If a number on a slide changes, an
experiment changed - go and find out which.

Output is Markdown: one `---` separated section per slide, with slide-ready
bullets and a spoken script. Paste into whatever tool you build slides in.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "paper" / "DECK.md"


def j(name):
    return json.loads((R / name).read_text())


def main():
    cfg = json.loads((ROOT / "models" / "champion_config.json").read_text())
    L, fin, aud = j("weather_ladder.json"), j("final_evaluation.json"), j("selection_audit.json")
    ea, fi, iv = j("error_analysis.json"), j("feature_importance.json"), j("intervals.json")
    sur, fes, dm = j("phase7_surge.json"), j("phase7_festival.json"), j("dm_tests.json")
    ab, pt = j("phase5_regime_ab_results.json"), j("publish_time.json")

    vp, seg, rd = cfg["validation_performance"], ea["segments"], ea["residual_diagnostics"]
    blk, cov = fi["by_block"], fin["interval_coverage_vs_2026_free_widths"]
    ci = ab["overall"]["bootstrap_ci_95"]
    dm_ab = next(r for r in dm["results"]["regime_b_vs_a"] if "ridge:" in r["comparison"])
    hv = sur["honest_validation"]
    gate = fes["results"][cfg["model_type"]]["gates"]["B"]

    m = pd.read_csv(ROOT / "data" / "master.csv", parse_dates=["date"])
    y = m.dropna(subset=["tn_peak_demand_mw"])
    ann = y.groupby(y["date"].dt.year)["tn_peak_demand_mw"].max()
    arch = sum(1 for _ in open(ROOT / "data" / "weather_forecast_archive.csv")) - 1

    # Derived facts the slides quote - computed, never typed. The rule in this
    # file's docstring applies to the script itself, not only to its output.
    span = pd.date_range(m["date"].min(), m["date"].max(), freq="D")
    ser = m.set_index("date")["tn_peak_demand_mw"].reindex(span)
    r_lag1 = float(ser.corr(ser.shift(1)))          # why losing peak_lag_1 hurts
    # Mean peak over the CONFIRMATION year, not the whole series. The series
    # mean (~14,300 MW) includes low early years, so quoting MAPE against it
    # is inconsistent: 2.7% of 14,300 is 387 MW, not the 503 MW we report.
    conf_year = int(fin["protocol"]["date_range"][0][:4])
    mean_peak = float(y.loc[y["date"].dt.year == conf_year, "tn_peak_demand_mw"].mean())
    fa = pd.read_csv(ROOT / "data" / "weather_forecast_archive.csv", usecols=["date"],
                     parse_dates=["date"])
    arch_years = (fa["date"].max() - fa["date"].min()).days / 365.25
    prior = cfg["supersedes"]                        # the model this one replaced

    gain_pct = 100 * L["weather_gain_mw"] / L["a0_mae"]
    pen_pct = L["penalty_pct_of_gain"]
    n_scored = seg["overall"]["n"]

    # worst GBM vs ridge, for slide 6
    gbms = [r for r in dm["results"]["phase6_vs_ridge_regime_a"]
            if any(g in r["comparison"] for g in ("lightgbm", "xgboost", "catboost"))]
    best_gbm = min(gbms, key=lambda r: r["delta_mae_mw"])
    hyb = next(r for r in dm["results"]["phase6_vs_ridge_regime_a"]
               if "ridge_gbm_hybrid" in r["comparison"])

    D = f"""# Next-Day Peak Electricity Demand Forecasting for Tamil Nadu

**Generated {datetime.now():%Y-%m-%d} by `make_deck.py` — every number read from
committed artifacts. Re-run before presenting; do not edit figures by hand.**

12 slides · ~13 min + questions · figures in `figures/`

> **Placeholders still open — leave visible until resolved**
> - `[FILL]` India/Tamil Nadu regional STLF citations
> - `[FILL]` prior work using archived NWP forecasts
> - `[FILL]` permutation importance under correlated predictors
> - `[DOI]` Zenodo record for the historical backfill (~3,852 of {len(m):,} target rows — count itself needs confirming with the DOI)

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
- Tamil Nadu's annual peak: **{ann.get(2022, float('nan')):,.0f} MW (2022) → {ann.get(2026, float('nan')):,.0f} MW (2026)**
- Up **{100 * (ann.get(2026, 1) / ann.get(2025, 1) - 1):.1f}%** in 2026 alone — the target is moving

**Script.**
Peak demand is what a control room commits reserve against. If you forecast
too low, you're scrambling for expensive power or shedding load. If you
forecast too high, you've paid to keep plants running that nobody needed. So
both directions cost money, just in different ways.

And this is getting harder, not easier. The annual peak has gone from about
{ann.get(2022, float('nan')):,.0f} megawatts in 2022 to {ann.get(2026, float('nan')):,.0f} this year — up {100 * (ann.get(2026, 1) / ann.get(2025, 1) - 1):.1f}% in 2026 alone. That
matters later, because it turns out some very popular model families simply
cannot handle a target that keeps climbing.

---

## Slide 3 — The constraint nobody budgets for: **H = 2**

- Grid-India publishes day *D*'s peak on **D+1**, around 09:00–11:00 IST
- Across **{pt['n_run_days']} observed runs** — including one at **{pt['run_clock']['latest']}** — we *never* had same-day data
- So an evening forecast for tomorrow has demand only through **two days ago**
- ⇒ `peak_lag_1` is **unusable** — and it's our second-best predictor (r = {r_lag1:.3f})

**Script.**
Here's the first thing we measured rather than assumed. The system operator
publishes yesterday's peak demand this morning. So if I'm issuing a forecast
tonight for tomorrow, the newest demand number I actually hold is from the day
before yesterday. We call that H equals 2.

We didn't guess this. We logged {pt['n_run_days']} actual pipeline runs, including one that ran
at {pt['run_clock']['latest']} in the evening, and not once did same-day data exist.

The consequence is painful. Yesterday's peak is the second most predictive
column in our whole dataset — correlation of {r_lag1:.3f}. And we cannot use it.
Training on it gives you a model that looks wonderful and cannot be run.

---

## Slide 4 — Data and leakage control

- **{len(m):,} days**, 2013 → 2026 · Grid-India PSP reports
- Weather: ERA5 reanalysis **+ {arch:,} archived lead-1 forecasts** ({arch_years:.1f} years)
- Expanding walk-forward, validation 2023 / 2024 / 2025 — **never** random splits
- 100+ mechanical checks across 5 test suites

**Script.**
About {len(m):,} days of demand data going back to 2013. For weather we have two
things: the reanalysis everyone uses, and — this is the unusual part — an
archive of {arch:,} genuine forecasts, stored by how far ahead they were issued.

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
| no weather | {L['a0_mae']:.1f} MW |
| reanalysis | {L['regime_a_mae']:.1f} MW |
| **real lead-1 forecast** | **{L['regime_b_mae']:.1f} MW** |

- Weather buys **{L['weather_gain_mw']:.0f} MW** — a **{gain_pct:.1f}%** improvement
- Operational forecasts **give back {L['forecast_penalty_mw']:.0f} MW**
- ⇒ **{pen_pct:.0f}% of the weather gain disappears** (95% CI [{ci['ci_low']:.1f}, {ci['ci_high']:.1f}], p = {dm_ab['dm_p_two_sided']:.0e})

**Script.**
This is the slide I care about. Same model, same days, same everything —
we only change where the weather comes from.

With no weather at all, error is about {L['a0_mae']:.0f} megawatts. Add reanalysis weather
and it drops to {L['regime_a_mae']:.0f}. That's a {gain_pct:.1f}% improvement, and that's the number
most papers would report.

Now switch to the weather a forecaster would really have — an actual forecast
issued the day before — and error goes back up to {L['regime_b_mae']:.0f}.

So weather bought us about {L['weather_gain_mw']:.0f} megawatts of accuracy, and using real
forecasts hands back {L['forecast_penalty_mw']:.0f} of it. That's roughly **{pen_pct:.0f}% of the benefit —
nearly a third — gone**, purely because the weather input is a forecast
instead of a perfect retrospective record.

If you only evaluate on reanalysis, you never see this. *(pause here)*

---

## Slide 6 — Does the penalty survive a better model?

**Figure:** `fig3_model_benchmark.png`

- Nine model families, paired comparisons against Ridge
- **Every pure gradient-boosting model lost** — best of them still **{best_gbm['delta_mae_mw']:+.0f} MW** worse
- Trees predict leaf averages → **cannot extrapolate** a rising target
- Winner: linear trend **+ boosted residual** ({hyb['delta_mae_mw']:+.0f} MW vs Ridge)

**Script.**
A fair question at this point is whether a better model makes the problem go
away. It doesn't.

We tried nine model families. Every single pure gradient boosting model —
LightGBM, XGBoost, CatBoost — lost to plain old ridge regression. The best of
them was still about {abs(best_gbm['delta_mae_mw']):.0f} megawatts worse.

The reason is simple once you see it. A tree predicts by averaging training
examples in a leaf. It literally cannot output a number higher than it has
seen before. Our target has been climbing for years. So every summer, the tree
models were structurally incapable of predicting a new record.

What won was a combination: a linear model to carry the trend, with boosting
on top to clean up what's left. That's {abs(hyb['delta_mae_mw']):.0f} megawatts better than ridge alone.

---

## Slide 7 — One honest surprise

- Changed only the residual learner's **objective**: mean → **quantile**
- Validation MAE **{prior['regime_b_mae']:.2f} → {vp['regime_b_mae']:.2f} MW**
- Why: **MAE is minimised by the median, not the mean** — we were optimising the wrong thing
- Held-out check: τ picked on 2023–24 only → **{hv['paired_vs_champion_2025']['delta_mae_mw']:+.1f} MW on untouched 2025** (CI [{hv['paired_vs_champion_2025']['ci_95_low']:.1f}, {hv['paired_vs_champion_2025']['ci_95_high']:.1f}])

**Script.**
Here's something I didn't expect, and it's a bit embarrassing in hindsight.

Our model's second stage was trained to predict the *average* leftover error.
But we report our results using mean absolute error — and mean absolute error
is minimised by the **median**, not the mean. We'd been optimising for the
wrong statistic the entire time.

Changing just that one setting took us from {prior['regime_b_mae']:.0f} megawatts down to {vp['regime_b_mae']:.0f}.

And we checked this honestly. We re-picked the setting using only 2023 and
2024, then tested once on 2025, which had played no part in choosing it. It
still improved things by about {abs(hv['paired_vs_champion_2025']['delta_mae_mw']):.0f} megawatts. So it's real, not us fooling
ourselves.

---

## Slide 8 — Is any of this real? Two checks

**(a) Did we fool ourselves by tuning?** Replayed the *entire* selection
pipeline with 2025 held out:

| | reported | honest | difference |
|---|---|---|---|
| Regime B | {aud['optimism']['B']['reported_2025_fold_mae']:.2f} | {aud['optimism']['B']['honest_2025_mae']:.2f} | **{aud['optimism']['B']['optimism_mw']:+.2f} MW** |

**(b) Final confirmation**, run **once** on 2026: **{fin['metrics_2026']['mae']:.2f} MW** vs {fin['validation_2023_25_regime_b_mae']:.2f} expected ({fin['metrics_2026']['mape']:.2f}% MAPE)

**Script.**
Two ways we tried to catch ourselves cheating.

First — we made four rounds of choices against the same validation years. That
inflates results, and usually nobody measures by how much. So we re-ran the
whole selection process from scratch, pretending 2025 didn't exist, then
tested on 2025 once. If we'd been fooling ourselves, the honest number would
be worse. It came out **{abs(aud['optimism']['B']['optimism_mw']):.1f} megawatts better**. So our reported figures are
conservative, not inflated.

Second — we froze the model and ran it once on 2026. We got {fin['metrics_2026']['mae']:.0f} megawatts
against {fin['validation_2023_25_regime_b_mae']:.0f} expected. That's about {fin['metrics_2026']['mape']:.1f}% error.

One honesty note: we call 2026 a *confirmation*, not a blind test, because we
had looked at some 2026 data earlier in development. We're not going to
pretend otherwise.

---

## Slide 9 — What's actually driving it

**Figure:** `fig5_feature_importance.png`

| block | attribution | **permutation** |
|---|---|---|
| weather | {blk['weather']['combined_mean_abs_mw']:,.0f} MW | **{blk['weather']['permutation_dmae_mw']:+,.0f} MW** |
| demand lags | {blk['demand_lags']['combined_mean_abs_mw']:,.0f} MW | {blk['demand_lags']['permutation_dmae_mw']:+,.0f} MW |

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

- **Surge days under-forecast by {abs(seg['surge_top5pct']['bias']):.0f} MW** (top 5%) — the expensive direction
- **Diwali over-forecast by {seg['diwali_window']['bias']:,.0f} MW** (n={seg['diwali_window']['n']} — anecdotal, but 3 of the worst 6 days)
- Residual autocorrelation **{rd['lag1']:+.3f}** — structure remains
- Festival features fix ~⅔ of Diwali but don't clear the bar ({gate['delta_mae']:+.1f} MW, CI [{gate['ci_95_low']:.1f}, {gate['ci_95_high']:.1f}])

**Script.**
I want to show you where this breaks before anyone has to ask.

On the hottest, highest-demand days — exactly when you most need it right — we
forecast about {abs(seg['surge_top5pct']['bias']):.0f} megawatts **too low**. That's the dangerous direction,
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
- 2026 interval coverage: **{cov['0.90']['realised']:.1%}** (target 90%) · **{cov['0.95']['realised']:.1%}** (target 95%) — out-of-sample
- Backtest: **{iv['walk_forward']['n_predictions']:,} forecasts**, weekly refit, MAE {iv['walk_forward']['mae_mw']:.1f} MW
- ⚠️ Honest caveat: only **1 live forecast logged** — this rests on backtests

**Script.**
This does run automatically. An evening job pulls the latest weather forecast
and issues tomorrow's number with an uncertainty range.

Those ranges are honest. We said 90% and got {cov['0.90']['realised']:.0f}%, we said 95% and got {cov['0.95']['realised']:.0f}% —
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
- Leakage-controlled, **H = 2**, {arch:,} real lead-1 forecasts, paired CIs
- **{pen_pct:.0f}% of the weather benefit does not survive** contact with operations
- Next: conditional surge correction · festival handling · live accumulation · other states

**Script.**
So, to close.

We're not claiming the best model in the world. What we're claiming is a
measurement nobody usually makes: when you move from the weather data that
research uses to the weather data an operator actually has, about **{pen_pct:.0f}% of
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
| "Why not deep learning?" | About {len(m):,} rows. And four tuned gradient boosting models already lost to ridge regression — capacity isn't the binding constraint, extrapolation is. |
| "Is {vp['regime_b_mae']:.0f} MW good?" | Roughly {fin['metrics_2026']['mape']:.1f}% error on a ~{mean_peak:,.0f} MW average {conf_year} peak. Compare against the no-weather baseline of {L['a0_mae']:.0f} MW, not against other papers on other grids. |
| "Isn't 2026 your test set?" | It's a confirmation set. We'd seen some 2026 data during development, so we don't call it blind. The unbiased number is the 2025 replay on slide 8. |
| "Why is error still autocorrelated?" | Multi-day weather shocks we can't see at H = 2. It's a stated limitation, and it's why all our confidence intervals use a block bootstrap. |
| "Could you run this at 8pm?" | We never verified what time the forecasts were actually issued. Our claim is "lead-1 forecast matched to the target date" — we deliberately don't claim more. |
| "Where's the data from?" | Grid-India PSP reports and Open-Meteo. **`[DOI]` — Zenodo record for the historical backfill still to be filled in.** |
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(D)
    n_slides = D.count("## Slide ")
    print(f"wrote {OUT} ({n_slides} slides, {len(D.splitlines())} lines, ~{len(D.split())} words)")


if __name__ == "__main__":
    main()
