# Complete Project Analysis

**Date:** 2026-11-15 · **Subject:** Next-Day Peak Electricity Demand Forecasting for Tamil Nadu

> ### ⚠️ Conflict of interest — read first
>
> I (Claude) built most of what is analysed here: the Phase 6/7 modelling, the
> serving path, the evaluation and interpretability code, the test suites, and
> most of this documentation. **A builder cannot neutrally audit their own
> work.** I have compensated by re-deriving every number from artifacts rather
> than from memory, and by weighting criticism toward my own contributions.
> Get a second opinion on §4 and §6 specifically.

---

## 1. Is the project finished?

**Engineering: yes. Research write-up: nearly. Submission: no — four blockers.**

| Dimension | State | Evidence |
|---|---|---|
| Data pipeline + automation | **Done** | daily/evening cron green, self-healing disk logic |
| Leakage control | **Done** | 147 mechanical checks across 5 suites |
| Feature engineering | **Done** | A0→A2 gated, A4 tested and rejected |
| Model selection | **Done** | 10 families benchmarked, champion frozen |
| Validity checks | **Done** | selection optimism measured; 2026 confirmation consumed |
| Interpretability | **Done** | two-part attribution + permutation cross-check |
| Uncertainty | **Done** | conformal intervals, out-of-sample coverage verified |
| Figures | **Done** | 6, regenerated from the champion |
| Paper draft | **Draft** | all 7 sections written, `paper/PAPER.md` |
| Deck | **Draft** | 12 slides + speaker scripts, `paper/DECK.md` |
| **Citations** | **Blocked** | 3 `[FILL]` gaps |
| **Data provenance** | **Blocked** | Zenodo DOI missing |
| **Live validation** | **Blocked** | 1 logged forecast; needs wall-clock |
| **Co-author review** | **Not started** | nobody but you has read this end to end |

**Honest verdict: ~90% complete.** What remains is not engineering. It is a
literature search, one missing identifier, and the passage of time.

---

## 2. What is genuinely strong

These are the things a referee should find impressive, ranked by how unusual
they are.

### 2.1 The contribution is a measurement, not a leaderboard entry ⭐

Most STLF papers report "our model achieves X MAE." This one reports something
the field mostly cannot see:

| condition | MAE |
|---|---|
| no weather | 696.3 MW |
| reanalysis (Regime A) | 571.6 MW |
| **real lead-1 forecast (Regime B)** | **612.0 MW** |

Weather buys 124.7 MW (17.9%). Operational forecasts give back 40.3 MW —
**32% of the gain**, 95% CI [25.9, 54.1], DM p = 1.4e−08.

This is hard to do and rarely done, because it requires a *forecast archive*,
not just weather history. You have 29,535 rows of lead-resolved forecasts
spanning 5.4 years. That archive is the single most valuable asset in the
project and it cannot be reconstructed retrospectively by anyone who did not
start logging years ago.

### 2.2 The horizon is measured, not assumed ⭐

H = 2 comes from 46 logged pipeline runs showing same-day demand data was
*never* available, including a run at 18:07. The consequence is costly and
accepted rather than quietly avoided: `peak_lag_1`, correlation 0.927 with the
target, is excluded.

Most papers would have used it. Most reviewers would not have noticed.

### 2.3 Validity was measured rather than asserted ⭐

Four sequential selection rounds ran against the same validation years. Rather
than claim the effect was small, the entire pipeline was replayed with 2025
held out:

| | reported | honest | optimism |
|---|---|---|---|
| Regime B | 520.55 | 514.75 | **−5.80 MW** |

Negative optimism — the reported figures are *conservative*. Very few applied
papers do this at all, and fewer report it when the answer could have been
embarrassing.

### 2.4 The interpretability result is better than an importance chart

| block | attribution | permutation ΔMAE |
|---|---|---|
| weather | 2,428 MW | **+1,739** |
| demand lags | 4,304 MW | +1,093 |

The two methods disagree, and the disagreement is the finding: demand lags are
mutually collinear so the model recovers when one is shuffled; weather has no
substitute. This answers "is weather actually doing anything?" more
convincingly than any ranking.

### 2.5 Reproducibility is genuinely excellent

`./reproduce.sh` rebuilds everything. Dependencies pinned. Environment
captured in the model config. Paper and deck **generated** from results so
prose cannot drift from evidence. 147 mechanical checks.

This is better than most submitted work and should be stated in the paper.

### 2.6 Negative results are reported, not buried

Four things were tried and rejected with evidence: recency weighting, the A4
festival feature set, the surge-bias quantile fix (as a *bias* fix), and — most
tellingly — four tuned GBM families that all lost to ridge regression.

---

## 3. What is weak

Ordered by how much damage each could do at review.

### 3.1 🔴 Nobody independent has reviewed this

The largest remaining risk is not technical. Your teammate owns Track A, you
own Track B, and an AI built much of Phase 6–8. **No human has read the
pipeline end to end with fresh eyes.** Every error found so far was found by
tooling or by the builder — which means the class of error that tooling
cannot see is entirely unexplored.

Two real errors shipped and survived human reading:
- a residual ACF of +0.41 belonging to a superseded model, quoted for months
- a headline claiming "7.1% of that gain" using the wrong denominator (true: 32%)

Both were caught late and by accident. Assume more exist.

### 3.2 🔴 Four submission blockers

| Blocker | Why it matters |
|---|---|
| Zenodo DOI missing | 3,852 of 4,909 target rows have no citable provenance. A referee will ask. |
| 3 `[FILL]` citations | India/TN regional STLF especially — an ICAIEA audience expects it |
| 1 live forecast logged | Deployment claims rest entirely on backtests |
| No co-author review | See §3.1 |

### 3.3 🟡 Festival days are the model's real weakness

29% of days fall within ±2 days of a holiday, and they carry **37% of total
absolute error**. Diwali is over-forecast by 1,766 MW.

Worse: the two Phase 7 results pull against each other. The quantile head
(which improved overall MAE) made Diwali *worse*, because it shifts predictions
up and Diwali is already over-forecast.

### 3.4 🟡 Surge days are under-forecast by 496 MW

The operationally expensive direction. A global quantile shift reduces this
in-sample but transfers poorly (7% on untouched data). Unsolved.

### 3.5 🟡 Residual autocorrelation of +0.481

Structure remains. **But see §6.4** — I tested the obvious fix and it does not
work, which is worth knowing before someone spends a week on it.

### 3.6 🟡 Single region, single target

No generality demonstrated. Fine for a first paper; state it.

### 3.7 🟢 2026 is a confirmation set, not a blind test

Already disclosed everywhere. Low risk *because* it is disclosed.

---

## 4. What is missing

| Missing | Severity | Effort |
|---|---|---|
| Zenodo DOI | **Blocker** | 30 min of searching |
| India/TN citations | **Blocker** | 3–4 h literature search |
| Independent review | **High** | 1 day of someone else's time |
| Live forecast accumulation | **Medium** | wall-clock only, now automated |
| Permutation-importance citation | Low | 30 min |
| Multi-state validation | Low (future work) | weeks |
| Hourly/sub-daily extension | Low (future work) | weeks |

**Not missing, despite feeling like it:** more models, deep learning, more
features. The benchmark settled the model-family question with confidence
intervals, and ~4,900 rows will not support a deep architecture. Adding more
candidates now would *increase* selection noise, not accuracy.

---

## 5. Where the error actually lives

Measured on the champion, Regime B, n = 1,066. "Total MW" = excess MAE × n,
i.e. how much each segment contributes above the overall average — this is
where improvement is genuinely available.

| segment | n | MAE | bias | excess | **total MW** |
|---|---|---|---|---|---|
| holiday | 70 | 888 | +401 | +390 | **27,329** |
| **diwali window** | 15 | 1,802 | +1,766 | +1,304 | **19,557** |
| monsoon Oct–Dec | 271 | 557 | +2 | +59 | 15,911 |
| long weekend | 26 | 999 | +551 | +501 | 13,022 |
| summer Mar–Jun | 366 | 530 | +59 | +32 | 11,684 |
| may day | 15 | 1,004 | +432 | +506 | 7,596 |
| dussehra/ayudha | 18 | 877 | +227 | +379 | 6,821 |
| pongal window | 13 | 787 | −404 | +289 | 3,755 |
| surge top-5% | 56 | 548 | −496 | +50 | 2,816 |

**Read this table before choosing what to work on.** Surge days feel urgent
because the bias is large and operationally dangerous — but they contribute
only 2,816 MW of excess error. Holidays contribute **ten times** that.

---

## 6. What would actually improve performance

Ranked by measured headroom, not by how interesting the technique is.

### 6.1 ⭐ Fix festival days — **up to −56.6 MW (−11%)**

If festival-window days performed like ordinary days, overall MAE would fall
from **497.9 → 441.3 MW**. That is the single largest available gain, by a
wide margin, and it is larger than everything else on this list combined.

**But the obvious approach does not work.** I tested removing the festival
*bias* alone: MAE got **worse** (+4.1 MW). The reason is that festival biases
point in opposite directions — Diwali +1,766, Pongal −404 — so any global or
averaged correction cancels itself.

What is likely to work, in order of promise:

1. **Per-festival effects, not a generic festival flag.** A4 already proved the
   mechanism: dedicated Diwali features cut its bias from +1,766 to +605. It
   failed the *aggregate* MAE gate only because 15 days cannot move a
   1,066-day average. Consider adopting it on segment evidence and saying so
   explicitly, or reporting a festival-specific model.
2. **Multi-day festival windows.** Diwali, Pongal and Dussehra are multi-day
   events with ramp-down and ramp-up. A ±2 day symmetric window is crude.
3. **Year-specific festival intensity.** Industrial shutdown length varies by
   year and by how the festival falls against the weekend.
4. **Long weekends / bridge days** (n=26, bias +551) — holidays adjacent to
   weekends behave differently from isolated ones.

> ⚠️ **Any of these re-opens model selection and invalidates the 2026
> confirmation**, which is a consumed one-shot. Decide deliberately: either
> accept the current model for this paper and list festivals as future work,
> or do the work now and re-run the confirmation with the change disclosed.

### 6.2 Surge-day bias — operationally important, numerically small

Bias −496 MW on 56 days. Worth fixing for *operational* reasons (it is the
dangerous direction) even though the MAE headroom is only ~2,816 MW total.

A global quantile shift does not work (tested: 31% in-sample → 7% out). What
might: heat-persistence features (consecutive days above 32 °C, 3-day rolling
CDD), or a two-stage surge-probability model with a conditional offset.

### 6.3 Weather features are underexploited relative to their value

Permutation importance says weather is the most load-bearing block, yet only
7 of 26 features are weather. Candidates: humidity interactions, heat-index
persistence, overnight minimum temperature as a cooling-load proxy, and
rainfall onset — which matters because the worst monsoon-transition errors are
sharp downward swings the model lags.

### 6.4 ❌ Residual error-correction — **tested, does not work**

The lag-1 ACF of +0.481 suggests exploitable structure. At H = 2 you genuinely
can observe the error from two days ago, so a correction is *feasible*.

I tested it. Correlations: lag-2 **+0.185**, lag-3 +0.111, lag-7 +0.025. Fitting
a linear correction on 2023–24 and applying it to 2025:

```
MAE before correction :  480.8 MW
MAE after  correction :  486.6 MW   (+5.9 MW WORSE)
```

**Do not spend time on this.** The autocorrelation is real but not
exploitable by a naive linear correction — most of it is probably the same
multi-day weather shocks the model already partially captures, so correcting
double-counts them.

### 6.5 Things that will NOT help

| Idea | Why not |
|---|---|
| Deep learning | ~4,900 rows. The GBM result shows capacity is not the constraint. |
| More model families | Settled with CIs. More candidates = more selection noise. |
| Recency weighting | Tested across 1–8 year half-lives. Monotonically worse. |
| More tuning | Ridge's L2 is inert; 41 of 50 trials spanned 0.01 MW. |
| Chasing lower MAE generally | The contribution is the *measurement*. A better MAE does not strengthen it. |

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Referee asks for the data DOI | **High** | Blocks acceptance | Find it |
| Referee expects Indian STLF literature | **High** | Weakens positioning | Literature search |
| Undiscovered error of the ACF/denominator class | **Medium** | Embarrassing | Independent review |
| Reviewer disputes "we are not aware of" novelty claim | Medium | Survivable | Claim is deliberately weak |
| Questioned on 1 live forecast | Medium | Survivable | Already disclosed |
| Data source changes format | Low | Pipeline breaks | Tests catch it |

---

## 8. Recommended path forward

**This week — unblock submission**
1. Find the Zenodo DOI *(only you can)*
2. India/TN literature search → fill 3 `[FILL]` markers
3. Read `paper/PAPER.md` end to end and rewrite §1–§2 in your own voice
4. Get Dharshini or a faculty member to review independently

**Before submitting**
5. Re-run `./reproduce.sh` — confirm no number moved
6. Build slides from `paper/DECK.md`
7. Decide the festival question (§6.1) deliberately and record the decision

**After submission / future work**
8. Let the cron accumulate live forecasts
9. Festival modelling as the next paper
10. Multi-state validation for generality

---

## 9. Bottom line

**The science is sound and the contribution is real.** The measurement is
novel, the protocol is unusually rigorous, and the validity checks are better
than most published work in this area.

**The weakness is not technical.** It is that this has been built by two people
and an AI, with no independent reading, and two real errors already escaped
that process. Before submission, the highest-value hour anyone can spend is
someone unfamiliar reading the paper and asking "how do you know that?"

**If you change one thing for performance,** make it festivals (§6.1) —
up to −11% MAE, roughly ten times the headroom of the surge-day problem that
looks more urgent.
