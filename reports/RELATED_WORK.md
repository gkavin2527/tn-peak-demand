# Related Work — positioning the forecast-gap contribution

> ### ⚠️ Verify every citation before submission
>
> The references below are ones I am confident exist and are correctly
> attributed, but **I have not fetched or re-read any of them here.** Check
> each against the publisher record — authors, year, venue, volume, pages —
> before it goes into a manuscript. Where I was *not* confident, I have left an
> explicit gap rather than filling it with something plausible. Those gaps are
> marked **[FILL]** and are genuine work, not decoration.

---

## 1. The argument this section has to make

The paper's contribution is **a measurement, not a leaderboard result**:

> Weather reduces next-day peak-demand MAE by 17.8% under reanalysis, but
> 7.1% of that gain is lost when only operationally available lead-1 forecasts
> are used — a penalty invisible to the reanalysis-based evaluations standard
> in the literature.

Related work therefore has to establish three things, in order:

1. **Weather-driven STLF is mature and well studied.** Nobody should think
   "use temperature" is the novelty.
2. **Evaluation in that literature overwhelmingly uses *observed* or
   *reanalysis* weather** for the target day — i.e. information a forecaster
   cannot have. This is the gap.
3. **Where forecast weather *is* used, the penalty is rarely isolated** as a
   separate, measured quantity under one fixed harness.

If §2 is established convincingly, the contribution follows almost
automatically. If it is not, a referee will say "everyone knows forecast
weather is worse" — so the section must show that *knowing* it and *measuring
it under a leak-free protocol* are different things.

---

## 2. Short-term load forecasting — the base

| Work | Why it is cited here |
|---|---|
| Hong & Fan (2016), *Probabilistic electric load forecasting: a tutorial review*, Int. J. Forecasting 32(3), 914–938 | The standard review. Establishes that temperature is the dominant exogenous driver and that regression-family models remain competitive — which supports this paper's finding that Ridge beats four tuned GBMs. |
| Hong, Pinson & Fan (2014), *Global Energy Forecasting Competition 2012*, IJF 30(2), 357–363 | GEFCom12 — the field's reference benchmark, and a case where temperature inputs were **actual**, not forecast. |
| Hong, Pinson, Fan et al. (2016), *Probabilistic energy forecasting: GEFCom2014 and beyond*, IJF 32(3) | GEFCom14 extended to probabilistic forecasting; useful for positioning the conformal intervals. |
| Taieb & Hyndman (2014), *A gradient boosting approach to the Kaggle load forecasting competition*, IJF 30(2), 382–394 | Directly comparable method family; a useful contrast given that gradient boosting **lost** to Ridge here, which this paper attributes to extrapolation on a trending target. |

**[FILL] India- and Tamil-Nadu-specific STLF.** I am not confident enough in
specific Indian load-forecasting papers to cite them without checking. This is
the most important gap: a referee at an Indian venue (ICAIEA) will expect the
regional literature. Search terms that should work: *"short-term load
forecasting India state load despatch"*, *"Tamil Nadu electricity demand
forecasting"*, *POSOCO/Grid-India report-based studies*, and the CEA's own
demand projection methodology. Aim for 3–5 references and state explicitly how
this work differs (daily peak, H=2, forecast-weather ladder).

---

## 3. The gap — weather information available at forecast time

This is the section that earns the paper.

The claim to make is narrow and defensible: **most published STLF evaluations
condition on target-day weather that a forecaster would not possess**, whether
observed station data or reanalysis. That is not a criticism of those papers —
it is a standard and often reasonable modelling choice — but it means the
reported errors are not operational errors.

| Work | Role |
|---|---|
| Hersbach et al. (2020), *The ERA5 global reanalysis*, Q. J. R. Meteorol. Soc. 146(730), 1999–2049 | Defines what reanalysis **is**: a retrospective best estimate assimilating observations that postdate the forecast moment. The cleanest way to explain why Regime A is unservable. |
| **[FILL]** 3–5 STLF papers whose experimental setup uses observed/reanalysis target-day weather | The evidence for the gap. Be specific and fair: quote the setup, do not characterise the paper. |
| **[FILL]** any STLF work that *does* use archived NWP forecasts | If these exist — and some likely do in the wind/solar forecasting literature — cite them and state precisely what this work adds: a **three-condition ladder** (none → reanalysis → lead-1 forecast) under one fixed leak-free harness with a paired CI on the difference. |

> **Honesty constraint.** Do not write "no prior work has done this." Write
> "we are not aware of a study that isolates the penalty as a separate
> measured quantity under a single harness." The weaker claim is both safer
> and sufficient.

A useful adjacent literature: **wind and solar** forecasting has long treated
NWP forecast error as a first-class component, because the resource *is* the
weather. **[FILL]** one or two references there, then note that load
forecasting has been slower to adopt the same discipline even though its
weather dependence is real.

---

## 4. Evaluation methodology — where this project is unusually strong

| Work | Role |
|---|---|
| Bergmeir & Benítez (2012), *On the use of cross-validation for time series predictor evaluation*, Information Sciences 191, 192–213 | Justifies the expanding walk-forward and the refusal to use random splits. |
| Tashman (2000), *Out-of-sample tests of forecasting accuracy: an analysis and review*, IJF 16(4), 437–450 | Rolling-origin evaluation; the standard reference for the protocol used here. |
| Diebold & Mariano (1995), *Comparing predictive accuracy*, J. Business & Economic Statistics 13(3), 253–263 | The DM test, reported alongside the bootstrap in every comparison table. |
| Harvey, Leybourne & Newbold (1997), *Testing the equality of prediction mean squared errors*, IJF 13(2), 281–291 | The small-sample correction applied to DM in `src/evaluation.py`. |
| Künsch (1989), *The jackknife and the bootstrap for general stationary observations*, Annals of Statistics 17(3), 1217–1241 | The moving-block bootstrap. Cite it to justify 7-day blocks: daily load errors are autocorrelated (this paper measures lag-1 ACF **+0.481**), so an iid bootstrap would give intervals roughly √7 too narrow. |

### 4.1 Selection bias — cite this, it is the paper's shield

The audit's most serious finding was that one validation set drove four
sequential selection rounds. The project now **measures** the resulting
optimism and finds it negative. That is a strong result, and it should be
framed with the literature that explains why anyone should care:

| Work | Role |
|---|---|
| Cawley & Talbot (2010), *On over-fitting in model selection and subsequent selection bias in performance evaluation*, JMLR 11, 2079–2107 | The canonical reference. Exactly the failure mode `run_selection_audit.py` tests for. |
| Varma & Simon (2006), *Bias in error estimation when using cross-validation for model selection*, BMC Bioinformatics 7, 91 | Nested cross-validation as the remedy; the structural analogue of the outer-2025 / inner-2023–24 design used here. |

Most applied forecasting papers do not do this. Saying "we quantified our own
selection optimism and report it" is a differentiator — *provided* the result
is reported whichever way it came out, which here it was.

---

## 5. Models and interpretation

| Work | Role |
|---|---|
| Chen & Guestrin (2016), *XGBoost: a scalable tree boosting system*, KDD | Benchmark member. |
| Ke et al. (2017), *LightGBM: a highly efficient gradient boosting decision tree*, NeurIPS | The residual head and the quantile objective. |
| Lundberg & Lee (2017), *A unified approach to interpreting model predictions*, NeurIPS | SHAP. |
| Lundberg et al. (2020), *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence 2, 56–67 | **TreeSHAP specifically** — this work uses LightGBM's exact `pred_contrib` implementation rather than the sampling estimator. |
| Koenker & Bassett (1978), *Regression quantiles*, Econometrica 46(1), 33–50 | The quantile objective on the residual head. Also the cleanest citation for the paper's own explanation of why it helped: **MAE is minimised by the conditional median, not the mean.** |

**[FILL]** a reference for permutation importance and its behaviour under
correlated features — this matters because the paper's strongest
interpretability result is precisely that attribution and permutation
*disagree* for collinear demand lags. Strobl et al. on conditional variable
importance is the kind of thing to look for; verify before citing.

---

## 6. Uncertainty quantification

| Work | Role |
|---|---|
| Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*, Springer | Conformal prediction, the foundational text. |
| Lei et al. (2018), *Distribution-free predictive inference for regression*, JASA 113(523), 1094–1111 | Split conformal — the method used in `calibrate_intervals.py`. Its distribution-free guarantee is the right justification given this paper's residuals are clearly non-Gaussian (skew +1.18, excess kurtosis +4.48). |

**[FILL]** a note on conformal validity under **temporal dependence** — the
exchangeability assumption is not strictly satisfied by autocorrelated daily
residuals. The paper should acknowledge this and point to the realised
coverage (91.1% / 93.9% against nominal 90/95 on 2026) as empirical support
rather than claiming the theoretical guarantee holds unmodified.

---

## 7. Drafting notes

- Lead §3 with the gap, not with the method. The method is ordinary; the
  measurement is not.
- Do **not** claim state of the art. Claim the measurement.
- State the limitations from [`PHASE7_RESULTS.md`](../PHASE7_RESULTS.md) §7 in
  the paper itself. Several — surge under-forecasting, the Diwali failure, the
  unverified issue hour — are the kind a referee finds anyway, and finding
  them already disclosed is worth more than hiding them.
- The reproducibility story is unusually strong (every headline number
  recomputed from saved predictions; pinned environment; `reproduce.sh`).
  One sentence in the paper, and a repository link.
