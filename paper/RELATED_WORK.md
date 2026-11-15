# Related Work — positioning the forecast-gap contribution

> ### ⚠️ Verify every citation before submission
>
> The three previously-open `[FILL]` gaps are **resolved and verified against
> Crossref** (title, authors, venue, year, DOI). The remaining references are
> ones I am confident exist and are correctly attributed but have not
> individually re-fetched — **verify volumes and page numbers against the
> publisher record before submission.** Nothing here is invented; anything
> unverified is still marked.

---

## 1. The argument this section has to make

The paper's contribution is **a measurement, not a leaderboard result**:

> Weather reduces next-day peak-demand MAE by 17.9% under reanalysis
> (696 → 572 MW), but **32.4% of that gain is given back**
> (612 MW, 95% CI on the penalty [25.9, 54.1]) when only operationally
> available lead-1 forecasts are used — a penalty invisible to the
> reanalysis-based evaluations standard in the literature.

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

### India and Tamil Nadu — resolved

| Work | Role |
|---|---|
| **Hunt & Bloomfield (2025)**, *Building and explaining data-driven energy demand models for Indian states*, Environmental Research: Energy, DOI [10.1088/2753-3751/adc7bc](https://doi.org/10.1088/2753-3751/adc7bc) | **The closest prior work, and the strongest anchor for our contribution.** Same authors as our backfill dataset, same Grid-India source, all 32 Indian states. |
| **Usha & Balamurugan (2014)**, *Predictive Data Mining Techniques for Forecasting Tamil Nadu Electricity Board (TNEB) Load Demand: An Empirical Study*, in *Intelligent Computing, Networking, and Informatics*, Springer India, pp. 1241–1247, DOI [10.1007/978-81-322-1665-0_129](https://doi.org/10.1007/978-81-322-1665-0_129) | Tamil Nadu specifically; regional precedent for data-mining approaches to TNEB load. |

**How to position against Hunt & Bloomfield — be precise, this matters.**
It is the nearest neighbour to our work and differs on four axes, every one of
which favours our framing rather than threatening it:

1. **Target.** They model daily *total energy*; we forecast daily *peak* demand.
2. **Task.** Theirs is a nowcast/hindcast — weather is **contemporaneous** with
   the demand it explains. Ours is a genuine forecast at H = 2.
3. **Weather.** **ERA5 reanalysis only.** No comparison against operational forecasts.
4. **Metric.** They report r² (median 0.78 across states); we report MAE with
   paired confidence intervals.

So the most relevant recent paper on Indian state demand — published 2025,
using the very dataset we backfill from — is itself an instance of the gap this
paper measures. Say that plainly and without snark: it is a legitimate,
well-executed explanatory study whose design simply does not answer the
operational question. That is the opening.

> Cite the dataset **and** the paper. Same authors, and the dataset is
> CC-BY-4.0, which requires attribution.
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
| **Hunt & Bloomfield (2025)** (above) | A concrete, recent, same-data instance of reanalysis-only evaluation for Indian states. |
| **Zhao, Duan & Liu (2018)**, *Uncertainty Analysis of Weather Forecast Data for Cooling Load Forecasting Based on the Monte Carlo Method*, Energies 11(7), DOI [10.3390/en11071900](https://doi.org/10.3390/en11071900) | Closest methodological precedent in the load domain. They **simulate** forecast uncertainty by Monte Carlo; we **measure** it against a real archive. That distinction is the sharpest single sentence available for positioning. |
| **Yang, Cerrai & Anagnostou (2021)**, *The Effect of Lead-Time Weather Forecast Uncertainty on Outage Prediction Modeling*, Forecasting 3(3), DOI [10.3390/forecast3030031](https://doi.org/10.3390/forecast3030031) | Adjacent power-system task where lead-time forecast uncertainty **is** treated as first-class — evidence the discipline exists elsewhere and load forecasting has been slower to adopt it. |

> **Honesty constraint.** Do not write "no prior work has done this." Write
> "we are not aware of a study that isolates the penalty as a separate
> measured quantity under a single harness." The weaker claim is both safer
> and sufficient — **and it is only honest because a search backs it.** See
> [`LITERATURE_SEARCH.md`](LITERATURE_SEARCH.md), which records what was
> searched, what was found, and explicitly what was *not* searched.

A useful adjacent literature: **wind and solar** forecasting has long treated
NWP forecast error as a first-class component, because the resource *is* the
weather. Yang et al. (2021) above is a power-system example; add one from wind
or solar if a reviewer asks. The point to make is that load forecasting has
been slower to adopt the same discipline despite a real weather dependence.

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
| Cawley & Talbot (2010), *On over-fitting in model selection and subsequent selection bias in performance evaluation*, JMLR 11, 2079–2107 | The canonical reference. Exactly the failure mode `experiments/run_selection_audit.py` tests for. |
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

**Strobl, Boulesteix, Kneib, Augustin & Zeileis (2008)**, *Conditional
variable importance for random forests*, BMC Bioinformatics 9:307, DOI
[10.1186/1471-2105-9-307](https://doi.org/10.1186/1471-2105-9-307).

This is the citation for our strongest interpretability result. Strobl et al.
show permutation importance behaves differently under correlated predictors —
exactly the mechanism behind our attribution/permutation disagreement, where
the five mutually collinear demand lags let the model recover when any one is
shuffled, while weather has no substitute.

---

## 6. Uncertainty quantification

| Work | Role |
|---|---|
| Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*, Springer | Conformal prediction, the foundational text. |
| Lei et al. (2018), *Distribution-free predictive inference for regression*, JASA 113(523), 1094–1111 | Split conformal — the method used in `calibrate_intervals.py`. Its distribution-free guarantee is the right justification given this paper's residuals are clearly non-Gaussian (skew +1.18, excess kurtosis +4.48). |

**Conformal validity under temporal dependence — state this as a caveat.**
Split conformal's finite-sample guarantee rests on exchangeability, which
autocorrelated daily residuals do not strictly satisfy (our lag-1 residual ACF
is +0.481). The honest position is therefore:

> We apply split conformal prediction and report *realised* coverage rather
> than relying on the theoretical guarantee, since the exchangeability
> assumption is not strictly satisfied by temporally dependent residuals.
> Realised coverage on the held-back year was 91.1% and 93.9% against nominal
> 90% and 95%, measured against interval widths fitted only on earlier data.

That is both defensible and stronger than a bare appeal to the theorem —
empirical coverage is what a reviewer actually wants to see, and quoting it
pre-empts the objection.

---

## 7. Drafting notes

- Lead §3 with the gap, not with the method. The method is ordinary; the
  measurement is not.
- Do **not** claim state of the art. Claim the measurement.
- State the limitations from [`PHASE7_RESULTS.md`](../docs/RESULTS.md) §7 in
  the paper itself. Several — surge under-forecasting, the Diwali failure, the
  unverified issue hour — are the kind a referee finds anyway, and finding
  them already disclosed is worth more than hiding them.
- The reproducibility story is unusually strong (every headline number
  recomputed from saved predictions; pinned environment; `reproduce.sh`).
  One sentence in the paper, and a repository link.
