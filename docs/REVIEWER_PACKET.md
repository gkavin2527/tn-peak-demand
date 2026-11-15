# Reviewer packet

**Repo state:** commit `2ce89d7` · **Revised 2026-11-15** after faculty review of v1.

> **What changed in v2.** A reviewer read v1 and found real problems in it.
> Claims 3 and 4 were overstated, the H=2 evidence was ambiguously described,
> one "known weakness" was actually an unresolved provenance issue, and the
> festival opportunity was over-claimed. All corrected below. **That review
> did its job — this version exists because of it.**

---

## 0. Definitions (you need these)

| Term | Meaning |
|---|---|
| **H = 2** | Forecast horizon. A forecast issued the evening of day *D* targets *D+1*, using demand observed only through *D−1*. |
| **Regime A** | Target-day weather from **ERA5 reanalysis** — a retrospective best estimate. Not obtainable at forecast time. |
| **Regime B** | Target-day weather from an **archived lead-1 forecast** — what a forecaster would really have held. |
| **MAE** | Mean absolute error, MW. The primary metric throughout. |
| **ACF** | Autocorrelation of forecast residuals. Ours is +0.481 at lag 1, meaning errors persist day to day. |
| **Gain** | 124.66 MW — how much weather improves MAE (696.27 → 571.61). |
| **Penalty** | 40.34 MW — how much is lost moving from reanalysis to real forecasts (571.61 → 611.95). |

**Severity scale for findings**

| | |
|---|---|
| **S1** | A number in the paper cannot be traced to an artifact, or is wrong |
| **S2** | A claim is stronger than its evidence supports |
| **S3** | Wording is ambiguous or could mislead |
| **S4** | Presentation / clarity |

S1 is not a nitpick. **Both past failures were S1.**

---

## 1. Why you are being asked

Built by two students and an AI. Everything found so far was found by tooling
or by the builder. **Two S1 errors escaped that process:**

- a residual ACF of +0.41 belonging to a superseded model, quoted as current
- "7.1% of that gain is lost" — wrong denominator; the true figure is 32%

Both were caught late and by accident. A third class of problem — overstated
claims — was found by the v1 reviewer, not by us. **Assume more exist.**

---

## 2. Priority order

If you only have an hour, do **Claim 2 and Claim 4**. They are the weakest.

---

## ⭐ Claim 2 — "H = 2, so `peak_lag_1` is unusable" *(highest priority)*

*Where:* `results/publish_time.json`, `docs/METHODOLOGY.md` §1

**Read this carefully, because v1 described it ambiguously.**

The evidence is **46 pipeline runs**. On each, we recorded the run's own clock
time and the newest data date obtained. On all 46, the newest data was **one
day behind the run day**.

- **18:07 is a JOB RUN TIME, not a publish time.** At 18:07 on some day *D*, we
  looked, and day-*D* data was not there.
- Run clock: earliest 06:00, **median 07:23**, latest 18:07.
- A second, independent probe: the report named `09.11.26_NLDC_PSP` was
  uploaded **2026-11-10 11:25** — next-morning publication.

**What to attack:**

1. The median run is 07:23, so **almost all 46 observations are morning runs.**
   The evening claim rests on essentially **one** observation at 18:07.
2. There is **no observation between 18:07 and 20:00**, which is when the
   forecast job actually runs. Is the extrapolation justified?
3. If publication ever drifts earlier, H would be 1 — and `peak_lag_1`
   (r = 0.927 with the target) becomes available. **How much of the paper
   changes if H = 1?**
4. A third source, `report_published` in `peak_demand.csv`, shows lag = 1 for
   all 1,057 rows — but that is **circular** (the extractor defines
   `observation_date = report_date − 1`) and was correctly rejected. Confirm it
   is not used anywhere as evidence.

---

## ⭐ Claim 4 — "2026 confirms the model generalises" *(highest priority)*

*Where:* `results/final_evaluation.json`

2026 MAE **503.49** vs validation **497.90** — a difference of **+5.59 MW**.

**We now believe v1 overstated this, and the paper is being corrected.** With
residual ACF +0.481, the effective sample size is far below the nominal 314:

```
nominal n            314
effective n (AR1)    110      (65% smaller)
SE of the 2026 MAE   45.0 MW
observed difference  +5.59 MW  =  0.12 SE
```

**This result says "not contradicted", not "confirmed."**

**What to attack:**

1. Does the paper still lean on 2026 as if it were independent evidence?
2. 2026 was **inspected during development**, so it is a confirmation set, not
   a blind test. Is that disclosed everywhere it is used?
3. **The hardest question: no genuinely blind evidence exists anywhere in this
   project.** 2026 was seen; the single live forecast used a stand-in model,
   not the champion. Should the paper say so explicitly?

---

## Claim 1 — "Weather buys 17.9%; operational forecasts give back 32% of it"

*Where:* `paper/PAPER.md` §5.1, `results/weather_ladder.json`

Arithmetic: 40.34 / 124.66 = 32.4%. The superseded "7.1%" was 40.34 / 571.61 —
the penalty over the **Regime A MAE** rather than over the **gain**.

**What to attack:**

1. **Wording, everywhere the figure appears.** "Gives back 32% of it" is only
   correct if "it" unambiguously means *the gain*. Check every instance —
   paper, deck, docs. *This is exactly where the last S1 error lived.*
2. Are all three conditions measured on the **same 1,066 days**?
3. §3.2 of METHODOLOGY admits a residual optimism: the four T−1-dependent
   features use complete ERA5 in **both** regimes, because the archive has no
   lead-0. Does that undercut the penalty estimate, or only the absolute
   Regime B number?

---

## Claim 3 — "Selection optimism is small or negative"

*Where:* `experiments/run_selection_audit.py`, `results/selection_audit.json`

**v1 claimed "no selection optimism". That was too absolute.** Measured value
is −5.80 MW (Regime B), but under real limitations:

1. The replay ranks model families on **default** hyperparameters. A family
   with a poor default never reaches tuning — `quantile_hybrid` at its default
   τ=0.9 scores 669 MW and is screened out.
2. The replay reuses feature sets (A0–A4) that were **themselves** selected
   using 2025. Is "held out" actually true?
3. **Sign-convention trap:** optimism = honest − reported. For MAE, negative
   means conservative. For any gain-style metric the same sign means the
   opposite. Check every table.

**What to attack:** is "small or negative under these limitations" defensible,
or should the claim be weaker still?

---

## 3. Five questions

1. Read **the abstract only**, cold. Write what you think was done. Then read
   §5. Do they match?
2. Pick any number in the deck. Trace it to a file in `results/`. Can you?
3. The novelty claim is *"we are not aware of a study that isolates…"*. That is
   weak in **form** — but is it honest? It is only honest if a documented
   literature search backs it. Is one documented?
4. `docs/ANALYSIS.md` §6 claims festivals offer up to −11% MAE. **We now think
   this is over-claimed** — see below. Do you agree?
5. **What evidence would change your mind about any claim here?**

---

## 4. Known-weak — documented, don't spend time rediscovering

| | |
|---|---|
| Surge days under-forecast by **496 MW** | **The operationally important failure.** Average MAE hides it, and the segment table understates it because it contributes only ~2,816 MW of *total* excess error. Unsolved. |
| Diwali over-forecast by 1,766 MW | n=15 — anecdotal on its own |
| Residual ACF +0.481 | A linear correction was tested and made 2025 **worse** by 5.9 MW |
| Only 1 live forecast logged | And it used the **stand-in**, not the champion. Fixed going forward; history unrecoverable |
| Single region, single target | Disclosed |

**Removed from this list in v2:** the 269 rows dated 2013. v1 wrongly listed it
as a known weakness to ignore. It was a genuine provenance question, and has
since been **resolved** — the Zenodo deposit was downloaded and does cover
2013-01-01 onward despite its title; all 269 rows match the source exactly.
A separate finding emerged: the deposit contains **7 dates with conflicting
duplicate values**, which our pipeline de-duplicates.

### On the festival claim — we think v1 over-claimed

`ANALYSIS.md` reports up to −56.6 MW (−11%) from fixing festival days. That
figure is a **ceiling under a counterfactual** (festival days performing like
ordinary days), not an achievable gain. The only festival intervention actually
built (A4) delivers **−3.37 MW with a CI spanning zero**. Seven festival
segments were examined, with n=15 and n=13 for the two headline ones, selected
after looking at the data. **Treat −11% as an upper bound on an idea, not a
forecast of a result.**

---

## 5. Running it

```bash
git clone <repo> && cd tn-peak-demand
git checkout 2ce89d7
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./reproduce.sh --tests      # 147 checks, ~36s
./reproduce.sh              # everything (~30 min)
git diff --stat results/ figures/ models/
```

**A clean diff proves the numbers reproduce — not that they are right.** The
two S1 errors both reproduced perfectly for months. Floating-point and platform
differences can also produce false diffs; judge magnitude, not presence.

---

## 6. Where to send findings

Open one GitHub issue per finding, titled `[S1]`…`[S4]` by severity. Or
annotate a copy and send it back. Blunt is useful; diplomatic is not.
