# Reviewer packet — for someone who did not build this

**Time needed: about one hour.** You do not need to understand the whole
codebase. You need to try to break four specific claims.

## Why you are being asked

This project was built by two students and an AI. Everything found so far was
found by tooling or by the builder. **Two real errors escaped that process and
survived months of people reading prose:**

- a residual autocorrelation of +0.41 that belonged to a superseded model and
  was quoted as current
- a headline claiming "7.1% of that gain is lost" that used the wrong
  denominator — the true figure is 32%

Both were caught late and by accident. The class of error that tooling cannot
see is the one you are here to find. **Assume more exist.**

You are not being asked to be polite.

---

## The four claims. Try to break them.

### Claim 1 — "Weather buys 17.9%, operational forecasts give back 32% of it"

*Where:* `paper/PAPER.md` §5.1, `results/weather_ladder.json`

Questions to ask:
- Are all three conditions measured on the **same days**? (Claimed: 1,066
  shared dates.) Check `run_phase7_festival.py:load_set` and the mask logic.
- Is 32% the right denominator? 40.34 MW of a 124.66 MW gain. Does the
  sentence's wording match the arithmetic? *(This is where the last error was.)*
- Does the Regime B weather genuinely postdate nothing? See
  `docs/METHODOLOGY.md` §3.2 on the admitted residual optimism.

### Claim 2 — "H = 2, so `peak_lag_1` is unusable"

*Where:* `results/publish_time.json`, `docs/METHODOLOGY.md` §1

- 46 runs is the whole evidence base. Is that enough?
- The latest observed run was 18:07. The evening job runs at 20:00. **Does the
  18:07 observation actually justify a claim about 20:00?**
- If H were really 1, how much of the paper collapses?

### Claim 3 — "No selection optimism; our numbers are conservative"

*Where:* `run_selection_audit.py`, `results/selection_audit.json`

- The replay ranks model families on **default** hyperparameters. A family
  with a bad default never reaches tuning — this is admitted in
  `docs/METHODOLOGY.md` §8.1. Does that invalidate the −5.80 MW result?
- The replay reuses A0/A1/A2/A4 feature sets that were **themselves** selected
  using 2025. Is the "held out" claim actually true?
- Optimism is `honest − reported`. Confirm the sign convention is used
  consistently everywhere it appears.

### Claim 4 — "2026 confirms the model generalises"

*Where:* `results/final_evaluation.json`

- It is called a confirmation, not a blind test, because 2026 was inspected
  during development. **Is the disclosure strong enough, or does the paper
  still lean on 2026 as if it were held out?**
- 314 days, MAE 503.49 vs 497.90 expected. Is +1.1% actually reassuring, or
  within noise either way?

---

## Five questions that would most help

1. Read `paper/PAPER.md` **abstract only**. Write down what you think was
   done. Does it match §5?
2. Pick any number in the deck. Trace it to a file in `results/`. Can you?
3. Is the novelty claim honest? It says *"we are not aware of a study that
   isolates…"* — deliberately weak. Is it weak enough?
4. `docs/ANALYSIS.md` §6 says festivals are the biggest opportunity (−11% MAE).
   Does the evidence support that, or is it over-claimed?
5. **What is the most embarrassing question a reviewer could ask?**

---

## Known-weak — you do not need to find these

Already documented, so don't spend time here:

| | |
|---|---|
| Surge days under-forecast by 496 MW | unsolved, disclosed |
| Diwali over-forecast by 1,766 MW | n=15, disclosed |
| Residual ACF +0.481 | structure remains; a correction was tested and made things **worse** |
| 269 rows dated 2013 attributed to a 2014–2024 dataset | **open**, flagged in README |
| Only 1 live forecast logged, and it used the stand-in | fixed going forward, but history can't be recovered |
| Single region, single target | disclosed |

---

## How to check it still runs

```bash
./reproduce.sh --tests    # 147 acceptance checks, ~36s
./reproduce.sh            # everything, including paper and deck
git diff --stat results/ figures/ models/
```

A clean diff means the committed numbers reproduce. Anything else is a finding.

---

## Where to write findings

Open a GitHub issue per finding, or annotate directly. **Severity you should
use freely:** if a number in the paper cannot be traced to an artifact, that is
not a nitpick — that is the exact failure mode that produced the last two
errors.
