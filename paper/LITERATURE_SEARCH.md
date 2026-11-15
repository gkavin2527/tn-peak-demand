# Literature search log

**Why this file exists.** The paper's novelty claim is *"we are not aware of a
study that isolates the operational weather penalty as a separate measured
quantity under a single fixed harness."* A reviewer correctly pointed out that
this phrasing is weak in **form** but is only **honest** if a real search backs
it. This is that record.

**It is not exhaustive.** It is one person's search, documented so a reader can
judge its reach and repeat it. If it missed something, the claim should be
weakened or dropped — not defended.

---

## Searches performed — 2026-11-15

| Query | Tool | What came back |
|---|---|---|
| `Zenodo dataset India state-wise daily peak electricity demand POSOCO Grid-India` | web | Located the backfill source (10.5281/zenodo.14983362) |
| `short-term electricity load forecasting India state level machine learning Tamil Nadu` | web | Regional STLF; mostly ARIMA/LSTM single-region studies |
| `short-term load forecasting India electricity demand state` | Crossref API | Poor precision — mostly non-India results |
| `Hunt Bloomfield India electricity demand weather` | Crossref API | **Found the closest prior work** (see below) |
| `conditional variable importance random forests correlated predictors` | Crossref API | Strobl et al. 2008 |
| `load forecasting numerical weather prediction forecast uncertainty impact accuracy` | Crossref API | Zhao et al. 2018; Yang et al. 2021 |

All citations used were verified against **Crossref** records (title, authors,
venue, year, DOI) rather than taken from search summaries.

---

## The decisive find

**Hunt & Bloomfield (2025)**, *Building and explaining data-driven energy
demand models for Indian states*, Environmental Research: Energy,
DOI 10.1088/2753-3751/adc7bc.

Same authors as our backfill dataset, same Grid-India source, all 32 states,
published 2025. Checked directly against the publisher record. It:

- models daily **total energy**, not peak
- is a **nowcast/hindcast** — weather is contemporaneous with demand
- uses **ERA5 reanalysis only**, with **no forecast comparison**
- reports r² (median 0.78), not paired error differences

This is the nearest neighbour to our work and it does **not** perform the
comparison we claim as novel.

## The nearest methodological precedent

**Zhao, Duan & Liu (2018)**, Energies 11(7), DOI 10.3390/en11071900 — weather
forecast uncertainty in **cooling load** forecasting. They **simulate**
uncertainty by Monte Carlo. We **measure** it against a 5.4-year archive. This
is the closest anything came, and the distinction is real.

**Yang, Cerrai & Anagnostou (2021)**, Forecasting 3(3), DOI
10.3390/forecast3030031 — lead-time forecast uncertainty treated as
first-class, but for **outage prediction**, not load.

---

## What was NOT searched — state this if challenged

- Non-English literature
- Indian conference proceedings not indexed by Crossref
- Utility and regulator technical reports (CEA, SLDC, POSOCO internal)
- Theses and dissertations
- The wind/solar forecasting literature systematically — only spot-checked,
  though NWP error is treated as first-class there and a reviewer may know of
  a direct analogue

**Any of these could contain a counterexample.** The claim is therefore
deliberately phrased as "we are not aware of", and should be withdrawn rather
than argued if a reviewer produces one.
