# Documentation index

Three places things live:

| Folder | Contents | Safe to delete? |
|---|---|---|
| **`/`** | `README.md` — the dataset and pipeline reference | No |
| **`paper/`** | manuscript, deck, related work, and their generators | **Yes, as a unit** — nothing else depends on it |
| **`docs/`** | project documentation (this folder) | No — contains the only record of several decisions |

---

## Start here

| Document | What it is | Status |
|---|---|---|
| **[ANALYSIS.md](ANALYSIS.md)** | **Complete project analysis** — what's done, strong, weak, missing, and what would actually improve the model | ⭐ **current** |
| [RESULTS.md](RESULTS.md) | Authoritative results. Every number generated from `results/*.json` | **current** |
| [METHODOLOGY.md](METHODOLOGY.md) | How everything works, in detail — horizon, regimes, protocol, serving | **current** |
| [CONTRACT.md](CONTRACT.md) | The Track A ↔ Track B interface. Read before touching the serving path | **current** |

## Historical record

These describe earlier phases. They are **not wrong**, but they are not
current — each carries a banner saying so. Kept because they are the only
record of why certain decisions were made.

| Document | What it is |
|---|---|
| [BENCHMARK.md](BENCHMARK.md) | Phase 6 model benchmark. Still the authoritative record of how the nine model families ranked |
| [ROADMAP.md](ROADMAP.md) | Phase-by-phase plan and task tracking |
| [CONCEPT.md](CONCEPT.md) | Original project concept and MVP design |
| [AUDIT.md](AUDIT.md) | Independent audit (2026-11-15) that triggered Phases 7–8 |
| [REMEDIATION.md](REMEDIATION.md) | The plan that answered the audit. Waves 0–3 are complete |

> Both `AUDIT.md` and `REMEDIATION.md` contain a **superseded** headline
> sentence ("7.1% of that gain"), preserved with a correction note above it.
> The correct figure is **32%** — see [ANALYSIS.md §2.1](ANALYSIS.md).

---

## Where the numbers come from

Nothing in `RESULTS.md`, `paper/PAPER.md` or `paper/DECK.md` is typed by hand.
All three are generated from committed artifacts:

```
results/*.json  ──┬─→ paper/make_paper.py ─→ paper/PAPER.md
                  └─→ paper/make_deck.py  ─→ paper/DECK.md
```

This exists because the project twice shipped a wrong number that survived
people reading prose — a residual ACF belonging to a superseded model, and a
headline that used the wrong denominator. Re-run before any submission:

```bash
./reproduce.sh          # everything, including paper and deck
./reproduce.sh --tests  # the 147 acceptance checks only
```

If a generated number changes, an experiment changed. Find out which.
