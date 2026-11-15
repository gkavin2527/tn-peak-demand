#!/usr/bin/env bash
#
# W3-2 - one-command reproduction of every committed result.
#
#     ./reproduce.sh            full run (tests + all experiments)
#     ./reproduce.sh --quick    tests + fast checks only, no tuning
#     ./reproduce.sh --tests    acceptance suites only
#
# Deliberately does NOT run:
#   run_final_evaluation.py   the 2026 confirmation is a ONE-SHOT and is
#                             already consumed. Re-running it after seeing the
#                             result turns a confirmation into another
#                             selection round. It refuses without --force.
#   build_dataset.py          rebuilds data/master.csv from raw sources and
#                             needs network. Run it deliberately.
#
# Ordering is load-bearing: the acceptance suites run FIRST, so a broken
# contract stops the run before it spends an hour producing numbers that would
# be wrong anyway.

set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
MODE="${1:-full}"
START=$(date +%s)

if [ ! -x "$PY" ]; then
  echo "error: no interpreter at $PY" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
run()  {
  printf '  -> %s\n' "$*"
  if ! "$PY" "$@"; then
    echo "FAILED: $*" >&2
    exit 1
  fi
}

step "environment"
"$PY" -c "import sys; print('  python', '.'.join(map(str, sys.version_info[:3])))"
"$PY" src/envinfo.py | sed 's/^/  /'

step "acceptance suites (these gate everything below)"
for t in test_contract test_phase5_gateway test_phase6_benchmark \
         test_serving test_phase7_festival; do
  printf '  -> %-28s' "$t"
  if "$PY" -m "tests.$t" >/dev/null 2>&1; then echo "PASS"; else echo "FAIL"; exit 1; fi
done

if [ "$MODE" = "--tests" ]; then
  echo; echo "tests only: done in $(( $(date +%s) - START ))s"; exit 0
fi

step "Phase 6 - model benchmark"
run -m experiments.run_phase6_benchmark

step "Phase 7 - festival gate (A2 vs A4)"
run -m experiments.run_phase7_festival

step "Phase 7 - surge-day bias sweep"
run -m experiments.run_phase7_surge

step "Phase 7 - recency weighting"
run -m experiments.run_phase7_recency

step "statistics - Diebold-Mariano alongside the bootstrap"
run -m experiments.run_dm_tests

step "analysis - segment errors and residual diagnostics"
run -m experiments.run_error_analysis

step "analysis - champion interpretability"
run -m experiments.run_interpretability

if [ "$MODE" = "--quick" ]; then
  step "figures"
  run -m experiments.make_figures
  echo; echo "quick run complete in $(( $(date +%s) - START ))s"
  echo "skipped: tuning, selection audit, interval calibration (slow)"
  exit 0
fi

step "Phase 6b - hyperparameter tuning"
run -m experiments.run_phase6_tuning

step "Phase 7 - champion re-freeze"
run -m experiments.run_phase7_refreeze

step "W0-1 - selection optimism audit"
run -m experiments.run_selection_audit

step "W2-3 - conformal interval calibration"
run calibrate_intervals.py

step "figures"
run -m experiments.make_figures

step "paper and deck (numbers read from results/, never typed)"
run paper/make_paper.py
run paper/make_deck.py

step "re-verify after regeneration"
for t in test_serving test_phase7_festival; do
  printf '  -> %-28s' "$t"
  if "$PY" -m "tests.$t" >/dev/null 2>&1; then echo "PASS"; else echo "FAIL"; exit 1; fi
done

ELAPSED=$(( $(date +%s) - START ))
step "done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
cat <<'NOTE'
  Artifacts refreshed in results/ and figures/.

  NOT run, by design:
    run_final_evaluation.py  - one-shot 2026 confirmation, already consumed
    build_dataset.py         - rebuilds master.csv from source, needs network

  Compare against the committed results with:
    git diff --stat results/ figures/ models/
  Any change there means a dependency moved or a seed drifted - investigate
  before publishing. requirements.txt pins the versions that produced them.
NOTE
