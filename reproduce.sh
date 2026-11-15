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
for t in test_contract.py test_phase5_gateway.py test_phase6_benchmark.py \
         test_serving.py test_phase7_festival.py; do
  printf '  -> %-28s' "$t"
  if "$PY" "$t" >/dev/null 2>&1; then echo "PASS"; else echo "FAIL"; exit 1; fi
done

if [ "$MODE" = "--tests" ]; then
  echo; echo "tests only: done in $(( $(date +%s) - START ))s"; exit 0
fi

step "Phase 6 - model benchmark"
run run_phase6_benchmark.py

step "Phase 7 - festival gate (A2 vs A4)"
run run_phase7_festival.py

step "Phase 7 - surge-day bias sweep"
run run_phase7_surge.py

step "Phase 7 - recency weighting"
run run_phase7_recency.py

step "statistics - Diebold-Mariano alongside the bootstrap"
run run_dm_tests.py

step "analysis - segment errors and residual diagnostics"
run run_error_analysis.py

step "analysis - champion interpretability"
run run_interpretability.py

if [ "$MODE" = "--quick" ]; then
  step "figures"
  run make_figures.py
  echo; echo "quick run complete in $(( $(date +%s) - START ))s"
  echo "skipped: tuning, selection audit, interval calibration (slow)"
  exit 0
fi

step "Phase 6b - hyperparameter tuning"
run run_phase6_tuning.py

step "Phase 7 - champion re-freeze"
run run_phase7_refreeze.py

step "W0-1 - selection optimism audit"
run run_selection_audit.py

step "W2-3 - conformal interval calibration"
run calibrate_intervals.py

step "figures"
run make_figures.py

step "re-verify after regeneration"
for t in test_serving.py test_phase7_festival.py; do
  printf '  -> %-28s' "$t"
  if "$PY" "$t" >/dev/null 2>&1; then echo "PASS"; else echo "FAIL"; exit 1; fi
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
