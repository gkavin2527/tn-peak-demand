"""
Cap BLAS/OpenMP threads. **Import this before numpy or sklearn.**

    import threadlimit  # noqa: F401  - must precede numpy/sklearn

Why this exists
---------------
This project's models fit on ~4,900 rows x 46 features. That is small, and
HistGradientBoostingRegressor's OpenMP parallelism is counterproductive at
that size - the per-split thread synchronisation costs more than the work it
distributes. Measured on this dataset, one fit of the production configuration
(max_iter=400):

    OMP_NUM_THREADS=1   ->  0.95 s
    OMP_NUM_THREADS=2   ->  1.66 s
    OMP_NUM_THREADS=4   ->  3.21 s
    OMP_NUM_THREADS=10  ->  9.68 s     (the default: all cores)

Ten times slower on all cores than on one. The walk-forward experiments fit
~700-1,200 models, so the default turned a 12-minute job into a two-hour one,
and running two such jobs concurrently made both worse.

Single-threaded fits also make the walk-forward trivially parallelisable
across configurations if that is ever wanted, and remove a source of
run-to-run timing noise.

Set TN_THREADS to override (e.g. for a genuinely large refit).
"""

import os

_N = os.environ.get("TN_THREADS", "1")

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _N)

THREADS = int(_N)
