"""Environment capture for result provenance.

The champion is reconstructed by re-fitting, not unpickled, so a saved
configuration only reproduces its numbers in combination with the library
versions that produced them. Every artifact that claims a metric should carry
this block, so a number that moves can be attributed to a version bump rather
than argued about.
"""

from __future__ import annotations

import platform
import sys

# Libraries whose version can move a reported metric. Deliberately not every
# installed package: requests or pdfplumber cannot change an MAE.
MODELLING_LIBS = ("numpy", "pandas", "scipy", "sklearn", "lightgbm",
                  "xgboost", "catboost", "optuna")

_DIST = {"sklearn": "scikit-learn"}


def capture_environment(libs=MODELLING_LIBS) -> dict:
    """Versions of the libraries that can affect a reported number."""
    env = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "platform": platform.platform(),
    }
    for name in libs:
        try:
            mod = __import__(name)
            env[_DIST.get(name, name)] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[_DIST.get(name, name)] = None
    return env


if __name__ == "__main__":
    import json
    print(json.dumps(capture_environment(), indent=2))
