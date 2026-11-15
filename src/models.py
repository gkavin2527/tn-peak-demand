"""
Phase 6 model zoo.

Every model exposes the same two methods as `src.ridge.TrackARidge`:

    fit(X: DataFrame, y: Series) -> self
    predict(X: DataFrame) -> np.ndarray

so `run_phase6_benchmark.py` never branches on model type. Adding a candidate
means adding one entry to `MODEL_ZOO`.

Why these models and not others
-------------------------------
The project's known failure mode is extrapolation: Tamil Nadu's peak demand
grew ~3.5%/yr and 2026 ran ~10% above 2025, so validation years routinely
contain demand levels above anything in training. Tree ensembles predict a
leaf average and therefore **cannot** output a value above their training
maximum, while a linear model extrapolates along its fitted trend. That single
fact shapes the whole benchmark:

  Naive / SeasonalNaive  floor. A model that cannot beat "last Tuesday" is
                         not learning anything.
  OLS                    isolates how much work the L2 penalty is doing. At
                         alpha=0.001 Ridge is nearly OLS, and demonstrating
                         that is a legitimate finding rather than a gap.
  TrackARidge            the incumbent champion and the reference every
                         candidate is paired against.
  LightGBM / XGBoost     two independent GBM implementations, so a verdict of
                         "trees underperform here" cannot be blamed on one
                         library's defaults.
  LightGBMLinearTree     LightGBM with linear models in the leaves. This is
                         the single most relevant option in the project: it
                         keeps non-linear splitting while restoring the
                         ability to extrapolate inside a leaf.
  CatBoost               ordered boosting, included for completeness. Lowest
                         marginal value on ~4,900 numeric rows.
  RidgeGBMHybrid         Ridge captures the trend and extrapolates; the GBM
                         models the non-linear weather residual. Directly
                         targets the failure mode rather than hoping a tree
                         works around it.

Threading
---------
`threadlimit` must be imported before numpy/sklearn/LightGBM. On ~4,900 rows
these libraries are an order of magnitude SLOWER on all cores than on one -
thread synchronisation dominates at this problem size. Every estimator below
is therefore pinned to a single thread; this is a speed decision, not a
correctness one, and does not change results.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy / sklearn / lightgbm

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from src.ridge import TrackARidge

SEED = RANDOM_SEED or 42

__all__ = ["NaivePersistence", "SeasonalNaive", "OLSRegression",
           "LightGBMModel", "XGBoostModel", "CatBoostModel",
           "RidgeGBMHybrid", "MODEL_ZOO", "build_model", "available_models"]


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

class _ColumnEcho:
    """Predicts by echoing one feature column. Base for the naive baselines.

    These fit nothing. They exist to establish the floor that any learned
    model must clear, and they read their column straight out of the feature
    matrix so they are scored on exactly the same rows as everything else.
    """

    column: str = ""

    def __init__(self, column: str | None = None):
        if column:
            self.column = column

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        if self.column not in X.columns:
            raise KeyError(f"{type(self).__name__} needs '{self.column}' in the "
                           f"feature set; got {list(X.columns)[:8]}...")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)


class NaivePersistence(_ColumnEcho):
    """Tomorrow equals the newest demand we actually hold.

    At H=2 that is `peak_lag_2` (T-2), NOT `peak_lag_1` - day T-1's peak is
    not published when the forecast is issued.
    """
    column = "peak_lag_2"


class SeasonalNaive(_ColumnEcho):
    """Tomorrow equals the same weekday last week (`peak_lag_7`, T-7).

    Usually the stronger of the two naive baselines here, because the
    weekday cycle is worth ~2,200 MW between Sunday and a working day.
    """
    column = "peak_lag_7"


class OLSRegression(TrackARidge):
    """Ordinary least squares, via the Ridge solver with a negligible penalty.

    Reuses TrackARidge's exact standardise/centre/closed-form path so the only
    difference from the incumbent is the penalty strength. A tiny ridge term
    (1e-10) is retained purely for numerical stability of the solve - with 26
    standardised features on ~3,000 rows the design matrix is well
    conditioned, so this is numerically OLS.
    """

    def __init__(self, alpha: float = 1e-10):
        super().__init__(alpha=alpha)


# --------------------------------------------------------------------------
# Gradient boosting
# --------------------------------------------------------------------------

class _SklearnStyle:
    """Thin adapter giving third-party estimators the TrackARidge interface."""

    def __init__(self, estimator):
        self.est = estimator
        self.features_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self.features_ = list(X.columns)
        if sample_weight is not None:
            self.est.fit(X, y, sample_weight=sample_weight)
        else:
            self.est.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est.predict(X), dtype=float).ravel()


def LightGBMModel(linear_tree: bool = False, **kw):
    """LightGBM with conservative defaults suited to ~3,000 training rows.

    Defaults are deliberately restrained rather than tuned: the benchmark's
    job is to rank model FAMILIES fairly, and only the top candidates are
    tuned afterwards. `num_leaves=15` and `min_child_samples=30` keep the
    trees shallow enough not to memorise a small daily series.

    `linear_tree=True` fits a linear model in each leaf, which restores
    extrapolation within a leaf - the property plain trees lack on this
    trending target.
    """
    import lightgbm as lgb
    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=15,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, n_jobs=1, verbose=-1)
    if linear_tree:
        # linear leaves need their own regularisation or they overfit badly
        params.update(linear_tree=True, linear_lambda=1.0)
    params.update(kw)
    return _SklearnStyle(lgb.LGBMRegressor(**params))


def XGBoostModel(**kw):
    import xgboost as xgb
    params = dict(n_estimators=600, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=SEED, n_jobs=1,
                  tree_method="hist", verbosity=0)
    params.update(kw)
    return _SklearnStyle(xgb.XGBRegressor(**params))


def CatBoostModel(**kw):
    from catboost import CatBoostRegressor
    params = dict(iterations=600, learning_rate=0.03, depth=4,
                  l2_leaf_reg=3.0, random_seed=SEED, thread_count=1,
                  verbose=0, allow_writing_files=False)
    params.update(kw)
    return _SklearnStyle(CatBoostRegressor(**params))


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------

class RidgeGBMHybrid:
    """Ridge for the trend, gradient boosting for the non-linear residual.

    Fit Ridge on the full feature set, then fit a GBM to the Ridge residuals
    and add the two predictions. The point is to keep the one property trees
    lack - extrapolation along a trend - while still capturing a non-linear
    temperature response.

    The residual GBM is deliberately weaker than the standalone GBMs
    (fewer, shallower trees): it is correcting a residual, not learning the
    series from scratch, and a strong residual learner would simply overwrite
    Ridge's extrapolation with leaf averages.
    """

    def __init__(self, alpha: float = 0.001, n_estimators: int = 300,
                 learning_rate: float = 0.03, num_leaves: int = 7,
                 half_life_years: float | None = None):
        self.alpha = alpha
        self.ridge = TrackARidge(alpha=alpha)
        self._gbm_kw = dict(n_estimators=n_estimators,
                            learning_rate=learning_rate, num_leaves=num_leaves)
        self.gbm = None
        self.features_: list[str] | None = None
        if half_life_years is not None and half_life_years <= 0:
            raise ValueError(
                f"half_life_years must be positive or None, got {half_life_years}")
        self.half_life_years = half_life_years

    def _recency_weights(self, X: pd.DataFrame, sample_weight=None):
        """Exponential decay by age, half-life `half_life_years`.

        Tamil Nadu's peak is not stationary - 2026 runs ~10% above 2025 - so
        the oldest rows describe a different system than the one being
        forecast. Weighting by recency lets the fit prefer recent structure
        without discarding history outright, which a hard training-window cut
        would do.

        Weights come from the frame's own DatetimeIndex rather than a
        row-order proxy: master.csv has 157 missing days, so position is not
        age. `None` means uniform, and must stay the default - every committed
        Phase 5/6 number was fitted unweighted.
        """
        if self.half_life_years is None:
            return sample_weight
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError(
                "half_life_years needs a DatetimeIndex to measure age from; "
                f"got {type(X.index).__name__}")
        age_days = (X.index.max() - X.index).days.to_numpy(dtype=float)
        w = 0.5 ** (age_days / (365.25 * self.half_life_years))
        if sample_weight is not None:
            w = w * np.asarray(sample_weight, dtype=float).ravel()
        return w

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        import lightgbm as lgb
        self.features_ = list(X.columns)
        y_arr = np.asarray(y, dtype=float).ravel()
        sample_weight = self._recency_weights(X, sample_weight)
        self.ridge.fit(X, y_arr, sample_weight=sample_weight)
        resid = y_arr - self.ridge.predict(X)
        self.gbm = lgb.LGBMRegressor(
            min_child_samples=30, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
            n_jobs=1, verbose=-1, **self._gbm_kw)
        self.gbm.fit(X, resid, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.ridge.predict(X) + np.asarray(self.gbm.predict(X), float).ravel()


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

class QuantileHybrid(RidgeGBMHybrid):
    """Ridge trend + a QUANTILE-regression GBM on the residual.

    Motivation, measured on the champion's Regime B residuals (2023-25):
    the top-5% surge days carry a bias of -549.0 MW on n=56, with 48 of 56
    under-forecast. The model is systematically low exactly when the reserve
    margin is thinnest, which is the operationally expensive direction to be
    wrong in.

    RidgeGBMHybrid fits the residual GBM to squared error, so it targets the
    conditional MEAN of the residual. This subclass changes only the residual
    objective to LightGBM's `quantile` at level `quantile`, so the head
    targets an upper conditional quantile instead.

    The trade-off is not free and should be stated before looking at the
    numbers: shifting the residual head upward reduces under-forecast bias
    everywhere, not only on surge days, so overall MAE must be expected to
    rise. `quantile=0.5` is the median control - it isolates "mean vs median"
    from "median vs upper quantile". Whether any level is worth adopting is a
    question about the cost asymmetry of over- versus under-forecasting, which
    the MAE gate alone cannot answer; run_phase7_surge.py reports the whole
    curve rather than a single point.
    """

    def __init__(self, alpha: float = 0.001, n_estimators: int = 300,
                 learning_rate: float = 0.03, num_leaves: int = 7,
                 quantile: float = 0.9, half_life_years: float | None = None):
        super().__init__(alpha=alpha, n_estimators=n_estimators,
                         learning_rate=learning_rate, num_leaves=num_leaves,
                         half_life_years=half_life_years)
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0,1), got {quantile}")
        self.quantile = float(quantile)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        import lightgbm as lgb
        self.features_ = list(X.columns)
        y_arr = np.asarray(y, dtype=float).ravel()
        sample_weight = self._recency_weights(X, sample_weight)
        self.ridge.fit(X, y_arr, sample_weight=sample_weight)
        resid = y_arr - self.ridge.predict(X)
        self.gbm = lgb.LGBMRegressor(
            objective="quantile", alpha=self.quantile,
            min_child_samples=30, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
            n_jobs=1, verbose=-1, **self._gbm_kw)
        self.gbm.fit(X, resid, sample_weight=sample_weight)
        return self


MODEL_ZOO: dict[str, dict] = {
    "naive_persistence": {
        "factory": lambda **kw: NaivePersistence(**kw),
        "family": "baseline",
        "note": "echoes peak_lag_2 (T-2), the newest demand available at H=2",
    },
    "seasonal_naive": {
        "factory": lambda **kw: SeasonalNaive(**kw),
        "family": "baseline",
        "note": "echoes peak_lag_7 (T-7), same weekday last week",
    },
    "ols": {
        "factory": lambda **kw: OLSRegression(**kw),
        "family": "linear",
        "note": "unpenalised; isolates the contribution of Ridge's L2 term",
    },
    "ridge": {
        "factory": lambda **kw: TrackARidge(**{"alpha": 0.001, **kw}),
        "family": "linear",
        "note": "incumbent champion and the paired-comparison reference",
    },
    "lightgbm": {
        "factory": lambda **kw: LightGBMModel(**kw),
        "family": "gbm",
        "note": "standard GBM; cannot extrapolate beyond training range",
    },
    "lightgbm_linear": {
        "factory": lambda **kw: LightGBMModel(**{"linear_tree": True, **kw}),
        "family": "gbm",
        "note": "linear models in leaves; restores within-leaf extrapolation",
    },
    "xgboost": {
        "factory": lambda **kw: XGBoostModel(**kw),
        "family": "gbm",
        "note": "independent GBM implementation as a cross-check on LightGBM",
    },
    "catboost": {
        "factory": lambda **kw: CatBoostModel(**kw),
        "family": "gbm",
        "note": "ordered boosting; lowest marginal value on this data size",
    },
    "ridge_gbm_hybrid": {
        "factory": lambda **kw: RidgeGBMHybrid(**kw),
        "family": "hybrid",
        "note": "Ridge trend + GBM residual; targets the extrapolation failure",
    },
    "quantile_hybrid": {
        "factory": lambda **kw: QuantileHybrid(**kw),
        "family": "hybrid",
        "note": ("Ridge trend + quantile-regression GBM residual; targets the "
                 "-549 MW surge-day under-forecast, at a cost in overall MAE"),
    },
}

REFERENCE_MODEL = "ridge"


def available_models() -> list[str]:
    """Zoo entries whose dependencies are importable in this environment.

    Reports missing dependencies rather than silently skipping a model - a
    silently absent candidate is how "Ridge is best" becomes an artefact of
    the environment instead of a finding.
    """
    ok = []
    for name in MODEL_ZOO:
        try:
            build_model(name)
            ok.append(name)
        except ImportError as e:
            print(f"  [SKIP] {name}: missing dependency ({e})")
    return ok


def build_model(name: str, **hyperparameters):
    """Instantiate a zoo model, optionally overriding its hyperparameters.

    Passing no kwargs reproduces the benchmark default, which is what the
    Phase 6 ranking used. Passing kwargs is how a tuned configuration is
    rebuilt - serving and tuning must construct the champion through this one
    function, because two construction paths is how a served model silently
    stops being the model that was selected.
    """
    if name not in MODEL_ZOO:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_ZOO)}")
    return MODEL_ZOO[name]["factory"](**hyperparameters)


def build_from_config(cfg: dict):
    """Rebuild the frozen champion from models/champion_config.json.

    The config is the single source of truth for model type, hyperparameters
    and - critically - the target parameterisation. Returns
    (model, features, parameterisation) so a caller cannot pick up the
    feature list without also picking up how the target was expressed.
    """
    name = cfg.get("model_type") or cfg.get("champion")
    if not name:
        raise KeyError("champion config names no model ('model_type')")
    features = list(cfg["features"])
    if not features:
        raise ValueError("champion config carries an empty feature list")
    if cfg.get("feature_count") not in (None, len(features)):
        raise ValueError(
            f"champion config is internally inconsistent: feature_count="
            f"{cfg['feature_count']} but {len(features)} features listed")

    param = cfg.get("target_parameterisation", "level")
    if param not in ("level", "differenced"):
        raise ValueError(f"unknown target_parameterisation {param!r}")
    if param == "differenced" and not cfg.get("anchor"):
        raise ValueError("differenced champion must name its 'anchor' column")

    model = build_model(name, **(cfg.get("hyperparameters") or {}))
    return model, features, param
