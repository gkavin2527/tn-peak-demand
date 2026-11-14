"""
Pure NumPy implementation of the verified Track A Ridge Regression baseline.

Preserves the exact mathematical formulation from Phase 3 & 4 (Notebooks 03 & 04):
- Standardises features: zero-mean, unit-variance on training data
- Centers the target: y_train - mean(y_train)
- Closed-form solution: w = (X_s^T @ X_s + alpha * I)^(-1) @ X_s^T @ y_c
- Prediction: X_val_s @ w + mean(y_train)

Requires only NumPy and Pandas (zero scikit-learn dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TrackARidge:
    """Standardised Ridge Regression matching Track A's exact closed-form setup."""

    def __init__(self, alpha: float = 0.001):
        self.alpha = float(alpha)
        self.mu_X_: np.ndarray | None = None
        self.sig_X_: np.ndarray | None = None
        self.mu_y_: float | None = None
        self.w_: np.ndarray | None = None
        self.features_: list[str] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray,
            sample_weight: np.ndarray | None = None) -> TrackARidge:
        if isinstance(X, pd.DataFrame):
            self.features_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.to_numpy(dtype=np.float64).ravel()
        else:
            y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.mu_X_ = X_arr.mean(axis=0)
        self.sig_X_ = X_arr.std(axis=0)
        self.sig_X_[self.sig_X_ == 0.0] = 1.0

        X_s = (X_arr - self.mu_X_) / self.sig_X_
        self.mu_y_ = float(y_arr.mean())
        y_c = y_arr - self.mu_y_

        p = X_arr.shape[1]
        I = np.eye(p, dtype=np.float64)

        if sample_weight is not None:
            w_diag = np.asarray(sample_weight, dtype=np.float64).ravel()
            w_diag = w_diag / w_diag.mean()
            X_w = X_s * np.sqrt(w_diag)[:, None]
            y_w = y_c * np.sqrt(w_diag)
            self.w_ = np.linalg.solve(X_w.T @ X_w + self.alpha * I, X_w.T @ y_w)
        else:
            self.w_ = np.linalg.solve(X_s.T @ X_s + self.alpha * I, X_s.T @ y_c)

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model must be fitted before predict()")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        X_s = (X_arr - self.mu_X_) / self.sig_X_
        return (X_s @ self.w_) + self.mu_y_

