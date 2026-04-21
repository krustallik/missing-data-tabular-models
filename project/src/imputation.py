"""Missing-value imputation strategies.

Supports every method the benchmark needs:

- ``mean``    — column mean from train.
- ``median``  — column median from train.
- ``knn``     — ``sklearn.impute.KNNImputer`` (``n_neighbors=5``).
- ``mice``    — ``sklearn.impute.IterativeImputer`` (MICE).
- ``mice_indicator`` — MICE plus binary missing-indicator features.
- ``none``    — pass-through: NaN stays in, used for foundation models that
  handle NaN natively (TabPFN / TabICL / CatBoost / XGBoost / LightGBM).

Every method fits on train only and applies the fitted transformer to both
train and test to avoid test leakage. Output is always a tuple of DataFrames
with identical columns (possibly extended by indicators).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from config import IMPUTATION_METHODS, RANDOM_STATE
from data_utils import coerce_features


__all__ = ["impute", "IMPUTATION_METHODS"]


def _align_test_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    return X_test.reindex(columns=X_train.columns)


def _append_indicators(
    X_train_imp: pd.DataFrame, X_test_imp: pd.DataFrame,
    X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate binary NaN indicators computed from the *raw* (pre-impute) frames."""
    ind_train = X_train_raw.isna().astype(float)
    ind_test = X_test_raw.isna().astype(float)
    ind_train.columns = [f"{c}__missing" for c in ind_train.columns]
    ind_test.columns = [f"{c}__missing" for c in ind_test.columns]
    out_train = pd.concat([X_train_imp.reset_index(drop=True), ind_train.reset_index(drop=True)], axis=1)
    out_test = pd.concat([X_test_imp.reset_index(drop=True), ind_test.reset_index(drop=True)], axis=1)
    return out_train, out_test


def _mean_impute(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    means = X_train.mean(numeric_only=True)
    return X_train.fillna(means).fillna(0.0), X_test.fillna(means).fillna(0.0)


def _median_impute(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    medians = X_train.median(numeric_only=True)
    return X_train.fillna(medians).fillna(0.0), X_test.fillna(medians).fillna(0.0)


def _knn_impute(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    imputer = KNNImputer(n_neighbors=5)
    X_tr = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index,
    )
    X_te = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index,
    )
    return X_tr.fillna(0.0), X_te.fillna(0.0)


def _mice_impute(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Enables experimental IterativeImputer. Imported lazily because sklearn
    # may emit a ConvergenceWarning which is fine but noisy.
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(
        random_state=RANDOM_STATE, max_iter=10, sample_posterior=False,
    )
    X_tr = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index,
    )
    X_te = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index,
    )
    return X_tr.fillna(0.0), X_te.fillna(0.0)


def impute(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit an imputer on ``X_train`` and return imputed train / test frames.

    ``method='none'`` is a pass-through (NaN values are preserved). All other
    methods return fully numeric dataframes with no NaN, ready to feed into
    ``StandardScaler`` and any downstream classifier.
    """
    if method not in IMPUTATION_METHODS:
        raise ValueError(f"Unknown imputation method: {method!r}. Known: {IMPUTATION_METHODS}")

    X_train = coerce_features(X_train)
    X_test = _align_test_columns(X_train, coerce_features(X_test))

    if method == "none":
        return X_train, X_test
    if method == "mean":
        return _mean_impute(X_train, X_test)
    if method == "median":
        return _median_impute(X_train, X_test)
    if method == "knn":
        return _knn_impute(X_train, X_test)
    if method == "mice":
        return _mice_impute(X_train, X_test)
    if method == "mice_indicator":
        X_tr_imp, X_te_imp = _mice_impute(X_train, X_test)
        return _append_indicators(X_tr_imp, X_te_imp, X_train, X_test)
    raise ValueError(f"Unhandled method: {method}")
