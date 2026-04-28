"""Model registry: classical sklearn models + pretrained foundation models.

Every model is addressed by its canonical snake_case key (see
:data:`config.ALL_MODELS`). Two public entry points:

- :func:`train_and_evaluate` — fit the chosen model, predict on test, and
  return metrics + training time + any error message.
- :func:`display_name` — canonical display name used in CSVs / reports /
  plots (e.g. ``"Logistic-Regression"``, ``"TabPFN"``).

Classical models go through ``StandardScaler`` when needed; foundation models
(TabPFN / TabICL) and the NaN-aware classical CatBoost are called with raw
numpy arrays since they bring their own preprocessing (and often their own
NaN handling).

Imbalance handling
~~~~~~~~~~~~~~~~~~
Controlled from ``config``:

- ``USE_CLASS_WEIGHT=True`` → ``class_weight='balanced'`` for
  LR / RF / SVM / LightGBM, ``scale_pos_weight=neg/pos`` for XGBoost,
  ``auto_class_weights='Balanced'`` for CatBoost,
  ``sample_weight`` for GradientBoosting. MLP and foundation models do not
  support class weighting.
- ``TUNE_THRESHOLD=True`` → for binary tasks, after a first fit on a
  stratified 80 % slice of train, search the decision threshold that
  maximises ``THRESHOLD_METRIC`` on the held-out 20 %; then refit on
  the full train and apply that threshold to test predictions.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    CLASSICAL_MODELS,
    FOUNDATION_MODELS,
    MODELS_ACCEPTING_NAN,
    RANDOM_STATE,
    THRESHOLD_METRIC,
    THRESHOLD_MIN_MINORITY,
    THRESHOLD_VAL_FRACTION,
    TUNE_THRESHOLD,
    USE_CLASS_WEIGHT,
)
from evaluation import compute_metrics


# ── Display names ────────────────────────────────────────────────────────────

DISPLAY_NAME = {
    "logistic_regression": "Logistic-Regression",
    "random_forest": "Random-Forest",
    "gradient_boosting": "Gradient-Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "svm": "SVM",
    "mlp": "MLP",
    "tabpfn": "TabPFN",
    "tabicl": "TabICL",
    "catboost": "CatBoost",
}


def display_name(model_key: str) -> str:
    """Return the canonical display name for a model key."""
    if model_key in DISPLAY_NAME.values():
        return model_key
    return DISPLAY_NAME.get(str(model_key).lower(), str(model_key))


def model_type(model_key: str) -> str:
    key = str(model_key).lower()
    if key in FOUNDATION_MODELS:
        return "Foundation"
    return "Classical"


def _is_gpu_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    hints = ["cuda", "cudnn", "cublas", "gpu", "opencl", "nvidia", "device", "driver"]
    return any(h in msg for h in hints)


# ── Imbalance helpers ────────────────────────────────────────────────────────

def _as_array(y) -> np.ndarray:
    return y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)


def _is_binary(y) -> bool:
    return len(np.unique(_as_array(y))) == 2


def _scale_pos_weight(y) -> float:
    """XGBoost ``scale_pos_weight`` = n_neg / n_pos (positive class is 1)."""
    y_arr = _as_array(y)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr != 1).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def _sample_weight_balanced(model_key: str, y) -> Optional[np.ndarray]:
    """Return balanced sample weights for models that need them via fit()."""
    if not USE_CLASS_WEIGHT:
        return None
    if model_key in {"gradient_boosting"}:
        return compute_sample_weight(class_weight="balanced", y=_as_array(y))
    return None


def _tune_threshold(y_true, proba_pos: np.ndarray) -> float:
    """Grid-search the decision threshold that maximises ``THRESHOLD_METRIC``.

    Candidates are the union of a fine grid in (0,1) and the empirical
    probabilities observed on the validation set. Ties are broken toward
    thresholds closer to 0.5 for stability.
    """
    y_true_arr = _as_array(y_true)
    if y_true_arr.size == 0 or proba_pos.size == 0:
        return 0.5
    grid = np.linspace(0.01, 0.99, 99)
    cand = np.unique(np.concatenate([grid, proba_pos]))
    cand = cand[(cand > 0.0) & (cand < 1.0)]
    if cand.size == 0:
        return 0.5

    best_thr = 0.5
    best_score = -np.inf
    best_dist = np.inf
    for thr in cand:
        y_pred = (proba_pos >= thr).astype(int)
        if THRESHOLD_METRIC == "balanced_accuracy":
            s = balanced_accuracy_score(y_true_arr, y_pred)
        else:
            s = f1_score(y_true_arr, y_pred, average="macro", zero_division=0)
        dist = abs(float(thr) - 0.5)
        if s > best_score + 1e-12 or (abs(s - best_score) <= 1e-12 and dist < best_dist):
            best_thr, best_score, best_dist = float(thr), float(s), dist
    return best_thr


def _safe_stratified_holdout(X_tr, y_tr) -> Optional[Tuple[Any, Any, Any, Any]]:
    """Return a stratified 80/20 split of train, or None if not feasible.

    We perform the actual split and then check whether the validation slice
    ended up with at least ``THRESHOLD_MIN_MINORITY`` minority samples — this
    avoids the floor-vs-ceil off-by-one you hit with very rare classes
    (e.g. 24 positives → ``int(24*0.2)=4`` but stratify actually puts 5 in val).
    """
    y_arr = _as_array(y_tr)
    if not _is_binary(y_arr):
        return None
    try:
        split = train_test_split(
            X_tr, y_tr,
            test_size=THRESHOLD_VAL_FRACTION,
            stratify=y_arr,
            random_state=RANDOM_STATE,
        )
    except Exception:
        return None

    _, _, _, y_val = split
    y_val_arr = _as_array(y_val)
    classes, counts = np.unique(y_val_arr, return_counts=True)
    if classes.size < 2 or int(counts.min()) < THRESHOLD_MIN_MINORITY:
        return None
    return split


def _fit_predict_with_optional_threshold(
    fit_fn: Callable[[Any, Any], Any],
    proba_fn: Callable[[Any, Any], Optional[np.ndarray]],
    predict_fn: Callable[[Any, Any], np.ndarray],
    X_tr,
    y_tr,
    X_te,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Fit + predict with optional threshold tuning.

    ``fit_fn(X, y)``          → trained model
    ``proba_fn(model, X)``    → Nx2 array or None
    ``predict_fn(model, X)``  → N array (default argmax at 0.5)

    If ``TUNE_THRESHOLD`` is on and the task is binary with enough minority
    samples, we:
      1. stratified 80/20 split of train;
      2. fit on 80 %, find best threshold on 20 % probas;
      3. refit on full train;
      4. predict probas on test, apply threshold.

    Returns ``(y_pred, y_proba, threshold_used)``.
    """
    threshold = 0.5
    do_tune = TUNE_THRESHOLD and _is_binary(y_tr)

    if do_tune:
        split = _safe_stratified_holdout(X_tr, y_tr)
        if split is not None:
            X_fit, X_val, y_fit, y_val = split
            tmp_model = fit_fn(X_fit, y_fit)
            proba_val = proba_fn(tmp_model, X_val)
            if proba_val is not None and proba_val.ndim == 2 and proba_val.shape[1] == 2:
                threshold = _tune_threshold(y_val, proba_val[:, 1])

    model = fit_fn(X_tr, y_tr)
    proba_test = proba_fn(model, X_te)
    if threshold != 0.5 and proba_test is not None and proba_test.ndim == 2 and proba_test.shape[1] == 2:
        y_pred = (proba_test[:, 1] >= threshold).astype(int)
    else:
        y_pred = predict_fn(model, X_te)
    return y_pred, proba_test, threshold


# ── Classical model builders ─────────────────────────────────────────────────

def _build_classical(model_key: str, use_gpu: bool = False, y_train=None):
    cw = "balanced" if USE_CLASS_WEIGHT else None

    if model_key == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs",
            class_weight=cw,
        )
    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight=cw,
        )
    if model_key == "gradient_boosting":
        # GBM has no class_weight param; we pass sample_weight at fit time.
        return GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    if model_key == "svm":
        return SVC(
            kernel="rbf", random_state=RANDOM_STATE, probability=True,
            class_weight=cw,
        )
    if model_key == "mlp":
        # MLPClassifier has neither class_weight nor sample_weight. Minority
        # recall relies solely on threshold tuning.
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
        )
    if model_key == "xgboost":
        import xgboost as xgb

        spw = _scale_pos_weight(y_train) if (USE_CLASS_WEIGHT and y_train is not None) else 1.0
        return xgb.XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric="logloss", verbosity=0,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            scale_pos_weight=spw,
        )
    if model_key == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            device_type="gpu" if use_gpu else "cpu",
            class_weight=cw,
        )
    raise ValueError(f"Unknown classical model: {model_key!r}")


def _train_classical(
    model_key: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "metrics": None, "training_time_seconds": None, "error": None,
    }
    needs_scaling = model_key in {"logistic_regression", "svm", "mlp"}

    if needs_scaling:
        X_tr_raw = X_train.to_numpy(dtype=float)
        X_te_raw = X_test.to_numpy(dtype=float)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
    else:
        X_tr = X_train
        X_te = X_test

    prefers_gpu = model_key in {"xgboost", "lightgbm"} and _detect_device(logger=None) == "cuda"

    def _fit_once(use_gpu: bool) -> Dict[str, Any]:
        start = time.time()

        def fit_fn(X, y):
            model = clone(_build_classical(model_key, use_gpu=use_gpu, y_train=y))
            sw = _sample_weight_balanced(model_key, y)
            if sw is not None:
                model.fit(X, y, sample_weight=sw)
            else:
                model.fit(X, y)
            return model

        def proba_fn(model, X):
            if hasattr(model, "predict_proba"):
                try:
                    return np.asarray(model.predict_proba(X))
                except Exception:
                    return None
            return None

        def predict_fn(model, X):
            return np.asarray(model.predict(X))

        y_pred, y_proba, thr = _fit_predict_with_optional_threshold(
            fit_fn, proba_fn, predict_fn, X_tr, y_train, X_te,
        )
        elapsed = time.time() - start
        metrics = compute_metrics(_as_array(y_test), y_pred, y_proba)
        metrics["threshold"] = thr
        return {
            "metrics": metrics,
            "training_time_seconds": elapsed,
            "error": None,
        }

    try:
        out = _fit_once(use_gpu=prefers_gpu)
        result.update(out)
    except Exception as exc:
        if prefers_gpu and _is_gpu_runtime_error(exc):
            try:
                out = _fit_once(use_gpu=False)
                result.update(out)
                return result
            except Exception as cpu_exc:
                result["error"] = f"GPU failed ({exc}); CPU fallback failed ({cpu_exc})"
                return result
        result["error"] = str(exc)
    return result


# ── Foundation models ────────────────────────────────────────────────────────

_TABPFN_PREFLIGHT: Optional[Dict[str, Any]] = None
_TABPFN_BLOCKED: Optional[str] = None


def _detect_device(logger: Optional[logging.Logger] = None) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            if logger is not None:
                try:
                    name = torch.cuda.get_device_name(0)
                except Exception:
                    name = "cuda"
                logger.info(f"GPU detected: {name}")
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def _tabpfn_preflight(logger: Optional[logging.Logger]) -> Dict[str, Any]:
    global _TABPFN_PREFLIGHT
    if _TABPFN_PREFLIGHT is not None:
        return dict(_TABPFN_PREFLIGHT)

    out: Dict[str, Any] = {"available": False, "error": None, "device": None}
    token = os.getenv("TABPFN_TOKEN")
    require_token = os.getenv("TABPFN_REQUIRE_TOKEN", "1") == "1"
    if require_token and not token:
        out["error"] = (
            "TABPFN_TOKEN is not set. TabPFN skipped to avoid interactive "
            "login prompts. Set TABPFN_TOKEN and rerun."
        )
        _TABPFN_PREFLIGHT = out
        if logger is not None:
            logger.warning(out["error"])
        return dict(out)
    try:
        from tabpfn import TabPFNClassifier  # noqa: F401

        out["available"] = True
        out["device"] = _detect_device(logger)
    except ImportError as exc:
        out["error"] = f"TabPFN not installed: {exc}. Run: pip install tabpfn"
    except Exception as exc:
        out["error"] = f"TabPFN preflight failed: {exc}"
    _TABPFN_PREFLIGHT = out
    if logger is not None and out["error"]:
        logger.warning(out["error"])
    return dict(out)


def _train_tabpfn(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    global _TABPFN_BLOCKED
    result: Dict[str, Any] = {"metrics": None, "training_time_seconds": None, "error": None}
    if _TABPFN_BLOCKED:
        result["error"] = _TABPFN_BLOCKED
        return result
    pre = _tabpfn_preflight(logger)
    if not pre["available"]:
        result["error"] = str(pre.get("error") or "TabPFN unavailable")
        return result
    try:
        from tabpfn import TabPFNClassifier

        device = str(pre["device"])
        X_tr = X_train.to_numpy(dtype=np.float32)
        X_te = X_test.to_numpy(dtype=np.float32)

        def fit_fn(X, y):
            try:
                model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
            except TypeError:
                model = TabPFNClassifier(device=device)
            model.fit(X, _as_array(y))
            return model

        def proba_fn(model, X):
            if hasattr(model, "predict_proba"):
                try:
                    return np.asarray(model.predict_proba(X))
                except Exception:
                    return None
            return None

        def predict_fn(model, X):
            return np.asarray(model.predict(X))

        start = time.time()
        y_pred, y_proba, thr = _fit_predict_with_optional_threshold(
            fit_fn, proba_fn, predict_fn, X_tr, y_train, X_te,
        )
        elapsed = time.time() - start
        metrics = compute_metrics(_as_array(y_test), y_pred, y_proba)
        metrics["threshold"] = thr
        result["metrics"] = metrics
        result["training_time_seconds"] = elapsed
    except Exception as exc:
        msg = str(exc)
        if any(s in msg.lower() for s in ("license", "api key", "tabpfn_token")):
            _TABPFN_BLOCKED = f"TabPFN disabled for remainder of run: {msg}"
            msg = _TABPFN_BLOCKED
        result["error"] = msg
    return result


def _train_tabicl(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"metrics": None, "training_time_seconds": None, "error": None}
    try:
        from tabicl import TabICLClassifier
    except ImportError as exc:
        result["error"] = f"TabICL not installed: {exc}. Run: pip install tabicl"
        return result
    except Exception as exc:
        result["error"] = f"TabICL import failed: {exc}"
        return result
    try:
        device = _detect_device(logger)

        def fit_fn(X, y):
            try:
                model = TabICLClassifier(device=device)
            except TypeError:
                model = TabICLClassifier()
            model.fit(X, y)
            return model

        def proba_fn(model, X):
            if hasattr(model, "predict_proba"):
                try:
                    return np.asarray(model.predict_proba(X))
                except Exception:
                    return None
            return None

        def predict_fn(model, X):
            return np.asarray(model.predict(X))

        start = time.time()
        y_pred, y_proba, thr = _fit_predict_with_optional_threshold(
            fit_fn, proba_fn, predict_fn, X_train, y_train, X_test,
        )
        elapsed = time.time() - start
        metrics = compute_metrics(_as_array(y_test), y_pred, y_proba)
        metrics["threshold"] = thr
        result["metrics"] = metrics
        result["training_time_seconds"] = elapsed
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _train_catboost(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"metrics": None, "training_time_seconds": None, "error": None}
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        result["error"] = f"CatBoost not installed: {exc}. Run: pip install catboost"
        return result
    prefers_gpu = _detect_device(logger=None) == "cuda"

    def _fit_once(use_gpu: bool) -> Dict[str, Any]:
        def fit_fn(X, y):
            kwargs = {
                "iterations": 200,
                "random_seed": RANDOM_STATE,
                "verbose": False,
                "allow_writing_files": False,
            }
            if USE_CLASS_WEIGHT:
                kwargs["auto_class_weights"] = "Balanced"
            if use_gpu:
                kwargs.update({"task_type": "GPU", "devices": "0"})
            model = CatBoostClassifier(**kwargs)
            model.fit(X, y)
            return model

        def proba_fn(model, X):
            if hasattr(model, "predict_proba"):
                try:
                    return np.asarray(model.predict_proba(X))
                except Exception:
                    return None
            return None

        def predict_fn(model, X):
            y_pred = model.predict(X)
            if hasattr(y_pred, "ravel"):
                y_pred = y_pred.ravel()
            return np.asarray(y_pred)

        start = time.time()
        y_pred, y_proba, thr = _fit_predict_with_optional_threshold(
            fit_fn, proba_fn, predict_fn, X_train, y_train, X_test,
        )
        elapsed = time.time() - start
        metrics = compute_metrics(_as_array(y_test), y_pred, y_proba)
        metrics["threshold"] = thr
        return {
            "metrics": metrics,
            "training_time_seconds": elapsed,
            "error": None,
        }

    try:
        out = _fit_once(use_gpu=prefers_gpu)
        result.update(out)
    except Exception as exc:
        if prefers_gpu and _is_gpu_runtime_error(exc):
            try:
                out = _fit_once(use_gpu=False)
                result.update(out)
                return result
            except Exception as cpu_exc:
                result["error"] = f"GPU failed ({exc}); CPU fallback failed ({cpu_exc})"
                return result
        result["error"] = str(exc)
    return result


# ── Public entry point ───────────────────────────────────────────────────────

def train_and_evaluate(
    model_key: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Dispatch to the right trainer and return a uniform result dict.

    Result keys: ``metrics`` (dict or None), ``training_time_seconds``, ``error``.
    ``metrics`` always includes ``threshold`` when available — the decision
    threshold actually used (0.5 if no tuning occurred).
    """
    key = str(model_key).lower()
    if key == "tabpfn":
        return _train_tabpfn(X_train, X_test, y_train, y_test, logger=logger)
    if key == "tabicl":
        return _train_tabicl(X_train, X_test, y_train, y_test, logger=logger)
    if key == "catboost":
        return _train_catboost(X_train, X_test, y_train, y_test, logger=logger)
    if key in CLASSICAL_MODELS:
        return _train_classical(key, X_train, X_test, y_train, y_test)
    return {
        "metrics": None,
        "training_time_seconds": None,
        "error": f"Unknown model: {model_key!r}",
    }


def accepts_nan(model_key: str) -> bool:
    """True when model can be trained on matrices with NaN directly."""
    return str(model_key).lower() in MODELS_ACCEPTING_NAN
