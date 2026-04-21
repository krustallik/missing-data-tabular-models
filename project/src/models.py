"""Model registry: classical sklearn models + pretrained foundation models.

Every model is addressed by its canonical snake_case key (see
:data:`config.ALL_MODELS`). Two public entry points:

- :func:`train_and_evaluate` — fit the chosen model, predict on test, and
  return metrics + training time + any error message.
- :func:`display_name` — canonical display name used in CSVs / reports /
  plots (e.g. ``"Logistic-Regression"``, ``"TabPFN"``).

Classical models go through ``StandardScaler`` when needed; foundation models
(TabPFN / TabICL / CatBoost) are called with raw numpy arrays since they
bring their own preprocessing (and often their own NaN handling).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    CLASSICAL_MODELS,
    FOUNDATION_MODELS,
    MODELS_ACCEPTING_NAN,
    RANDOM_STATE,
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
    """Heuristic check whether an exception is GPU/CUDA related."""
    msg = str(exc).lower()
    hints = [
        "cuda",
        "cudnn",
        "cublas",
        "gpu",
        "opencl",
        "nvidia",
        "device",
        "driver",
    ]
    return any(h in msg for h in hints)


# ── Classical model builders ─────────────────────────────────────────────────

def _build_classical(model_key: str, use_gpu: bool = False):
    if model_key == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    if model_key == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    if model_key == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    if model_key == "svm":
        return SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
    if model_key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
        )
    if model_key == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric="logloss", verbosity=0,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
        )
    if model_key == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            device_type="gpu" if use_gpu else "cpu",
        )
    raise ValueError(f"Unknown classical model: {model_key!r}")


def _train_classical(
    model_key: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test,
) -> Dict[str, Any]:
    """Train a classical model with default hyperparameters (no grid search).

    Adds ``StandardScaler`` for distance-based / linear models (LR, SVM, MLP);
    tree-based models use raw inputs. When the input contains NaN and the
    model does not accept NaN natively, ``StandardScaler`` fails fast and the
    error is recorded (expected for ``imputation='none'`` paired with a
    classical model that does not handle NaN).
    """
    result: Dict[str, Any] = {
        "metrics": None, "training_time_seconds": None, "error": None,
    }
    needs_scaling = model_key in {"logistic_regression", "svm", "mlp"}
    X_tr_raw = X_train.to_numpy(dtype=float) if not hasattr(X_train, "to_numpy") else X_train.to_numpy(dtype=float)
    X_te_raw = X_test.to_numpy(dtype=float)

    if needs_scaling:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
    else:
        X_tr = X_tr_raw
        X_te = X_te_raw

    prefers_gpu = model_key in {"xgboost", "lightgbm"} and _detect_device(logger=None) == "cuda"

    def _fit_once(use_gpu: bool) -> Dict[str, Any]:
        start = time.time()
        model = clone(_build_classical(model_key, use_gpu=use_gpu))
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
        elapsed = time.time() - start
        y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test
        return {
            "metrics": compute_metrics(y_true, y_pred, y_proba),
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
        start = time.time()
        try:
            model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
        except TypeError:
            model = TabPFNClassifier(device=device)
        model.fit(X_tr, y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train)
        y_pred = model.predict(X_te)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_te)
            except Exception:
                y_proba = None
        elapsed = time.time() - start
        y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test
        result["metrics"] = compute_metrics(y_true, y_pred, y_proba)
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
        start = time.time()
        try:
            model = TabICLClassifier(device=device)
        except TypeError:
            model = TabICLClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = np.asarray(model.predict_proba(X_test))
            except Exception:
                y_proba = None
        elapsed = time.time() - start
        y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test
        result["metrics"] = compute_metrics(y_true, y_pred, y_proba)
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
        start = time.time()
        kwargs = {
            "iterations": 200,
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False,
        }
        if use_gpu:
            kwargs.update({"task_type": "GPU", "devices": "0"})
        model = CatBoostClassifier(**kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(y_pred, "ravel"):
            y_pred = y_pred.ravel()
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = np.asarray(model.predict_proba(X_test))
            except Exception:
                y_proba = None
        elapsed = time.time() - start
        y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test
        return {
            "metrics": compute_metrics(y_true, y_pred, y_proba),
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
    """
    key = str(model_key).lower()
    if key in CLASSICAL_MODELS:
        return _train_classical(key, X_train, X_test, y_train, y_test)
    if key == "tabpfn":
        return _train_tabpfn(X_train, X_test, y_train, y_test, logger=logger)
    if key == "tabicl":
        return _train_tabicl(X_train, X_test, y_train, y_test, logger=logger)
    if key == "catboost":
        return _train_catboost(X_train, X_test, y_train, y_test, logger=logger)
    return {
        "metrics": None,
        "training_time_seconds": None,
        "error": f"Unknown model: {model_key!r}",
    }


def accepts_nan(model_key: str) -> bool:
    """True when model can be trained on matrices with NaN directly."""
    return str(model_key).lower() in MODELS_ACCEPTING_NAN
