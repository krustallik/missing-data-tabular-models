"""Shared utilities for Student 1 phases 3.3, 3.4, 3.5."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from data_utils import load_dataset_from_csv
from missingness import inject_mar, inject_mcar, inject_mnar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TARGET_COLUMN = "target"

DATASETS = {
    "taiwan_bankruptcy": PROJECT_ROOT / "data" / "processed" / "taiwan_bankruptcy.csv",
    "polish_1year": PROJECT_ROOT / "data" / "processed" / "polish_1year.csv",
    "slovak_manufacture_13": PROJECT_ROOT / "data" / "processed" / "slovak_manufacture_13.csv",
}

IMPUTATION_METHODS = ["mean", "median", "knn"]
BASELINE_MODELS = ["logistic_regression", "random_forest"]
MISSING_MECHANISMS = ["MCAR", "MAR", "MNAR"]
MISSING_RATES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

# Phase tags used to match Student 2 consolidated CSV layout.
PHASE_TAG_IMPUTATION = "3.3"
PHASE_TAG_BASELINE = "3.4"
PHASE_TAG_FOUNDATION = "3.5"

# Model -> model_type mapping consistent with Student 2 (Classical / Foundation).
MODEL_TYPES = {
    "logistic_regression": "Classical",
    "random_forest": "Classical",
    "tabpfn": "Foundation",
}


def detect_device(logger: Optional[logging.Logger] = None) -> str:
    """Return 'cuda' when a CUDA-capable GPU is available, otherwise 'cpu'.

    Always safe: imports torch lazily and falls back to CPU on any failure.
    """
    try:
        import torch  # local import to avoid hard dep for callers that do not need it

        if torch.cuda.is_available():
            if logger is not None:
                try:
                    name = torch.cuda.get_device_name(0)
                except Exception:
                    name = "cuda"
                logger.info(f"GPU detected: {name}")
            return "cuda"
        if logger is not None:
            logger.info("No GPU detected; using CPU")
        return "cpu"
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Device detection failed ({exc}); defaulting to CPU")
        return "cpu"


def setup_logging(experiment_name: str) -> logging.Logger:
    """Configure logging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"experiment_{experiment_name}_{timestamp}.log"

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def splits_present() -> bool:
    for name in DATASETS:
        if not (SPLITS_DIR / f"{name}_train.csv").exists():
            return False
        if not (SPLITS_DIR / f"{name}_test.csv").exists():
            return False
    return True


def ensure_splits(logger: logging.Logger) -> None:
    """Ensure train/test splits exist; run Phase 3.1 setup only if needed."""
    if splits_present():
        logger.info("Precomputed splits present; skipping Phase 3.1 setup.")
        return

    logger.info("Splits missing; running Phase 3.1 dataset setup (run_dataset_setup.run_setup).")
    from run_dataset_setup import run_setup
    run_setup()


def load_precomputed_split(dataset_name: str, logger: logging.Logger) -> Optional[Tuple]:
    train_path = SPLITS_DIR / f"{dataset_name}_train.csv"
    test_path = SPLITS_DIR / f"{dataset_name}_test.csv"
    if not train_path.exists() or not test_path.exists():
        logger.warning(f"Precomputed splits not found for {dataset_name}")
        return None
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
            logger.warning("Split CSVs missing target column")
            return None
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error loading precomputed split: {e}")
        return None


def coerce_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == object:
            cleaned = out[col].astype(str).str.replace(",", ".", regex=False)
            out[col] = pd.to_numeric(cleaned, errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def inject_missingness(X: pd.DataFrame, mechanism: str, rate: float, random_state: int) -> pd.DataFrame:
    """Inject MCAR/MAR/MNAR missingness into features."""
    if mechanism == "MCAR":
        return inject_mcar(X, rate, random_state=random_state)
    if mechanism == "MAR":
        return inject_mar(X, rate, random_state=random_state)
    if mechanism == "MNAR":
        return inject_mnar(X, rate, random_state=random_state)
    raise ValueError(f"Unknown mechanism: {mechanism}")


def impute(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str):
    """Fit simpler imputer on training data only, then apply to both splits."""
    if method == "mean":
        means = X_train.mean(numeric_only=True)
        X_tr = X_train.fillna(means)
        X_te = X_test.fillna(means)
        return X_tr.fillna(0.0), X_te.fillna(0.0)

    if method == "median":
        medians = X_train.median(numeric_only=True)
        X_tr = X_train.fillna(medians)
        X_te = X_test.fillna(medians)
        return X_tr.fillna(0.0), X_te.fillna(0.0)

    if method == "knn":
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=5)
        X_tr = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_te = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        return X_tr.fillna(0.0), X_te.fillna(0.0)

    raise ValueError(f"Unknown imputation method: {method}")


def compute_metrics(y_test, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def build_baseline(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    raise ValueError(f"Unknown baseline model: {model_name}")


def train_baseline(model_name: str, X_train, X_test, y_train, y_test, logger: logging.Logger) -> Dict[str, Any]:
    # Sklearn Logistic Regression and Random Forest do not use GPU; device is always cpu.
    result = {"metrics": None, "training_time_seconds": None, "error": None, "device": "cpu"}
    try:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        model = build_baseline(model_name)
        start = time.time()
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)
        y_proba = model.predict_proba(X_te_s) if hasattr(model, "predict_proba") else None
        elapsed = time.time() - start

        result["metrics"] = compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        result["training_time_seconds"] = elapsed
        logger.info(f"    {model_name} done in {elapsed:.2f}s | acc={result['metrics']['accuracy']:.4f}")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"{model_name} failed: {e}", exc_info=True)
    return result


def test_tabpfn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    logger: logging.Logger,
    allow_nan: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate TabPFN.

    When ``allow_nan`` is False the callers must pass already-imputed inputs;
    receiving any NaN then raises and is recorded as an error. When True the
    raw NaN-containing arrays are handed to TabPFN directly (TabPFN v2+ has
    a built-in NaN handling preprocessing step).
    """
    result = {
        "available": False,
        "error": None,
        "metrics": None,
        "training_time_seconds": None,
        "device": None,
        "native_nan_input": bool(allow_nan),
        "n_train_used": int(X_train.shape[0]),
        "n_features_used": int(X_train.shape[1]),
    }
    try:
        import os

        token = os.getenv("TABPFN_TOKEN")
        if not token:
            logger.warning(
                "TABPFN_TOKEN not set; TabPFN model download may fail. "
                "Set TABPFN_TOKEN to enable the foundation model."
            )

        from tabpfn import TabPFNClassifier

        result["available"] = True
        logger.info("TabPFN imported successfully")

        device = detect_device(logger)
        result["device"] = device
        logger.info(f"TabPFN device selected: {device}")

        X_train_np = X_train.to_numpy(dtype=np.float32)
        X_test_np = X_test.to_numpy(dtype=np.float32)

        has_nan = bool(np.isnan(X_train_np).any() or np.isnan(X_test_np).any())
        if has_nan and not allow_nan:
            raise ValueError(
                "TabPFN received NaN values with allow_nan=False. "
                "Apply imputation first or set preprocessing='none'."
            )

        start = time.time()
        try:
            model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
        except TypeError:
            # Backward-compatible fallback for older TabPFN versions.
            model = TabPFNClassifier(device=device)
        model.fit(X_train_np, y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train)
        y_pred = model.predict(X_test_np)

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_np)
            except Exception:
                pass

        elapsed = time.time() - start
        result["metrics"] = compute_metrics(
            y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test,
            y_pred,
            y_proba,
        )
        result["training_time_seconds"] = elapsed
        logger.info(f"    tabpfn done in {elapsed:.2f}s | acc={result['metrics']['accuracy']:.4f}")

    except ImportError as e:
        result["error"] = f"TabPFN not available: {e}. Run: pip install tabpfn"
        logger.warning(result["error"])
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"TabPFN failed: {e}", exc_info=True)

    return result


def parse_scenario_key(scenario_key: str) -> Tuple[str, int]:
    if scenario_key == "native":
        return "native", 0
    mechanism, pct = scenario_key.split("_")
    return mechanism, int(pct.replace("pct", ""))


def load_dataset_and_split(dataset_name: str, dataset_path: Path, logger: logging.Logger):
    """Load processed dataset metadata + split dataframes."""
    X, y = load_dataset_from_csv(dataset_path, target_column=TARGET_COLUMN)
    loaded = load_precomputed_split(dataset_name, logger)
    if loaded is None:
        raise RuntimeError(f"No precomputed split found for {dataset_name}")

    X_train, X_test, y_train, y_test = loaded
    X_train = coerce_features(X_train)
    X_test = coerce_features(X_test)
    return X, y, X_train, X_test, y_train, y_test
