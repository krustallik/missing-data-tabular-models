"""Student 1 pipeline runner - Tasks 3.3, 3.4, 3.5.

Student 1 responsibilities:
- 3.3: Simpler missing-data handling methods (mean, median, kNN imputation)
- 3.4: Baseline classification models (Logistic Regression, Random Forest)
- 3.5: Foundation model (TabPFN)

Design:
- Reuses precomputed train/test splits from data/splits/ (Phase 3.1).
  If splits are missing, falls back to running dataset setup once.
- Does NOT invoke Student 2 phases (MNAR generation, Gradient Boosting
  robustness, TabICL/CatBoost, Phase 4.5-4.7 analysis/reporting).
- Keeps file layout and coding style consistent with existing
  run_phase4_*_experiments.py modules.

Outputs:
- results/tables/student1_pipeline_results.json
- results/logs/experiment_student1_pipeline_*.log
"""

from __future__ import annotations

import json
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

# Task 3.3 - simpler imputation methods owned by Student 1
IMPUTATION_METHODS = ["mean", "median", "knn"]

# Task 3.4 - baseline classifiers owned by Student 1
BASELINE_MODELS = ["logistic_regression", "random_forest"]


def _setup_logging(experiment_name: str) -> logging.Logger:
    """Configure logging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"experiment_{experiment_name}_{timestamp}.log"

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def _splits_present() -> bool:
    for name in DATASETS:
        if not (SPLITS_DIR / f"{name}_train.csv").exists():
            return False
        if not (SPLITS_DIR / f"{name}_test.csv").exists():
            return False
    return True


def _ensure_splits(logger: logging.Logger) -> None:
    """Ensure train/test splits exist; run Phase 3.1 setup only if needed."""
    if _splits_present():
        logger.info("Precomputed splits present; skipping Phase 3.1 setup.")
        return

    logger.info("Splits missing; running Phase 3.1 dataset setup (run_dataset_setup.run_setup).")
    from run_dataset_setup import run_setup
    run_setup()


def _load_precomputed_split(dataset_name: str, logger: logging.Logger) -> Optional[Tuple]:
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


def _coerce_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == object:
            cleaned = out[col].astype(str).str.replace(",", ".", regex=False)
            out[col] = pd.to_numeric(cleaned, errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ── Task 3.3 - simpler imputation methods ────────────────────────────────────

def _impute(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str):
    """Fit imputer on training data only, then apply to both splits."""
    if method == "mean":
        means = X_train.mean(numeric_only=True)
        return X_train.fillna(means), X_test.fillna(means)
    if method == "median":
        medians = X_train.median(numeric_only=True)
        return X_train.fillna(medians), X_test.fillna(medians)
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
        return X_tr, X_te
    raise ValueError(f"Unknown imputation method: {method}")


def _compute_metrics(y_test, y_pred, y_proba=None) -> Dict[str, float]:
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


# ── Task 3.4 - baseline classifiers ──────────────────────────────────────────

def _build_baseline(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    raise ValueError(f"Unknown baseline model: {model_name}")


def _train_baseline(model_name: str, X_train, X_test, y_train, y_test,
                    logger: logging.Logger) -> Dict[str, Any]:
    result = {"metrics": None, "training_time_seconds": None, "error": None}
    try:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        model = _build_baseline(model_name)

        start = time.time()
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)
        y_proba = model.predict_proba(X_te_s) if hasattr(model, "predict_proba") else None
        elapsed = time.time() - start

        result["metrics"] = _compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        result["training_time_seconds"] = elapsed
        logger.info(
            f"{model_name} done in {elapsed:.2f}s | acc={result['metrics']['accuracy']:.4f}"
        )
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"{model_name} failed: {e}", exc_info=True)
    return result


# ── Task 3.5 - TabPFN foundation model ───────────────────────────────────────

def _test_tabpfn(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train, y_test, logger: logging.Logger) -> Dict[str, Any]:
    """Train and evaluate TabPFN on already-imputed numeric features."""
    result = {
        "available": False,
        "error": None,
        "metrics": None,
        "training_time_seconds": None,
    }
    try:
        from tabpfn import TabPFNClassifier
        result["available"] = True
        logger.info("TabPFN imported successfully")

        start = time.time()
        model = TabPFNClassifier()
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = model.predict(X_test.to_numpy())

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test.to_numpy())
            except Exception:
                pass

        elapsed = time.time() - start
        result["metrics"] = _compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        result["training_time_seconds"] = elapsed
        logger.info(
            f"TabPFN done in {elapsed:.2f}s | acc={result['metrics']['accuracy']:.4f}"
        )

    except ImportError as e:
        result["error"] = f"TabPFN not available: {e}. Run: pip install tabpfn"
        logger.warning(result["error"])
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"TabPFN failed: {e}", exc_info=True)

    return result


# ── Per-dataset experiment ───────────────────────────────────────────────────

def run_student1_experiment(dataset_path: Path, dataset_name: str,
                            logger: logging.Logger) -> Dict[str, Any]:
    logger.info(f"\n{'='*80}\nStudent 1: {dataset_name}\n{'='*80}")

    X, y = load_dataset_from_csv(dataset_path, target_column=TARGET_COLUMN)
    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

    loaded = _load_precomputed_split(dataset_name, logger)
    if loaded is None:
        logger.error(f"No precomputed split found for {dataset_name} - skipping")
        return {}

    X_train, X_test, y_train, y_test = loaded
    X_train = _coerce_features(X_train)
    X_test = _coerce_features(X_test)

    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),
        "split_source": "data/splits",
        "imputation": {},
    }

    for method in IMPUTATION_METHODS:
        logger.info(f"\n--- Imputation: {method} (Task 3.3) ---")
        method_block: Dict[str, Any] = {
            "train_missing_fraction_before": float(
                X_train.isna().sum().sum() / max(X_train.size, 1)
            ),
            "baselines": {},
            "tabpfn": None,
            "error": None,
        }

        try:
            X_tr_imp, X_te_imp = _impute(X_train.copy(), X_test.copy(), method)
        except Exception as e:
            method_block["error"] = str(e)
            logger.error(f"Imputation {method} failed: {e}", exc_info=True)
            results["imputation"][method] = method_block
            continue

        method_block["train_missing_fraction_after"] = float(
            X_tr_imp.isna().sum().sum() / max(X_tr_imp.size, 1)
        )

        # Task 3.4
        for model_name in BASELINE_MODELS:
            logger.info(f"  baseline: {model_name}")
            method_block["baselines"][model_name] = _train_baseline(
                model_name, X_tr_imp, X_te_imp, y_train, y_test, logger,
            )

        # Task 3.5
        logger.info("  foundation: tabpfn")
        method_block["tabpfn"] = _test_tabpfn(
            X_tr_imp, X_te_imp, y_train, y_test, logger,
        )

        results["imputation"][method] = method_block

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = _setup_logging("student1_pipeline")
    logger.info("=" * 80)
    logger.info("STUDENT 1 PIPELINE - Tasks 3.3, 3.4, 3.5")
    logger.info("=" * 80)

    _ensure_splits(logger)

    all_results = []
    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        try:
            res = run_student1_experiment(dataset_path, dataset_name, logger)
            if res:
                all_results.append(res)
        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}", exc_info=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "student1_pipeline_results.json"
    out_file.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Student 1 pipeline complete")
    print("=" * 80)
    print(f"  Results: {out_file}")
    print(f"  Logs:    {LOGS_DIR}")


if __name__ == "__main__":
    main()
