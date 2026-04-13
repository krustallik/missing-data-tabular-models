"""Phase 4.4 - Foundation Model Testing (Student 2).

Tests pretrained tabular foundation models:
- TabPFN
- TabICL

Also includes CatBoost as an additional bonus comparison.

Tasks:
- Test TabPFN and TabICL on same datasets and splits as other models
- Test TabPFN with imputation
- Test TabICL with and without explicit imputation
- Record computation time and performance metrics

Outputs:
- results/tables/phase4_4_tabicl_results.json
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
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, TEST_SIZE
from data_utils import load_dataset_from_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TARGET_COLUMN = "target"

PREPROCESSING_METHODS = ["median", "mice", "mice_indicator"]


def _setup_logging(experiment_name: str) -> logging.Logger:
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


def _impute(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str):
    """Impute missing values. Returns (X_train_imputed, X_test_imputed) as DataFrames."""
    if method == "median":
        medians = X_train.median(numeric_only=True)
        return X_train.fillna(medians), X_test.fillna(medians)
    elif method == "mice":
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
            X_train_imp = pd.DataFrame(
                imputer.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_imp = pd.DataFrame(
                imputer.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )
            return X_train_imp, X_test_imp
        except Exception as exc:
            print(f"MICE failed ({exc}), falling back to median")
            return _impute(X_train, X_test, "median")
    else:
        raise ValueError(f"Unknown method: {method}")


def _append_missing_indicators(X_missing: pd.DataFrame, X_imputed: pd.DataFrame) -> pd.DataFrame:
    """Append binary indicators marking where values were originally missing."""
    indicator = X_missing.isna().astype(float)
    indicator.columns = [f"{c}__was_missing" for c in X_missing.columns]
    return pd.concat([X_imputed, indicator], axis=1)


def _compute_metrics(y_test, y_pred, y_proba=None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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


# ── TabICL ────────────────────────────────────────────────────────────────────

def _test_tabicl(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train, y_test, logger: logging.Logger) -> Dict[str, Any]:
    """Train and evaluate TabICLClassifier. Accepts DataFrames (may contain NaN)."""
    result = {"available": False, "error": None, "metrics": None, "training_time_seconds": None}
    try:
        from tabicl import TabICLClassifier  # pip install tabicl
        result["available"] = True
        logger.info("TabICL imported successfully")

        start = time.time()
        model = TabICLClassifier()
        # TabICL expects DataFrames; it handles NaN internally
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass

        elapsed = time.time() - start
        result["metrics"] = _compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        result["training_time_seconds"] = elapsed
        logger.info(f"TabICL done in {elapsed:.2f}s | acc={result['metrics']['accuracy']:.4f}")

    except ImportError as e:
        result["error"] = f"TabICL not available: {e}. Run: pip install tabicl"
        logger.warning(result["error"])
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"TabICL failed: {e}", exc_info=True)

    return result

# TABFN

def _test_tabpfn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Train and evaluate TabPFN.
    TabPFN usually requires numeric arrays and does NOT reliably handle NaN directly,
    so use this mainly with imputed data.
    """
    result = {
        "available": False,
        "error": None,
        "metrics": None,
        "training_time_seconds": None,
    }

    try:
        import os

        token = os.getenv("TABPFN_TOKEN")
        if not token:
            raise RuntimeError(
                "TABPFN_TOKEN is not set. Set it before running, e.g. "
                "PowerShell: $env:TABPFN_TOKEN='your_token'"
            )
        logger.info("TABPFN_TOKEN detected in environment")

        from tabpfn import TabPFNClassifier
        result["available"] = True
        logger.info("TabPFN imported successfully")

        # TabPFN expects numeric input
        X_train_np = X_train.to_numpy(dtype=np.float32)
        X_test_np = X_test.to_numpy(dtype=np.float32)

        # Safety check: TabPFN generally should not receive NaN here
        if np.isnan(X_train_np).any() or np.isnan(X_test_np).any():
            raise ValueError("TabPFN received NaN values. Use imputation before TabPFN.")

        start = time.time()

        # Minimal constructor compatible with newer versions
        model = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        model.fit(X_train_np, y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train)

        y_pred = model.predict(X_test_np)

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_np)
            except Exception:
                pass

        elapsed = time.time() - start
        result["metrics"] = _compute_metrics(
            y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test,
            y_pred,
            y_proba,
        )
        result["training_time_seconds"] = elapsed
        logger.info(f"TabPFN done in {elapsed:.2f}s | acc={result['metrics']['accuracy']:.4f}")

    except ImportError as e:
        result["error"] = f"TabPFN not available: {e}. Run: pip install tabpfn"
        logger.warning(result["error"])
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"TabPFN failed: {e}", exc_info=True)

    return result

# ── CatBoost (bonus baseline) ─────────────────────────────────────────────────

def _test_catboost(X_train_np, X_test_np, y_train, y_test,
                   logger: logging.Logger) -> Dict[str, Any]:
    """Train and evaluate CatBoost (bonus — handles NaN natively)."""
    result = {"available": False, "error": None, "metrics": None, "training_time_seconds": None}
    try:
        from catboost import CatBoostClassifier
        result["available"] = True

        start = time.time()
        model = CatBoostClassifier(
            iterations=100, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=False, nan_mode="Min",
        )
        model.fit(X_train_np, y_train)
        y_pred = model.predict(X_test_np)
        y_proba = model.predict_proba(X_test_np) if hasattr(model, "predict_proba") else None

        elapsed = time.time() - start
        result["metrics"] = _compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        result["training_time_seconds"] = elapsed
        logger.info(f"CatBoost done in {elapsed:.2f}s | acc={result['metrics']['accuracy']:.4f}")

    except ImportError as e:
        result["error"] = f"CatBoost not available: {e}"
        logger.warning(result["error"])
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"CatBoost failed: {e}", exc_info=True)

    return result


# ── Main experiment ───────────────────────────────────────────────────────────

def run_phase4_4_experiment(dataset_path: Path, dataset_name: str,
                             logger: logging.Logger) -> Dict:
    logger.info(f"\n{'='*80}\nPhase 4.4: {dataset_name}\n{'='*80}")

    X, y = load_dataset_from_csv(dataset_path, target_column=TARGET_COLUMN)
    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

    loaded = _load_precomputed_split(dataset_name, logger)
    if loaded is None:
        logger.error("No precomputed split found — skipping")
        return {}
    X_train, X_test, y_train, y_test = loaded

    # Coerce to numeric
    X_train = _coerce_features(X_train)
    X_test = _coerce_features(X_test)

    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),

        # TabPFN
        "tabpfn_with_imputation": {},

        # TabICL
        "tabicl_with_imputation": {},
        "tabicl_without_imputation": {},

        # CatBoost bonus
        "catboost_with_imputation": {},
        "catboost_without_imputation": {},
    }

    # 0. TabPFN WITH imputation
    logger.info("\n--- TabPFN WITH imputation ---")
    for method in PREPROCESSING_METHODS:
        logger.info(f"  preprocessing: {method}")
        if method == "mice_indicator":
            X_tr_base, X_te_base = _impute(X_train.copy(), X_test.copy(), "mice")
            X_tr_imp = _append_missing_indicators(X_train, X_tr_base)
            X_te_imp = _append_missing_indicators(X_test, X_te_base)
        else:
            X_tr_imp, X_te_imp = _impute(X_train.copy(), X_test.copy(), method)

        results["tabpfn_with_imputation"][method] = _test_tabpfn(
            X_tr_imp, X_te_imp, y_train, y_test, logger
        )

    # 1. TabICL WITH imputation
    logger.info("\n--- TabICL WITH imputation ---")
    for method in PREPROCESSING_METHODS:
        logger.info(f"  preprocessing: {method}")
        if method == "mice_indicator":
            X_tr_base, X_te_base = _impute(X_train.copy(), X_test.copy(), "mice")
            X_tr_imp = _append_missing_indicators(X_train, X_tr_base)
            X_te_imp = _append_missing_indicators(X_test, X_te_base)
        else:
            X_tr_imp, X_te_imp = _impute(X_train.copy(), X_test.copy(), method)
        results["tabicl_with_imputation"][method] = _test_tabicl(
            X_tr_imp, X_te_imp, y_train, y_test, logger
        )

    # 2. TabICL WITHOUT imputation (raw NaN — TabICL handles this natively)
    logger.info("\n--- TabICL WITHOUT imputation (raw NaN) ---")
    results["tabicl_without_imputation"]["raw_nan"] = _test_tabicl(
        X_train.copy(), X_test.copy(), y_train, y_test, logger
    )

    # 3. CatBoost WITH imputation (bonus comparison)
    logger.info("\n--- CatBoost WITH imputation (bonus) ---")
    for method in PREPROCESSING_METHODS:
        logger.info(f"  preprocessing: {method}")
        if method == "mice_indicator":
            X_tr_base, X_te_base = _impute(X_train.copy(), X_test.copy(), "mice")
            X_tr_imp = _append_missing_indicators(X_train, X_tr_base)
            X_te_imp = _append_missing_indicators(X_test, X_te_base)
        else:
            X_tr_imp, X_te_imp = _impute(X_train.copy(), X_test.copy(), method)
        results["catboost_with_imputation"][method] = _test_catboost(
            X_tr_imp.to_numpy(), X_te_imp.to_numpy(), y_train, y_test, logger
        )

    # 4. CatBoost WITHOUT imputation (native NaN)
    logger.info("\n--- CatBoost WITHOUT imputation (raw NaN, bonus) ---")
    results["catboost_without_imputation"]["raw_nan"] = _test_catboost(
        X_train.to_numpy(), X_test.to_numpy(), y_train, y_test, logger
    )

    return results


def main():
    logger = _setup_logging("phase4_4_tabicl")
    logger.info("=" * 80)
    logger.info("PHASE 4.4: TabICL Foundation Model (Student 2)")
    logger.info("=" * 80)

    datasets = {
        "taiwan_bankruptcy":    PROJECT_ROOT / "data" / "processed" / "taiwan_bankruptcy.csv",
        "polish_1year":         PROJECT_ROOT / "data" / "processed" / "polish_1year.csv",
        "slovak_manufacture_13": PROJECT_ROOT / "data" / "processed" / "slovak_manufacture_13.csv",
    }

    all_results = []
    for dataset_name, dataset_path in datasets.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        try:
            result = run_phase4_4_experiment(dataset_path, dataset_name, logger)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}", exc_info=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "phase4_4_tabicl_results.json"
    out_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✓ Phase 4.4 complete. Results: {out_file}")


if __name__ == "__main__":
    main()
