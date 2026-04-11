"""Phase 4.3 - Gradient Boosting robustness testing with missingness injection (Student 2).

This phase evaluates XGBoost and LightGBM models across scenarios with injected
missingness at varying rates (5%-40%) and mechanisms (MCAR/MAR/MNAR) to assess
sensitivity to missing data.

Outputs:
- results/tables/phase4_3_gradient_boosting_results.json
- results/tables/phase4_3_sensitivity_summary.csv
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, TEST_SIZE
from config_phase4_3 import (
    CV_FOLDS,
    GRADIENT_BOOSTING_MODELS,
    FALLBACK_MODELS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    PREPROCESSING_METHODS,
)
from data_utils import load_dataset_from_csv
from missingness import inject_mcar, inject_mar, inject_mnar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TARGET_COLUMN = "target"


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


def _load_precomputed_split(dataset_name: str, logger: logging.Logger) -> Optional[Tuple]:
    """Load train/test splits from data/splits if available; otherwise return None."""
    train_path = SPLITS_DIR / f"{dataset_name}_train.csv"
    test_path = SPLITS_DIR / f"{dataset_name}_test.csv"
    if not train_path.exists() or not test_path.exists():
        return None

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
        logger.warning("Split CSVs missing target column; falling back to internal split")
        return None

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def _inject_missingness(X: pd.DataFrame, mechanism: str, rate: float, random_state: int) -> pd.DataFrame:
    """Inject missingness using MCAR/MAR/MNAR."""
    if mechanism == "MCAR":
        return inject_mcar(X, rate, random_state=random_state)
    elif mechanism == "MAR":
        return inject_mar(X, rate, random_state=random_state)
    elif mechanism == "MNAR":
        return inject_mnar(X, rate, random_state=random_state)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def _get_model(model_name: str):
    """Instantiate a model (Gradient Boosting preferred, falls back to SVM/MLP)."""
    try:
        import xgboost as xgb
        if model_name == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        if model_name == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            )
    except ImportError:
        pass

    # Fallback models if Gradient Boosting not available
    if model_name == "svm":
        from sklearn.svm import SVC
        return SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True)
    elif model_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE, early_stopping=True)

    raise ValueError(f"Model {model_name} not available")


def _coerce_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert feature table to numeric form."""
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == object:
            cleaned = out[col].astype(str).str.replace(",", ".", regex=False)
            out[col] = pd.to_numeric(cleaned, errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _preprocess_and_impute(X_numeric: pd.DataFrame, method: str, fit: bool = True, scaler=None, imputer=None):
    """Apply preprocessing and imputation."""
    if method == "median":
        if fit:
            scaler = StandardScaler()
            X_imputed = X_numeric.fillna(X_numeric.median(numeric_only=True))
            X_scaled = scaler.fit_transform(X_imputed)
            return X_scaled, scaler, None
        else:
            X_imputed = X_numeric.fillna(X_numeric.median(numeric_only=True))
            X_scaled = scaler.transform(X_imputed)
            return X_scaled, scaler, None
    elif method == "mice":
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            if fit:
                scaler = StandardScaler()
                imputer = IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
                X_imputed = imputer.fit_transform(X_numeric)
                X_scaled = scaler.fit_transform(X_imputed)
                return X_scaled, scaler, imputer
            else:
                X_imputed = imputer.transform(X_numeric)
                X_scaled = scaler.transform(X_imputed)
                return X_scaled, scaler, imputer
        except Exception as exc:
            print(f"MICE failed: {exc}; falling back to median")
            return _preprocess_and_impute(X_numeric, "median", fit, scaler, None)
    else:
        raise ValueError(f"Unknown preprocessing: {method}")


def run_phase4_3_experiment(dataset_path: Path, dataset_name: str, logger: logging.Logger, models_to_use: List[str] = None) -> Dict:
    """Run Phase 4.3 Gradient Boosting robustness test."""
    if models_to_use is None:
        models_to_use = GRADIENT_BOOSTING_MODELS
    
    logger.info(f"Starting Phase 4.3 experiment on {dataset_name}")
    logger.info(f"Using models: {models_to_use}")

    # Load data
    X, y = load_dataset_from_csv(dataset_path, target_column=TARGET_COLUMN)
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # Use precomputed splits
    loaded = _load_precomputed_split(dataset_name, logger)
    if loaded is not None:
        X_train, X_test, y_train, y_test = loaded
        logger.info(f"Using precomputed split from data/splits")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X.shape[1],
        "n_classes": y.nunique(),
        "split_source": "data/splits" if loaded is not None else "internal",
        "scenarios": {},
    }

    # Test each scenario: missing_mechanism x missing_rate x preprocessing x model
    for mechanism in MISSING_MECHANISMS:
        for rate in MISSING_RATES:
            scenario_key = f"{mechanism}_{int(rate*100)}pct"
            logger.info(f"\n=== Scenario: {scenario_key} ===")
            results["scenarios"][scenario_key] = {}

            # Inject missingness into train set
            X_train_missing = _inject_missingness(X_train.copy(), mechanism, rate, RANDOM_STATE)
            logger.info(f"Injected {mechanism} at {rate*100:.1f}%")

            for preprocessing in PREPROCESSING_METHODS:
                logger.info(f"  Preprocessing: {preprocessing}")
                results["scenarios"][scenario_key][preprocessing] = {"models": {}}

                for model_name in models_to_use:
                    logger.info(f"    Model: {model_name}")
                    try:
                        # Coerce to numeric
                        X_train_numeric = _coerce_features(X_train_missing)
                        X_test_numeric = _coerce_features(X_test)

                        # Preprocess/impute train
                        X_train_scaled, scaler, imputer = _preprocess_and_impute(
                            X_train_numeric, preprocessing, fit=True
                        )

                        # Preprocess/impute test
                        X_test_scaled, _, _ = _preprocess_and_impute(
                            X_test_numeric, preprocessing, fit=False, scaler=scaler, imputer=imputer
                        )

                        # Train model
                        model = _get_model(model_name)
                        model.fit(X_train_scaled, y_train)

                        # Evaluate
                        y_pred = model.predict(X_test_scaled)
                        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

                        accuracy = float(accuracy_score(y_test, y_pred))
                        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

                        try:
                            if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                                roc_auc = float(roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
                            else:
                                roc_auc = np.nan
                        except Exception:
                            roc_auc = np.nan

                        results["scenarios"][scenario_key][preprocessing]["models"][model_name] = {
                            "accuracy": accuracy,
                            "f1": f1,
                            "roc_auc": roc_auc,
                        }

                        logger.info(f"      Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

                    except Exception as e:
                        logger.error(f"Error training {model_name}: {e}", exc_info=True)
                        results["scenarios"][scenario_key][preprocessing]["models"][model_name] = {
                            "error": str(e)
                        }

    return results


def main():
    """Run Phase 4.3 experiments."""
    experiment_name = "phase4_3_gradient_boosting"
    logger = _setup_logging(experiment_name)

    logger.info("=" * 80)
    logger.info("PHASE 4.3: Gradient Boosting Robustness (Student 2)")
    logger.info("=" * 80)

    # Check which models are available
    try:
        import xgboost  # noqa: F401
        logger.info("✓ XGBoost available")
        models_to_use = list(GRADIENT_BOOSTING_MODELS)
    except ImportError:
        logger.warning("✗ XGBoost not available, using fallback models (SVM/MLP)")
        models_to_use = list(FALLBACK_MODELS)
        
        # Override global config for this run
        import config_phase4_3
        config_phase4_3.GRADIENT_BOOSTING_MODELS = FALLBACK_MODELS

    datasets = {
        "taiwan_bankruptcy": PROJECT_ROOT / "data" / "processed" / "taiwan_bankruptcy.csv",
        "polish_1year": PROJECT_ROOT / "data" / "processed" / "polish_1year.csv",
        "slovak_manufacture_13": PROJECT_ROOT / "data" / "processed" / "slovak_manufacture_13.csv",
    }

    all_results = []

    for dataset_name, dataset_path in datasets.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        try:
            result = run_phase4_3_experiment(dataset_path, dataset_name, logger, models_to_use=models_to_use)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}", exc_info=True)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "phase4_3_gradient_boosting_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    print(f"\n✓ Phase 4.3 experiments complete. Results: {results_file}")


if __name__ == "__main__":
    main()
