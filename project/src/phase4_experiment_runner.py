"""
Phase 4.1 - Enhanced Experiment Runner with Model Evaluation.

This module provides a complete framework for running classification experiments
with multiple models, cross-validation, hyperparameter tuning, and comprehensive
evaluation metrics logging.

Improvements over Phase 3:
- Model selection and hyperparameter tuning
- K-fold cross-validation
- Detailed experiment logging
- Results persistence and reporting
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from config_phase4 import CV_FOLDS, PREPROCESSING_METHODS
from data_utils import load_dataset_from_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


def _setup_logging(experiment_name: str) -> logging.Logger:
    """Configure logging for experiment tracking."""
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


class ModelEvaluator:
    """Evaluate classification models with multiple metrics."""
    
    MODELS = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }
    
    PARAM_GRIDS = {
        "logistic_regression": {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "penalty": ["l2"],
        },
        "random_forest": {
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "n_estimators": [50, 100, 200],
        },
    }
    
    def __init__(self, model_name: str, preprocessing: str, logger: logging.Logger):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        if preprocessing not in PREPROCESSING_METHODS:
            raise ValueError(
                f"Unknown preprocessing: {preprocessing}. Available: {list(PREPROCESSING_METHODS)}"
            )
        self.model_name = model_name
        self.preprocessing = preprocessing
        self.logger = logger
        self.model = self.MODELS[model_name]
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.impute_values = None
        self.imputer = None

    @staticmethod
    def _coerce_features(X: pd.DataFrame) -> pd.DataFrame:
        """Convert feature table to numeric form, coercing comma-decimal strings."""
        out = X.copy()
        for col in out.columns:
            if out[col].dtype == object:
                cleaned = out[col].astype(str).str.replace(",", ".", regex=False)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception as e:
                self.logger.warning(f"ROC-AUC computation failed: {e}")
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
        
        return metrics
    
    def train_with_cv(self, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5) -> Dict:
        """Train model with cross-validation and hyperparameter tuning."""
        self.logger.info(f"Training {self.model_name} with {cv_folds}-fold CV and grid search...")

        X_scaled = self._prepare_features(X_train, fit=True)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            self.model,
            self.PARAM_GRIDS[self.model_name],
            cv=cv_folds,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_scaled, y_train)
        
        self.best_model = grid_search.best_estimator_
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate best model on test set."""
        if self.best_model is None:
            raise ValueError("Model not yet trained. Call train_with_cv() first.")

        X_scaled = self._prepare_features(X_test, fit=False)
        y_pred = self.best_model.predict(X_scaled)
        
        if hasattr(self.best_model, "predict_proba"):
            y_pred_proba = self.best_model.predict_proba(X_scaled)
        else:
            y_pred_proba = None
        
        return self.compute_metrics(y_test.to_numpy(), y_pred, y_pred_proba)


    def _prepare_features(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """Coerce to numeric, apply imputation/scaling, and optionally add missing indicators."""
        X_numeric = self._coerce_features(X)
        if fit:
            self.feature_columns = list(X_numeric.columns)
        elif self.feature_columns is not None:
            X_numeric = X_numeric.reindex(columns=self.feature_columns)

        add_indicator = self.preprocessing == "mice_indicator"
        indicator = None
        if add_indicator:
            indicator = X_numeric.isna().to_numpy().astype(float)

        if self.preprocessing == "median":
            if fit:
                self.impute_values = X_numeric.median(numeric_only=True)
                X_imputed = X_numeric.fillna(self.impute_values)
                X_scaled = self.scaler.fit_transform(X_imputed)
            else:
                X_imputed = X_numeric.fillna(self.impute_values)
                X_scaled = self.scaler.transform(X_imputed)
        else:
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer

                if fit:
                    self.imputer = IterativeImputer(
                        random_state=RANDOM_STATE,
                        max_iter=10,
                        sample_posterior=False,
                    )
                    X_imputed = self.imputer.fit_transform(X_numeric)
                    X_scaled = self.scaler.fit_transform(X_imputed)
                else:
                    if self.imputer is None:
                        raise ValueError("MICE imputer not initialized")
                    X_imputed = self.imputer.transform(X_numeric)
                    X_scaled = self.scaler.transform(X_imputed)
            except Exception as exc:
                self.logger.warning(
                    f"MICE preprocessing failed ({exc}); falling back to median imputation."
                )
                if fit or self.impute_values is None:
                    self.impute_values = X_numeric.median(numeric_only=True)
                X_imputed = X_numeric.fillna(self.impute_values)
                X_scaled = self.scaler.fit_transform(X_imputed) if fit else self.scaler.transform(X_imputed)

        if indicator is not None:
            X_scaled = np.hstack([X_scaled, indicator])

        return X_scaled


def run_phase4_experiment(dataset_path: Path, dataset_name: str, logger: logging.Logger) -> Dict:
    """Run complete Phase 4.1 experiment on a single dataset."""
    logger.info(f"Starting Phase 4.1 experiment on {dataset_name}")
    logger.info(f"Dataset path: {dataset_path}")
    
    # Load data
    X, y = load_dataset_from_csv(dataset_path, target_column="target")
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Simple train/test split (no stratification for now in Phase 4.1 basic version)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
        "n_classes": y.nunique(),
        "preprocessing": {},
    }
    
    model_names = ["logistic_regression", "random_forest"]
    for preprocessing in PREPROCESSING_METHODS:
        logger.info(f"\n=== Preprocessing: {preprocessing} ===")
        results["preprocessing"][preprocessing] = {"models": {}}

        for model_name in model_names:
            logger.info(f"\n--- Evaluating {model_name} ---")
            try:
                evaluator = ModelEvaluator(model_name, preprocessing=preprocessing, logger=logger)
                cv_info = evaluator.train_with_cv(X_train, y_train, cv_folds=CV_FOLDS)
                test_metrics = evaluator.evaluate(X_test, y_test)

                logger.info(f"Test metrics: {test_metrics}")

                results["preprocessing"][preprocessing]["models"][model_name] = {
                    "best_params": cv_info["best_params"],
                    "best_cv_score": cv_info["best_cv_score"],
                    "test_metrics": test_metrics,
                }
            except Exception as e:
                logger.error(f"Error training {model_name} ({preprocessing}): {e}", exc_info=True)
                results["preprocessing"][preprocessing]["models"][model_name] = {"error": str(e)}
    
    return results


def main():
    """Run Phase 4.1 experiments on all datasets."""
    experiment_name = "phase4_model_eval"
    logger = _setup_logging(experiment_name)
    
    logger.info("=== Starting Phase 4.1 Experiment Runner ===")
    logger.info(f"Random state: {RANDOM_STATE}")
    
    # Dataset paths (from processed directory)
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
            result = run_phase4_experiment(dataset_path, dataset_name, logger)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}", exc_info=True)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "phase4_experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    print(f"\n✓ Phase 4.1 experiments complete. See {results_file} for results.")


if __name__ == "__main__":
    main()
