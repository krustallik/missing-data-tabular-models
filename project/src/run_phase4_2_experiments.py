"""Phase 4.2 experiment runner - Extended models."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE
from config_phase4_2 import CV_FOLDS, PREPROCESSING_METHODS
from data_utils import load_dataset_from_csv
from phase4_2_additional_models import ExtendedModelEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


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


def run_phase4_2_experiment(dataset_path: Path, dataset_name: str, logger: logging.Logger) -> Dict:
    """Run Phase 4.2 experiment on a dataset."""
    logger.info(f"Starting Phase 4.2 experiment on {dataset_name}")
    
    # Load data
    X, y = load_dataset_from_csv(dataset_path, target_column="target")
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Split
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
    
    # Evaluate extended models
    model_names = ["svm", "mlp"]
    
    # Add optional models if available
    try:
        import xgboost
        model_names.append("xgboost")
    except ImportError:
        logger.info("XGBoost not available")
    
    try:
        import lightgbm
        model_names.append("lightgbm")
    except ImportError:
        logger.info("LightGBM not available")
    
    for preprocessing in PREPROCESSING_METHODS:
        logger.info(f"\n=== Preprocessing: {preprocessing} ===")
        results["preprocessing"][preprocessing] = {"models": {}}

        for model_name in model_names:
            logger.info(f"\n--- Evaluating {model_name} ---")
            try:
                evaluator = ExtendedModelEvaluator(model_name, preprocessing=preprocessing, logger=logger)
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
    """Run Phase 4.2 experiments."""
    experiment_name = "phase4_2_extended_models"
    logger = _setup_logging(experiment_name)
    
    logger.info("=== Starting Phase 4.2 Extended Models Experiment ===")
    
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
            result = run_phase4_2_experiment(dataset_path, dataset_name, logger)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}", exc_info=True)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "phase4_2_experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    print(f"\n✓ Phase 4.2 experiments complete. Results: {results_file}")


if __name__ == "__main__":
    main()
