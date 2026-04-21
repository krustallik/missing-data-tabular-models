"""Student 1 - Task 3.4: baseline models on Student 1 missingness scenarios.

Baseline models:
- logistic_regression
- random_forest

Preprocessing options come from Task 3.3:
- mean
- median
- knn

Scenarios are aligned with Student 2:
- native
- MCAR/MAR/MNAR at 5%, 10%, 15%, 20%, 30%, 40%

Outputs:
- results/tables/student1_3_4_baseline_results.json
- results/tables/student1_3_4_baseline_results.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import RANDOM_STATE
from student1_common import (
    BASELINE_MODELS,
    DATASETS,
    IMPUTATION_METHODS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    MODEL_TYPES,
    PHASE_TAG_BASELINE,
    RESULTS_DIR,
    detect_device,
    ensure_splits,
    impute,
    inject_missingness,
    load_dataset_and_split,
    parse_scenario_key,
    setup_logging,
    train_baseline,
)


def run_task_3_4(dataset_path: Path, dataset_name: str, logger):
    logger.info(f"\n{'=' * 80}\nStudent 1 - Task 3.4: {dataset_name}\n{'=' * 80}")

    X, y, X_train, X_test, y_train, y_test = load_dataset_and_split(dataset_name, dataset_path, logger)

    result = {
        "dataset": dataset_name,
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),
        "split_source": "data/splits",
        "preprocessing_methods": list(IMPUTATION_METHODS),
        "baseline_models": list(BASELINE_MODELS),
        "missing_mechanisms": list(MISSING_MECHANISMS),
        "missing_rates": list(MISSING_RATES),
        "scenarios": {},
    }

    result["scenarios"]["native"] = {}
    for method in IMPUTATION_METHODS:
        logger.info(f"Scenario native | preprocessing {method}")
        result["scenarios"]["native"][method] = {"models": {}, "error": None}
        try:
            X_tr_imp, X_te_imp = impute(X_train.copy(), X_test.copy(), method)
            for model_name in BASELINE_MODELS:
                result["scenarios"]["native"][method]["models"][model_name] = train_baseline(
                    model_name, X_tr_imp, X_te_imp, y_train, y_test, logger
                )
        except Exception as exc:
            result["scenarios"]["native"][method]["error"] = str(exc)
            logger.error(f"Baseline run failed for native/{method}: {exc}", exc_info=True)

    for mechanism in MISSING_MECHANISMS:
        for rate in MISSING_RATES:
            scenario_key = f"{mechanism}_{int(rate * 100)}pct"
            logger.info(f"Scenario {scenario_key}")
            result["scenarios"][scenario_key] = {}
            try:
                X_train_missing = inject_missingness(X_train.copy(), mechanism, rate, RANDOM_STATE)
            except Exception as exc:
                result["scenarios"][scenario_key]["error"] = str(exc)
                logger.error(f"Injection failed for {scenario_key}: {exc}", exc_info=True)
                continue

            for method in IMPUTATION_METHODS:
                result["scenarios"][scenario_key][method] = {"models": {}, "error": None}
                try:
                    X_tr_imp, X_te_imp = impute(X_train_missing.copy(), X_test.copy(), method)
                    for model_name in BASELINE_MODELS:
                        result["scenarios"][scenario_key][method]["models"][model_name] = train_baseline(
                            model_name, X_tr_imp, X_te_imp, y_train, y_test, logger
                        )
                except Exception as exc:
                    result["scenarios"][scenario_key][method]["error"] = str(exc)
                    logger.error(f"Baseline run failed for {scenario_key}/{method}: {exc}", exc_info=True)

    return result


def _flatten_results(all_results):
    rows = []
    for ds in all_results:
        dataset = ds.get("dataset")
        for scenario_key, scenario_block in ds.get("scenarios", {}).items():
            if not isinstance(scenario_block, dict):
                continue
            mechanism, rate = parse_scenario_key(scenario_key)
            for preproc, cell in scenario_block.items():
                if not isinstance(cell, dict):
                    continue
                models = cell.get("models", {}) or {}
                for model_name, model_block in models.items():
                    if not isinstance(model_block, dict):
                        continue
                    metrics = model_block.get("metrics") or {}
                    rows.append(
                        {
                            "phase": PHASE_TAG_BASELINE,
                            "dataset": dataset,
                            "model": model_name,
                            "model_type": MODEL_TYPES.get(model_name, "Classical"),
                            "preprocessing": preproc,
                            "accuracy": metrics.get("accuracy"),
                            "f1": metrics.get("f1"),
                            "precision": metrics.get("precision"),
                            "recall": metrics.get("recall"),
                            "roc_auc": metrics.get("roc_auc"),
                            "training_time_seconds": model_block.get("training_time_seconds"),
                            "missing_mechanism": mechanism,
                            "missing_rate": rate,
                            "device": model_block.get("device"),
                            "error": model_block.get("error"),
                        }
                    )
    # Column order matches Student 2 phase4_5_consolidated_results.csv plus
    # Student 1-specific columns (device, error) at the end for traceability.
    columns = [
        "phase", "dataset", "model", "model_type", "preprocessing",
        "accuracy", "f1", "precision", "recall", "roc_auc",
        "training_time_seconds", "missing_mechanism", "missing_rate",
        "device", "error",
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.reindex(columns=columns)
    return df


def main():
    logger = setup_logging("student1_3_4_baselines")
    logger.info("=" * 80)
    logger.info("STUDENT 1 - TASK 3.4: BASELINE MODELS")
    logger.info("=" * 80)
    # Baseline models are CPU-only sklearn estimators; we still log what is
    # available so the document can report "where GPU was used".
    detect_device(logger)
    logger.info("Baseline models (LogReg / RandomForest) run on CPU by design.")

    ensure_splits(logger)

    all_results = []
    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        try:
            all_results.append(run_task_3_4(dataset_path, dataset_name, logger))
        except Exception as exc:
            logger.error(f"Task 3.4 failed for {dataset_name}: {exc}", exc_info=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "student1_3_4_baseline_results.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = RESULTS_DIR / "student1_3_4_baseline_results.csv"
    _flatten_results(all_results).to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("Student 1 Task 3.4 complete")
    print("=" * 80)
    print(f"  Results JSON: {out_json}")
    print(f"  Results CSV : {out_csv}")


if __name__ == "__main__":
    main()
