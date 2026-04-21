"""Student 1 - Task 3.5: TabPFN foundation model on Student 1 scenarios.

Evaluates TabPFN across:
- preprocessing methods from Task 3.3: mean, median, knn
- scenarios aligned with Student 2: native + MCAR/MAR/MNAR rates

Outputs:
- results/tables/student1_3_5_tabpfn_results.json
- results/tables/student1_3_5_tabpfn_results.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import RANDOM_STATE
from student1_common import (
    DATASETS,
    IMPUTATION_METHODS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    MODEL_TYPES,
    PHASE_TAG_FOUNDATION,
    RESULTS_DIR,
    detect_device,
    ensure_splits,
    impute,
    inject_missingness,
    load_dataset_and_split,
    parse_scenario_key,
    setup_logging,
    test_tabpfn,
)

# Preprocessing options for TabPFN: the three classical imputers plus a
# "none" option that feeds NaN directly to TabPFN (native NaN handling).
TABPFN_PREPROCESSING = list(IMPUTATION_METHODS) + ["none"]


def run_task_3_5(dataset_path: Path, dataset_name: str, logger):
    logger.info(f"\n{'=' * 80}\nStudent 1 - Task 3.5: {dataset_name}\n{'=' * 80}")

    X, y, X_train, X_test, y_train, y_test = load_dataset_and_split(dataset_name, dataset_path, logger)

    result = {
        "dataset": dataset_name,
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),
        "split_source": "data/splits",
        "preprocessing_methods": list(TABPFN_PREPROCESSING),
        "missing_mechanisms": list(MISSING_MECHANISMS),
        "missing_rates": list(MISSING_RATES),
        "scenarios": {},
    }

    def _run_for_method(X_tr: pd.DataFrame, X_te: pd.DataFrame, method: str):
        """Run TabPFN for a given preprocessing method.

        method == 'none' means no imputation; NaN is passed directly to TabPFN
        (native NaN handling in TabPFN v2+).
        """
        if method == "none":
            return test_tabpfn(X_tr, X_te, y_train, y_test, logger, allow_nan=True)
        X_tr_imp, X_te_imp = impute(X_tr.copy(), X_te.copy(), method)
        return test_tabpfn(X_tr_imp, X_te_imp, y_train, y_test, logger, allow_nan=False)

    result["scenarios"]["native"] = {}
    for method in TABPFN_PREPROCESSING:
        logger.info(f"Scenario native | preprocessing {method}")
        result["scenarios"]["native"][method] = {"model": None, "error": None}
        try:
            result["scenarios"]["native"][method]["model"] = _run_for_method(
                X_train, X_test, method
            )
        except Exception as exc:
            result["scenarios"]["native"][method]["error"] = str(exc)
            logger.error(f"TabPFN run failed for native/{method}: {exc}", exc_info=True)

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

            for method in TABPFN_PREPROCESSING:
                result["scenarios"][scenario_key][method] = {"model": None, "error": None}
                try:
                    result["scenarios"][scenario_key][method]["model"] = _run_for_method(
                        X_train_missing, X_test, method
                    )
                except Exception as exc:
                    result["scenarios"][scenario_key][method]["error"] = str(exc)
                    logger.error(f"TabPFN run failed for {scenario_key}/{method}: {exc}", exc_info=True)

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
                model_block = cell.get("model") or {}
                metrics = model_block.get("metrics") or {}
                rows.append(
                    {
                        "phase": PHASE_TAG_FOUNDATION,
                        "dataset": dataset,
                        "model": "tabpfn",
                        "model_type": MODEL_TYPES.get("tabpfn", "Foundation"),
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
                        "available": model_block.get("available"),
                        "native_nan_input": model_block.get("native_nan_input"),
                        "n_train_used": model_block.get("n_train_used"),
                        "n_features_used": model_block.get("n_features_used"),
                        "error": model_block.get("error") or cell.get("error"),
                    }
                )
    columns = [
        "phase", "dataset", "model", "model_type", "preprocessing",
        "accuracy", "f1", "precision", "recall", "roc_auc",
        "training_time_seconds", "missing_mechanism", "missing_rate",
        "device", "available", "native_nan_input",
        "n_train_used", "n_features_used", "error",
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.reindex(columns=columns)
    return df


def main():
    logger = setup_logging("student1_3_5_tabpfn")
    logger.info("=" * 80)
    logger.info("STUDENT 1 - TASK 3.5: TABPFN FOUNDATION MODEL")
    logger.info("=" * 80)
    # Log hardware selection once upfront for the documentation section
    # "where GPU was actually used".
    detect_device(logger)

    ensure_splits(logger)

    all_results = []
    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        try:
            all_results.append(run_task_3_5(dataset_path, dataset_name, logger))
        except Exception as exc:
            logger.error(f"Task 3.5 failed for {dataset_name}: {exc}", exc_info=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "student1_3_5_tabpfn_results.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = RESULTS_DIR / "student1_3_5_tabpfn_results.csv"
    _flatten_results(all_results).to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("Student 1 Task 3.5 complete")
    print("=" * 80)
    print(f"  Results JSON: {out_json}")
    print(f"  Results CSV : {out_csv}")


if __name__ == "__main__":
    main()
