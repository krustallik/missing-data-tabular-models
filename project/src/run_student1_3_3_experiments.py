"""Student 1 - Task 3.3: simple imputation methods on missingness scenarios.

Evaluates imputation behaviour for:
- mean
- median
- knn

Across scenarios aligned with Student 2:
- native (no injection)
- MCAR/MAR/MNAR at 5%, 10%, 15%, 20%, 30%, 40%

Outputs:
- results/tables/student1_3_3_imputation_results.json
- results/tables/student1_3_3_imputation_results.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from config import RANDOM_STATE
from student1_common import (
    DATASETS,
    IMPUTATION_METHODS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    PHASE_TAG_IMPUTATION,
    RESULTS_DIR,
    ensure_splits,
    impute,
    inject_missingness,
    load_dataset_and_split,
    parse_scenario_key,
    setup_logging,
)


def run_task_3_3(dataset_path: Path, dataset_name: str, logger):
    logger.info(f"\n{'=' * 80}\nStudent 1 - Task 3.3: {dataset_name}\n{'=' * 80}")

    X, y, X_train, X_test, y_train, y_test = load_dataset_and_split(dataset_name, dataset_path, logger)

    result = {
        "dataset": dataset_name,
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),
        "split_source": "data/splits",
        "imputation_methods": list(IMPUTATION_METHODS),
        "missing_mechanisms": list(MISSING_MECHANISMS),
        "missing_rates": list(MISSING_RATES),
        "scenarios": {},
    }

    result["scenarios"]["native"] = {}
    for method in IMPUTATION_METHODS:
        logger.info(f"Scenario native | preprocessing {method}")
        before = float(X_train.isna().sum().sum() / max(X_train.size, 1))
        start = time.time()
        X_tr_imp, X_te_imp = impute(X_train.copy(), X_test.copy(), method)
        elapsed = time.time() - start
        result["scenarios"]["native"][method] = {
            "train_missing_fraction_before": before,
            "train_missing_fraction_after": float(X_tr_imp.isna().sum().sum() / max(X_tr_imp.size, 1)),
            "test_missing_fraction_after": float(X_te_imp.isna().sum().sum() / max(X_te_imp.size, 1)),
            "imputation_time_seconds": elapsed,
            "n_train": int(X_tr_imp.shape[0]),
            "n_test": int(X_te_imp.shape[0]),
            "n_features_after": int(X_tr_imp.shape[1]),
            "error": None,
        }

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
                before = float(X_train_missing.isna().sum().sum() / max(X_train_missing.size, 1))
                start = time.time()
                try:
                    X_tr_imp, X_te_imp = impute(X_train_missing.copy(), X_test.copy(), method)
                    elapsed = time.time() - start
                    result["scenarios"][scenario_key][method] = {
                        "train_missing_fraction_before": before,
                        "train_missing_fraction_after": float(X_tr_imp.isna().sum().sum() / max(X_tr_imp.size, 1)),
                        "test_missing_fraction_after": float(X_te_imp.isna().sum().sum() / max(X_te_imp.size, 1)),
                        "imputation_time_seconds": elapsed,
                        "n_train": int(X_tr_imp.shape[0]),
                        "n_test": int(X_te_imp.shape[0]),
                        "n_features_after": int(X_tr_imp.shape[1]),
                        "error": None,
                    }
                except Exception as exc:
                    result["scenarios"][scenario_key][method] = {
                        "train_missing_fraction_before": before,
                        "train_missing_fraction_after": None,
                        "test_missing_fraction_after": None,
                        "imputation_time_seconds": None,
                        "n_train": None,
                        "n_test": None,
                        "n_features_after": None,
                        "error": str(exc),
                    }
                    logger.error(f"Imputation failed for {scenario_key}/{method}: {exc}", exc_info=True)

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
                rows.append(
                    {
                        "phase": PHASE_TAG_IMPUTATION,
                        "dataset": dataset,
                        "missing_mechanism": mechanism,
                        "missing_rate": rate,
                        "preprocessing": preproc,
                        "train_missing_fraction_before": cell.get("train_missing_fraction_before"),
                        "train_missing_fraction_after": cell.get("train_missing_fraction_after"),
                        "test_missing_fraction_after": cell.get("test_missing_fraction_after"),
                        "imputation_time_seconds": cell.get("imputation_time_seconds"),
                        "n_train": cell.get("n_train"),
                        "n_test": cell.get("n_test"),
                        "n_features_after": cell.get("n_features_after"),
                        "error": cell.get("error"),
                    }
                )
    return pd.DataFrame(rows)


def main():
    logger = setup_logging("student1_3_3_imputation")
    logger.info("=" * 80)
    logger.info("STUDENT 1 - TASK 3.3: SIMPLE IMPUTATION METHODS")
    logger.info("=" * 80)

    ensure_splits(logger)

    all_results = []
    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        try:
            all_results.append(run_task_3_3(dataset_path, dataset_name, logger))
        except Exception as exc:
            logger.error(f"Task 3.3 failed for {dataset_name}: {exc}", exc_info=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "student1_3_3_imputation_results.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = RESULTS_DIR / "student1_3_3_imputation_results.csv"
    _flatten_results(all_results).to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("Student 1 Task 3.3 complete")
    print("=" * 80)
    print(f"  Results JSON: {out_json}")
    print(f"  Results CSV : {out_csv}")


if __name__ == "__main__":
    main()
