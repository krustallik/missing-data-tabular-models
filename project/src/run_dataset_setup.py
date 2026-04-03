"""
Phase 3.1 dataset setup runner.

Why random_state=42:
- Fixing a random seed makes train/test splits reproducible across runs.

Why stratified split for classification:
- Stratification preserves class proportions in train and test sets, reducing sampling bias.

Why Accuracy and F1 are mandatory:
- Accuracy gives a global correctness view; F1 balances precision/recall and is more robust on imbalanced labels.

When ROC-AUC should be used:
- ROC-AUC is useful when probability/ranking quality matters, especially for binary classification and class imbalance analysis.
"""

from pathlib import Path

import pandas as pd

from config import METRICS, RANDOM_STATE, RESULT_COLUMNS, TEST_SIZE
from data_utils import (
    class_distribution,
    ensure_project_dirs,
    load_dataset_from_csv,
    make_train_test_split,
    save_split,
    summarize_dataset,
    validate_dataset,
)
from dataset_registry import DATASET_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_results_template() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def run_setup() -> None:
    """Run full dataset preparation workflow for phase 3.1 using local raw files."""
    ensure_project_dirs()
    print("Using local raw files only (remote download disabled).")

    overview_rows = []
    experiment_rows = []

    for dataset_name, meta in DATASET_REGISTRY.items():
        raw_path = Path(meta["raw_path"])
        target_column = meta.get("target_column", "target")

        if not raw_path.exists():
            print(f"[SKIP] {dataset_name}: file not found at {raw_path}")
            continue

        print(f"\n=== Processing dataset: {dataset_name} ===")

        try:
            df = pd.read_csv(raw_path)
            validate_dataset(df, target_column=target_column)
            summary = summarize_dataset(df, target_column=target_column)

            X, y = load_dataset_from_csv(raw_path, target_column=target_column)
            print("Class distribution:")
            print(class_distribution(y))

            X_train, X_test, y_train, y_test = make_train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=True,
            )

            train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
            test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

            train_path, test_path = save_split(train_df, test_df, dataset_name)
            print(f"Saved split: {train_path}")
            print(f"Saved split: {test_path}")

            overview_rows.append(
                {
                    "dataset": dataset_name,
                    "raw_path": str(raw_path),
                    "n_samples": summary["n_samples"],
                    "n_features": summary["n_features"],
                    "n_classes": summary["n_classes"],
                    "target_column": target_column,
                    "task_type": summary["task_type"],
                    "missing_cells": summary["missing_cells"],
                    "missing_fraction": summary["missing_fraction"],
                }
            )

            experiment_rows.append(
                {
                    "dataset": dataset_name,
                    "random_state": RANDOM_STATE,
                    "test_size": TEST_SIZE,
                    "metrics": ",".join(METRICS),
                    "target_column": target_column,
                    "n_samples": summary["n_samples"],
                    "n_features": summary["n_features"],
                    "n_classes": summary["n_classes"],
                    "task_type": summary["task_type"],
                    "native_missing_values_present": bool(summary["missing_cells"] > 0),
                }
            )

        except Exception as exc:
            print(f"[ERROR] Failed processing {dataset_name}: {exc}")

    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    overview_df = pd.DataFrame(overview_rows)
    overview_path = tables_dir / "dataset_overview.csv"
    overview_df.to_csv(overview_path, index=False)

    experiment_df = pd.DataFrame(experiment_rows)
    experiment_path = tables_dir / "experiment_setup.csv"
    experiment_df.to_csv(experiment_path, index=False)

    template_df = _build_results_template()
    template_path = tables_dir / "results_template.csv"
    template_df.to_csv(template_path, index=False)

    print("\n=== Setup complete ===")
    print(f"Dataset overview table: {overview_path}")
    print(f"Experiment setup table: {experiment_path}")
    print(f"Results template table: {template_path}")


if __name__ == "__main__":
    run_setup()
