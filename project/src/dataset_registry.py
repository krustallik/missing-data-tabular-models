"""Dataset registry for phase 3.1 dataset preparation."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_REGISTRY = {
    "taiwan_bankruptcy": {
        "name": "taiwan_bankruptcy",
        "source": "Manual local source",
        "target_column": "target",
        "raw_path": PROJECT_ROOT / "data" / "processed" / "taiwan_bankruptcy.csv",
        "description": "Taiwanese Bankruptcy Prediction standardized dataset.",
        "task_type": "classification",
        "label_type": "binary",
        "native_missing_values_expected": True,
    },
    "polish_1year": {
        "name": "polish_1year",
        "source": "Manual local source (ARFF converted)",
        "target_column": "target",
        "raw_path": PROJECT_ROOT / "data" / "processed" / "polish_1year.csv",
        "description": "Polish bankruptcy 1st year standardized dataset.",
        "task_type": "classification",
        "label_type": "binary",
        "native_missing_values_expected": True,
    },
    "slovak_manufacture_13": {
        "name": "slovak_manufacture_13",
        "source": "Manual local source (merged CSV files)",
        "target_column": "target",
        "raw_path": PROJECT_ROOT / "data" / "processed" / "slovak_manufacture_13.csv",
        "description": "Slovak manufacture subset (bankrupt + nonbankrupt) standardized dataset.",
        "task_type": "classification",
        "label_type": "binary",
        "native_missing_values_expected": True,
    },
}
