"""Run Phase 3.2 missing-data scenario generation on existing train/test split files."""

from pathlib import Path
from typing import List

import pandas as pd

from config import RANDOM_STATE
from data_utils import ensure_project_dirs
from missingness import (
    calculate_missing_fraction,
    inject_mar,
    inject_mcar,
    inject_mnar,
    summarize_missingness_change,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SCENARIOS_DIR = PROJECT_ROOT / "data" / "scenarios"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

DATASETS = [
    "taiwan_bankruptcy",
    "polish_1year",
    "slovak_manufacture_13",
]
SPLITS = ["train", "test"]
TARGET_RATES = [0.1, 0.2, 0.3, 0.4]
TARGET_COLUMN = "target"


def _ensure_dirs() -> None:
    ensure_project_dirs()
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _load_split(dataset_name: str, split_name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{dataset_name}_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in split file: {path}")
    return df


def _save_scenario(df: pd.DataFrame, dataset_name: str, split_name: str, missing_type: str, rate: float) -> Path:
    suffix = int(round(rate * 100))
    out_dir = SCENARIOS_DIR / dataset_name / missing_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}_{split_name}_{missing_type.lower()}_{suffix}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _run_for_one(df: pd.DataFrame, missing_type: str, rate: float, seed: int) -> pd.DataFrame:
    features = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    exclude = []
    if missing_type == "MCAR":
        masked_X = inject_mcar(features, target_final_missing_rate=rate, random_state=seed, exclude_columns=exclude)
    elif missing_type == "MAR":
        masked_X = inject_mar(features, target_final_missing_rate=rate, random_state=seed, exclude_columns=exclude)
    elif missing_type == "MNAR":
        masked_X = inject_mnar(features, target_final_missing_rate=rate, random_state=seed, exclude_columns=exclude)
    else:
        raise ValueError(f"Unsupported missing type: {missing_type}")

    out = pd.concat([masked_X, y], axis=1)
    return out


def _write_methodology() -> Path:
    text = """Phase 3.2 Missingness Methodology

MCAR generation:
- Randomly masks eligible observed cells from numeric feature columns.
- Adds enough missing cells to approximate the requested FINAL missing fraction.

MAR generation:
- For each masked numeric feature, chooses another numeric feature as a control.
- Rows with control values above the median receive higher masking probability.
- Samples observed candidate cells with weighted probabilities to approximate target FINAL missing fraction.

MNAR generation (synthetic):
- For each masked numeric feature, uses that feature's own observed values as the control.
- Rows with values above the median receive higher masking probability.
- Then masks those selected cells (making them missing), approximating the target FINAL missing fraction.

General rules:
- Target column is excluded from masking.
- Only numeric feature columns are masked.
- Existing native missing values are preserved and counted as current state.
- If current missing fraction is already >= requested target rate, no extra masking is added.
- Requested rates refer to approximate FINAL missing fraction over numeric feature columns.
"""
    out_path = LOGS_DIR / "missingness_methodology.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def run_generation() -> None:
    _ensure_dirs()

    report_rows: List[dict] = []

    for dataset_name in DATASETS:
        for split_name in SPLITS:
            try:
                base_df = _load_split(dataset_name, split_name)
            except Exception as exc:
                print(f"[ERROR] {dataset_name}/{split_name}: {exc}")
                continue

            base_X = base_df.drop(columns=[TARGET_COLUMN])
            initial_rate = calculate_missing_fraction(base_X, exclude_columns=[])

            for missing_type in ["MCAR", "MAR", "MNAR"]:
                for rate in TARGET_RATES:
                    seed = RANDOM_STATE + int(rate * 1000) + (0 if split_name == "train" else 1)

                    try:
                        scenario_df = _run_for_one(base_df, missing_type=missing_type, rate=rate, seed=seed)
                        out_path = _save_scenario(
                            scenario_df,
                            dataset_name=dataset_name,
                            split_name=split_name,
                            missing_type=missing_type,
                            rate=rate,
                        )

                        stats = summarize_missingness_change(
                            before_df=base_df.drop(columns=[TARGET_COLUMN]),
                            after_df=scenario_df.drop(columns=[TARGET_COLUMN]),
                            exclude_columns=[],
                        )

                        report_rows.append(
                            {
                                "dataset": dataset_name,
                                "split": split_name,
                                "missing_type": missing_type,
                                "target_rate": rate,
                                "initial_missing_rate": initial_rate,
                                "final_missing_rate": stats["final_missing_rate"],
                                "added_missing_cells": stats["added_missing_cells"],
                            }
                        )

                        print(
                            f"[OK] {dataset_name}/{split_name}/{missing_type}/{int(rate*100)} "
                            f"-> final_missing={stats['final_missing_rate']:.4f}, saved={out_path}"
                        )
                    except Exception as exc:
                        print(
                            f"[ERROR] Failed {dataset_name}/{split_name}/{missing_type}/{rate}: {exc}"
                        )

    report_df = pd.DataFrame(report_rows, columns=[
        "dataset",
        "split",
        "missing_type",
        "target_rate",
        "initial_missing_rate",
        "final_missing_rate",
        "added_missing_cells",
    ])
    report_path = TABLES_DIR / "missingness_report.csv"
    report_df.to_csv(report_path, index=False)

    methodology_path = _write_methodology()

    print("\n=== Missingness generation complete ===")
    print(f"Report: {report_path}")
    print(f"Methodology log: {methodology_path}")


if __name__ == "__main__":
    run_generation()

