"""Reusable utilities for dataset preparation and validation."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_dirs() -> None:
    """Create required project directories if they do not exist."""
    required_dirs = [
        PROJECT_ROOT / "data" / "raw" / "taiwan",
        PROJECT_ROOT / "data" / "raw" / "polish",
        PROJECT_ROOT / "data" / "raw" / "slovak",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "splits",
        PROJECT_ROOT / "results" / "tables",
        PROJECT_ROOT / "results" / "logs",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "src",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset_from_csv(path: Path, target_column: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """Load a dataset CSV and split into X and y."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    validate_dataset(df, target_column=target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def validate_dataset(df: pd.DataFrame, target_column: str = "target") -> None:
    """Validate basic dataset requirements for classification tasks."""
    if target_column not in df.columns:
        raise ValueError(f"Missing required target column: '{target_column}'")

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 1 feature column and 1 target column.")

    y = df[target_column]
    n_classes = y.nunique(dropna=True)
    if n_classes < 2:
        raise ValueError("Target must contain at least 2 classes for classification.")


def summarize_dataset(df: pd.DataFrame, target_column: str = "target") -> Dict[str, float]:
    """Compute and print a clean dataset summary."""
    validate_dataset(df, target_column=target_column)

    n_samples = int(df.shape[0])
    n_features = int(df.shape[1] - 1)
    n_classes = int(df[target_column].nunique(dropna=True))
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1]) if df.shape[0] and df.shape[1] else 0
    missing_fraction = float(missing_cells / total_cells) if total_cells else 0.0
    task_type = detect_task_type(df[target_column])

    print("-" * 72)
    print(f"Shape: {df.shape}")
    print(f"Samples: {n_samples}")
    print(f"Features: {n_features}")
    print(f"Classes: {n_classes}")
    print(f"Task type: {task_type}")
    print(f"Missing values (cells): {missing_cells}")
    print(f"Missing fraction: {missing_fraction:.6f}")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "task_type": task_type,
        "missing_cells": missing_cells,
        "missing_fraction": missing_fraction,
    }


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    stratify: bool = True,
):
    """Create reproducible train/test split, stratified for classification by default."""
    stratify_y = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )


def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str) -> Tuple[Path, Path]:
    """Save train/test CSV files in a consistent naming format."""
    split_dir = PROJECT_ROOT / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / f"{dataset_name}_train.csv"
    test_path = split_dir / f"{dataset_name}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


def count_missing_values(df: pd.DataFrame) -> int:
    """Return total number of missing cells in a dataframe."""
    return int(df.isna().sum().sum())


def detect_task_type(y: pd.Series) -> str:
    """Detect whether classification target is binary or multiclass."""
    unique_classes = y.dropna().nunique()
    return "binary" if unique_classes == 2 else "multiclass"


def class_distribution(y: pd.Series) -> pd.Series:
    """Return normalized class distribution."""
    return y.value_counts(dropna=False, normalize=True).sort_index()

