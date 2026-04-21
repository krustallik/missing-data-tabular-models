"""Dataset I/O and split utilities.

Everything that deals with *complete* (non-missing-injected) dataframes lives
here: loading CSVs, validating them, creating train/test splits, and loading
precomputed splits back on every run.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

from config import (
    DROP_UNNAMED_COLUMNS,
    LOGS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
)


# ── Directory + logging helpers ──────────────────────────────────────────────

def ensure_project_dirs() -> None:
    """Create every directory the pipeline may write to."""
    for d in (RAW_DIR, PROCESSED_DIR, SPLITS_DIR, TABLES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ensure_output_dirs()


def setup_logging(step_name: str) -> logging.Logger:
    """Create a file logger for one pipeline step."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(step_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOGS_DIR / f"{step_name}_{ts}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler for real-time visibility during long runs (steps 2/3/4).
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset_from_csv(
    path: Path, target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    validate_dataset(df, target_column=target_column)
    X = df.drop(columns=[target_column])
    X = sanitize_feature_columns(X)
    return X, df[target_column]


def validate_dataset(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Missing required target column: '{target_column}'")
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 1 feature column and 1 target.")
    if df[target_column].nunique(dropna=True) < 2:
        raise ValueError("Target must contain at least 2 classes for classification.")


def summarize_dataset(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Dict[str, float]:
    validate_dataset(df, target_column=target_column)
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1]) if df.shape[0] and df.shape[1] else 0
    return {
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "n_classes": int(df[target_column].nunique(dropna=True)),
        "task_type": detect_task_type(df[target_column]),
        "missing_cells": missing_cells,
        "missing_fraction": float(missing_cells / total_cells) if total_cells else 0.0,
    }


def detect_task_type(y: pd.Series) -> str:
    return "binary" if y.dropna().nunique() == 2 else "multiclass"


def class_distribution(y: pd.Series) -> pd.Series:
    return y.value_counts(dropna=False, normalize=True).sort_index()


# ── Numeric coercion (shared) ────────────────────────────────────────────────

def coerce_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert every feature column to numeric, handling comma decimals.

    Non-parseable values become ``NaN`` so downstream imputers can decide how
    to treat them.
    """
    out = X.copy()
    for col in out.columns:
        # Handle both classic object dtype and pandas StringDtype columns where
        # decimals may be stored with commas (e.g. "29,11").
        if not is_numeric_dtype(out[col]):
            cleaned = out[col].astype(str).str.replace(",", ".", regex=False)
            out[col] = pd.to_numeric(cleaned, errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def sanitize_feature_columns(
    X: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
    context: str = "",
) -> pd.DataFrame:
    """Drop technical index-artifact columns like ``Unnamed: 0`` when enabled."""
    out = X.copy()
    if not DROP_UNNAMED_COLUMNS:
        return out
    unnamed_cols = [c for c in out.columns if str(c).lower().startswith("unnamed:")]
    if unnamed_cols:
        out = out.drop(columns=unnamed_cols)
        if logger is not None:
            suffix = f" ({context})" if context else ""
            logger.info(f"Dropped columns{suffix}: {unnamed_cols}")
    return out


# ── Splits ───────────────────────────────────────────────────────────────────

def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    stratify: bool = True,
):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str) -> Tuple[Path, Path]:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS_DIR / f"{dataset_name}_train.csv"
    test_path = SPLITS_DIR / f"{dataset_name}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def load_precomputed_split(dataset_name: str, logger: Optional[logging.Logger] = None) -> Optional[Tuple]:
    """Return (X_train, X_test, y_train, y_test) for ``dataset_name`` or None."""
    train_path = SPLITS_DIR / f"{dataset_name}_train.csv"
    test_path = SPLITS_DIR / f"{dataset_name}_test.csv"
    if not train_path.exists() or not test_path.exists():
        if logger is not None:
            logger.warning(f"Precomputed split missing for {dataset_name}")
        return None
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
        if logger is not None:
            logger.warning("Split CSVs missing target column")
        return None
    X_train = sanitize_feature_columns(
        coerce_features(train_df.drop(columns=[TARGET_COLUMN])),
        logger=logger,
        context=f"{dataset_name} train",
    )
    y_train = train_df[TARGET_COLUMN]
    X_test = sanitize_feature_columns(
        coerce_features(test_df.drop(columns=[TARGET_COLUMN])),
        logger=logger,
        context=f"{dataset_name} test",
    )
    y_test = test_df[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def splits_present(dataset_names) -> bool:
    for name in dataset_names:
        if not (SPLITS_DIR / f"{name}_train.csv").exists():
            return False
        if not (SPLITS_DIR / f"{name}_test.csv").exists():
            return False
    return True


# ── Misc ─────────────────────────────────────────────────────────────────────

def count_missing_values(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())
