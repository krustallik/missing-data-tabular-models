"""Missing-data scenario generation utilities (Phase 3.2)."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _resolve_excluded_columns(exclude_columns: Optional[Sequence[str]]) -> set:
    return set(exclude_columns or [])


def _numeric_feature_columns(df: pd.DataFrame, exclude_columns: Optional[Sequence[str]] = None) -> List[str]:
    excluded = _resolve_excluded_columns(exclude_columns)
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    if not cols:
        raise ValueError("No eligible numeric feature columns found for missingness injection.")
    return cols


def calculate_missing_fraction(X: pd.DataFrame, exclude_columns=None) -> float:
    """Calculate missing-cell fraction over eligible numeric feature columns only."""
    cols = _numeric_feature_columns(X, exclude_columns=exclude_columns)
    subset = X[cols]
    total_cells = subset.shape[0] * subset.shape[1]
    if total_cells == 0:
        raise ValueError("No eligible cells available to compute missing fraction.")
    return float(subset.isna().sum().sum() / total_cells)


def _eligible_observed_positions(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    observed_mask = ~df[cols].isna().to_numpy()
    return np.where(observed_mask)


def _target_missing_cells(total_cells: int, target_final_missing_rate: float) -> int:
    if not (0.0 <= target_final_missing_rate <= 1.0):
        raise ValueError("target_final_missing_rate must be between 0 and 1.")
    return int(round(total_cells * target_final_missing_rate))


def inject_mcar(
    X: pd.DataFrame,
    target_final_missing_rate: float,
    random_state: int = 42,
    exclude_columns=None,
) -> pd.DataFrame:
    """Inject MCAR missingness to reach an approximate final missing fraction."""
    df = X.copy(deep=True)
    cols = _numeric_feature_columns(df, exclude_columns=exclude_columns)
    subset = df[cols]

    total_cells = subset.shape[0] * subset.shape[1]
    current_missing = int(subset.isna().sum().sum())
    desired_missing = _target_missing_cells(total_cells, target_final_missing_rate)

    if current_missing >= desired_missing:
        print(
            f"[WARN][MCAR] Current missing fraction ({current_missing / total_cells:.4f}) "
            f">= target ({target_final_missing_rate:.4f}). No additional masking applied."
        )
        return df

    missing_to_add = desired_missing - current_missing
    row_idx, col_idx = _eligible_observed_positions(df, cols)

    if len(row_idx) == 0:
        print("[WARN][MCAR] No observed numeric cells available for additional masking.")
        return df

    if missing_to_add > len(row_idx):
        print(
            f"[WARN][MCAR] Requested {missing_to_add} new missing cells, "
            f"but only {len(row_idx)} observed cells available. Masking all available."
        )
        missing_to_add = len(row_idx)

    rng = np.random.default_rng(random_state)
    chosen = rng.choice(len(row_idx), size=missing_to_add, replace=False)

    for idx in chosen:
        r = row_idx[idx]
        c = col_idx[idx]
        df.iat[r, df.columns.get_loc(cols[c])] = np.nan

    return df


def _weighted_row_candidates(control_values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    observed_rows = control_values.dropna().index.to_numpy()
    if len(observed_rows) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    med = float(control_values.median(skipna=True))
    vals = control_values.loc[observed_rows]

    # Higher probability above median to satisfy MAR dependence.
    weights = np.where(vals.to_numpy() > med, 0.8, 0.2).astype(float)
    if weights.sum() == 0:
        weights = np.ones_like(weights, dtype=float)
    probs = weights / weights.sum()
    return observed_rows, probs


def inject_mar(
    X: pd.DataFrame,
    target_final_missing_rate: float,
    random_state: int = 42,
    exclude_columns=None,
) -> pd.DataFrame:
    """Inject MAR missingness where masking in one feature depends on another feature."""
    df = X.copy(deep=True)
    cols = _numeric_feature_columns(df, exclude_columns=exclude_columns)
    subset = df[cols]

    total_cells = subset.shape[0] * subset.shape[1]
    current_missing = int(subset.isna().sum().sum())
    desired_missing = _target_missing_cells(total_cells, target_final_missing_rate)

    if current_missing >= desired_missing:
        print(
            f"[WARN][MAR] Current missing fraction ({current_missing / total_cells:.4f}) "
            f">= target ({target_final_missing_rate:.4f}). No additional masking applied."
        )
        return df

    missing_to_add = desired_missing - current_missing
    rng = np.random.default_rng(random_state)

    # Build candidate (row, feature) pairs with MAR control probabilities.
    candidate_pairs: List[Tuple[int, str]] = []
    candidate_weights: List[float] = []

    for i, masked_col in enumerate(cols):
        control_col = cols[(i + 1) % len(cols)] if len(cols) > 1 else cols[i]
        control_values = df[control_col]

        rows, row_probs = _weighted_row_candidates(control_values)
        if len(rows) == 0:
            continue

        observed_in_masked = df.loc[rows, masked_col].notna().to_numpy()
        rows = rows[observed_in_masked]
        row_probs = row_probs[observed_in_masked]
        if len(rows) == 0:
            continue

        row_probs = row_probs / row_probs.sum()

        for r, p in zip(rows, row_probs):
            candidate_pairs.append((int(r), masked_col))
            candidate_weights.append(float(p))

    if not candidate_pairs:
        print("[WARN][MAR] No eligible MAR candidates found for additional masking.")
        return df

    weights_arr = np.array(candidate_weights, dtype=float)
    if weights_arr.sum() == 0:
        weights_arr = np.ones_like(weights_arr, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()

    max_addable = len(candidate_pairs)
    if missing_to_add > max_addable:
        print(
            f"[WARN][MAR] Requested {missing_to_add} new missing cells, "
            f"but only {max_addable} MAR candidates available. Masking all available."
        )
        missing_to_add = max_addable

    chosen_idx = rng.choice(np.arange(max_addable), size=missing_to_add, replace=False, p=weights_arr)

    for idx in chosen_idx:
        row_i, col_name = candidate_pairs[int(idx)]
        df.at[row_i, col_name] = np.nan

    return df


def summarize_missingness_change(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    exclude_columns=None,
) -> Dict[str, float]:
    """Return initial/final missing rates and how many new missing cells were added."""
    cols = _numeric_feature_columns(before_df, exclude_columns=exclude_columns)

    before_missing = int(before_df[cols].isna().sum().sum())
    after_missing = int(after_df[cols].isna().sum().sum())
    added = after_missing - before_missing

    return {
        "initial_missing_rate": calculate_missing_fraction(before_df, exclude_columns=exclude_columns),
        "final_missing_rate": calculate_missing_fraction(after_df, exclude_columns=exclude_columns),
        "added_missing_cells": int(max(0, added)),
    }

