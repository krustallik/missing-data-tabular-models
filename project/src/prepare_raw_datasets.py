"""Standardize manually downloaded raw datasets into unified processed CSV files.

This script prepares three datasets for classification experiments:
- Taiwan from data/raw/taiwan/data.csv
- Polish (1 year) from data/raw/polish/1year.arff
- Slovak manufacture subset by merging one bankrupt and one nonbankrupt CSV

Output files are saved to data/processed/ with target column named exactly 'target'.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _ensure_output_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _detect_target_column(df: pd.DataFrame, preferred: List[str] = None) -> str:
    if preferred is None:
        preferred = ["target", "class", "label", "y", "bankrupt?", "bankrupt", "status"]

    columns = list(df.columns)
    lower_map = {c.lower(): c for c in columns}

    for key in preferred:
        if key.lower() in lower_map:
            return lower_map[key.lower()]

    # Fallback: use a low-cardinality column with at least 2 classes.
    candidates = []
    for col in columns:
        n_unique = df[col].nunique(dropna=True)
        if 2 <= n_unique <= 20:
            candidates.append((col, n_unique))

    if len(candidates) == 1:
        return candidates[0][0]

    if "class" in lower_map:
        return lower_map["class"]

    raise ValueError(
        "Could not reliably detect target column. "
        f"Columns available: {columns}. Please rename target to 'target' in raw data or extend detection rules."
    )


def _decode_bytes_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(
                lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x
            )
    return out


def _standardize_target_values(y: pd.Series) -> pd.Series:
    # Convert bytes and strings to a clean comparable representation.
    y_clean = y.copy()
    if y_clean.dtype == object:
        y_clean = y_clean.apply(
            lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x
        )
    y_clean = y_clean.astype(str).str.strip()

    # Map common binary labels to 0/1 where possible.
    mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "n": 0,
        "y": 1,
        "nonbankrupt": 0,
        "bankrupt": 1,
    }
    mapped = y_clean.str.lower().map(mapping)

    if mapped.notna().all():
        return mapped.astype(int)

    # Fallback: factorize to integers while preserving class count.
    codes, uniques = pd.factorize(y_clean)
    if len(uniques) < 2:
        raise ValueError("Target must contain at least 2 classes after standardization.")
    return pd.Series(codes, index=y.index, name=y.name)


def _print_stats(name: str, df: pd.DataFrame) -> None:
    n_samples = int(df.shape[0])
    n_features = int(df.shape[1] - 1)
    n_classes = int(df["target"].nunique(dropna=True))
    missing_cells = int(df.isna().sum().sum())
    print(f"[{name}] shape={df.shape}, features={n_features}, classes={n_classes}, missing_cells={missing_cells}")


def _validate_standardized(df: pd.DataFrame, dataset_name: str) -> None:
    if "target" not in df.columns:
        raise ValueError(f"{dataset_name}: standardized dataframe missing 'target' column.")
    if df.shape[1] < 2:
        raise ValueError(f"{dataset_name}: must contain at least one feature plus target.")
    if df["target"].nunique(dropna=True) < 2:
        raise ValueError(f"{dataset_name}: target must contain at least 2 classes.")


def prepare_taiwan() -> Tuple[Path, str]:
    input_path = RAW_DIR / "taiwan" / "data.csv"
    output_path = PROCESSED_DIR / "taiwan_bankruptcy.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Taiwan file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = _normalize_column_names(df)

    detected_target = _detect_target_column(df)
    if detected_target != "target":
        df = df.rename(columns={detected_target: "target"})

    df["target"] = _standardize_target_values(df["target"])

    _validate_standardized(df, "taiwan_bankruptcy")
    df.to_csv(output_path, index=False)
    _print_stats("taiwan_bankruptcy", df)
    return output_path, detected_target


def prepare_polish_1year() -> Tuple[Path, str]:
    input_path = RAW_DIR / "polish" / "1year.arff"
    output_path = PROCESSED_DIR / "polish_1year.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Polish file not found: {input_path}")

    data, _meta = arff.loadarff(str(input_path))
    df = pd.DataFrame(data)
    df = _decode_bytes_columns(df)
    df = _normalize_column_names(df)

    detected_target = _detect_target_column(df)
    if detected_target != "target":
        df = df.rename(columns={detected_target: "target"})

    df["target"] = _standardize_target_values(df["target"])

    _validate_standardized(df, "polish_1year")
    df.to_csv(output_path, index=False)
    _print_stats("polish_1year", df)
    return output_path, detected_target


def _read_slovak_csv_with_fallback(path: Path) -> pd.DataFrame:
    # Slovak CSV files may use semicolon delimiter or ragged rows.
    attempts = [
        {"sep": ",", "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": None, "engine": "python"},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Failed to parse Slovak file {path}. Last error: {last_error}")


def prepare_slovak_manufacture_13() -> Tuple[Path, bool]:
    bankrupt_path = RAW_DIR / "slovak" / "bankrupt_manufacture_13_year_10_11_12.csv"
    nonbankrupt_path = RAW_DIR / "slovak" / "nonbankrupt_manufacture_13_year_10_11_12.csv"
    output_path = PROCESSED_DIR / "slovak_manufacture_13.csv"

    if not bankrupt_path.exists():
        raise FileNotFoundError(f"Slovak bankrupt file not found: {bankrupt_path}")
    if not nonbankrupt_path.exists():
        raise FileNotFoundError(f"Slovak nonbankrupt file not found: {nonbankrupt_path}")

    df_b = _normalize_column_names(_read_slovak_csv_with_fallback(bankrupt_path))
    df_n = _normalize_column_names(_read_slovak_csv_with_fallback(nonbankrupt_path))

    if list(df_b.columns) != list(df_n.columns):
        raise ValueError(
            "Slovak bankrupt/nonbankrupt files have different columns and cannot be merged safely. "
            f"bankrupt_cols={list(df_b.columns)} nonbankrupt_cols={list(df_n.columns)}"
        )

    df_b = df_b.copy()
    df_n = df_n.copy()
    df_b["target"] = 1
    df_n["target"] = 0

    merged = pd.concat([df_b, df_n], axis=0, ignore_index=True)
    _validate_standardized(merged, "slovak_manufacture_13")

    merged.to_csv(output_path, index=False)
    _print_stats("slovak_manufacture_13", merged)
    return output_path, True


def main() -> None:
    _ensure_output_dir()

    print("Preparing standardized datasets from local raw files...")

    taiwan_out, taiwan_target = prepare_taiwan()
    polish_out, polish_target = prepare_polish_1year()
    slovak_out, slovak_ok = prepare_slovak_manufacture_13()

    print("\nPreparation complete.")
    print(f"Taiwan target detected: {taiwan_target}")
    print(f"Polish target detected: {polish_target}")
    print(f"Slovak merge succeeded: {slovak_ok}")
    print(f"Saved: {taiwan_out}")
    print(f"Saved: {polish_out}")
    print(f"Saved: {slovak_out}")


if __name__ == "__main__":
    main()

