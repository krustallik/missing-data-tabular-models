"""Sanity check for the missingness injectors.

For every (dataset, mechanism, rate) combination we inject missing values
into the *training* split only and compare the realised missing fraction
(computed over eligible numeric feature columns) against the requested
target. Differences above :data:`config.MISSINGNESS_TOLERANCE` are flagged.

Output: ``results/tables/missingness_verification.csv``.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from config import (
    DATASETS,
    EXPERIMENT_SEEDS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    MISSINGNESS_TOLERANCE,
    OUTPUT_FILES,
    RANDOM_STATES,
)
from data_utils import load_precomputed_split, setup_logging
from missingness import (
    calculate_missing_fraction,
    inject_mar,
    inject_mcar,
    inject_mnar,
    summarize_missingness_change,
)


INJECTORS = {
    "MCAR": inject_mcar,
    "MAR": inject_mar,
    "MNAR": inject_mnar,
}


def _verify_one(
    dataset_name: str, X_train, mechanism: str, rate: float, rs: int,
    logger: logging.Logger,
) -> Dict:
    inject_fn = INJECTORS[mechanism]
    X_missing = inject_fn(X_train.copy(), rate, random_state=rs)
    realized = calculate_missing_fraction(X_missing)
    delta = realized - rate
    row = {
        "dataset": dataset_name,
        "mechanism": mechanism,
        "target_rate": rate,
        "realized_rate": float(realized),
        "delta": float(delta),
        "abs_delta": float(abs(delta)),
        "tolerance": MISSINGNESS_TOLERANCE,
        "within_tolerance": bool(abs(delta) <= MISSINGNESS_TOLERANCE),
    }
    try:
        summary = summarize_missingness_change(X_train, X_missing)
        row["added_missing_cells"] = int(summary.get("added_missing_cells", 0))
        row["affected_columns"] = int(summary.get("affected_columns", 0))
    except Exception:
        row["added_missing_cells"] = None
        row["affected_columns"] = None
    return row


def run_missingness_verification(logger: logging.Logger = None) -> pd.DataFrame:
    """Run the verification loop and persist the resulting CSV."""
    if logger is None:
        logger = setup_logging("missingness_check")
    rows: List[Dict] = []
    for dataset_name in DATASETS:
        for split_seed in RANDOM_STATES:
            loaded = load_precomputed_split(dataset_name, random_state=split_seed, logger=logger)
            if loaded is None:
                logger.warning(f"Skipping {dataset_name}: no precomputed split for random_state={split_seed}")
                continue
            X_train, _, _, _ = loaded
            logger.info(
                f"Verifying {dataset_name}: split_seed={split_seed} "
                f"X_train shape={X_train.shape}"
            )
            for seed in EXPERIMENT_SEEDS:
                for mechanism in MISSING_MECHANISMS:
                    for rate in MISSING_RATES:
                        try:
                            row = _verify_one(dataset_name, X_train, mechanism, rate, seed, logger)
                            row["split_seed"] = split_seed
                            row["seed"] = seed
                            rows.append(row)
                            tag = "ok" if rows[-1]["within_tolerance"] else "OUT-OF-TOL"
                            logger.info(
                                f"  split={split_seed} seed={seed} {mechanism} @ {int(rate*100):>2}%: "
                                f"realized={rows[-1]['realized_rate']:.4f} "
                                f"delta={rows[-1]['delta']:+.4f} [{tag}]"
                            )
                        except Exception as exc:
                            logger.error(
                                f"  split={split_seed} seed={seed} {mechanism} @ "
                                f"{int(rate*100)}%: injection failed: {exc}",
                                exc_info=True,
                            )
                            rows.append({
                                "dataset": dataset_name,
                                "split_seed": split_seed,
                                "seed": seed,
                                "mechanism": mechanism,
                                "target_rate": rate,
                                "realized_rate": None,
                                "delta": None,
                                "abs_delta": None,
                                "tolerance": MISSINGNESS_TOLERANCE,
                                "within_tolerance": False,
                                "added_missing_cells": None,
                                "affected_columns": None,
                                "error": str(exc),
                            })

    df = pd.DataFrame(rows)
    out_path = OUTPUT_FILES["missingness_verification"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path.name}: {len(df)} rows")
    return df


if __name__ == "__main__":
    run_missingness_verification()
