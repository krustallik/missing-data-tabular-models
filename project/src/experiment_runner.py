"""Core experiment loop: inject missingness, impute, train, evaluate.

The single iteration order is:

    for dataset in DATASETS:
        for mechanism in ['native', MCAR, MAR, MNAR]:
            for rate in MISSING_RATES (or [0.0] for native):
                X_train_missing = inject(X_train, mechanism, rate)
                for imputation in IMPUTATION_METHODS:
                    X_tr, X_te = impute(X_train_missing, X_test, imputation)
                    for model in selected models:
                        result = train_and_evaluate(model, X_tr, X_te, y_train, y_test)

Skipped combinations (recorded as ``skipped``):

- ``imputation='none'`` with a classical model that does not accept NaN.
- Any combination where imputation itself raises (e.g. the KNN matrix-size
  guard); the error is recorded and downstream models are skipped.

Outputs are written incrementally to ``experiment_results.csv`` so that a
crash mid-way still leaves a usable partial table.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from config import (
    ALL_MODELS,
    DATASETS,
    IMPUTATION_METHODS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    OUTPUT_FILES,
    RANDOM_STATE,
    RESULT_COLUMNS,
    ensure_output_dirs,
)
from data_utils import load_precomputed_split, setup_logging
from imputation import impute
from missingness import inject_mar, inject_mcar, inject_mnar
from models import accepts_nan, display_name, model_type, train_and_evaluate


_INJECTORS = {
    "MCAR": inject_mcar,
    "MAR": inject_mar,
    "MNAR": inject_mnar,
}


def _apply_missingness(X, mechanism: str, rate: float, random_state: int):
    if mechanism == "native":
        return X
    return _INJECTORS[mechanism](X.copy(), rate, random_state=random_state)


def _should_skip(model_key: str, imputation: str) -> Optional[str]:
    """Return a short reason string if this combination should be skipped."""
    if imputation == "none" and not accepts_nan(model_key):
        return "classical model cannot consume NaN"
    return None


def _record(
    dataset: str, mechanism: Optional[str], rate: Optional[float],
    imputation: str, model_key: str, outcome: Dict, skipped_reason: Optional[str] = None,
) -> Dict:
    metrics = outcome.get("metrics") if outcome else None
    row = {
        "dataset": dataset,
        "missing_mechanism": mechanism,
        "missing_rate": None if rate is None else round(float(rate) * 100, 2),
        "imputation": imputation,
        "model": display_name(model_key),
        "model_type": model_type(model_key),
        "accuracy": metrics.get("accuracy") if metrics else None,
        "f1": metrics.get("f1") if metrics else None,
        "precision": metrics.get("precision") if metrics else None,
        "recall": metrics.get("recall") if metrics else None,
        "roc_auc": metrics.get("roc_auc") if metrics else None,
        "balanced_accuracy": metrics.get("balanced_accuracy") if metrics else None,
        "f1_macro": metrics.get("f1_macro") if metrics else None,
        "recall_class1": metrics.get("recall_class1") if metrics else None,
        "pr_auc": metrics.get("pr_auc") if metrics else None,
        "threshold": metrics.get("threshold") if metrics else None,
        "training_time_seconds": outcome.get("training_time_seconds") if outcome else None,
        "error": skipped_reason or (outcome.get("error") if outcome else None),
    }
    return row


def _write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_path, index=False)


def run_experiments(
    datasets: Optional[Iterable[str]] = None,
    mechanisms: Optional[Iterable[str]] = None,
    rates: Optional[Iterable[float]] = None,
    imputations: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    include_native: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Run the full experiment matrix (or any filtered subset).

    Every argument defaults to the full grid declared in :mod:`config`.
    Progress is logged per inner iteration, and the resulting table is
    written to :data:`config.OUTPUT_FILES["experiment_results"]` after each
    dataset finishes so partial runs are preserved.
    """
    ensure_output_dirs()
    if logger is None:
        logger = setup_logging("experiments")

    datasets = list(datasets) if datasets is not None else list(DATASETS.keys())
    mechanisms = list(mechanisms) if mechanisms is not None else list(MISSING_MECHANISMS)
    rates = list(rates) if rates is not None else list(MISSING_RATES)
    imputations = list(imputations) if imputations is not None else list(IMPUTATION_METHODS)
    models_ = list(models) if models is not None else list(ALL_MODELS)

    logger.info("=" * 80)
    logger.info(f"Datasets     : {datasets}")
    logger.info(f"Mechanisms   : {mechanisms} (native={'yes' if include_native else 'no'})")
    logger.info(f"Rates        : {rates}")
    logger.info(f"Imputations  : {imputations}")
    logger.info(f"Models       : {models_}")
    logger.info("=" * 80)

    rows: List[Dict] = []
    out_path = OUTPUT_FILES["experiment_results"]
    out_json = OUTPUT_FILES["experiment_results_json"]

    total_scenarios = len(datasets) * (
        (1 if include_native else 0) + len(mechanisms) * len(rates)
    )
    scenario_idx = 0

    for dataset in datasets:
        if dataset not in DATASETS:
            logger.warning(f"Dataset {dataset!r} not in registry; skipping")
            continue
        loaded = load_precomputed_split(dataset, logger=logger)
        if loaded is None:
            logger.error(f"No split for {dataset}; skipping")
            continue
        X_train_full, X_test, y_train, y_test = loaded
        logger.info(
            f"\n[{dataset}] train={X_train_full.shape} test={X_test.shape}"
        )

        scenarios: List = []
        if include_native:
            scenarios.append(("native", 0.0))
        for mech in mechanisms:
            for rate in rates:
                scenarios.append((mech, float(rate)))

        for mech, rate in scenarios:
            scenario_idx += 1
            label = "native" if mech == "native" else f"{mech}_{int(rate*100)}pct"
            logger.info(f"[{scenario_idx}/{total_scenarios}] {dataset} scenario={label}")
            try:
                X_train_miss = _apply_missingness(X_train_full, mech, rate, RANDOM_STATE)
            except Exception as exc:
                logger.error(f"  injection failed: {exc}", exc_info=True)
                for imp in imputations:
                    for m in models_:
                        rows.append(_record(
                            dataset, None if mech == "native" else mech,
                            None if mech == "native" else rate,
                            imp, m, {"error": f"injection failed: {exc}"},
                        ))
                continue

            for imp in imputations:
                try:
                    X_tr, X_te = impute(X_train_miss, X_test, imp)
                    impute_error = None
                except Exception as exc:
                    X_tr = X_te = None
                    impute_error = str(exc)
                    logger.warning(f"  imputation {imp} failed: {exc}")

                for m in models_:
                    if impute_error is not None:
                        rows.append(_record(
                            dataset, None if mech == "native" else mech,
                            None if mech == "native" else rate,
                            imp, m, {"error": f"imputation failed: {impute_error}"},
                        ))
                        continue
                    skip = _should_skip(m, imp)
                    if skip is not None:
                        rows.append(_record(
                            dataset, None if mech == "native" else mech,
                            None if mech == "native" else rate,
                            imp, m, {}, skipped_reason=f"skipped: {skip}",
                        ))
                        continue
                    outcome = train_and_evaluate(
                        m, X_tr, X_te, y_train, y_test, logger=logger
                    )
                    rows.append(_record(
                        dataset, None if mech == "native" else mech,
                        None if mech == "native" else rate,
                        imp, m, outcome,
                    ))
                    if outcome.get("error"):
                        logger.info(f"    {m:20s} [{imp:15s}] -> error: {outcome['error'][:80]}")
                    elif outcome.get("metrics"):
                        mt = outcome["metrics"]
                        acc = mt.get("accuracy", float("nan"))
                        bacc = mt.get("balanced_accuracy", float("nan"))
                        f1m = mt.get("f1_macro", float("nan"))
                        prauc = mt.get("pr_auc", float("nan"))
                        thr = mt.get("threshold", 0.5)
                        t = outcome.get("training_time_seconds") or 0.0
                        logger.info(
                            f"    {m:20s} [{imp:15s}] "
                            f"acc={acc:.4f} bacc={bacc:.4f} f1m={f1m:.4f} prauc={prauc:.4f} "
                            f"thr={thr:.2f} t={t:.1f}s"
                        )

        # Persist partial table after each dataset.
        _write_csv(rows, out_path)
        logger.info(f"Saved partial results: {out_path.name} ({len(rows)} rows)")

    _write_csv(rows, out_path)

    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"generated_at": datetime.now().isoformat(), "rows": rows}, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not write {out_json.name}: {exc}")

    logger.info(f"Done. Total rows: {len(rows)}")
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


if __name__ == "__main__":
    run_experiments()
