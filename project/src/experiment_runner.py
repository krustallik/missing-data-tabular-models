"""Core experiment loop: inject missingness, impute, train, evaluate.

The single iteration order is:

    for dataset in DATASETS:
        for split_seed in RANDOM_STATES:
            load split created with split_seed
            for seed in EXPERIMENT_SEEDS:
                for mechanism in ['native', MCAR, MAR, MNAR]:
                    for rate in MISSING_RATES (or [0.0] for native):
                        X_train_missing = inject(X_train, mechanism, rate, seed)
                        for imputation in IMPUTATION_METHODS:
                            X_tr, X_te = impute(..., random_state=seed)
                            for model in selected models:
                                result = train_and_evaluate(..., random_state=seed)

Skipped combinations (recorded as ``skipped``):

- ``imputation='none'`` with a classical model that does not accept NaN.
- Any combination where imputation itself raises (e.g. the KNN matrix-size
  guard); the error is recorded and downstream models are skipped.

Outputs are written incrementally to ``experiment_results.csv`` after every
single model finishes, so an aborted run can be resumed by re-running with
``resume=True`` (the default).

Resume protocol
~~~~~~~~~~~~~~~

A finished experiment is uniquely identified by the 7-tuple:

    (dataset, split_seed, seed, missing_mechanism, missing_rate,
     imputation, model)

For the *native* scenario we store a stable representation
(``missing_mechanism="native"``, ``missing_rate=0.0``) instead of NaN so the
key comparison stays robust across CSV round-trips. Legacy CSVs that still
have NaN for native rows are normalised on load.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from config import (
    ALL_MODELS,
    DATASETS,
    EXPERIMENT_SEEDS,
    IMPUTATION_METHODS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    OUTPUT_FILES,
    RANDOM_STATES,
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


# Sentinel values used in CSV for the no-injection scenario. Kept as plain
# strings/floats so the resume key remains stable across pandas read/write
# cycles (None/NaN comparisons are fragile in tuple sets).
NATIVE_MECHANISM = "native"
NATIVE_RATE = 0.0

# Key columns that uniquely identify a single trained model row.
KEY_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "split_seed",
    "seed",
    "missing_mechanism",
    "missing_rate",
    "imputation",
    "model",
)

ExperimentKey = Tuple[str, int, int, str, float, str, str]


def _apply_missingness(X, mechanism: str, rate: float, random_state: int):
    if mechanism == NATIVE_MECHANISM:
        return X
    return _INJECTORS[mechanism](X.copy(), rate, random_state=random_state)


def _should_skip(model_key: str, imputation: str) -> Optional[str]:
    """Return a short reason string if this combination should be skipped."""
    if imputation == "none" and not accepts_nan(model_key):
        return "classical model cannot consume NaN"
    return None


def _scenario_columns(
    mechanism: str, rate: float,
) -> Tuple[str, float]:
    """Return ``(mechanism, rate_pct)`` as they appear in the CSV.

    ``native`` is normalised to ``("native", 0.0)``; all other scenarios use
    the percentage form (``5.0``, ``10.0``, ...) for ``missing_rate``.
    """
    if mechanism == NATIVE_MECHANISM:
        return NATIVE_MECHANISM, NATIVE_RATE
    return mechanism, round(float(rate) * 100, 2)


def _record(
    dataset: str,
    split_seed: int,
    seed: int,
    mechanism: str,
    rate: float,
    imputation: str,
    model_key: str,
    outcome: Dict,
    skipped_reason: Optional[str] = None,
) -> Dict:
    metrics = outcome.get("metrics") if outcome else None
    row = {
        "dataset": dataset,
        "split_seed": int(split_seed),
        "seed": int(seed),
        "missing_mechanism": str(mechanism),
        "missing_rate": float(rate),
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


def _row_key(row: Dict) -> ExperimentKey:
    """Build the canonical resume key from a row dict.

    Values are cast to deterministic types so the same tuple is produced
    whether the row was just created in-memory or round-tripped through a
    pandas-written CSV.
    """
    mech_raw = row.get("missing_mechanism")
    if mech_raw is None or (isinstance(mech_raw, float) and pd.isna(mech_raw)):
        mech = NATIVE_MECHANISM
    else:
        mech = str(mech_raw)
    rate_raw = row.get("missing_rate")
    if rate_raw is None or (isinstance(rate_raw, float) and pd.isna(rate_raw)):
        rate = NATIVE_RATE
    else:
        rate = round(float(rate_raw), 2)
    return (
        str(row["dataset"]),
        int(row["split_seed"]),
        int(row["seed"]),
        mech,
        rate,
        str(row["imputation"]),
        str(row["model"]),
    )


def _normalize_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a previously saved CSV to the new stable native representation.

    - ``missing_mechanism`` NaN → ``"native"``.
    - ``missing_rate`` NaN for native rows → ``0.0``.
    """
    if df.empty:
        return df
    df = df.copy()
    if "missing_mechanism" in df.columns:
        df["missing_mechanism"] = df["missing_mechanism"].where(
            df["missing_mechanism"].notna(), NATIVE_MECHANISM,
        )
    if "missing_rate" in df.columns and "missing_mechanism" in df.columns:
        native_mask = df["missing_mechanism"] == NATIVE_MECHANISM
        df.loc[native_mask & df["missing_rate"].isna(), "missing_rate"] = NATIVE_RATE
    return df


def _load_existing_results(
    out_path: Path, logger: logging.Logger,
) -> Tuple[List[Dict], Set[ExperimentKey]]:
    """Load and normalise the existing results CSV for resume.

    Returns ``(rows, completed_keys)``. ``rows`` is the list of previously
    saved row dicts (already normalised); ``completed_keys`` is the set of
    7-tuples we already have a record for.
    """
    if not out_path.exists():
        logger.info(f"Resume: no existing CSV at {out_path} (starting fresh)")
        return [], set()
    try:
        df = pd.read_csv(out_path)
    except Exception as exc:
        logger.warning(f"Resume: could not read {out_path.name}: {exc}; starting fresh")
        return [], set()

    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            f"Resume: {out_path.name} is missing key columns {missing}; starting fresh"
        )
        return [], set()

    df = _normalize_existing(df)
    rows = df.to_dict("records")
    keys = {_row_key(r) for r in rows}
    logger.info(
        f"Resume: loaded {len(rows)} rows from {out_path.name} "
        f"({len(keys)} unique completed keys)"
    )
    return rows, keys


def _atomic_write_csv(rows: List[Dict], out_path: Path) -> None:
    """Write ``rows`` to ``out_path`` atomically via a ``.tmp`` sibling.

    A power loss between the two ``Path.replace`` boundaries leaves either
    the previous CSV intact or the new one — never a truncated half-write.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


# Backwards-compat shim: a few callers/tests still use this name.
def _write_csv(rows: List[Dict], out_path: Path) -> None:
    _atomic_write_csv(rows, out_path)


def run_experiments(
    datasets: Optional[Iterable[str]] = None,
    mechanisms: Optional[Iterable[str]] = None,
    rates: Optional[Iterable[float]] = None,
    imputations: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
    random_states: Optional[Iterable[int]] = None,
    include_native: bool = True,
    resume: bool = True,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Run the full experiment matrix (or any filtered subset).

    Parameters
    ----------
    resume : bool, default True
        If ``True`` and an ``experiment_results.csv`` already exists, the
        loop reads the previously completed (dataset, split_seed, seed,
        mechanism, rate, imputation, model) combinations and skips them.
    force : bool, default False
        Override ``resume``: ignore the existing CSV entirely and recompute
        every requested combination. Equivalent to deleting the file.

    Progress is persisted after **every** model finishes via an atomic
    ``.tmp`` → ``replace`` write, so a crash at any point leaves a usable,
    non-truncated CSV.
    """
    ensure_output_dirs()
    if logger is None:
        logger = setup_logging("experiments")

    datasets = list(datasets) if datasets is not None else list(DATASETS.keys())
    mechanisms = list(mechanisms) if mechanisms is not None else list(MISSING_MECHANISMS)
    rates = list(rates) if rates is not None else list(MISSING_RATES)
    imputations = list(imputations) if imputations is not None else list(IMPUTATION_METHODS)
    models_ = list(models) if models is not None else list(ALL_MODELS)
    experiment_seeds = list(seeds) if seeds is not None else list(EXPERIMENT_SEEDS)
    split_seeds = list(random_states) if random_states is not None else list(RANDOM_STATES)

    out_path = OUTPUT_FILES["experiment_results"]
    out_json = OUTPUT_FILES["experiment_results_json"]

    if force:
        rows: List[Dict] = []
        completed_keys: Set[ExperimentKey] = set()
        resume_active = False
        logger.info("=" * 80)
        logger.info(f"Resume       : DISABLED (force=True) — ignoring {out_path.name}")
    elif resume:
        rows, completed_keys = _load_existing_results(out_path, logger)
        resume_active = True
        logger.info("=" * 80)
        logger.info(f"Resume       : ENABLED (csv={out_path})")
    else:
        rows = []
        completed_keys = set()
        resume_active = False
        logger.info("=" * 80)
        logger.info(f"Resume       : DISABLED (no-resume) — overwriting {out_path.name}")

    logger.info(f"Datasets     : {datasets}")
    logger.info(f"Split seeds  : {split_seeds}")
    logger.info(f"Seeds        : {experiment_seeds}")
    logger.info(f"Mechanisms   : {mechanisms} (native={'yes' if include_native else 'no'})")
    logger.info(f"Rates        : {rates}")
    logger.info(f"Imputations  : {imputations}")
    logger.info(f"Models       : {models_}")
    logger.info(f"Already done : {len(completed_keys)} keys")
    logger.info("=" * 80)

    # If resume is off, wipe the CSV up-front so the first failure does not
    # leave a stale half-table around. (When resume is on we keep the file
    # because the in-memory ``rows`` already contains its contents and
    # ``_atomic_write_csv`` will refresh it on first new record.)
    if not resume_active and out_path.exists():
        try:
            out_path.unlink()
        except OSError as exc:
            logger.warning(f"Could not remove stale {out_path.name}: {exc}")

    scenarios_per_dataset = (
        (1 if include_native else 0) + len(mechanisms) * len(rates)
    )
    total_scenarios = (
        len(datasets) * scenarios_per_dataset
        * len(split_seeds) * len(experiment_seeds)
    )
    scenario_idx = 0
    n_skipped_resume = 0
    n_new_records = 0

    def _persist(row: Dict, key: ExperimentKey, label: str) -> None:
        """Append ``row``, register its key, and atomically flush to disk."""
        nonlocal n_new_records
        rows.append(row)
        completed_keys.add(key)
        n_new_records += 1
        try:
            _atomic_write_csv(rows, out_path)
            logger.debug(f"SAVED {label} (total rows in CSV: {len(rows)})")
        except Exception as exc:
            logger.error(f"Failed to persist {out_path.name}: {exc}", exc_info=True)

    for dataset in datasets:
        if dataset not in DATASETS:
            logger.warning(f"Dataset {dataset!r} not in registry; skipping")
            continue

        scenarios: List[Tuple[str, float]] = []
        if include_native:
            scenarios.append((NATIVE_MECHANISM, NATIVE_RATE))
        for mech in mechanisms:
            for rate in rates:
                scenarios.append((mech, float(rate)))

        for split_seed in split_seeds:
            loaded = None  # lazily loaded only if there is real work to do
            split_loaded = False
            X_train_full = X_test = y_train = y_test = None

            for seed in experiment_seeds:
                logger.info(f"  --- split_seed={split_seed} seed={seed} ---")
                for mech, rate in scenarios:
                    scenario_idx += 1
                    csv_mech, csv_rate = _scenario_columns(mech, rate)
                    label = (
                        "native"
                        if mech == NATIVE_MECHANISM
                        else f"{mech}_{int(rate * 100)}pct"
                    )
                    logger.info(
                        f"[{scenario_idx}/{total_scenarios}] "
                        f"{dataset} split_seed={split_seed} seed={seed} scenario={label}"
                    )

                    # Fast-path: if every (imputation × model) for this
                    # scenario is already in the CSV, skip injection entirely.
                    pending: List[Tuple[str, str]] = []
                    for imp in imputations:
                        for m in models_:
                            key: ExperimentKey = (
                                dataset, int(split_seed), int(seed),
                                csv_mech, float(csv_rate),
                                imp, display_name(m),
                            )
                            if key in completed_keys:
                                n_skipped_resume += 1
                                logger.debug(
                                    f"SKIP completed: dataset={dataset}, "
                                    f"split_seed={split_seed}, seed={seed}, "
                                    f"mechanism={csv_mech}, rate={csv_rate}, "
                                    f"imputation={imp}, model={display_name(m)}"
                                )
                                continue
                            pending.append((imp, m))
                    if not pending:
                        logger.info(
                            f"  scenario fully cached for split_seed={split_seed} "
                            f"seed={seed} {label}; skipping"
                        )
                        continue

                    # We have at least one new combination — make sure the
                    # split is loaded.
                    if not split_loaded:
                        loaded = load_precomputed_split(
                            dataset, random_state=split_seed, logger=logger,
                        )
                        if loaded is None:
                            logger.error(
                                f"No split for {dataset} with "
                                f"random_state={split_seed}; skipping seed"
                            )
                            break  # break out of `for seed` for this split_seed
                        X_train_full, X_test, y_train, y_test = loaded
                        split_loaded = True
                        logger.info(
                            f"\n[{dataset}] split_seed={split_seed} "
                            f"train={X_train_full.shape} test={X_test.shape}"
                        )

                    try:
                        X_train_miss = _apply_missingness(
                            X_train_full, mech, rate, seed,
                        )
                    except Exception as exc:
                        logger.error(f"  injection failed: {exc}", exc_info=True)
                        for imp, m in pending:
                            row = _record(
                                dataset, split_seed, seed,
                                csv_mech, csv_rate,
                                imp, m, {"error": f"injection failed: {exc}"},
                            )
                            _persist(row, _row_key(row), f"{label} {imp} {m}")
                        continue

                    # Group pending by imputation so we only impute once per
                    # method even if several models remain to be trained.
                    by_imp: Dict[str, List[str]] = {}
                    for imp, m in pending:
                        by_imp.setdefault(imp, []).append(m)

                    for imp, model_list in by_imp.items():
                        try:
                            X_tr, X_te = impute(
                                X_train_miss, X_test, imp, random_state=seed,
                            )
                            impute_error = None
                        except Exception as exc:
                            X_tr = X_te = None
                            impute_error = str(exc)
                            logger.warning(f"  imputation {imp} failed: {exc}")

                        for m in model_list:
                            run_label = (
                                f"dataset={dataset} split_seed={split_seed} "
                                f"seed={seed} {label} imp={imp} model={display_name(m)}"
                            )
                            if impute_error is not None:
                                row = _record(
                                    dataset, split_seed, seed,
                                    csv_mech, csv_rate, imp, m,
                                    {"error": f"imputation failed: {impute_error}"},
                                )
                                _persist(row, _row_key(row), run_label)
                                continue
                            skip = _should_skip(m, imp)
                            if skip is not None:
                                row = _record(
                                    dataset, split_seed, seed,
                                    csv_mech, csv_rate, imp, m,
                                    {}, skipped_reason=f"skipped: {skip}",
                                )
                                _persist(row, _row_key(row), run_label)
                                continue
                            logger.info(f"RUN {run_label}")
                            outcome = train_and_evaluate(
                                m, X_tr, X_te, y_train, y_test,
                                random_state=seed, logger=logger,
                            )
                            row = _record(
                                dataset, split_seed, seed,
                                csv_mech, csv_rate, imp, m, outcome,
                            )
                            _persist(row, _row_key(row), run_label)
                            if outcome.get("error"):
                                logger.info(
                                    f"    split_seed={split_seed} seed={seed} "
                                    f"{m:20s} [{imp:15s}] -> error: {outcome['error'][:80]}"
                                )
                            elif outcome.get("metrics"):
                                mt = outcome["metrics"]
                                acc = mt.get("accuracy", float("nan"))
                                bacc = mt.get("balanced_accuracy", float("nan"))
                                f1m = mt.get("f1_macro", float("nan"))
                                prauc = mt.get("pr_auc", float("nan"))
                                thr = mt.get("threshold", 0.5)
                                t = outcome.get("training_time_seconds") or 0.0
                                logger.info(
                                    f"    split_seed={split_seed} seed={seed} {m:20s} "
                                    f"[{imp:15s}] acc={acc:.4f} bacc={bacc:.4f} "
                                    f"f1m={f1m:.4f} prauc={prauc:.4f} thr={thr:.2f} t={t:.1f}s"
                                )

        logger.info(
            f"Dataset {dataset} done so far: {len(rows)} rows on disk "
            f"({n_new_records} new, {n_skipped_resume} skipped via resume)"
        )

    _atomic_write_csv(rows, out_path)

    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"generated_at": datetime.now().isoformat(), "rows": rows}, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not write {out_json.name}: {exc}")

    logger.info(
        f"Done. Total rows: {len(rows)} "
        f"(new this run: {n_new_records}, skipped via resume: {n_skipped_resume})"
    )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


if __name__ == "__main__":
    run_experiments()
