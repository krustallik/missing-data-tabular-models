"""Main entry point for the whole benchmark.

Runs the full pipeline end-to-end in the order the project plan requires:

1. **Prepare splits** - build stratified train/test CSVs from the processed
   datasets if they are not already present.
2. **Verify missingness injection** - check that every (mechanism, rate)
   hits the requested fraction within tolerance.
3. **Benchmark imputations** - measure imputation time and success per
   scenario (so the main loop can skip failing combinations fast).
4. **Run experiments** - the main matrix
   ``datasets x mechanisms x rates x imputations x models``.
5. **Consolidate** - single consolidated CSV, classical/foundation splits,
   robustness analysis, and the four standard visualisations.
6. **Generate reports** - Data/Methods, Results/Discussion, Practical
   usability, deployment guide, interpretation guide and presentation points.

All steps write results under ``results/``. A partial crash keeps partial
CSVs on disk so the pipeline is safe to re-run without starting over.

Usage::

    python src/run_experiments.py             # full pipeline
    python src/run_experiments.py --step 4    # run only the experiment loop
    python src/run_experiments.py --from 4    # run step 4 and everything after

Optional filters for step 4 (experiment loop)::

    --models logistic_regression random_forest tabpfn
    --imputations median mice_indicator none
    --mechanisms MCAR MAR
    --rates 0.05 0.10 0.20
    --datasets taiwan_bankruptcy
    --no-native
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import (
    ALL_MODELS,
    DATASETS,
    IMPUTATION_METHODS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    OUTPUT_FILES,
    PROCESSED_DIR,
    RANDOM_STATE,
    SPLITS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    ensure_output_dirs,
)
from consolidation import consolidate
from data_utils import (
    load_dataset_from_csv,
    make_train_test_split,
    save_split,
    setup_logging,
    splits_present,
    summarize_dataset,
)
from experiment_runner import run_experiments as _run_experiments
from imputation import impute
from missingness import inject_mar, inject_mcar, inject_mnar
from missingness_check import run_missingness_verification
from reporting import generate_reports


# ── Step 1: Splits ───────────────────────────────────────────────────────────

def step_prepare_splits(logger: logging.Logger) -> bool:
    """Build stratified splits from ``data/processed/*.csv``."""
    logger.info("=" * 80)
    logger.info("STEP 1: PREPARE TRAIN/TEST SPLITS")
    logger.info("=" * 80)

    if splits_present(DATASETS.keys()):
        logger.info("All splits already present; skipping split creation.")
    else:
        for dataset_name, path in DATASETS.items():
            if not path.exists():
                logger.error(f"Processed dataset missing: {path}")
                continue
            X, y = load_dataset_from_csv(path, target_column=TARGET_COLUMN)
            X_train, X_test, y_train, y_test = make_train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=True,
            )
            train_df = pd.concat([X_train.reset_index(drop=True),
                                  y_train.reset_index(drop=True)], axis=1)
            test_df = pd.concat([X_test.reset_index(drop=True),
                                 y_test.reset_index(drop=True)], axis=1)
            tp, ep = save_split(train_df, test_df, dataset_name)
            logger.info(f"Saved {tp.name} and {ep.name}")

    # Always refresh the dataset overview so it reflects the current data.
    overview_rows = []
    for dataset_name, path in DATASETS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        s = summarize_dataset(df, target_column=TARGET_COLUMN)
        overview_rows.append({
            "dataset": dataset_name,
            "path": str(path),
            **s,
        })
    if overview_rows:
        pd.DataFrame(overview_rows).to_csv(OUTPUT_FILES["dataset_overview"], index=False)
        logger.info(f"Saved {OUTPUT_FILES['dataset_overview'].name}")

    setup_rows = [{
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "mechanisms": ",".join(MISSING_MECHANISMS),
        "rates": ",".join(str(r) for r in MISSING_RATES),
        "imputations": ",".join(IMPUTATION_METHODS),
        "models": ",".join(ALL_MODELS),
    }]
    pd.DataFrame(setup_rows).to_csv(OUTPUT_FILES["experiment_setup"], index=False)
    logger.info(f"Saved {OUTPUT_FILES['experiment_setup'].name}")
    return True


# ── Step 2: Missingness verification ────────────────────────────────────────

def step_verify_missingness(logger: logging.Logger) -> bool:
    logger.info("=" * 80)
    logger.info("STEP 2: VERIFY MISSINGNESS INJECTION RATES")
    logger.info("=" * 80)
    try:
        run_missingness_verification(logger=logger)
        return True
    except Exception as exc:
        logger.error(f"Verification step failed: {exc}", exc_info=True)
        return False


# ── Step 3: Imputation benchmark ─────────────────────────────────────────────

def step_benchmark_imputations(logger: logging.Logger) -> bool:
    """Measure imputation time and success per scenario for every dataset."""
    logger.info("=" * 80)
    logger.info("STEP 3: BENCHMARK IMPUTATION METHODS")
    logger.info("=" * 80)
    from data_utils import load_precomputed_split

    rows = []
    injectors = {"MCAR": inject_mcar, "MAR": inject_mar, "MNAR": inject_mnar}
    for dataset_name in DATASETS:
        loaded = load_precomputed_split(dataset_name, logger=logger)
        if loaded is None:
            continue
        X_train, X_test, _, _ = loaded
        for mech in MISSING_MECHANISMS:
            for rate in MISSING_RATES:
                X_train_missing = injectors[mech](X_train.copy(), rate, random_state=RANDOM_STATE)
                for method in IMPUTATION_METHODS:
                    row = {
                        "dataset": dataset_name,
                        "mechanism": mech,
                        "missing_rate": round(rate * 100, 2),
                        "imputation": method,
                        "duration_seconds": None,
                        "success": False,
                        "error": None,
                    }
                    start = time.time()
                    try:
                        impute(X_train_missing, X_test, method)
                        row["success"] = True
                    except Exception as exc:
                        row["error"] = str(exc)
                    row["duration_seconds"] = round(time.time() - start, 4)
                    rows.append(row)
                    status = "ok " if row["success"] else "err"
                    logger.info(
                        f"  {dataset_name:24s} {mech} @ {int(rate*100):>2}% "
                        f"{method:15s} {status} t={row['duration_seconds']}s"
                    )

    df = pd.DataFrame(rows)
    out = OUTPUT_FILES["imputation_benchmark"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved {out.name}: {len(df)} rows")
    return True


# ── Step 4: Main experiment loop ─────────────────────────────────────────────

def step_run_experiments(
    logger: logging.Logger,
    datasets: Optional[List[str]] = None,
    mechanisms: Optional[List[str]] = None,
    rates: Optional[List[float]] = None,
    imputations: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    include_native: bool = True,
) -> bool:
    logger.info("=" * 80)
    logger.info("STEP 4: RUN FULL EXPERIMENT MATRIX")
    logger.info("=" * 80)
    _run_experiments(
        datasets=datasets, mechanisms=mechanisms, rates=rates,
        imputations=imputations, models=models,
        include_native=include_native, logger=logger,
    )
    return True


# ── Step 5: Consolidation + visualisations ───────────────────────────────────

def step_consolidate(logger: logging.Logger) -> bool:
    logger.info("=" * 80)
    logger.info("STEP 5: CONSOLIDATE + VISUALIZE")
    logger.info("=" * 80)
    consolidate(logger=logger)
    return True


# ── Step 6: Reports ──────────────────────────────────────────────────────────

def step_reports(logger: logging.Logger) -> bool:
    logger.info("=" * 80)
    logger.info("STEP 6: GENERATE REPORTS")
    logger.info("=" * 80)
    generate_reports(logger=logger)
    return True


# ── Entry point ──────────────────────────────────────────────────────────────

STEPS = {
    1: ("prepare_splits", step_prepare_splits),
    2: ("verify_missingness", step_verify_missingness),
    3: ("benchmark_imputations", step_benchmark_imputations),
    4: ("run_experiments", step_run_experiments),
    5: ("consolidate", step_consolidate),
    6: ("generate_reports", step_reports),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full tabular-missingness benchmark.")
    p.add_argument("--step", type=int, default=None,
                   help="Run only this single step (1..6).")
    p.add_argument("--from", dest="from_step", type=int, default=None,
                   help="Run this step and every subsequent step.")
    p.add_argument("--to", dest="to_step", type=int, default=None,
                   help="Stop at this step (inclusive) when using --from.")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Subset of datasets for step 4.")
    p.add_argument("--mechanisms", nargs="+", default=None,
                   help="Subset of missingness mechanisms for step 4.")
    p.add_argument("--rates", nargs="+", type=float, default=None,
                   help="Subset of missing rates (as floats) for step 4.")
    p.add_argument("--imputations", nargs="+", default=None,
                   help="Subset of imputation methods for step 4.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of models for step 4.")
    p.add_argument("--no-native", action="store_true",
                   help="Skip the 'native' (no-injection) scenario in step 4.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ensure_output_dirs()
    logger = setup_logging("pipeline")

    if args.step is not None:
        step_ids = [args.step]
    elif args.from_step is not None:
        end = args.to_step if args.to_step is not None else max(STEPS.keys())
        step_ids = [i for i in STEPS if args.from_step <= i <= end]
    else:
        step_ids = sorted(STEPS.keys())

    logger.info(f"Pipeline steps to run: {step_ids}")
    started = time.time()

    for step_id in step_ids:
        name, fn = STEPS[step_id]
        print(f"\n>>> Step {step_id}: {name}")
        logger.info(f">>> Step {step_id}: {name}")
        try:
            if step_id == 4:
                ok = step_run_experiments(
                    logger=logger,
                    datasets=args.datasets,
                    mechanisms=args.mechanisms,
                    rates=args.rates,
                    imputations=args.imputations,
                    models=args.models,
                    include_native=not args.no_native,
                )
            else:
                ok = fn(logger)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            return 130
        except Exception as exc:
            logger.error(f"Step {step_id} ({name}) failed: {exc}", exc_info=True)
            print(f"Step {step_id} ({name}) failed: {exc}")
            return 1
        if not ok:
            logger.error(f"Step {step_id} ({name}) returned False")
            return 1

    logger.info(f"Pipeline finished in {time.time() - started:.1f}s")
    print("\n" + "=" * 80)
    print("Pipeline complete.")
    print("=" * 80)
    print("Outputs:")
    print(f"  splits      : {SPLITS_DIR}")
    print(f"  tables      : {TABLES_DIR}")
    print(f"  reports     : {OUTPUT_FILES['data_methods_report'].parent}")
    print(f"  plots       : {OUTPUT_FILES['viz_stability_heatmap'].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
