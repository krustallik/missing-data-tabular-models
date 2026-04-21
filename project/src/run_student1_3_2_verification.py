"""Student 1 - Task 3.2: verify that MCAR / MAR / MNAR injection produces the
requested final missingness fraction.

For every dataset in ``student1_common.DATASETS`` and every (mechanism, rate)
pair we:

- inject missingness into the precomputed training split,
- measure the actual final missing fraction over numeric feature columns,
- compare with the requested target rate,
- record both absolute and relative deviation.

Outputs (artifacts):
- results/tables/student1_3_2_missingness_verification.json
- results/tables/student1_3_2_missingness_verification.csv
- results/logs/experiment_student1_3_2_verification_<ts>.log
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import RANDOM_STATE
from missingness import calculate_missing_fraction
from student1_common import (
    DATASETS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    RESULTS_DIR,
    ensure_splits,
    inject_missingness,
    load_dataset_and_split,
    setup_logging,
)


TOLERANCE_ABS = 0.01  # 1 percentage point absolute deviation considered OK


def run_task_3_2(dataset_path: Path, dataset_name: str, logger):
    logger.info(f"\n{'=' * 80}\nStudent 1 - Task 3.2 verification: {dataset_name}\n{'=' * 80}")

    _, _, X_train, _, _, _ = load_dataset_and_split(dataset_name, dataset_path, logger)

    initial_rate = float(calculate_missing_fraction(X_train))
    logger.info(f"Initial numeric missing fraction (train): {initial_rate:.6f}")

    rows = []
    for mechanism in MISSING_MECHANISMS:
        for rate in MISSING_RATES:
            try:
                X_missing = inject_missingness(X_train.copy(), mechanism, rate, RANDOM_STATE)
                final_rate = float(calculate_missing_fraction(X_missing))
                abs_dev = float(final_rate - rate)
                rel_dev = float(abs_dev / rate) if rate > 0 else 0.0
                if initial_rate >= rate:
                    # Injection is intentionally a no-op when native missingness
                    # already exceeds the requested target (see missingness.py).
                    status = "skipped_native_above_target"
                    passed = True
                elif abs(abs_dev) <= TOLERANCE_ABS:
                    status = "passed"
                    passed = True
                else:
                    status = "failed"
                    passed = False
                logger.info(
                    f"  {mechanism} target={rate:.2f} -> final={final_rate:.4f} "
                    f"(abs_dev={abs_dev:+.4f}, rel_dev={rel_dev:+.2%}, status={status})"
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "missing_mechanism": mechanism,
                        "missing_rate": int(round(rate * 100)),
                        "target_rate": rate,
                        "initial_missing_rate": initial_rate,
                        "final_missing_rate": final_rate,
                        "absolute_deviation": abs_dev,
                        "relative_deviation": rel_dev,
                        "tolerance_abs": TOLERANCE_ABS,
                        "status": status,
                        "passed": passed,
                        "n_train": int(X_train.shape[0]),
                        "n_features": int(X_train.shape[1]),
                        "error": None,
                    }
                )
            except Exception as exc:
                logger.error(f"  {mechanism} rate={rate}: injection failed: {exc}", exc_info=True)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "missing_mechanism": mechanism,
                        "missing_rate": int(round(rate * 100)),
                        "target_rate": rate,
                        "initial_missing_rate": initial_rate,
                        "final_missing_rate": None,
                        "absolute_deviation": None,
                        "relative_deviation": None,
                        "tolerance_abs": TOLERANCE_ABS,
                        "status": "error",
                        "passed": False,
                        "n_train": int(X_train.shape[0]),
                        "n_features": int(X_train.shape[1]),
                        "error": str(exc),
                    }
                )

    return rows


def main():
    logger = setup_logging("student1_3_2_verification")
    logger.info("=" * 80)
    logger.info("STUDENT 1 - TASK 3.2: MISSINGNESS INJECTION VERIFICATION")
    logger.info("=" * 80)

    ensure_splits(logger)

    all_rows = []
    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        try:
            all_rows.extend(run_task_3_2(dataset_path, dataset_name, logger))
        except Exception as exc:
            logger.error(f"Task 3.2 failed for {dataset_name}: {exc}", exc_info=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "student1_3_2_missingness_verification.json"
    out_csv = RESULTS_DIR / "student1_3_2_missingness_verification.csv"

    out_json.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    total = len(all_rows)
    by_status = df["status"].value_counts().to_dict() if total else {}

    logger.info("=" * 80)
    logger.info(f"SUMMARY: tolerance={TOLERANCE_ABS:.2f} abs; "
                f"total={total}; {by_status}")
    logger.info("=" * 80)

    print("\n" + "=" * 80)
    print("Student 1 Task 3.2 verification complete")
    print("=" * 80)
    print(f"  Total scenarios: {total}")
    for status, count in by_status.items():
        print(f"  - {status}: {count}")
    print(f"  Results JSON: {out_json}")
    print(f"  Results CSV : {out_csv}")


if __name__ == "__main__":
    main()
