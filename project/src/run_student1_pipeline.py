"""Student 1 end-to-end pipeline (tasks 3.3, 3.4, 3.5).

Mirrors the structure of run_phase4_pipeline.py used by Student 2: this file
is only an orchestrator. Each task lives in its own ``run_student1_*`` module
and is executed in a separate subprocess so that failures are isolated and
logs are kept per-phase.

Preprocessing (task 3.3) is invoked first, then the downstream phases that
use the imputed data (tasks 3.4 and 3.5).

Missingness scenarios used by every Student 1 phase come from
``student1_common.py`` and include MCAR, MAR and MNAR on the same rate grid
as Student 2 (5%, 10%, 15%, 20%, 30%, 40%).
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_student1_3_2():
    """Run Student 1 task 3.2 verification (MCAR/MAR/MNAR rate check)."""
    print("\n" + "=" * 80)
    print("STUDENT 1 - TASK 3.2: MISSINGNESS INJECTION VERIFICATION")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_student1_3_2_verification.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Student 1 task 3.2 verification failed")
        return False

    print("[OK] Student 1 task 3.2 verification completed successfully")
    return True


def run_student1_3_3():
    """Run Student 1 task 3.3 (simple imputations: mean, median, kNN)."""
    print("\n" + "=" * 80)
    print("STUDENT 1 - TASK 3.3: SIMPLE IMPUTATIONS (MEAN / MEDIAN / KNN)")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_student1_3_3_experiments.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Student 1 task 3.3 failed")
        return False

    print("[OK] Student 1 task 3.3 completed successfully")
    return True


def run_student1_3_4():
    """Run Student 1 task 3.4 (baseline models: Logistic Regression + Random Forest)."""
    print("\n" + "=" * 80)
    print("STUDENT 1 - TASK 3.4: BASELINE MODELS (LOGISTIC REGRESSION / RANDOM FOREST)")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_student1_3_4_experiments.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Student 1 task 3.4 failed")
        return False

    print("[OK] Student 1 task 3.4 completed successfully")
    return True


def run_student1_3_5():
    """Run Student 1 task 3.5 (foundation model TabPFN)."""
    print("\n" + "=" * 80)
    print("STUDENT 1 - TASK 3.5: FOUNDATION MODEL (TABPFN)")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_student1_3_5_experiments.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Student 1 task 3.5 failed")
        return False

    print("[OK] Student 1 task 3.5 completed successfully")
    return True


def main():
    """Run the complete Student 1 pipeline."""
    print("\n" + "=" * 80)
    print("STUDENT 1 PIPELINE")
    print("=" * 80)
    print("This script will execute:")
    print("  1. Task 3.2: Missingness injection verification (MCAR / MAR / MNAR)")
    print("  2. Task 3.3: Simple imputations (mean / median / knn)")
    print("  3. Task 3.4: Baseline models (Logistic Regression, Random Forest)")
    print("  4. Task 3.5: Foundation model (TabPFN)")
    print("Scenarios per phase: native + MCAR / MAR / MNAR at 5-40% rates.")
    print("Student 2 phases (4.x) are NOT executed by this runner.")

    if not run_student1_3_2():
        sys.exit(1)

    if not run_student1_3_3():
        sys.exit(1)

    if not run_student1_3_4():
        sys.exit(1)

    if not run_student1_3_5():
        sys.exit(1)

    print("\n" + "=" * 80)
    print("[OK] STUDENT 1 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nResults saved to:")
    print(f"  - 3.2 JSON: {PROJECT_ROOT}/results/tables/student1_3_2_missingness_verification.json")
    print(f"  - 3.2 CSV : {PROJECT_ROOT}/results/tables/student1_3_2_missingness_verification.csv")
    print(f"  - 3.3 JSON: {PROJECT_ROOT}/results/tables/student1_3_3_imputation_results.json")
    print(f"  - 3.3 CSV : {PROJECT_ROOT}/results/tables/student1_3_3_imputation_results.csv")
    print(f"  - 3.4 JSON: {PROJECT_ROOT}/results/tables/student1_3_4_baseline_results.json")
    print(f"  - 3.4 CSV : {PROJECT_ROOT}/results/tables/student1_3_4_baseline_results.csv")
    print(f"  - 3.5 JSON: {PROJECT_ROOT}/results/tables/student1_3_5_tabpfn_results.json")
    print(f"  - 3.5 CSV : {PROJECT_ROOT}/results/tables/student1_3_5_tabpfn_results.csv")
    print(f"  - Logs    : {PROJECT_ROOT}/results/logs/")


if __name__ == "__main__":
    main()
