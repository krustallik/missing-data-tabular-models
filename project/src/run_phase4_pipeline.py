"""End-to-end pipeline for dataset preparation, Phases 4.1-4.6."""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_raw_standardization():
    """Convert raw files into standardized processed CSV datasets."""
    print("\n" + "=" * 80)
    print("RAW DATA STANDARDIZATION")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "prepare_raw_datasets.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Raw data standardization failed")
        return False

    print("✓ Raw data standardization completed successfully")
    return True


def run_phase_3_setup():
    """Run Phase 3.1 dataset preparation."""
    print("\n" + "=" * 80)
    print("PHASE 3.1: Dataset Preparation")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_dataset_setup.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Phase 3.1 failed")
        return False

    print("✓ Phase 3.1 completed successfully")
    return True


def run_phase_4_experiments():
    """Run Phase 4.1 model evaluation."""
    print("\n" + "=" * 80)
    print("PHASE 4.1: Model Training and Evaluation")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "phase4_experiment_runner.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Phase 4.1 failed")
        return False

    print("✓ Phase 4.1 completed successfully")
    return True


def run_phase_4_2_experiments():
    """Run Phase 4.2 extended model evaluation."""
    print("\n" + "=" * 80)
    print("PHASE 4.2: Extended Model Training and Evaluation")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_phase4_2_experiments.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Phase 4.2 failed")
        return False

    print("✓ Phase 4.2 completed successfully")
    return True


def run_phase_4_3_experiments():
    """Run Phase 4.3 Gradient Boosting robustness testing (Student 2)."""
    print("\n" + "=" * 80)
    print("PHASE 4.3: Gradient Boosting Robustness with Missingness Injection")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_phase4_3_experiments.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Phase 4.3 failed")
        return False

    print("✓ Phase 4.3 completed successfully")
    return True


def run_phase_4_4_experiments():
    """Run Phase 4.4 CatBoost foundation model testing."""
    print("\n" + "=" * 80)
    print("PHASE 4.4: CatBoost Foundation Model Testing")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_phase4_4_experiments.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Phase 4.4 failed")
        return False

    print("✓ Phase 4.4 completed successfully")
    return True


def run_phase_4_5_analysis():
    """Run Phase 4.5 final analysis and visualization."""
    print("\n" + "=" * 80)
    print("PHASE 4.5: Final Analysis and Visualization")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_phase4_5_experiments.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Phase 4.5 failed")
        return False

    print("✓ Phase 4.5 completed successfully")
    return True


def run_phase_4_6_documentation():
    """Run Phase 4.6 Documentation and Report Generation."""
    print("\n" + "=" * 80)
    print("PHASE 4.6: Documentation and Report Generation")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_phase4_6_experiments.py")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("ERROR: Phase 4.6 failed")
        return False

    print("✓ Phase 4.6 completed successfully")
    return True


def main():
    """Run complete end-to-end pipeline."""
    print("\n" + "=" * 80)
    print("END-TO-END PIPELINE")
    print("=" * 80)
    print("This script will execute:")
    print("  1. Raw data standardization")
    print("  2. Phase 3.1: Dataset setup and splits")
    print("  3. Phase 4.1: Model training and evaluation (baseline + MICE)")
    print("  4. Phase 4.2: Extended model training (SVM, MLP, XGBoost, LightGBM)")
    print("  5. Phase 4.3: Gradient Boosting robustness with missingness injection (MCAR/MAR/MNAR)")
    print("  6. Phase 4.4: CatBoost foundation model testing")
    print("  7. Phase 4.5: Final analysis and visualization")
    print("  8. Phase 4.6: Documentation and Report Generation")

    if not run_raw_standardization():
        sys.exit(1)

    if not run_phase_3_setup():
        sys.exit(1)

    if not run_phase_4_experiments():
        sys.exit(1)

    if not run_phase_4_2_experiments():
        sys.exit(1)

    if not run_phase_4_3_experiments():
        sys.exit(1)

    if not run_phase_4_4_experiments():
        sys.exit(1)

    if not run_phase_4_5_analysis():
        sys.exit(1)

    if not run_phase_4_6_documentation():
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✓ ALL PHASES COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nResults saved to:")
    print(f"  - Dataset overview: {PROJECT_ROOT}/results/tables/dataset_overview.csv")
    print(f"  - Split indices: {PROJECT_ROOT}/results/tables/split_indices.json")
    print(f"  - Phase 4.1 results: {PROJECT_ROOT}/results/tables/phase4_experiment_results.json")
    print(f"  - Phase 4.2 results: {PROJECT_ROOT}/results/tables/phase4_2_experiment_results.json")
    print(f"  - Phase 4.3 results: {PROJECT_ROOT}/results/tables/phase4_3_gradient_boosting_results.json")
    print(f"  - Phase 4.4 results: {PROJECT_ROOT}/results/tables/phase4_4_catboost_results.json")
    print(f"  - Phase 4.5 consolidated: {PROJECT_ROOT}/results/tables/phase4_5_consolidated_results.csv")
    print(f"  - Phase 4.5 robustness: {PROJECT_ROOT}/results/tables/phase4_5_robustness_analysis.csv")
    print(f"  - Phase 4.6 report: {PROJECT_ROOT}/results/reports/phase4_6_student2_report.md")
    print(f"  - Phase 4.6 interpretation: {PROJECT_ROOT}/results/reports/phase4_6_interpretation_guide.md")
    print(f"  - Phase 4.6 presentation: {PROJECT_ROOT}/results/reports/phase4_6_presentation_points.txt")
    print(f"  - Logs: {PROJECT_ROOT}/results/logs/")


if __name__ == "__main__":
    main()
