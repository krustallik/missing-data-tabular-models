"""Global configuration for the tabular-foundation-models benchmark.

One place for every knob used by the experiment pipeline:

- reproducibility (seed, test fraction)
- the three datasets and where their CSVs live
- the controlled missingness grid (mechanisms + rates)
- imputation strategies (simple + complex)
- model registry split into Classical / Foundation
- output paths (tables, reports, visualizations, logs)

Everything else in ``src/`` imports from here so that there is a single
source of truth and renaming / extending the benchmark is a one-file change.
"""

from __future__ import annotations

from pathlib import Path


# ── Reproducibility ──────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ── Metrics ──────────────────────────────────────────────────────────────────

METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]
PRIMARY_METRICS = ["accuracy", "f1", "roc_auc"]

# Column schema used by every results CSV in this project.
RESULT_COLUMNS = [
    "dataset",
    "missing_mechanism",
    "missing_rate",
    "imputation",
    "model",
    "model_type",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "training_time_seconds",
    "error",
]


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = RESULTS_DIR / "logs"
VIZ_DIR = RESULTS_DIR / "visualizations"


# ── Datasets ─────────────────────────────────────────────────────────────────

TARGET_COLUMN = "target"

DATASETS = {
    "taiwan_bankruptcy": PROCESSED_DIR / "taiwan_bankruptcy.csv",
    "polish_1year": PROCESSED_DIR / "polish_1year.csv",
    "slovak_manufacture_13": PROCESSED_DIR / "slovak_manufacture_13.csv",
}


# ── Missingness grid ─────────────────────────────────────────────────────────

MISSING_MECHANISMS = ["MCAR", "MAR", "MNAR"]
MISSING_RATES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

# Absolute tolerance when comparing realised vs. requested missing fraction.
MISSINGNESS_TOLERANCE = 0.01


# ── Imputation methods ───────────────────────────────────────────────────────

# "none" means: feed raw NaN into the model (foundation models may handle it).
IMPUTATION_METHODS = ["mean", "median", "knn", "mice", "mice_indicator", "none"]

# Soft cap for kNN imputation to avoid pathological runtime on wide matrices.
# Override with the ``BENCHMARK_KNN_MAX_CELLS`` env variable if needed.
KNN_MAX_TRAIN_CELLS = 250_000


# ── Models ───────────────────────────────────────────────────────────────────

# Model names here are the *canonical* keys used everywhere (CSV, reports,
# visualizations). Display-friendly versions are produced in ``models.display_name``.
CLASSICAL_MODELS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "svm",
    "mlp",
]

FOUNDATION_MODELS = [
    "tabpfn",
    "tabicl",
    "catboost",
]

ALL_MODELS = CLASSICAL_MODELS + FOUNDATION_MODELS

# Models that accept raw NaN via their own preprocessing and therefore may
# pair with the ``none`` imputation method.
MODELS_ACCEPTING_NAN = {"tabpfn", "tabicl", "catboost", "xgboost", "lightgbm"}


# ── Output file names (no phase / student prefixes) ──────────────────────────

OUTPUT_FILES = {
    # tables
    "dataset_overview": TABLES_DIR / "dataset_overview.csv",
    "experiment_setup": TABLES_DIR / "experiment_setup.csv",
    "missingness_verification": TABLES_DIR / "missingness_verification.csv",
    "imputation_benchmark": TABLES_DIR / "imputation_benchmark.csv",
    "experiment_results": TABLES_DIR / "experiment_results.csv",
    "experiment_results_json": TABLES_DIR / "experiment_results.json",
    "consolidated_results": TABLES_DIR / "consolidated_results.csv",
    "classical_models": TABLES_DIR / "classical_models.csv",
    "foundation_models": TABLES_DIR / "foundation_models.csv",
    "robustness_analysis": TABLES_DIR / "robustness_analysis.csv",
    # reports
    "data_methods_report": REPORTS_DIR / "data_methods_report.md",
    "results_discussion_report": REPORTS_DIR / "results_discussion_report.md",
    "practical_usability_report": REPORTS_DIR / "practical_usability_report.md",
    "deployment_guide": REPORTS_DIR / "deployment_guide.md",
    "deployment_complexity": REPORTS_DIR / "deployment_complexity.csv",
    "suitability_matrix": REPORTS_DIR / "suitability_matrix.csv",
    "interpretation_guide": REPORTS_DIR / "interpretation_guide.md",
    "presentation_points": REPORTS_DIR / "presentation_points.txt",
    # visualizations
    "viz_missing_rate_MCAR": VIZ_DIR / "missing_rate_MCAR.png",
    "viz_missing_rate_MAR": VIZ_DIR / "missing_rate_MAR.png",
    "viz_missing_rate_MNAR": VIZ_DIR / "missing_rate_MNAR.png",
    "viz_model_comparison": VIZ_DIR / "classical_vs_foundation.png",
    "viz_stability_heatmap": VIZ_DIR / "stability_heatmap.png",
    "viz_per_dataset_ranking": VIZ_DIR / "per_dataset_ranking.png",
}


# ── Report metadata ──────────────────────────────────────────────────────────

REPORT_TITLE = "Performance of Pretrained Tabular Foundation Models on Incomplete Data"
REPORT_AUTHOR = "Project team"
REPORT_INSTITUTION = "University"

PERFORMANCE_EXCELLENT = 0.95
PERFORMANCE_GOOD = 0.90
PERFORMANCE_ACCEPTABLE = 0.85


def ensure_output_dirs() -> None:
    """Create every output directory used by the pipeline (idempotent)."""
    for p in (TABLES_DIR, REPORTS_DIR, LOGS_DIR, VIZ_DIR, SPLITS_DIR):
        p.mkdir(parents=True, exist_ok=True)
