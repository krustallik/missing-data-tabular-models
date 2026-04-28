# Interpretation guide

How to read the tables and plots produced by this pipeline.

## Files in `results/tables/`

- `dataset_overview.csv` - per-dataset size and class balance.
- `experiment_setup.csv` - random seed, test fraction, metrics used.
- `missingness_verification.csv` - realised vs target missing rates.
- `imputation_benchmark.csv` - imputation time and errors per scenario.
- `experiment_results.csv` - full benchmark table (one row per run).
- `consolidated_results.csv` - same rows with the canonical column schema used by Results / Discussion.
- `classical_models.csv` / `foundation_models.csv` - convenience splits.
- `robustness_analysis.csv` - per-(model, mechanism) aggregates.

## Files in `results/visualizations/`

- `missing_rate_MCAR.png` / `missing_rate_MAR.png` / `missing_rate_MNAR.png` - primary metrics (accuracy, balanced_accuracy, f1_macro, pr_auc) as functions of missing rate for each mechanism.
- `classical_vs_foundation.png` - distribution boxplot across the primary metrics.
- `stability_heatmap.png` - balanced_accuracy heatmap across missing rates (the metric that actually reflects minority-class detection).
- `per_dataset_ranking.png` - bar chart ranking on the native scenario by balanced_accuracy.

## Files in `results/reports/`

- `data_methods_report.md` - Data / Setup / Methods.
- `results_discussion_report.md` - quantitative results + discussion.
- `practical_usability_report.md` - practical recommendations.
- `deployment_guide.md` - per-model install / GPU / size notes.
- `deployment_complexity.csv` / `suitability_matrix.csv` - tables behind the practical report.

## Quick numbers at hand

- Consolidated rows: 1140
- Datasets: slovak_manufacture_13
- Models: CatBoost, Gradient-Boosting, LightGBM, Logistic-Regression, MLP, Random-Forest, SVM, TabICL, TabPFN, XGBoost
