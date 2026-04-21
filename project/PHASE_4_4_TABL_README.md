# Phase 4.4 - CatBoost Foundation Model Testing

## Overview

**Phase 4.4** tests **CatBoost** (Categorical Boosting), a gradient boosting library designed for tabular data classification with native support for:
- Categorical features
- Missing values (NaN handling)
- Large datasets
- Fast training

## Configuration

File: `src/config_phase4_4.py`

```python
TEST_WITH_IMPUTATION = True          # Test with median & MICE imputation
TEST_WITHOUT_IMPUTATION = True       # Test on raw data with NaN values
PREPROCESSING_METHODS = ["median", "mice"]
```

## Execution

Run Phase 4.4 standalone:
```bash
cd project
python src/run_phase4_4_experiments.py
```

Or as part of full pipeline:
```bash
python src/run_phase4_pipeline.py
```

## Test Coverage

### 1. With Imputation
- **Median imputation**: Simple median-based missing value handling
- **MICE imputation**: Multiple Imputation by Chained Equations
- Tests CatBoost's performance after explicit preprocessing

### 2. Without Imputation (Raw Data)
- Tests CatBoost's native capability to handle missing values (NaN)
- CatBoost's `nan_mode="Min"` treats NaN as a separate category
- Evaluates the model's robustness to raw data with missing values
- Useful for understanding if explicit preprocessing improves or hinders performance

## Output Format

File: `results/tables/phase4_4_catboost_results.json`

```json
[
  {
    "dataset": "taiwan_bankruptcy",
    "timestamp": "2026-04-11T13:25:27.652045",
    "n_train": 5455,
    "n_test": 1364,
    "n_features": 95,
    "n_classes": 2,
    "catboost_available": true,
    "with_imputation": {
      "median": {
        "available": true,
        "error": null,
        "metrics": {
          "accuracy": 0.96,
          "precision": 0.96,
          "recall": 0.95,
          "f1": 0.96,
          "roc_auc": 0.989
        },
        "training_time_seconds": 2.34
      },
      "mice": {...}
    },
    "without_imputation": {
      "raw_with_nan": {...}
    }
  }
]
```

### Result Fields

- **available**: Boolean indicating if CatBoost was successfully trained
- **error**: Error message if training failed
- **metrics**:
  - `accuracy`: Overall accuracy score
  - `precision`: Weighted precision score
  - `recall`: Weighted recall score
  - `f1`: Weighted F1 score
  - `roc_auc`: ROC-AUC score (binary classification)
  - `training_time_seconds`: Computation time (seconds)

## Installation

CatBoost is available on PyPI. Install with:

```bash
pip install catboost
```

Or with GPU support:
```bash
pip install catboost-gpu
```

## Expected Results

CatBoost typically achieves:
- **Accuracy**: 0.94-0.97 on tabular classification tasks
- **Training Time**: 1-5 seconds per dataset (depends on size)
- **Robustness**: Good performance both with explicit imputation and raw NaN data

### Comparison Baseline
- Phase 4.1 (LR/RF with imputation): ~0.95 accuracy
- Phase 4.2 (SVM/MLP with imputation): ~0.96 accuracy
- Phase 4.3 (Robustness across 5-40% missing): ~0.85-0.96 accuracy (varies with missing %)
- **Phase 4.4 (CatBoost)**: Expected ~0.96+ accuracy

## Advantages of CatBoost

1. **Native NaN Handling**: Treats missing values as a separate category; no explicit imputation required
2. **Categorical Features**: Automatically handles categorical variables without encoding
3. **Fast Training**: Typically faster than traditional gradient boosting on medium datasets
4. **Regularization**: Built-in L1/L2 regularization and overfitting prevention
5. **Interpretability**: Feature importance scores available for model analysis
6. **Symmetrical Trees**: More stable predictions and better generalization

## Limitations & Characteristics

1. **Hyperparameter Tuning**: Default parameters work well but may require tuning for optimal performance

2. **Memory Usage**: Moderate (less than LightGBM, more than linear models)

3. **Categorical Limit**: Best for data with moderate number of categorical features

4. **NaN Pattern**: Treats all NaN values identically (no distinction between MCAR/MAR/MNAR to the model)

5. **Determinism**: `random_state` ensures reproducibility across runs

## Integration with Other Phases

- **Phase 4.1** (Baseline): LogisticRegression, RandomForest with median/MICE/mice_indicator preprocessing
- **Phase 4.2** (Extended): SVM, MLP with same preprocessing strategies
- **Phase 4.3** (Robustness): Tests sensitivity to injected missingness (MCAR/MAR/MNAR)
- **Phase 4.4** (Foundation): Tests CatBoost (gradient boosting) on same splits with explicit and implicit missing value handling

## Research Questions

Phase 4.4 addresses:
1. How does CatBoost (gradient boosting) compare to traditional ML models on tabular data?
2. Does explicit preprocessing (imputation) improve CatBoost performance, or does native NaN handling suffice?
3. What are the computational trade-offs (time, memory) vs. performance gains?
4. Can CatBoost handle different missingness patterns without explicit imputation?
5. Is CatBoost more robust to missing data than traditional approaches?
6. How does performance compare across different imputation strategies (median vs. MICE)?

## Files

- Configuration: [src/config_phase4_4.py](../src/config_phase4_4.py)
- Runner: [src/run_phase4_4_experiments.py](../src/run_phase4_4_experiments.py)
- Results: [results/tables/phase4_4_catboost_results.json](../results/tables/phase4_4_catboost_results.json)
- Logs: [results/logs/experiment_phase4_4_catboost_*.log](../results/logs/)
