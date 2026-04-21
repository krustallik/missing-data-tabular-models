# Phase 4.2 - Missing Indicator Requirement

## Overview
Phase 4.2 evaluates extended models (SVM, MLP, XGBoost, LightGBM) with three preprocessing strategies.

## Required: mice_indicator Strategy

**Missing Indicator Preprocessing** (`mice_indicator`) is **REQUIRED** in Phase 4.2 for the following reasons:

1. **Sensitivity Analysis (Student 2)**
   - Tests if models detect missingness patterns through explicit binary features
   - Each feature gets an additional binary indicator: 1 if original value was missing, 0 otherwise

2. **Comparison with MICE**
   - Isolates the effect of "knowing" which values were imputed
   - Helps determine if model performance improves with explicit missingness information

3. **Robustness Metrics**
   - Enables assessment of how model robustness changes across:
     - Different missing mechanisms (MCAR, MAR, MNAR)
     - Different missing rates (5%-40%)
     - Different preprocessing methods

## Configuration

File: `src/config_phase4_2.py`

```python
PREPROCESSING_METHODS = [
    "median",           # Simple median imputation
    "mice",             # Multiple Imputation by Chained Equations
    "mice_indicator",   # MICE + binary missing indicators (REQUIRED)
]
```

## Execution

Run Phase 4.2 with all preprocessing methods:
```bash
cd project
python src/run_phase4_2_experiments.py
```

Results saved to: `results/tables/phase4_2_experiment_results.json`

## Output Structure

Results include metrics for each combination:
- Dataset: {taiwan_bankruptcy, polish_1year, slovak_manufacture_13}
- Preprocessing: {median, mice, mice_indicator}
- Models: {svm, mlp, xgboost, lightgbm}
- Metrics: {accuracy, f1, precision, recall, roc_auc}

## Notes for Student 2

The `mice_indicator` strategy is your key contribution to understanding:
- Model sensitivity to missingness patterns
- Whether explicit indicators help or hurt model performance
- How different imputation strategies compare under systematic missing data
