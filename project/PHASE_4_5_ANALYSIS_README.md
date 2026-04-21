# Phase 4.5 - Final Analysis and Visualization

## Overview

**Phase 4.5** is the culmination of the experimental pipeline. It consolidates results from all phases (4.1-4.4) and generates:
- **Unified comparison tables** of classical vs. foundation models
- **Missingness sensitivity plots** showing how models perform across different missing data rates and mechanisms
- **Robustness analysis** evaluating model stability
- **Performance comparisons** highlighting strengths and weaknesses

## Research Questions Addressed

1. **Classical vs. Foundation Models**: Where does CatBoost outperform traditional ML algorithms?
2. **Missingness Type Sensitivity**: How does each model respond to MCAR, MAR, and MNAR patterns?
3. **Missing Data Tolerance**: What missing rate causes model degradation?
4. **Preprocessing Effectiveness**: Does explicit imputation (median, MICE) improve performance?
5. **Model Stability**: Which models are most robust across different scenarios?

## Data Consolidation

### Phase 4.1 & 4.2: Classical Models
- **Algorithms**: LogisticRegression, RandomForest, SVM, MLP
- **Preprocessing**: median, MICE, mice_indicator (Phase 4.2 only)
- **Datasets**: 3 (Taiwan, Polish, Slovak)
- **Source**: `phase4_experiment_results.json`, `phase4_2_experiment_results.json`

### Phase 4.3: Robustness Testing
- **Algorithms**: SVM, MLP (fallback from XGBoost/LightGBM)
- **Missingness Mechanisms**: MCAR, MAR, MNAR
- **Missing Rates**: 5%, 10%, 15%, 20%, 30%, 40%
- **Preprocessing**: median, MICE
- **Source**: `phase4_3_gradient_boosting_results.json`
- **Note**: Systematic missingness injection on test set after model training

### Phase 4.4: Foundation Models
- **Model**: CatBoost (gradient boosting with native NaN handling)
- **Testing Modes**: 
  - With imputation (median, MICE)
  - Without imputation (raw NaN - tests native capability)
- **Source**: `phase4_4_catboost_results.json`

## Outputs

### 1. Consolidated Results Tables

**File**: `phase4_5_consolidated_results.csv`

Unified table combining all models across all phases:

```
phase,dataset,model,model_type,preprocessing,accuracy,f1,precision,recall,roc_auc
4.1,taiwan_bankruptcy,LogisticRegression,Classical,median,0.945,0.934,...
4.1,taiwan_bankruptcy,LogisticRegression,Classical,mice,0.948,0.937,...
4.2,taiwan_bankruptcy,SVM,Classical,median,0.952,0.941,...
4.3,taiwan_bankruptcy,SVM,Classical,median,0.920,0.905,...  [with missing_mechanism=MCAR, missing_rate=5, ...]
4.4,taiwan_bankruptcy,CatBoost,Foundation,median,0.968,0.962,...
4.4,taiwan_bankruptcy,CatBoost,Foundation,raw_nan,0.971,0.966,...
...
```

**Subsets**:
- `phase4_5_classical_models.csv` - Only Phase 4.1 & 4.2
- `phase4_5_foundation_models.csv` - Only Phase 4.4

### 2. Robustness Analysis

**File**: `phase4_5_robustness_analysis.csv`

Summary statistics for each model across missingness mechanisms:

```
model,missing_mechanism,accuracy_mean,accuracy_std,accuracy_min,accuracy_max,f1_mean,f1_std
SVM,MCAR,0.920,0.015,0.895,0.938,...
SVM,MAR,0.915,0.018,0.887,0.932,...
SVM,MNAR,0.910,0.022,0.875,0.928,...
MLP,MCAR,0.925,0.012,0.903,0.941,...
...
```

**Interpretation**:
- **accuracy_mean**: Average performance across missing rates
- **accuracy_std**: Stability (lower = more stable)
- **accuracy_min/max**: Performance range
- **Impact**: Models with low std are more robust

### 3. Visualizations

#### Plot 1: Missing Rate vs. Model Performance (Separate for MCAR/MAR/MNAR)

**File**: `phase4_5_missing_rate_MCAR.png`, `...MAR.png`, `...MNAR.png`

Three metrics plotted (Accuracy, F1, ROC-AUC) against missing rate (X-axis):
- Classical models: dashed lines
- Foundation models: solid lines
- Shows performance degradation as missing rate increases

**Key Insights**:
- Which models degrade fastest?
- Are foundation models more robust?
- Does robustness differ by missingness type?

**Example Pattern**:
```
Accuracy vs. Missing Rate (MCAR)
│
1.0 ├─ CatBoost (solid, stable)
    ├─ SVM (dashed, moderate decline)
    ├─ MLP (dashed, steeper decline)
    │
0.8 └─ RF (dashed, fastest decline)
    
    0%  10%  20%  30%  40%  50%
        Missing Rate
```

#### Plot 2: Classical vs. Foundation Model Comparison

**File**: `phase4_5_model_comparison.png`

Box plots comparing model types across all datasets and preprocessing:
- **Accuracy**: Distribution across models
- **F1, ROC-AUC**: Additional metrics
- **Training Time**: Computational cost comparison

**Visualization**:
- X-axis: Model Type (Classical, Foundation)
- Y-axis: Metric value
- Shows median, quartiles, and outliers

**Key Insight**: Can identify if foundation models consistently outperform or underperform classical approaches.

#### Plot 3: Stability Heatmap

**File**: `phase4_5_stability_heatmap.png`

Heatmap of accuracy across models (rows) and missing rates (columns):
- Color intensity indicates performance (green = high, red = low)
- Allows visual identification of stable vs. unstable models
- Shows which models handle high missing rates best

**Example**:
```
          5%    10%   15%   20%   30%   40%
SVM      0.95  0.92  0.90  0.87  0.82  0.78
MLP      0.94  0.91  0.88  0.85  0.80  0.75
CatBoost 0.97  0.96  0.95  0.94  0.92  0.90
```

Models closer to the right with high values are most robust.

## Execution

### Standalone

```bash
cd project
python src/run_phase4_5_experiments.py
```

### As Part of Full Pipeline

```bash
python src/run_phase4_pipeline.py
```

The full pipeline will:
1. Standardize raw data
2. Create splits
3. Run Phase 4.1 (baseline models)
4. Run Phase 4.2 (extended models)
5. Run Phase 4.3 (robustness testing)
6. Run Phase 4.4 (CatBoost testing)
7. **Run Phase 4.5 (analysis & visualization)**

## Configuration

File: `src/config_phase4_5.py`

```python
# Enable/disable components
CONSOLIDATION_ENABLED = True          # Create consolidated tables
VISUALIZATION_ENABLED = True          # Generate plots
STABILITY_ANALYSIS_ENABLED = True     # Compute robustness statistics

# Metrics to analyze
ANALYSIS_METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]

# Model categorization
CLASSICAL_MODELS = ["LogisticRegression", "RandomForest", "SVM", "MLP"]
FOUNDATION_MODELS = ["CatBoost"]

# Visualization settings
PLOT_DPI = 150                         # Resolution
PLOT_STYLE = "seaborn-v0_8-darkgrid"  # Matplotlib style
```

## Interpretation Guide

### What to Look For

1. **Missingness Sensitivity**:
   - Are all models affected equally? Or some more than others?
   - At what missing rate does performance collapse?
   - Is MNAR worse than MCAR/MAR?

2. **Foundation Model Advantage**:
   - Does CatBoost outperform classical models at high missing rates?
   - Does raw NaN handling (no imputation) work as well as explicit imputation?
   - Is CatBoost more stable (lower variance)?

3. **Preprocessing Impact**:
   - Does median imputation hurt, help, or make no difference?
   - Is MICE beneficial for foundation models?
   - Does mice_indicator (missing flags) add value?

4. **Model-Specific Patterns**:
   - Which classical model is most robust? (Compare SVM, MLP, RF, LR)
   - Is there a "consistency" winner across all scenarios?
   - Are expensive models (SVM, MLP) worth the computational cost?

### Example Findings

**Scenario 1: CatBoost Dominance**
```
Result: CatBoost maintains >0.95 accuracy even at 40% missing
Classical: Most models drop to <0.80 at 40% missing
Implication: Use CatBoost for data with significant missingness
```

**Scenario 2: MNAR vs. MCAR**
```
Result: MNAR causes 5-10% accuracy drop more than MCAR
Classical models affected more than CatBoost
Implication: Models struggle with informative missingness; explicit handling needed
```

**Scenario 3: Raw NaN vs. Imputation**
```
Result: CatBoost with raw NaN > CatBoost with median imputation
Implication: Trust CatBoost's native NaN handling; avoid unnecessary preprocessing
```

## Files Generated

```
results/
├── tables/
│   ├── phase4_5_consolidated_results.csv       (Full unified table)
│   ├── phase4_5_classical_models.csv           (Phase 4.1/4.2)
│   ├── phase4_5_foundation_models.csv          (Phase 4.4)
│   └── phase4_5_robustness_analysis.csv        (Summary statistics)
├── visualizations/
│   ├── phase4_5_missing_rate_MCAR.png          (MCAR comparison)
│   ├── phase4_5_missing_rate_MAR.png           (MAR comparison)
│   ├── phase4_5_missing_rate_MNAR.png          (MNAR comparison)
│   ├── phase4_5_model_comparison.png           (Classical vs. Foundation)
│   └── phase4_5_stability_heatmap.png          (Robustness heatmap)
└── logs/
    └── experiment_phase4_5_analysis_*.log      (Execution log)
```

## Requirements

- **pandas**: Data manipulation
- **matplotlib** / **seaborn**: Visualization (*optional; analysis works without*)
- **numpy**: Numerical computation

If matplotlib/seaborn unavailable:
- Consolidation and analysis still run ✓
- Plots are skipped ⚠️
- All CSV results are still generated ✓

## Integration with Experiment Design

**Phase 4.1-4.4** generate raw experimental results. **Phase 4.5** synthesizes these into actionable insights:

```
Phase 4.1 (Baseline)
      ↓
Phase 4.2 (Extended)  ────┐
      ↓                    ├──> Classical baseline performance
Phase 4.1-4.2 Data ───────┤
      ↓                    
(Results consolidated)    
      ↓
Phase 4.3 (Robustness)  ──┤
      ↓                    ├──> Missingness sensitivity
Phase 4.3 Data ───────────┤
      ↓                    
(Robustness tracked)      
      ↓
Phase 4.4 (CatBoost) ─────┤
      ↓                    ├──> Foundation model performance
Phase 4.4 Data ───────────┤
      ↓                    
(Foundation results)      
      
Phase 4.5 (Analysis)
      ↓
Unified Table + Visualizations
      ↓
Actionable Insights: "Use CatBoost for <X% missing", "MNAR requires...", etc.
```

## Next Steps (Beyond Phase 4.5)

Future phases could include:
- **Phase 4.6**: Hyperparameter tuning based on findings
- **Phase 4.7**: Feature importance analysis from CatBoost
- **Phase 4.8**: Ensemble methods combining classical + foundation models
- **Phase 5**: Deployment optimization for production use

## Troubleshooting

**Issue**: Some plots missing?
- Check `VISUALIZATION_ENABLED` in `config_phase4_5.py`
- Verify matplotlib/seaborn installed: `pip install matplotlib seaborn`

**Issue**: Consolidated table has fewer rows than expected?
- Some phases may have had failures; check logs in `results/logs/`
- Verify all result JSON files exist: `ls results/tables/phase4_*.json`

**Issue**: Robustness analysis shows NaN?
- Phase 4.3 data may be incomplete
- Check if MCAR/MAR/MNAR patterns are in `phase4_3_gradient_boosting_results.json`

## References

- Phase 4.1: [Baseline Model Configuration](../src/config_phase4.py)
- Phase 4.2: [Extended Model Configuration](../src/config_phase4_2.py)
- Phase 4.3: [Robustness Testing](../src/run_phase4_3_experiments.py)
- Phase 4.4: [CatBoost Configuration](../src/config_phase4_4.py)
- Phase 4.5: [Analysis Configuration](../src/config_phase4_5.py)
