# Tabular Missing-Data Benchmark (Phase 4.1)

This project provides a complete reproducible pipeline for bankruptcy prediction experiments on tabular data, from dataset preparation to model training and evaluation.

## Phases

**Phase 3.1: Dataset Preparation** — Standardizes and splits 3 classification datasets

**Phase 4.1: Model Training & Evaluation** — Trains models with hyperparameter tuning and comprehensive metrics

## Datasets

- Taiwanese Bankruptcy Prediction (UCI id=572)
- Polish Companies Bankruptcy (UCI id=365) 
- Slovak financial ratios dataset

## Quick Start

```bash
pip install -r requirements.txt
python src/run_phase4_pipeline.py
```

This runs the complete Phase 4.1 pipeline (Phase 3.1 + Phase 4.1).

### Individual Phases

```bash
# Phase 3.1 only: dataset preparation
python src/run_dataset_setup.py

# Phase 4.1 only: model training  
python src/phase4_experiment_runner.py
```

## Results

Results are saved to:
- `results/tables/` - dataset overviews and experiment setup
- `results/tables/phase4_experiment_results.json` - model evaluation results
- `results/logs/` - detailed experiment logs

## Features (Phase 4.1)

✓ Multiple classification models (Logistic Regression, Random Forest)
✓ K-fold cross-validation with grid search
✓ Comprehensive metrics (Accuracy, F1, Precision, Recall, ROC-AUC)
✓ Experiment logging and tracking
✓ Reproducible results (random_state=42)

For detailed documentation, see [README_PHASE4.md](README_PHASE4.md).

