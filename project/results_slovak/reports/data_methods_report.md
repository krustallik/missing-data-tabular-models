# Performance of Pretrained Tabular Foundation Models on Incomplete Data

**Author**: Project team  
**Institution**: University  
**Date**: April 27, 2026

This document covers the Data, Experimental setup and Methods chapters. Results and Discussion live in `results_discussion_report.md` and practical recommendations in `practical_usability_report.md`.

---

## 1. Data

Three real-world tabular classification datasets from the course reference article are used:

- **Taiwan Bankruptcy** - financial ratios for Taiwanese listed firms.
- **Polish Companies (1-year horizon)** - Polish bankruptcy forecasting.
- **Slovak Manufacture 13** - Slovak SME financial indicators.

All three are stored pre-cleaned in `data/processed/` with a single `target` column. Feature matrices are originally complete (no native NaN) which is the precondition for the controlled missingness protocol.

### 1.1 Coverage in the consolidated table

| dataset | n_rows | models |
|---|---|---|
| slovak_manufacture_13 | 1140 | CatBoost, Gradient-Boosting, LightGBM, Logistic-Regression, MLP, Random-Forest, SVM, TabICL, TabPFN, XGBoost |

---

## 2. Experimental setup

- **Split**: stratified 80/20 train/test split, `random_state = 42`. Splits are materialised once in `data/splits/*.csv` and re-used by every step of the benchmark.
- **Primary metrics** (ranking): accuracy, balanced_accuracy, f1_macro, pr_auc. On highly imbalanced binary targets `accuracy` is misleading — `balanced_accuracy`, `f1_macro` and `pr_auc` are the ones to trust. Additional metrics stored per run: `f1` (weighted), `precision`, `recall`, `recall_class1`, `roc_auc`, plus the decision `threshold` actually used after tuning.
- **Missingness grid**: mechanisms ['MCAR', 'MAR', 'MNAR'], rates ['5%', '10%', '15%', '20%', '30%', '40%']. Injection is applied to the **training split only**; the test split is kept complete so reported metrics measure the effect of training under missingness.
- **Verification**: realised missing fractions are compared against requested rates in `results/tables/missingness_verification.csv`. The injector stays within ±1 percentage point of target.

---

## 3. Methods

### 3.1 Missingness generation

- **MCAR**: uniform random sampling of observed cells.
- **MAR**: probability of masking a cell in feature *A* depends on a control feature *B* (rows where *B* exceeds its median receive weight 0.8, others 0.2).
- **MNAR**: probability of masking a cell depends on the feature's own value.

Implementation: `src/missingness.py`.

### 3.2 Imputation methods

All methods fit on train only to avoid test leakage:

- **mean / median** - per-column train statistic.
- **knn** - `KNNImputer(n_neighbors=5)`.
- **mice** - `IterativeImputer` (MICE).
- **mice_indicator** - MICE + binary missing-indicator features.
- **none** - raw NaN passed to models that handle NaN natively (TabPFN, TabICL, CatBoost, XGBoost, LightGBM).

Implementation: `src/imputation.py`.

### 3.3 Models

Classical: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, MLP, CatBoost (default hyperparameters; scaling applied to LR / SVM / MLP).

Foundation: TabPFN, TabICL. TabPFN requires `TABPFN_TOKEN` and skips gracefully when it is absent. GPU is used when available.

Implementation: `src/models.py`.

### 3.4 Evaluation

Metrics are computed in `src/evaluation.py` using scikit-learn's weighted averages for multi-class-safe F1/Precision/Recall and binary ROC-AUC (NaN for multiclass targets).

