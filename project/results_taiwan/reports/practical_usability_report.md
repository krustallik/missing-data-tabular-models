# Performance of Pretrained Tabular Foundation Models on Incomplete Data - Practical Usability

_Generated on 2026-04-27 20:53_

This document translates the benchmark into practical guidance for real-world use: at what missingness level is each model usable, and what does it take to deploy it.

---

## 1. Suitability matrix

Mean **balanced_accuracy** by missingness bucket (all mechanisms, all imputations). `balanced_accuracy` is used by default as it is honest on imbalanced datasets (0.5 ≈ random guessing).

| model | low (<=10%) | medium (<=20%) | high (>20%) |
|---|---|---|---|
| CatBoost | 0.8594 | 0.8458 | 0.8509 |
| Gradient-Boosting | 0.8658 | 0.8650 | 0.8612 |
| LightGBM | 0.8648 | 0.8680 | 0.8563 |
| Logistic-Regression | 0.8168 | 0.8199 | 0.8079 |
| MLP | 0.6801 | 0.6817 | 0.6384 |
| Random-Forest | 0.8606 | 0.8592 | 0.8579 |
| SVM | 0.8664 | 0.8412 | 0.8213 |
| TabICL | 0.8668 | 0.8622 | 0.8608 |
| TabPFN | 0.8788 | 0.8716 | 0.8655 |
| XGBoost | 0.8776 | 0.8743 | 0.8654 |

## 2. Performance bands

- Excellent: accuracy >= 0.95
- Good:      accuracy >= 0.90
- Acceptable: accuracy >= 0.85


## 3. Foundation models - when to use

- **TabPFN** is strongest for small-to-medium datasets where labeled data is scarce. It does not need imputation and gives competitive accuracy at low-to-medium missingness rates. License token required.
- **TabICL** competes with TabPFN on small/medium tabular tasks with the advantage of NaN handling; runtime and size limits depend on the particular implementation version.


## 4. Foundation models - when NOT to use

- Very large datasets (millions of rows) -> use XGBoost/LightGBM.
- High feature counts above TabPFN's pretraining limit -> classical boosting is safer.
- Environments without GPU / without license access -> CatBoost or XGBoost with MICE + indicator (CatBoost is the most robust NaN-aware classical tree ensemble and a good baseline when TabPFN / TabICL are not an option).


## 5. Deployment complexity

See also `deployment_complexity.csv` and `deployment_guide.md`.

| model | install | gpu_required | typical_training_seconds | size_limits | notes |
|---|---|---|---|---|---|
| Logistic-Regression | sklearn only | no | <1 | scales linearly; no practical cap | Needs imputation and scaling; weakest classical baseline on nonlinear data. |
| Random-Forest | sklearn only | no | 1-30 | millions of rows ok with n_jobs=-1 | Robust default; does not need scaling; needs imputation. |
| Gradient-Boosting | sklearn only | no | 5-60 | up to ~100k rows comfortably | Sensitive to hyperparameters; needs imputation. |
| XGBoost | pip install xgboost | optional | 1-30 | very large (millions of rows) | Handles NaN natively; strong default for tabular tasks. |
| LightGBM | pip install lightgbm | optional | 1-20 | very large (millions of rows) | Handles NaN natively; fastest among boosted trees here. |
| SVM | sklearn only | no | 10-300 | tens of thousands (O(n^2) kernel) | Requires scaling and imputation; slow on large datasets. |
| MLP | sklearn only | no (torch-backed MLPs do) | 10-180 | hundreds of thousands | Requires scaling and imputation; sensitive to architecture. |
| TabPFN | pip install tabpfn + TABPFN_TOKEN | recommended (CUDA) | 1-60 per call (pretrained, no training) | up to ~10k rows, ~500 features per call | Consumes raw NaN; requires license token; best for small-to-medium tabular. |
| TabICL | pip install tabicl | recommended (CUDA) | 5-120 per call | similar to TabPFN; in-context learning | Pretrained, handles NaN; verify availability in the target environment. |
| CatBoost | pip install catboost | optional | 10-120 | very large | Handles NaN and categorical features natively; robust default. |
