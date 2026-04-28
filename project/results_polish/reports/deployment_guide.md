# Deployment Guide

Quick-reference installation and operational notes for every model in the benchmark.

## Logistic-Regression

- **Install**: `sklearn only`
- **GPU required**: no
- **Typical training time**: <1
- **Size limits**: scales linearly; no practical cap
- **Notes**: Needs imputation and scaling; weakest classical baseline on nonlinear data.

## Random-Forest

- **Install**: `sklearn only`
- **GPU required**: no
- **Typical training time**: 1-30
- **Size limits**: millions of rows ok with n_jobs=-1
- **Notes**: Robust default; does not need scaling; needs imputation.

## Gradient-Boosting

- **Install**: `sklearn only`
- **GPU required**: no
- **Typical training time**: 5-60
- **Size limits**: up to ~100k rows comfortably
- **Notes**: Sensitive to hyperparameters; needs imputation.

## XGBoost

- **Install**: `pip install xgboost`
- **GPU required**: optional
- **Typical training time**: 1-30
- **Size limits**: very large (millions of rows)
- **Notes**: Handles NaN natively; strong default for tabular tasks.

## LightGBM

- **Install**: `pip install lightgbm`
- **GPU required**: optional
- **Typical training time**: 1-20
- **Size limits**: very large (millions of rows)
- **Notes**: Handles NaN natively; fastest among boosted trees here.

## SVM

- **Install**: `sklearn only`
- **GPU required**: no
- **Typical training time**: 10-300
- **Size limits**: tens of thousands (O(n^2) kernel)
- **Notes**: Requires scaling and imputation; slow on large datasets.

## MLP

- **Install**: `sklearn only`
- **GPU required**: no (torch-backed MLPs do)
- **Typical training time**: 10-180
- **Size limits**: hundreds of thousands
- **Notes**: Requires scaling and imputation; sensitive to architecture.

## TabPFN

- **Install**: `pip install tabpfn + TABPFN_TOKEN`
- **GPU required**: recommended (CUDA)
- **Typical training time**: 1-60 per call (pretrained, no training)
- **Size limits**: up to ~10k rows, ~500 features per call
- **Notes**: Consumes raw NaN; requires license token; best for small-to-medium tabular.

## TabICL

- **Install**: `pip install tabicl`
- **GPU required**: recommended (CUDA)
- **Typical training time**: 5-120 per call
- **Size limits**: similar to TabPFN; in-context learning
- **Notes**: Pretrained, handles NaN; verify availability in the target environment.

## CatBoost

- **Install**: `pip install catboost`
- **GPU required**: optional
- **Typical training time**: 10-120
- **Size limits**: very large
- **Notes**: Handles NaN and categorical features natively; robust default.
