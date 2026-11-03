# Experimental design and data notes

## Task and data

The benchmark is a binary classification study. In the bundled processed CSVs, `target = 1` denotes bankrupt and `target = 0` denotes non-bankrupt. Data statistics below were recalculated from `project/data/processed/`.

| Dataset | Rows | Features | Bankrupt count | Positive share | Native missing feature cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Polish companies, 1-year subset | 7,027 | 64 | 271 | 3.86% | 1.30% |
| Slovak manufacturing, 13-year subset | 4,107 | 63 | 30 | 0.73% | 5.00% |
| Taiwan bankruptcy | 6,819 | 95 | 220 | 3.23% | 0.00% |

The Polish files correspond to the `1year.arff` case in the [UCI Polish Companies Bankruptcy dataset](https://archive.ics.uci.edu/dataset/365/polish%2Bcompanies%2Bbankruptcy%2Bdata). Taiwan's bundled source is [UCI Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese%2Bbankruptcy%2Bprediction). The Slovak manufacturing file names match the [Mendeley Data, version 2](https://data.mendeley.com/datasets/j89csb932y/2) source. Each upstream dataset page specifies CC BY 4.0. See [data attribution](data-attribution.md) for dataset citations and credit details.

## Missingness scenarios

The main configuration requests three synthetic mechanisms at six target final feature-cell rates: 5%, 10%, 15%, 20%, 30%, and 40%. The configured tolerance for the achieved rate is 0.01 absolute fraction.

- **MCAR:** randomly selects currently observed feature cells.
- **MAR:** chooses a control feature and gives higher masking probability to rows whose control value is above its median. This is the repository's concrete simulation rule, not a general implementation of every MAR process.
- **MNAR:** uses an observed feature's own value to bias which cells of that feature are masked. The source code explicitly treats this as a synthetic proxy; true MNAR depends on unobserved values.

Rates are total missingness after injection, including any native missing cells already present. If native missingness is already at or above a requested rate, the injector does not add more cells. The 72-row verification run on Taiwan (two split seeds, two experiment seeds, three mechanisms, six rates) passed the configured tolerance for every row.

## Models, imputers, and scoring

The model registry contains eight classical estimators and two pretrained tabular models. The configured imputation strategies are mean, median, KNN, iterative imputation (MICE), MICE plus missingness indicators, and no explicit imputation for estimators marked as NaN-capable.

The primary ranking metric is average precision, stored as `pr_auc`. This is suitable for inspection alongside the severe class imbalance, but it does not by itself describe the operational cost of false positives or false negatives. The benchmark also calculates balanced accuracy, macro F1, positive-class recall, ROC-AUC, and weighted metrics.

Classical estimators use class weighting where implemented. With `TUNE_THRESHOLD` enabled, the model code chooses a classification threshold on a stratified holdout drawn from the training data, using balanced accuracy by default. This means reported threshold-based predictions are not based on an unexamined fixed 0.5 threshold. The search pipeline is distinct: it evaluates sampled classical-model settings with five-fold stratified CV and ranks candidates by average precision.

## Reproduction caveats

1. `config.DATASETS` currently enables only Taiwan for the six-step main pipeline. The three processed datasets are not all included in its default experiment matrix.
2. The random-search script separately enables all three datasets. It uses a precomputed training split when available, but otherwise reads the full CSV. The split step currently prepares only Taiwan, so the Polish and Slovak searches fall back to full data by default.
3. No consolidated benchmark outputs are committed. The scripts and inputs are present, but final model rankings cannot be independently inspected from the clone alone.
4. The default randomized search is computationally expensive: 150 candidates × five folds × three seeds, across eight models and three datasets (about 54,000 requested fits before failures or skips).
5. The package list has no version pins and no declared Python version. Foundation-model access and hardware requirements can vary by package version and environment.
6. The dataset source citations and CC BY 4.0 terms are documented, but the repository does not specify a license for its own code.

A technically defensible report should include the exact dataset scope, random seeds, software versions, trained models, and generated result tables. Avoid treating the single-run smoke checks described in the README as benchmark findings.
