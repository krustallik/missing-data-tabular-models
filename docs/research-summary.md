# Co-authored study: reported results and contribution notes

## Source and status

This summary describes the co-authored technical report *Hodnotenie výkonnosti pretrained tabular foundation modelov pri práci s chýbajúcimi dátami* by Taras Havryliuk and Ihor Havryliv (Technical University of Košice, FEI, semester project). The report is not included in this repository. Its full text and figures are not redistributed here because no publication or reuse license for the co-authored document was confirmed.

The numbers below are historical results reported in that document. They have not been reproduced from the current checkout. The current six-step experiment runner enables Taiwan only, and the repository does not contain the report's consolidated result tables. The results should therefore be read as a summary of the co-authored report, not as a claim that a fresh run of this repository produces them.

## Study design

The report compares TabPFN and TabICL with eight classical estimators on three bankruptcy datasets:

| Dataset | Rows | Bankrupt companies | Positive share |
| --- | ---: | ---: | ---: |
| Polish Companies, 1-year | 7,027 | 271 | 3.86% |
| Slovak Manufacture 13 | 4,107 | 30 | 0.73% |
| Taiwan Bankruptcy | 6,819 | 220 | 3.23% |

The experiment uses stratified 80/20 train/test splits, two split random states, and two experiment seeds. Synthetic MCAR, MAR, and MNAR missingness is injected into training data at target rates from 5% to 40%; no synthetic masking is applied to test data, though source datasets retain their native missingness. Imputers are fitted on training data only. PR-AUC is the primary metric because of class imbalance; balanced accuracy, macro F1, and minority-class recall are secondary measures.

The report states that the Slovak split leaves only 24 bankrupt cases in training and six in testing. That small positive test count makes its scores especially sensitive to individual predictions. It also reports 5,835 initially missing cells (1.28%) for Polish, 12,928 (4.84%) for Slovak, and none for Taiwan. These historical figures differ slightly from missingness recalculated from the CSV files currently committed here (1.30%, 5.00%, and 0.00%, respectively); they have not been reconciled and must not be treated as independently reproduced.

## Selected reported results

Group-average PR-AUC reported for each dataset:

| Dataset | Foundation models (TabPFN, TabICL) | Classical models |
| --- | ---: | ---: |
| Polish Companies, 1-year | 0.6483 | 0.4030 |
| Slovak Manufacture 13 | 0.2760 | 0.1795 |
| Taiwan Bankruptcy | 0.4740 | 0.4185 |

Best individual model PR-AUC reported by dataset:

| Dataset | Reported best model | PR-AUC |
| --- | --- | ---: |
| Polish Companies, 1-year | TabPFN | 0.7117 |
| Slovak Manufacture 13 | TabICL | 0.3042 |
| Taiwan Bankruptcy | XGBoost | 0.5002 |

Average PR-AUC across models, mechanisms, and missingness rates for the reported imputation comparison:

| Strategy | Polish | Slovak | Taiwan |
| --- | ---: | ---: | ---: |
| No explicit imputation (none) | 0.6247 | 0.2645 | 0.4754 |
| MICE with missingness indicators | 0.5396 | 0.2045 | 0.4242 |
| Mean | 0.4524 | 0.1906 | 0.4263 |
| MICE | 0.4516 | 0.1908 | 0.4304 |
| Median | 0.4035 | 0.1896 | 0.4261 |
| kNN | 0.3528 | 0.1953 | 0.4235 |

The report's selected missingness-response values are also shown in newly drawn charts embedded in the README. For example, it reports Polish TabPFN PR-AUC declining from 0.8923 at 5% missingness to 0.4275 at 40%; Polish TabICL declines from 0.8339 to 0.2318. For Taiwan, the report gives 5%→40% values of 0.5194→0.4640 for XGBoost, 0.5191→0.4644 for LightGBM, 0.5064→0.4493 for TabICL, and 0.4942→0.4363 for TabPFN. On Slovak Manufacture 13, reported TabICL PR-AUC is 0.3087 under MCAR, 0.3041 under MAR, and 0.2998 under MNAR. These are selected report-level values; the connector lines in the chart join only reported endpoints and do not assert unreported intermediate scores.

These are report-level aggregates, not values recalculated from repository artifacts. The reported comparison is not a controlled test of equal tuning budgets: classical models were tuned with dataset-specific randomized greedy search using PR-AUC, while TabPFN and TabICL were used as pretrained models without comparable hyperparameter tuning. The conclusion is therefore about the practical pipelines evaluated in the report, not an intrinsic ranking under equal optimization budgets. The report also notes that MICE sometimes reached its configured 30-iteration limit without convergence.

## Authorship and contribution context

The work was co-authored by Taras Havryliuk and Ihor Havryliv, and both authors remain credited. The Git history shows Taras's recorded code contributions concentrated in the early algorithm-development period. Later history records substantial work by Ihor on connecting and refactoring modules, expanding the experiment runners and repeated-seed workflow, debugging, result handling, and reporting.

Git commits do not represent the full experimental contribution. Ihor coordinated the wider execution and analysis, ran the main body of the calculations across two computers, and directed a subset of runs that Taras executed on Taras's hardware. Taras also contributed early algorithm components. The integration, cross-model and cross-dataset comparisons, analysis, and later corrections were primarily led by Ihor. No exact percentage is assigned because neither commit counts nor compute-run counts alone measure the full research contribution.

## Reproduction boundary

The repository currently contains processed input datasets and experiment code, but not the report's consolidated benchmark outputs or complete environment lock. Its main experiment matrix is configured for Taiwan only. The random-search script covers all three datasets but falls back to full processed data when Polish or Slovak split files are absent. The report's outcomes cannot be independently verified from this checkout until the exact historical splits, result artifacts, software versions, and execution settings are available and checked.
