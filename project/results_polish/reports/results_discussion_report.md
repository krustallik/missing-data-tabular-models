# Performance of Pretrained Tabular Foundation Models on Incomplete Data - Results and Discussion

_Generated on 2026-04-27 20:52_

This document reports the numerical outcomes of the experiment matrix (datasets x mechanisms x rates x imputations x models) consolidated in `consolidated_results.csv`.

---

## 1. Summary

- Total runs recorded: **1140**
- Successful runs:      **1037**
- Failed / skipped:     **103**

- Unique models: **10**

- Mean **accuracy** across successful runs: **0.8304**  _(misleading on imbalanced data — see balanced_accuracy below)_

- Mean **balanced_accuracy**: **0.7865**  _(honest summary — 0.5 = random on binary tasks)_

- Mean **pr_auc** (binary tasks): **0.4899**  _(how well the positive class is ranked above negatives)_

- Mean **recall_class1**: **0.7390**  _(fraction of true positives actually caught)_


## 2. Best configuration per model

Mean metrics per model (averaged over datasets and scenarios), ranked by **balanced_accuracy** — the honest summary on imbalanced data. `accuracy` is kept for comparison but is not the primary ranking.

| model | accuracy | f1_weighted | runs | balanced_accuracy | f1_macro | pr_auc | recall_class1 | performance |
|---|---|---|---|---|---|---|---|---|
| TabPFN | 0.9549 | 0.9633 | 112 | 0.9169 | 0.8130 | 0.8403 | 0.8757 | Excellent |
| LightGBM | 0.8584 | 0.8970 | 114 | 0.8560 | 0.6379 | 0.6364 | 0.8535 | Acceptable |
| XGBoost | 0.8797 | 0.9111 | 112 | 0.8483 | 0.6619 | 0.6160 | 0.8143 | Acceptable |
| TabICL | 0.9088 | 0.9308 | 114 | 0.8373 | 0.7159 | 0.6272 | 0.7597 | Good |
| Gradient-Boosting | 0.8427 | 0.8861 | 93 | 0.8313 | 0.6165 | 0.5674 | 0.8190 | Below target |
| CatBoost | 0.8398 | 0.8838 | 114 | 0.8313 | 0.6158 | 0.6126 | 0.8221 | Below target |
| Random-Forest | 0.8345 | 0.8806 | 93 | 0.7982 | 0.5980 | 0.4092 | 0.7589 | Below target |
| Logistic-Regression | 0.7247 | 0.8037 | 95 | 0.6865 | 0.5038 | 0.1646 | 0.6452 | Below target |
| SVM | 0.6173 | 0.7167 | 95 | 0.6823 | 0.4410 | 0.1833 | 0.7526 | Below target |
| MLP | 0.7890 | 0.7837 | 95 | 0.5109 | 0.4245 | 0.0749 | 0.2096 | Below target |

## 3. Classical vs Foundation

| model_type | accuracy_mean | accuracy_std | f1_weighted_mean | n | balanced_accuracy_mean | balanced_accuracy_std | f1_macro_mean | pr_auc_mean |
|---|---|---|---|---|---|---|---|---|
| Classical | 0.8021 | 0.1690 | 0.8486 | 811 | 0.7614 | 0.1248 | 0.5673 | 0.4222 |
| Foundation | 0.9317 | 0.0662 | 0.9469 | 226 | 0.8767 | 0.0786 | 0.7640 | 0.7328 |

## 4. Robustness across missingness mechanisms

Mean **balanced_accuracy** per (model, mechanism) across all rates and imputations:

| model | MAR | MCAR | MNAR |
|---|---|---|---|
| CatBoost | 0.8397 | 0.8266 | 0.8197 |
| Gradient-Boosting | 0.8371 | 0.8311 | 0.8253 |
| LightGBM | 0.8589 | 0.8471 | 0.8544 |
| Logistic-Regression | 0.6794 | 0.7024 | 0.6805 |
| MLP | 0.5141 | 0.5124 | 0.4985 |
| Random-Forest | 0.7973 | 0.7982 | 0.7966 |
| SVM | 0.6937 | 0.6872 | 0.6643 |
| TabICL | 0.8358 | 0.8299 | 0.8318 |
| TabPFN | 0.9217 | 0.9191 | 0.9055 |
| XGBoost | 0.8531 | 0.8399 | 0.8488 |

## 5. Balanced Accuracy vs missing rate

Mean **balanced_accuracy** per (model, missing_rate) averaged over mechanisms, imputations, and datasets:

| model | 5.0 | 10.0 | 15.0 | 20.0 | 30.0 | 40.0 |
|---|---|---|---|---|---|---|
| CatBoost | 0.8747 | 0.8664 | 0.8519 | 0.8245 | 0.8071 | 0.7474 |
| Gradient-Boosting | 0.8693 | 0.8605 | 0.8489 | 0.8369 | 0.8042 | 0.7674 |
| LightGBM | 0.8968 | 0.8861 | 0.8757 | 0.8511 | 0.8336 | 0.7775 |
| Logistic-Regression | 0.6835 | 0.6986 | 0.7001 | 0.6992 | 0.6848 | 0.6583 |
| MLP | 0.5471 | 0.4947 | 0.4976 | 0.4934 | 0.5103 | 0.5069 |
| Random-Forest | 0.8470 | 0.8331 | 0.8191 | 0.7953 | 0.7665 | 0.7231 |
| SVM | 0.7084 | 0.7033 | 0.6800 | 0.6806 | 0.6572 | 0.6608 |
| TabICL | 0.9189 | 0.8943 | 0.8546 | 0.8314 | 0.7707 | 0.7252 |
| TabPFN | 0.9452 | 0.9389 | 0.9359 | 0.9170 | 0.9020 | 0.8538 |
| XGBoost | 0.8944 | 0.8722 | 0.8673 | 0.8537 | 0.8242 | 0.7719 |

## 6. Imputation method ranking

Mean **balanced_accuracy** averaged across models / datasets / mechanisms / rates:

| imputation | mean | std | count |
|---|---|---|---|
| none | 0.8678 | 0.0692 | 95 |
| mice_indicator | 0.8124 | 0.1243 | 186 |
| mean | 0.7843 | 0.1331 | 190 |
| mice | 0.7809 | 0.1316 | 186 |
| median | 0.7672 | 0.1258 | 190 |
| knn | 0.7477 | 0.1126 | 190 |

## 7. Discussion

- **Foundation models vs classical baselines**: see table in §3 and the boxplot in `visualizations/classical_vs_foundation.png`.
- **Imputation sensitivity**: §6 ranks imputation methods by mean accuracy; `mice_indicator` is expected to shine with higher rates while simple mean/median are competitive at low rates.
- **Missingness mechanism**: MNAR is the hardest scenario (value itself drives missingness), so gaps between mechanisms in §4 are informative for real-world applicability.


## 8. Conclusion

The consolidated table produced by `experiment_runner.py` gives a single source of truth for answering the project's question - which combination of (imputation, model) is most robust across the MCAR/MAR/MNAR grid. Practical recommendations follow in `practical_usability_report.md`.

