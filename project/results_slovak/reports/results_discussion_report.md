# Performance of Pretrained Tabular Foundation Models on Incomplete Data - Results and Discussion

_Generated on 2026-04-27 20:52_

This document reports the numerical outcomes of the experiment matrix (datasets x mechanisms x rates x imputations x models) consolidated in `consolidated_results.csv`.

---

## 1. Summary

- Total runs recorded: **1140**
- Successful runs:      **1045**
- Failed / skipped:     **95**

- Unique models: **10**

- Mean **accuracy** across successful runs: **0.7710**  _(misleading on imbalanced data — see balanced_accuracy below)_

- Mean **balanced_accuracy**: **0.7172**  _(honest summary — 0.5 = random on binary tasks)_

- Mean **pr_auc** (binary tasks): **0.2032**  _(how well the positive class is ranked above negatives)_

- Mean **recall_class1**: **0.6625**  _(fraction of true positives actually caught)_


## 2. Best configuration per model

Mean metrics per model (averaged over datasets and scenarios), ranked by **balanced_accuracy** — the honest summary on imbalanced data. `accuracy` is kept for comparison but is not the primary ranking.

| model | accuracy | f1_weighted | runs | balanced_accuracy | f1_macro | pr_auc | recall_class1 | performance |
|---|---|---|---|---|---|---|---|---|
| TabICL | 0.8574 | 0.9140 | 114 | 0.8353 | 0.5076 | 0.3731 | 0.8129 | Acceptable |
| TabPFN | 0.9001 | 0.9397 | 114 | 0.7966 | 0.5258 | 0.2568 | 0.6915 | Good |
| XGBoost | 0.8028 | 0.8786 | 114 | 0.7802 | 0.4805 | 0.2070 | 0.7573 | Below target |
| LightGBM | 0.8685 | 0.9189 | 114 | 0.7662 | 0.5178 | 0.2169 | 0.6623 | Acceptable |
| Logistic-Regression | 0.7217 | 0.8086 | 95 | 0.7466 | 0.4356 | 0.1223 | 0.7719 | Below target |
| CatBoost | 0.8403 | 0.8988 | 114 | 0.7418 | 0.5093 | 0.2293 | 0.6418 | Below target |
| Gradient-Boosting | 0.7749 | 0.8522 | 95 | 0.7133 | 0.4640 | 0.1774 | 0.6509 | Below target |
| Random-Forest | 0.9212 | 0.9506 | 95 | 0.6999 | 0.5363 | 0.1623 | 0.4754 | Good |
| SVM | 0.6935 | 0.7942 | 95 | 0.6532 | 0.4175 | 0.2269 | 0.6123 | Below target |
| MLP | 0.2471 | 0.3031 | 95 | 0.3717 | 0.1566 | 0.0064 | 0.4982 | Below target |

## 3. Classical vs Foundation

| model_type | accuracy_mean | accuracy_std | f1_weighted_mean | n | balanced_accuracy_mean | balanced_accuracy_std | f1_macro_mean | pr_auc_mean |
|---|---|---|---|---|---|---|---|---|
| Classical | 0.7410 | 0.2659 | 0.8075 | 817 | 0.6896 | 0.1495 | 0.4441 | 0.1720 |
| Foundation | 0.8787 | 0.0797 | 0.9268 | 228 | 0.8159 | 0.0814 | 0.5167 | 0.3149 |

## 4. Robustness across missingness mechanisms

Mean **balanced_accuracy** per (model, mechanism) across all rates and imputations:

| model | MAR | MCAR | MNAR |
|---|---|---|---|
| CatBoost | 0.7565 | 0.7524 | 0.7168 |
| Gradient-Boosting | 0.7322 | 0.7028 | 0.7047 |
| LightGBM | 0.7975 | 0.7567 | 0.7341 |
| Logistic-Regression | 0.7697 | 0.7066 | 0.7508 |
| MLP | 0.3745 | 0.3791 | 0.3578 |
| Random-Forest | 0.7374 | 0.6800 | 0.6882 |
| SVM | 0.6476 | 0.6382 | 0.6746 |
| TabICL | 0.8315 | 0.8249 | 0.8525 |
| TabPFN | 0.7642 | 0.8254 | 0.8079 |
| XGBoost | 0.7835 | 0.7850 | 0.7734 |

## 5. Balanced Accuracy vs missing rate

Mean **balanced_accuracy** per (model, missing_rate) averaged over mechanisms, imputations, and datasets:

| model | 5.0 | 10.0 | 15.0 | 20.0 | 30.0 | 40.0 |
|---|---|---|---|---|---|---|
| CatBoost | 0.7398 | 0.7247 | 0.7120 | 0.7504 | 0.7450 | 0.7795 |
| Gradient-Boosting | 0.7147 | 0.7018 | 0.6956 | 0.7138 | 0.7312 | 0.7223 |
| LightGBM | 0.7909 | 0.7223 | 0.7606 | 0.7994 | 0.7436 | 0.7600 |
| Logistic-Regression | 0.8228 | 0.7908 | 0.7673 | 0.6884 | 0.7360 | 0.6489 |
| MLP | 0.3947 | 0.3592 | 0.3615 | 0.3580 | 0.3687 | 0.3806 |
| Random-Forest | 0.6653 | 0.6608 | 0.6997 | 0.6979 | 0.7568 | 0.7306 |
| SVM | 0.6483 | 0.6583 | 0.6617 | 0.6728 | 0.6657 | 0.6140 |
| TabICL | 0.8171 | 0.8740 | 0.8745 | 0.8059 | 0.8265 | 0.8198 |
| TabPFN | 0.7505 | 0.7973 | 0.8113 | 0.8158 | 0.7861 | 0.8338 |
| XGBoost | 0.7730 | 0.8137 | 0.7642 | 0.8028 | 0.7710 | 0.7591 |

## 6. Imputation method ranking

Mean **balanced_accuracy** averaged across models / datasets / mechanisms / rates:

| imputation | mean | std | count |
|---|---|---|---|
| none | 0.8014 | 0.0888 | 95 |
| knn | 0.7147 | 0.1440 | 190 |
| mice | 0.7119 | 0.1401 | 190 |
| mean | 0.7083 | 0.1494 | 190 |
| mice_indicator | 0.7049 | 0.1623 | 190 |
| median | 0.7039 | 0.1503 | 190 |

## 7. Discussion

- **Foundation models vs classical baselines**: see table in §3 and the boxplot in `visualizations/classical_vs_foundation.png`.
- **Imputation sensitivity**: §6 ranks imputation methods by mean accuracy; `mice_indicator` is expected to shine with higher rates while simple mean/median are competitive at low rates.
- **Missingness mechanism**: MNAR is the hardest scenario (value itself drives missingness), so gaps between mechanisms in §4 are informative for real-world applicability.


## 8. Conclusion

The consolidated table produced by `experiment_runner.py` gives a single source of truth for answering the project's question - which combination of (imputation, model) is most robust across the MCAR/MAR/MNAR grid. Practical recommendations follow in `practical_usability_report.md`.

