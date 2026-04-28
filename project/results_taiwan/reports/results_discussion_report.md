# Performance of Pretrained Tabular Foundation Models on Incomplete Data - Results and Discussion

_Generated on 2026-04-27 20:53_

This document reports the numerical outcomes of the experiment matrix (datasets x mechanisms x rates x imputations x models) consolidated in `consolidated_results.csv`.

---

## 1. Summary

- Total runs recorded: **1140**
- Successful runs:      **1045**
- Failed / skipped:     **95**

- Unique models: **10**

- Mean **accuracy** across successful runs: **0.8215**  _(misleading on imbalanced data — see balanced_accuracy below)_

- Mean **balanced_accuracy**: **0.8406**  _(honest summary — 0.5 = random on binary tasks)_

- Mean **pr_auc** (binary tasks): **0.4438**  _(how well the positive class is ranked above negatives)_

- Mean **recall_class1**: **0.8609**  _(fraction of true positives actually caught)_


## 2. Best configuration per model

Mean metrics per model (averaged over datasets and scenarios), ranked by **balanced_accuracy** — the honest summary on imbalanced data. `accuracy` is kept for comparison but is not the primary ranking.

| model | accuracy | f1_weighted | runs | balanced_accuracy | f1_macro | pr_auc | recall_class1 | performance |
|---|---|---|---|---|---|---|---|---|
| TabPFN | 0.8698 | 0.9078 | 114 | 0.8741 | 0.6194 | 0.5641 | 0.8786 | Acceptable |
| XGBoost | 0.8631 | 0.9036 | 114 | 0.8711 | 0.6143 | 0.5150 | 0.8798 | Acceptable |
| Gradient-Boosting | 0.8045 | 0.8663 | 95 | 0.8651 | 0.5641 | 0.4318 | 0.9299 | Below target |
| TabICL | 0.8631 | 0.9036 | 114 | 0.8642 | 0.6104 | 0.5364 | 0.8654 | Acceptable |
| LightGBM | 0.8035 | 0.8653 | 114 | 0.8633 | 0.5649 | 0.5318 | 0.9272 | Below target |
| Random-Forest | 0.8530 | 0.8972 | 95 | 0.8590 | 0.6009 | 0.4459 | 0.8653 | Acceptable |
| CatBoost | 0.8060 | 0.8668 | 114 | 0.8525 | 0.5653 | 0.4793 | 0.9021 | Below target |
| SVM | 0.7836 | 0.8525 | 95 | 0.8447 | 0.5472 | 0.3167 | 0.9100 | Below target |
| Logistic-Regression | 0.7533 | 0.8315 | 95 | 0.8157 | 0.5255 | 0.3169 | 0.8825 | Below target |
| MLP | 0.7955 | 0.8304 | 95 | 0.6714 | 0.5290 | 0.2184 | 0.5388 | Below target |

## 3. Classical vs Foundation

| model_type | accuracy_mean | accuracy_std | f1_weighted_mean | n | balanced_accuracy_mean | balanced_accuracy_std | f1_macro_mean | pr_auc_mean |
|---|---|---|---|---|---|---|---|---|
| Classical | 0.8090 | 0.1020 | 0.8652 | 817 | 0.8326 | 0.0740 | 0.5651 | 0.4141 |
| Foundation | 0.8664 | 0.0346 | 0.9057 | 228 | 0.8691 | 0.0159 | 0.6149 | 0.5503 |

## 4. Robustness across missingness mechanisms

Mean **balanced_accuracy** per (model, mechanism) across all rates and imputations:

| model | MAR | MCAR | MNAR |
|---|---|---|---|
| CatBoost | 0.8495 | 0.8550 | 0.8516 |
| Gradient-Boosting | 0.8614 | 0.8642 | 0.8665 |
| LightGBM | 0.8565 | 0.8658 | 0.8668 |
| Logistic-Regression | 0.8151 | 0.8176 | 0.8120 |
| MLP | 0.6141 | 0.7381 | 0.6479 |
| Random-Forest | 0.8568 | 0.8579 | 0.8629 |
| SVM | 0.8483 | 0.8395 | 0.8411 |
| TabICL | 0.8602 | 0.8658 | 0.8638 |
| TabPFN | 0.8695 | 0.8700 | 0.8763 |
| XGBoost | 0.8694 | 0.8733 | 0.8746 |

## 5. Balanced Accuracy vs missing rate

Mean **balanced_accuracy** per (model, missing_rate) averaged over mechanisms, imputations, and datasets:

| model | 5.0 | 10.0 | 15.0 | 20.0 | 30.0 | 40.0 |
|---|---|---|---|---|---|---|
| CatBoost | 0.8581 | 0.8608 | 0.8430 | 0.8487 | 0.8559 | 0.8458 |
| Gradient-Boosting | 0.8657 | 0.8659 | 0.8630 | 0.8670 | 0.8634 | 0.8591 |
| LightGBM | 0.8657 | 0.8639 | 0.8676 | 0.8683 | 0.8565 | 0.8560 |
| Logistic-Regression | 0.8170 | 0.8167 | 0.8150 | 0.8249 | 0.8152 | 0.8005 |
| MLP | 0.6735 | 0.6867 | 0.7012 | 0.6621 | 0.6436 | 0.6331 |
| Random-Forest | 0.8601 | 0.8610 | 0.8610 | 0.8574 | 0.8607 | 0.8551 |
| SVM | 0.8666 | 0.8661 | 0.8476 | 0.8348 | 0.8307 | 0.8119 |
| TabICL | 0.8722 | 0.8614 | 0.8626 | 0.8618 | 0.8598 | 0.8618 |
| TabPFN | 0.8843 | 0.8732 | 0.8710 | 0.8722 | 0.8718 | 0.8592 |
| XGBoost | 0.8742 | 0.8810 | 0.8711 | 0.8775 | 0.8717 | 0.8592 |

## 6. Imputation method ranking

Mean **balanced_accuracy** averaged across models / datasets / mechanisms / rates:

| imputation | mean | std | count |
|---|---|---|---|
| none | 0.8691 | 0.0179 | 95 |
| mice_indicator | 0.8416 | 0.0519 | 190 |
| median | 0.8394 | 0.0768 | 190 |
| knn | 0.8370 | 0.0657 | 190 |
| mice | 0.8367 | 0.0721 | 190 |
| mean | 0.8339 | 0.0802 | 190 |

## 7. Discussion

- **Foundation models vs classical baselines**: see table in §3 and the boxplot in `visualizations/classical_vs_foundation.png`.
- **Imputation sensitivity**: §6 ranks imputation methods by mean accuracy; `mice_indicator` is expected to shine with higher rates while simple mean/median are competitive at low rates.
- **Missingness mechanism**: MNAR is the hardest scenario (value itself drives missingness), so gaps between mechanisms in §4 are informative for real-world applicability.


## 8. Conclusion

The consolidated table produced by `experiment_runner.py` gives a single source of truth for answering the project's question - which combination of (imputation, model) is most robust across the MCAR/MAR/MNAR grid. Practical recommendations follow in `practical_usability_report.md`.

