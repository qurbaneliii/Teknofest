# Model Performance Improvement Report

## 1. Current Baseline Performance
The current LightGBM OOF baseline at the F1-macro threshold has ROC-AUC 0.8375, PR-AUC 0.9011, F1-macro 0.7569, MCC 0.5219, pathogenic recall 0.9001, and specificity 0.5889.

## 2. Improved Model Performance
The selected final profile is `conservative_regularized` with threshold strategy `profile_f1_macro_opt` and threshold 0.4710. MASTER metrics are ROC-AUC 0.8475, PR-AUC 0.9025, F1-macro 0.7764, and MCC 0.5548. Panel-combined metrics are ROC-AUC 0.8725, PR-AUC 0.8251, F1-macro 0.7708, and MCC 0.5825.

## 3. Before vs After Comparison
Previous threshold-optimized profile: MASTER F1-macro 0.7564, MCC 0.5169, panel F1-macro 0.7565, panel MCC 0.5647.

Final selected profile: MASTER F1-macro 0.7764, MCC 0.5548, panel F1-macro 0.7708, panel MCC 0.5825.

## 4. Best Single Model
Best single-model recommendation: LightGBM with the `conservative_regularized` configuration. It was selected because it improves the final selection score after considering decision metrics, panel behavior, overfitting gap, and threshold stability.

## 5. Best Ensemble Model
The saved OOF ensemble stack was evaluated and is included in `reports/tables/all_evaluation_metrics.csv` and `reports/tables/experiment_comparison.csv`. It improves OOF ranking/decision metrics over several baselines, but the final recommendation remains LightGBM because the final selected LightGBM profile has explicit panel evaluation and final artifact support.

## 6. Best Threshold
Recommended final threshold: 0.4710. Threshold comparison:

| threshold_strategy    |   threshold |   f1_macro |      mcc |   recall |   specificity |
|:----------------------|------------:|-----------:|---------:|---------:|--------------:|
| default_0.5           |       0.5   |   0.477412 | 0.20475  | 0.996896 |     0.0714286 |
| f1_macro_opt          |       0.631 |   0.756388 | 0.516937 | 0.885164 |     0.609164  |
| mcc_opt               |       0.585 |   0.749295 | 0.533766 | 0.947858 |     0.508086  |
| youden_j              |       0.681 |   0.74794  | 0.507493 | 0.77157  |     0.762803  |
| balanced_accuracy_opt |       0.681 |   0.74794  | 0.507493 | 0.77157  |     0.762803  |

## 7. Best Calibration Method
Best MASTER Brier score method: `isotonic` with Brier 0.1416. Calibration is reported but not selected for the final decision model because panel trade-offs and decision metrics did not justify replacing the uncalibrated decision scores. See `reports/calibration_decision_review.md`.

## 8. Feature Selection Decision
Feature ablations were evaluated where saved OOF results exist. Computed feature ablations:

| configuration                         |   master_cv_roc_auc |   master_cv_f1_macro | generalization_flag                               |
|:--------------------------------------|--------------------:|---------------------:|:--------------------------------------------------|
| EK-only                               |            0.724734 |             0.661108 | panel metrics unavailable for this saved ablation |
| AL-only engineered frequency features |            0.805148 |             0.747452 | panel metrics unavailable for this saved ablation |
| All engineered features               |            0.8271   |             0.76086  | panel metrics unavailable for this saved ablation |
| All features                          |            0.838855 |             0.764827 | panel metrics unavailable for this saved ablation |
| All except AA chemistry               |            0.837096 |             0.762913 | panel metrics unavailable for this saved ablation |
| All except CAT_1 decomposition        |            0.840141 |             0.765511 | panel metrics unavailable for this saved ablation |

Top-k and some subgroup-only ablations remain explicitly marked as queued/not retrained; their metrics were not fabricated.

## 9. Leakage Audit Summary
Validation remains contamination-aware: MASTER variants shared with panels are excluded from validation folds and panel-unique subsets are evaluated separately. Variant_ID is not used as a predictive feature. Leakage and feature-quality outputs are available in EDA and Phase 9/10 reports; no perfect-separation leakage feature was accepted as final.

## 10. Medical Metric Report
The complete medical metric table is `reports/tables/all_evaluation_metrics.csv`. It includes ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, F1-macro, F1-weighted, MCC, log loss, Brier score, and confusion matrix counts.

## 11. Error Analysis Summary
Final error analysis found 195 false-negative rows and 314 false-positive rows across saved final predictions. Feature summaries by TP/TN/FP/FN are saved in `reports/tables/final_error_group_feature_summary.csv`. Panel-specific final metrics:

| evaluation_split   |   roc_auc |   pr_auc |   f1_macro |      mcc |   recall |   specificity | panel_role   |
|:-------------------|----------:|---------:|-----------:|---------:|---------:|--------------:|:-------------|
| CFTR_unique        |  0.972318 | 0.973536 |   0.911688 | 0.824958 | 0.882353 |      0.941176 | strongest    |
| KANSER_unique      |  0.902176 | 0.808062 |   0.733611 | 0.543684 | 0.933333 |      0.649485 | weakest      |
| PAH_unique         |  0.815126 | 0.843003 |   0.745098 | 0.528486 | 0.926471 |      0.55102  | intermediate |

## 12. Final Hidden-Test Strategy
Use the selected LightGBM final artifacts under `artifacts/models` and `artifacts/preprocessors`, apply the saved final threshold in `artifacts/metrics/final_threshold.json`, preserve calibration as report-only unless the competition format specifically requires calibrated probabilities, and prioritize panel-unique robustness over small OOF-only gains.

## Rejected Or Deferred Changes
Calibration was not selected for final decision scoring because it improved some probability losses but did not clearly preserve panel decision behavior. Uncomputed feature-group experiments are marked as queued rather than reported. Accuracy-only selection was rejected in favor of medical metrics.
