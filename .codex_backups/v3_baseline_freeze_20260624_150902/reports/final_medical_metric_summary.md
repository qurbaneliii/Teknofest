# Final Medical Metric Summary

This audit recomputes clinical metrics from the saved final prediction artifacts. It does not retrain a model, modify probabilities, or change the saved threshold.

## Saved Final Threshold

Threshold: `0.471000` using `profile_f1_macro_opt`.

## MASTER Consistency Check

The saved MASTER predictions match the existing selected Phase 10 ROC-AUC, PR-AUC, F1-macro, and MCC values within 1e-9.

| metric   |   saved_prediction_audit |   phase10_selected_reference |   difference |
|:---------|-------------------------:|-----------------------------:|-------------:|
| roc_auc  |                 0.847536 |                     0.847536 |            0 |
| pr_auc   |                 0.902494 |                     0.902494 |            0 |
| f1_macro |                 0.776393 |                     0.776393 |            0 |
| mcc      |                 0.554754 |                     0.554754 |            0 |

## Medical Metrics By Evaluation Split

| evaluation_split            |   n_samples |   threshold |   roc_auc |   pr_auc |   balanced_accuracy |   pathogenic_recall |   specificity |   f1_macro |      mcc |   brier_score |   medical_utility_score |   clinical_safety_score |
|:----------------------------|------------:|------------:|----------:|---------:|--------------------:|--------------------:|--------------:|-----------:|---------:|--------------:|------------------------:|------------------------:|
| MASTER_CV_saved_predictions |        2353 |       0.471 |  0.847536 | 0.902494 |            0.769402 |            0.885164 |      0.653639 |   0.776393 | 0.554754 |     0.140572  |                0.774675 |                0.738335 |
| CFTR_unique                 |          34 |       0.471 |  0.972318 | 0.973536 |            0.911765 |            0.882353 |      0.941176 |   0.911688 | 0.824958 |     0.0709215 |                0.916968 |                0.882789 |
| KANSER_unique               |         142 |       0.471 |  0.902176 | 0.808062 |            0.791409 |            0.933333 |      0.649485 |   0.733611 | 0.543684 |     0.178631  |                0.765027 |                0.70722  |
| PAH_unique                  |         117 |       0.471 |  0.815126 | 0.843003 |            0.738745 |            0.926471 |      0.55102  |   0.745098 | 0.528486 |     0.171498  |                0.742066 |                0.704183 |
| panel_unique_combined       |         293 |       0.471 |  0.872534 | 0.825099 |            0.786692 |            0.923077 |      0.650307 |   0.770808 | 0.582499 |     0.163284  |                0.774898 |                0.727627 |

MedicalUtilityScore is the specified weighted combination of ROC-AUC, PR-AUC, F1-macro, MCC, balanced accuracy, pathogenic recall, and specificity. ClinicalSafetyScore increases the relative importance of pathogenic recall and calibration quality.
