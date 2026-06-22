# Model Zoo Summary

All challenger models use the existing fold-safe FeatureEngineer and contamination-aware folds. The LightGBM conservative reference is copied from the immutable saved Phase 10 OOF artifact, so its result is directly reproducible without altering that model.

All decision metrics use the unchanged saved final threshold `0.471000`. No model is selected as final in this phase.

## OOF Metrics

| model_id                          |   roc_auc |   pr_auc |   f1_macro |      mcc |   medical_utility_score |   pathogenic_recall |   specificity |
|:----------------------------------|----------:|---------:|-----------:|---------:|------------------------:|--------------------:|--------------:|
| lightgbm_conservative_regularized |  0.847536 | 0.902494 |   0.776393 | 0.554754 |                0.774675 |            0.885164 |      0.653639 |
| catboost                          |  0.850874 | 0.910881 |   0.764586 | 0.567485 |                0.773083 |            0.959032 |      0.522911 |
| xgboost                           |  0.844757 | 0.904976 |   0.764451 | 0.557466 |                0.769108 |            0.945996 |      0.540431 |
| lightgbm_high_capacity_controlled |  0.842399 | 0.903164 |   0.762971 | 0.532104 |                0.763604 |            0.896958 |      0.606469 |
| extra_trees                       |  0.835306 | 0.904187 |   0.740725 | 0.484002 |                0.744372 |            0.867784 |      0.59973  |
| elasticnet_logistic_regression    |  0.815246 | 0.887633 |   0.722969 | 0.453419 |                0.725868 |            0.767846 |      0.706199 |

## Per-Metric Leaders

- ROC-AUC: `catboost` (0.850874)
- PR-AUC: `catboost` (0.910881)
- F1-macro: `lightgbm_conservative_regularized` (0.776393)
- MCC: `catboost` (0.567485)
- MedicalUtilityScore: `lightgbm_conservative_regularized` (0.774675)
