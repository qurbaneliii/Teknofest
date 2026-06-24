# Medical Optuna Optimization Report

## Audit And Resume Decision

Historical AUC-only study: `101` completed trials; best AUC `0.855274`.
Historical study timeout loss detected: `False`. No RUNNING historical trials were found.
Preserve the 101-trial AUC-only history and use a separate medical-utility study in the same SQLite store. Its AUC alone cannot establish a safe final-model improvement.

## Medical Study

Completed before resume: `0`; newly completed: `40`; completed total: `40`.
Best medical objective: `0.787323`. The objective is the requested MedicalUtilityScore and each completed trial stores ROC-AUC, PR-AUC, F1-macro, MCC, balanced accuracy, pathogenic recall, specificity, and fold stability attributes in SQLite.

## Best Parameters

```json
{
  "colsample_bytree": 0.5865857808774373,
  "learning_rate": 0.02723317124110314,
  "max_depth": 8,
  "min_child_samples": 105,
  "min_split_gain": 0.07649420721400307,
  "n_estimators": 1150,
  "num_leaves": 24,
  "reg_alpha": 0.010269883693999423,
  "reg_lambda": 1.7260735570455419,
  "scale_pos_weight": 0.3662203933390258,
  "subsample": 0.690107305692775
}
```

## Before Vs After

| model_id                                    | selection_status   |   threshold |   threshold_stability |   mean_roc_auc_gap |   master_roc_auc |   master_pr_auc |   master_f1_macro |   master_mcc |   master_balanced_accuracy |   master_pathogenic_recall |   master_specificity |   master_medical_utility_score |   panel_roc_auc |   panel_pr_auc |   panel_f1_macro |   panel_mcc |   panel_balanced_accuracy |   panel_pathogenic_recall |   panel_specificity |   panel_medical_utility_score |
|:--------------------------------------------|:-------------------|------------:|----------------------:|-------------------:|-----------------:|----------------:|------------------:|-------------:|---------------------------:|---------------------------:|---------------------:|-------------------------------:|----------------:|---------------:|-----------------:|------------:|--------------------------:|--------------------------:|--------------------:|------------------------------:|
| lightgbm_conservative_regularized_reference | preserved_final    |    0.471    |             0.0422054 |          0.0822681 |         0.847536 |        0.902494 |          0.776393 |     0.554754 |                   0.769402 |                   0.885164 |             0.653639 |                       0.774675 |        0.872534 |       0.825099 |         0.770808 |    0.582499 |                  0.786692 |                  0.923077 |            0.650307 |                      0.774898 |
| optuna_medical_candidate                    | rejected           |    0.386429 |             0.105739  |          0.0812943 |         0.831179 |        0.896741 |          0.70334  |     0.452693 |                   0.685838 |                   0.94041  |             0.431267 |                       0.721329 |        0.895186 |       0.853246 |         0.76593  |    0.599983 |                  0.788296 |                  0.969231 |            0.607362 |                      0.788543 |

## Decision

Candidate status: `rejected`. The deployed final model was not modified by this optimization run.
Required improvements observed: panel MCC.
Rejection reason: fewer than two required metrics improved; panel f1_macro worsened; threshold stability worsened.

## Provenance

All candidate metrics were recomputed from the newly saved candidate OOF and panel prediction CSV files. Panels were evaluated only after fitting on MASTER and their labels were never used for tuning or threshold selection.
