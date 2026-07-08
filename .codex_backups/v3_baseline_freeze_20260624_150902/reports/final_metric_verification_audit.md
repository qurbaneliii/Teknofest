# Final Metric Verification Audit

Final metrics were recomputed from `artifacts/predictions/final_master_cv_predictions.csv` and `artifacts/predictions/final_panel_predictions.csv`.

| split                 |   reported_roc_auc |   recomputed_roc_auc |   reported_pr_auc |   recomputed_pr_auc |   reported_f1_macro |   recomputed_f1_macro |   reported_mcc |   recomputed_mcc |   roc_auc_abs_diff |   pr_auc_abs_diff |   f1_macro_abs_diff |   mcc_abs_diff | status   |
|:----------------------|-------------------:|---------------------:|------------------:|--------------------:|--------------------:|----------------------:|---------------:|-----------------:|-------------------:|------------------:|--------------------:|---------------:|:---------|
| MASTER_CV             |           0.847536 |             0.847536 |          0.902494 |            0.902494 |            0.776393 |              0.776393 |       0.554754 |         0.554754 |                  0 |                 0 |                   0 |              0 | match    |
| panel_unique_combined |           0.872534 |             0.872534 |          0.825099 |            0.825099 |            0.770808 |              0.770808 |       0.582499 |         0.582499 |                  0 |                 0 |                   0 |              0 | match    |

Small differences, if present, reflect regenerated deterministic fold predictions versus cached profile summary rows.
