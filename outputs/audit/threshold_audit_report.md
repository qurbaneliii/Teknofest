# Threshold Audit

Selected threshold is `0.438150`, the median of fold-specific F1-macro optima. It was selected from MASTER-only OOF evidence; no panel or official test label was used. This is model-selection evidence, not an independent final test estimate.

## Default 0.50

| evaluation_split            | model_name               |   threshold |   n_samples |   n_pathogenic |   n_benign |   precision |   recall |   specificity |   f1_binary |   f1_macro |      mcc |   pr_auc |   roc_auc |   tn |   fp |   fn |   tp | source_prediction_file                                        | verified_timestamp               |
|:----------------------------|:-------------------------|------------:|------------:|---------------:|-----------:|------------:|---------:|--------------:|------------:|-----------:|---------:|---------:|----------:|-----:|-----:|-----:|-----:|:--------------------------------------------------------------|:---------------------------------|
| MASTER_ONLY_CV_DEFAULT_0.50 | audited_master_only_lgbm |         0.5 |        2353 |           1611 |        742 |    0.866202 | 0.847921 |      0.715633 |    0.856964 |   0.778284 | 0.556979 | 0.904085 |  0.847814 |  531 |  211 |  245 | 1366 | artifacts\predictions\audited_master_only_oof_predictions.csv | 2026-06-24T16:16:21.621400+00:00 |

## Selected

| evaluation_split   | model_name               |   threshold |   n_samples |   n_pathogenic |   n_benign |   precision |   recall |   specificity |   f1_binary |   f1_macro |      mcc |   pr_auc |   roc_auc |   tn |   fp |   fn |   tp | source_prediction_file                                        | verified_timestamp               |
|:-------------------|:-------------------------|------------:|------------:|---------------:|-----------:|------------:|---------:|--------------:|------------:|-----------:|---------:|---------:|----------:|-----:|-----:|-----:|-----:|:--------------------------------------------------------------|:---------------------------------|
| MASTER_ONLY_CV     | audited_master_only_lgbm |     0.43815 |        2353 |           1611 |        742 |    0.854897 | 0.877716 |       0.67655 |    0.866156 |   0.781447 | 0.563587 | 0.904085 |  0.847814 |  502 |  240 |  197 | 1414 | artifacts\predictions\audited_master_only_oof_predictions.csv | 2026-06-24T16:16:21.621400+00:00 |
