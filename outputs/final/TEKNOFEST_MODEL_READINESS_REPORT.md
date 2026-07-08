# TEKNOFEST Model Readiness Report

## Executive summary

**CONDITIONAL GO.** The audited model is reproducible, has saved recomputed metrics, excludes supplied-panel overlap from its MASTER training/validation population, and completes label-free inference. The official test CSV and official submission schema are absent.

## Final selected model

`audited_master_only_lgbm`; threshold `0.438150`; artifact `C:\Users\qurba\OneDrive\Documents\Teknofest\artifacts\models\audited_master_only_lgbm\full_model.joblib`.

## Dataset and validation status

MASTER-only 5-fold stratified OOF; KANSER/PAH/CFTR unique panels are external-like labelled checks. No hidden-test metrics are claimed.

## Metric verification and subgroup performance

| evaluation_split      | model_name               |   threshold |   n_samples |   n_pathogenic |   n_benign |   precision |   recall |   specificity |   f1_binary |   f1_macro |      mcc |   pr_auc |   roc_auc |   tn |   fp |   fn |   tp | source_prediction_file                                        | verified_timestamp               |
|:----------------------|:-------------------------|------------:|------------:|---------------:|-----------:|------------:|---------:|--------------:|------------:|-----------:|---------:|---------:|----------:|-----:|-----:|-----:|-----:|:--------------------------------------------------------------|:---------------------------------|
| MASTER_ONLY_CV        | audited_master_only_lgbm |     0.43815 |        2353 |           1611 |        742 |    0.854897 | 0.877716 |      0.67655  |    0.866156 |   0.781447 | 0.563587 | 0.904085 |  0.847814 |  502 |  240 |  197 | 1414 | artifacts\predictions\audited_master_only_oof_predictions.csv | 2026-06-24T16:16:21.621400+00:00 |
| CFTR_UNIQUE           | audited_master_only_lgbm |     0.43815 |          34 |             17 |         17 |    0.882353 | 0.882353 |      0.882353 |    0.882353 |   0.882353 | 0.764706 | 0.961641 |  0.961938 |   15 |    2 |    2 |   15 | artifacts\predictions\audited_panel_unique_predictions.csv    | 2026-06-24T16:16:21.621400+00:00 |
| KANSER_UNIQUE         | audited_master_only_lgbm |     0.43815 |         142 |             45 |         97 |    0.617647 | 0.933333 |      0.731959 |    0.743363 |   0.786886 | 0.619632 | 0.800442 |  0.906071 |   71 |   26 |    3 |   42 | artifacts\predictions\audited_panel_unique_predictions.csv    | 2026-06-24T16:16:21.621400+00:00 |
| PAH_UNIQUE            | audited_master_only_lgbm |     0.43815 |         117 |             68 |         49 |    0.766234 | 0.867647 |      0.632653 |    0.813793 |   0.755211 | 0.520365 | 0.854474 |  0.821128 |   31 |   18 |    9 |   59 | artifacts\predictions\audited_panel_unique_predictions.csv    | 2026-06-24T16:16:21.621400+00:00 |
| PANEL_UNIQUE_COMBINED | audited_master_only_lgbm |     0.43815 |         293 |            130 |        163 |    0.716049 | 0.892308 |      0.717791 |    0.794521 |   0.795219 | 0.609639 | 0.825512 |  0.876168 |  117 |   46 |   14 |  116 | artifacts\predictions\audited_panel_unique_predictions.csv    | 2026-06-24T16:16:21.621400+00:00 |

## Threshold optimization

Default 0.50 and selected threshold metrics are preserved in `outputs/audit/threshold_audit_report.md`; full sweep is `outputs/final/threshold_sweep.csv`.

## Errors, explainability, and inference

See error, explainability, and inference outputs in this directory. Label-free dry-run outputs were generated from MASTER schema only; Label was ignored.

## Remaining limitations

- OOF threshold selection is not a third independent test set.
- Panel sample sizes are limited, especially CFTR.
- Distribution shift and organizer submission format remain unknown.

## Reproduce

```powershell
python src/run_final_pipeline.py
python src/sanity_check_final.py
python src/predict_final.py --input path/to/official_test.csv --output outputs/final/submission.csv --basic-submission
```

## Go / No-Go

**CONDITIONAL GO** pending organizer test input and submission template confirmation.
