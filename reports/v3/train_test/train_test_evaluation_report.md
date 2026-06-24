# Controlled V3 Local-Holdout Training and Evaluation

## Executive summary

This is an **internal local-holdout evaluation**, not official hidden-test performance. All learned feature fitting, imputation, scaling, and model fitting used MASTER `train` rows only. Thresholds were selected on `validation` only; `local_test` was not used for feature fitting, threshold selection, or candidate promotion. The protected baseline remains final.

## Data files and official-test status

Labeled competition training files: YARISMA_TRAIN_MASTER.csv, YARISMA_TRAIN_KANSER.csv, YARISMA_TRAIN_CFTR.csv, YARISMA_TRAIN_PAH.csv.

| file_path   | exists   | has_label_column   | action_taken               | metrics_computed   | prediction_file_created   | notes                              |
|:------------|:---------|:-------------------|:---------------------------|:-------------------|:--------------------------|:-----------------------------------|
| none        | False    | False              | no official test CSV found | False              | False                     | no official test metric is claimed |

No official test metric is claimed.

## Split and class distribution

| split      |    n |   benign |   pathogenic |   pathogenic_rate |
|:-----------|-----:|---------:|-------------:|------------------:|
| train      | 1875 |      500 |         1375 |          0.733333 |
| validation |  469 |      125 |          344 |          0.733475 |
| local_test |  587 |      157 |          430 |          0.732538 |
| MASTER     | 2931 |      782 |         2149 |          0.733197 |
| KANSER     |  388 |      120 |          268 |          0.690722 |
| CFTR       |  111 |       21 |           90 |          0.810811 |
| PAH        |  372 |       62 |          310 |          0.833333 |

The requested 80/20 MASTER split was created first; its 80% training side was split into train/validation (80/20) solely for validation-derived thresholds.

## Feature sets

| feature_set           |   feature_count | equivalent_to   |
|:----------------------|----------------:|:----------------|
| v3_safe_minimal       |             352 | none            |
| v3_no_target_encoding |             352 | v3_safe_minimal |
| v3_frequency_heavy    |             363 | none            |
| v3_panel_robust       |              21 | none            |

All outputs are numeric; `Variant_ID` and `Label` are excluded from features. An equivalence entry identifies feature matrices that are exactly identical rather than presenting them as independent evidence.

## Model-family handling and warnings

| model_family           | available   | status    |   reason |
|:-----------------------|:------------|:----------|---------:|
| logistic_regression    | True        | available |      nan |
| extratrees             | True        | available |      nan |
| hist_gradient_boosting | True        | available |      nan |
| lightgbm               | True        | available |      nan |
| xgboost                | True        | available |      nan |
| catboost               | True        | available |      nan |

Logistic Regression uses `SimpleImputer → StandardScaler → LogisticRegression(lbfgs, max_iter=5000, class_weight=balanced)`. Captured warnings:

No training warnings captured.

## Best internal results

Best local-holdout F1-macro: `hist_gradient_boosting__v3_safe_minimal` at `fixed_0_50` — F1-macro 0.7655, MCC 0.5454.

Best local-holdout MCC: `hist_gradient_boosting__v3_safe_minimal` at `protected_baseline_0_471` — F1-macro 0.7597, MCC 0.5456.

Best exploratory robust-selection candidate: `hist_gradient_boosting__v3_safe_minimal` at `validation_f1_macro` — score 0.7238. This is **not** a final-selection result.

## Panel results, KANSER, and worst panel

| evaluation_split   |   n |   f1_macro |      mcc |   roc_auc |   pr_auc |   precision |   recall |   specificity |   tn |   fp |   fn |   tp |
|:-------------------|----:|-----------:|---------:|----------:|---------:|------------:|---------:|--------------:|-----:|-----:|-----:|-----:|
| KANSER             | 388 |   0.847931 | 0.708464 |  0.914801 | 0.953218 |    0.874576 | 0.962687 |      0.691667 |   83 |   37 |   10 |  258 |
| CFTR               | 111 |   0.840517 | 0.691174 |  0.945503 | 0.988256 |    0.964286 | 0.9      |      0.857143 |   18 |    3 |    9 |   81 |
| PAH                | 372 |   0.734286 | 0.492234 |  0.81821  | 0.935797 |    0.89521  | 0.964516 |      0.435484 |   27 |   35 |   11 |  299 |

Combined panel: F1-macro 0.8166; MCC 0.6425; PR-AUC 0.9506. Worst-panel values are F1-macro 0.7343 and MCC 0.4922.

## Confusion-matrix and threshold interpretation

The best robust candidate's local-holdout confusion matrix is TN 98, FP 59, FN 46, TP 384. Threshold variants (0.50, protected 0.471, validation F1-macro, validation MCC) are recorded without optimizing on local_test.

## Robust comparison and final decision

Every candidate is rejected from final replacement. The protected baseline has MASTER OOF F1-macro 0.7764 and MCC 0.5548 with panel-unique F1-macro 0.7708 and MCC 0.5825. Internal holdout performance is not protocol-comparable, and no candidate can pass the robust replacement gates from this experiment. **Protected baseline remains final.**

## Limitations and next action

- Internal evaluation is not official hidden-test performance.
- Panel training files are labeled external checks, but their sizes and disease distributions differ from MASTER.
- No official test CSV was found locally; no submission predictions were produced.
- Do not run Optuna from this phase. The next valid action, if requested, is a repeated contamination-aware V3 validation of one candidate under the protected baseline's protocol.

## Reproduce

`python scripts/v3_train_test_evaluate.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main --output-dir reports/v3/train_test --test-size 0.20 --random-state 42 --feature-sets v3_safe_minimal,v3_no_target_encoding,v3_frequency_heavy,v3_panel_robust --models logistic_regression,extratrees,hist_gradient_boosting,lightgbm,xgboost,catboost --quick`
