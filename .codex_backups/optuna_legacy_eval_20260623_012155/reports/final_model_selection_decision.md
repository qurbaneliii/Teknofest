# Final Model Selection Decision

## Selected Configuration

- Selected model: `lightgbm_conservative_regularized` (current_final_model).
- Selected threshold: `0.471000` using `profile_f1_macro_opt`.
- Ensemble replaced LightGBM: `false`.
- Calibration decision: `none`.

## Why This Is Safest

The preserved LightGBM result is selected because it has direct saved OOF and panel-unique evidence, plus documented calibration and error-analysis audits, at the unchanged threshold. No Phase 2 or Phase 3 challenger satisfies every conservative replacement gate: at least two required metric improvements, non-degraded OOF and panel MedicalUtilityScore, non-degraded panel F1-macro and MCC, and no material fold-instability increase.

## Exact OOF Metrics

| metric                |       value |
|:----------------------|------------:|
| roc_auc               |    0.847536 |
| pr_auc                |    0.902494 |
| accuracy              |    0.812155 |
| balanced_accuracy     |    0.769402 |
| precision             |    0.847296 |
| pathogenic_recall     |    0.885164 |
| specificity           |    0.653639 |
| f1                    |    0.865817 |
| f1_macro              |    0.776393 |
| mcc                   |    0.554754 |
| brier_score           |    0.140572 |
| log_loss              |    0.444047 |
| ppv                   |    0.847296 |
| npv                   |    0.723881 |
| tn                    |  485        |
| fp                    |  257        |
| fn                    |  185        |
| tp                    | 1426        |
| medical_utility_score |    0.774675 |
| clinical_safety_score |    0.738335 |

## Exact Combined Panel Metrics

| metric                |      value |
|:----------------------|-----------:|
| roc_auc               |   0.872534 |
| pr_auc                |   0.825099 |
| accuracy              |   0.771331 |
| balanced_accuracy     |   0.786692 |
| precision             |   0.677966 |
| pathogenic_recall     |   0.923077 |
| specificity           |   0.650307 |
| f1                    |   0.781759 |
| f1_macro              |   0.770808 |
| mcc                   |   0.582499 |
| brier_score           |   0.163284 |
| log_loss              |   0.508685 |
| ppv                   |   0.677966 |
| npv                   |   0.913793 |
| tn                    | 106        |
| fp                    |  57        |
| fn                    |  10        |
| tp                    | 120        |
| medical_utility_score |   0.774898 |
| clinical_safety_score |   0.727627 |

## Calibration Decision

Calibration is retained as `none` for the final decision model. The calibration comparison is evidence only; it is not directly comparable to the Phase 2/3 candidate scores because it uses its own archived threshold review.

| calibration_method   | decision                          |   brier_score |   log_loss |   f1_macro |      mcc |   panel_f1_macro |   panel_mcc |
|:---------------------|:----------------------------------|--------------:|-----------:|-----------:|---------:|-----------------:|------------:|
| none                 | selected_for_final_decision_model |      0.172732 |   0.524124 |   0.756388 | 0.516937 |         0.756545 |    0.564707 |
| sigmoid              | reported_only                     |      0.145517 |   0.458323 |   0.756388 | 0.516937 |         0.756545 |    0.564707 |
| isotonic             | rejected_for_panel_tradeoff       |      0.141613 |   0.46347  |   0.754085 | 0.513844 |         0.756545 |    0.564707 |

## Panel-Unique Metrics

| evaluation_split   |   roc_auc |   pr_auc |   accuracy |   balanced_accuracy |   precision | pathogenic_recall   |   specificity |       f1 |   f1_macro |      mcc | brier_score   | log_loss   | ppv   | npv   |   tn |   fp |   fn |   tp | medical_utility_score   | clinical_safety_score   |
|:-------------------|----------:|---------:|-----------:|--------------------:|------------:|:--------------------|--------------:|---------:|-----------:|---------:|:--------------|:-----------|:------|:------|-----:|-----:|-----:|-----:|:------------------------|:------------------------|
| CFTR_unique        |  0.972318 | 0.973536 |   0.911765 |            0.911765 |    0.9375   |                     |      0.941176 | 0.909091 |   0.911688 | 0.824958 |               |            |       |       |   16 |    1 |    2 |   15 |                         |                         |
| KANSER_unique      |  0.902176 | 0.808062 |   0.739437 |            0.791409 |    0.552632 |                     |      0.649485 | 0.694215 |   0.733611 | 0.543684 |               |            |       |       |   63 |   34 |    3 |   42 |                         |                         |
| PAH_unique         |  0.815126 | 0.843003 |   0.769231 |            0.738745 |    0.741176 |                     |      0.55102  | 0.823529 |   0.745098 | 0.528486 |               |            |       |       |   27 |   22 |    5 |   63 |                         |                         |

## Error Analysis

Saved false-negative case rows: 195. Saved false-positive case rows: 314. The detailed error-group summary has 320 rows and remains available at `C:\Users\qurba\OneDrive\Documents\Teknofest\reports\tables\final_error_group_feature_summary.csv`.
The selected OOF confusion counts are TN=485, FP=257, FN=185, TP=1426.
The archived FN/FP case-file row counts do not match the immutable Phase 1 OOF confusion counts, so they are retained as qualitative error-analysis evidence and are not used in model ranking.

## Rejected Experiments

| candidate_id                      | candidate_kind   |   roc_auc |   pr_auc |   f1_macro |      mcc |   medical_utility_score |   panel_f1_macro |   panel_mcc |   panel_medical_utility_score | rejection_reason                                                                                                                                        |
|:----------------------------------|:-----------------|----------:|---------:|-----------:|---------:|------------------------:|-----------------:|------------:|------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| lightgbm_high_capacity_controlled | model_zoo        |  0.842399 | 0.903164 |   0.762971 | 0.532104 |                0.763604 |         0.759611 |    0.577476 |                      0.772388 | lower OOF MedicalUtilityScore; lower panel MedicalUtilityScore; lower panel F1-macro; lower panel MCC; fewer than two required metric improvements.     |
| catboost                          | model_zoo        |  0.850874 | 0.910881 |   0.764586 | 0.567485 |                0.773083 |         0.714547 |    0.53012  |                      0.754736 | lower OOF MedicalUtilityScore; lower panel MedicalUtilityScore; lower panel F1-macro; lower panel MCC.                                                  |
| xgboost                           | model_zoo        |  0.844757 | 0.904976 |   0.764451 | 0.557466 |                0.769108 |         0.736411 |    0.564706 |                      0.76104  | lower OOF MedicalUtilityScore; lower panel MedicalUtilityScore; lower panel F1-macro; lower panel MCC.                                                  |
| extra_trees                       | model_zoo        |  0.835306 | 0.904187 |   0.740725 | 0.484002 |                0.744372 |         0.748557 |    0.566097 |                      0.762351 | lower OOF MedicalUtilityScore; lower panel MedicalUtilityScore; lower panel F1-macro; lower panel MCC; fewer than two required metric improvements.     |
| elasticnet_logistic_regression    | model_zoo        |  0.815246 | 0.887633 |   0.722969 | 0.453419 |                0.725868 |         0.780216 |    0.562271 |                      0.765174 | lower OOF MedicalUtilityScore; lower panel MedicalUtilityScore; lower panel MCC; fewer than two required metric improvements; less stable across folds. |
| simple_average                    | ensemble         |  0.851712 | 0.909867 |   0.770394 | 0.556547 |                0.773756 |         0.759611 |    0.577476 |                      0.777013 | lower OOF MedicalUtilityScore; lower panel F1-macro; lower panel MCC.                                                                                   |
| rank_average                      | ensemble         |  0.851361 | 0.909891 |   0.728244 | 0.490246 |                0.747705 |         0.822424 |    0.657917 |                      0.812531 | lower OOF MedicalUtilityScore.                                                                                                                          |
| weighted_average                  | ensemble         |  0.851712 | 0.90988  |   0.770394 | 0.556547 |                0.773758 |         0.759611 |    0.577476 |                      0.777084 | lower OOF MedicalUtilityScore; lower panel F1-macro; lower panel MCC.                                                                                   |
| mcc_weighted_average              | ensemble         |  0.851712 | 0.909867 |   0.770394 | 0.556547 |                0.773756 |         0.759611 |    0.577476 |                      0.777013 | lower OOF MedicalUtilityScore; lower panel F1-macro; lower panel MCC.                                                                                   |
| medical_utility_weighted_average  | ensemble         |  0.851712 | 0.909867 |   0.770394 | 0.556547 |                0.773756 |         0.759611 |    0.577476 |                      0.777013 | lower OOF MedicalUtilityScore; lower panel F1-macro; lower panel MCC.                                                                                   |
| logistic_stacking                 | ensemble         |  0.84929  | 0.903832 |   0.770548 | 0.559285 |                0.77274  |         0.745277 |    0.556819 |                      0.768543 | lower OOF MedicalUtilityScore; lower panel MedicalUtilityScore; lower panel F1-macro; lower panel MCC.                                                  |
| ridge_stacking                    | ensemble         |  0.848954 | 0.90535  |   0.768918 | 0.570147 |                0.774048 |         0.73686  |    0.559892 |                      0.767808 | lower OOF MedicalUtilityScore; lower panel MedicalUtilityScore; lower panel F1-macro; lower panel MCC.                                                  |
| elasticnet_stacking               | ensemble         |  0.849451 | 0.903807 |   0.768203 | 0.553946 |                0.770823 |         0.745277 |    0.556819 |                      0.768602 | lower OOF MedicalUtilityScore; lower panel MedicalUtilityScore; lower panel F1-macro; lower panel MCC; fewer than two required metric improvements.     |

## Remaining Risks

- Hidden-test prevalence, disease panel mix, population composition, and annotation quality can differ from MASTER and the supplied panels.
- Calibration was not selected for decision scores; probability calibration should be revalidated before any non-competition use.
- Panel-unique results are a distribution-shift proxy, not an independent prospective clinical validation.
- This repository supports a competition model only and does not establish clinical deployment readiness.
