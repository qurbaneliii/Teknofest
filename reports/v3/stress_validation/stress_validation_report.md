# HistGradientBoosting Candidate Stress Validation

## Executive summary

Candidate: `hist_gradient_boosting` with `v3_safe_minimal`. This is repeated internal validation, **not official hidden-test performance**. Every builder and model was fitted only on each fold's MASTER training rows; panels were evaluation-only.

## Protocol

Five stratified folds were repeated over five seeds (25 fold validations). Thresholds 0.50, 0.471, validation F1-macro, and validation MCC were evaluated. Validation-derived thresholds were selected from validation scores only and never from panels.

## Repeated MASTER validation

At validation-F1 thresholds: F1-macro 0.7656 ± 0.0182; MCC 0.5397 ± 0.0358; ROC-AUC 0.8378; PR-AUC 0.9034.

## Panel stress result

Combined panels: F1-macro 0.7995; MCC 0.6074. KANSER: F1-macro 0.8344; MCC 0.6858. PAH: F1-macro 0.7319; MCC 0.4770.

## Threshold stability

Validation F1 threshold mean 0.6412, std 0.0574, IQR 0.0700; range 0.5100–0.7400.

## Baseline comparison and decision

Protected baseline reference is MASTER OOF F1-macro 0.7764/MCC 0.5548 and panel-unique F1-macro 0.7708/MCC 0.5825. The protocol is not identical, so this comparison is directional only. Decision: `reject_candidate`. Repeated MASTER decision metrics do not establish a robust improvement over the protected OOF baseline; protocol differences prevent promotion, and worst-panel/PAH performance requires caution.

The protected LightGBM baseline remains final. HistGradientBoosting remains exploratory. No official hidden-test metric is claimed.

## Reproduce

`python scripts/v3_stress_validate_candidate.py --data-dir teknofest2026_artificialintelligenceinhealtcare-main --output-dir reports/v3/stress_validation --model hist_gradient_boosting --feature-set v3_safe_minimal --n-splits 5 --seeds 42,2026,777,123,999 --threshold-strategy validation_f1 --compare-baseline yes`
