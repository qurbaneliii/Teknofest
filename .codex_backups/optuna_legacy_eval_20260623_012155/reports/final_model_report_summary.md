# Final Model Report Summary

## Problem Definition
Binary classification of clinical genomics missense variants as Pathogenic or Benign for TEKNOFEST 2026 Healthcare AI, University level.

## Dataset Summary
MASTER has 2931 variants. KANSER, PAH, and CFTR contain 388, 372, and 111 variants. MASTER-shared variants are excluded from validation folds.

## Method Summary
The pipeline uses leakage-safe feature engineering, contamination-aware cross-validation, LightGBM-centered tabular modeling, panel-unique external checks, and SHAP explainability.

## Feature Engineering Summary
Features include missingness indicators/PCA, population allele-frequency aggregates, ACMG-inspired BA1/BS1/PM2/BS2 flags, EK interactions/evidence counts, CAT decompositions, and AA physicochemical features.

## Validation Strategy
Primary validation is 5-fold StratifiedKFold on MASTER with MASTER-shared variants removed from validation folds. Secondary validation evaluates KANSER-unique, PAH-unique, and CFTR-unique subsets.

## Model Architecture
Baselines include majority class, ACMG rule engine, and EK-only logistic regression. Main models include LightGBM plus optional CatBoost/XGBoost/ExtraTrees/LR stack. The final LightGBM parameters come from the resumable Optuna study with 101 complete trials and best contamination-aware CV AUC of 0.85527.

## Metrics Table References
See `reports/tables/all_evaluation_metrics.csv`, `baseline_results.csv`, `main_model_cv_results.csv`, `panel_generalization_results.csv`, and `threshold_results.csv`.

## Required Visualization References
Phase 9.5 visualizations are saved under `reports/figures`: `correlation_matrix_top_features.png`, confusion matrices for MASTER and panel-unique splits, `roc_curve_master.png`, `roc_curve_panel_unique.png`, `pr_curve_master.png`, `pr_curve_panel_unique.png`, `threshold_optimization.png`, `model_comparison_metrics.png`, `feature_importance_top30.png`, `class_distribution_by_dataset.png`, `missingness_by_feature_group.png`, and `error_analysis_key_features.png`.

## Explainability Summary
See `reports/tables/feature_importance.csv`, `acmg_feature_mapping.csv`, and `reports/figures/feature_importance.png`.

## Error Analysis Summary
Panel-unique false positives and false negatives are saved in `reports/tables/error_analysis.csv`; final selected-model FP/FN cases are saved in `reports/tables/final_false_positive_cases.csv` and `reports/tables/final_false_negative_cases.csv`.

## Limitations And Next Steps
Final competition performance depends on the hidden external validation set distribution.

## Phase 10 Final Selection
The final model is classified as moderate rather than weak or strong. The selected threshold strategy is `profile_f1_macro_opt` with threshold 0.471. The main technical weakness is thresholding/probability calibration rather than feature learning. See `reports/final_performance_analysis.md`, `reports/tables/final_model_selection_table.csv`, and `reports/tables/advanced_threshold_comparison.csv`.

## Phase 11 Final Audit
The final audit verifies saved predictions against the report tables, compares the previous threshold-optimized model with the conservative regularized final model, reviews calibration as reported-only, analyzes FP/FN cases, summarizes panel-specific behavior, updates feature interpretation, and preserves the model-strength statement as moderate-to-good but conservatively reported as moderate. See `reports/final_metric_verification_audit.md`, `reports/calibration_decision_review.md`, `reports/panel_specific_final_interpretation.md`, `reports/final_feature_interpretation.md`, and `reports/final_model_strength_statement.md`.

