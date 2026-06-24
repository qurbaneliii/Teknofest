# TEKNOFEST Master Prompt Implementation Report

## Critical Checklist

| requirement                          | completed   | evidence_or_limitation                                                                                         |
|:-------------------------------------|:------------|:---------------------------------------------------------------------------------------------------------------|
| AL_185 dropped                       | True        | AL_186 kept and AL_185 absent after preparation.                                                               |
| Overlap maps computed                | True        | overlap_summary.csv written.                                                                                   |
| Missingness flags before imputation  | True        | FeatureEngineer creates flags before model matrix imputation.                                                  |
| n_pops from AL_1:26                  | True        | FeatureEngineer.al_raw is AL_1..AL_26.                                                                         |
| EK negative values preserved         | True        | No abs() is used on EK columns.                                                                                |
| EK_3 left as NaN                     | True        | Tree models consume NaN natively; LR-only baseline imputes only its local matrix.                              |
| CAT_1 training-fold target encoding  | True        | FeatureEngineer is fit on each CV train fold.                                                                  |
| CAT_1 multipop flag                  | True        | cat1_multipop implemented.                                                                                     |
| BLOSUM62 scores                      | True        | blosum62_approx implemented.                                                                                   |
| AA physicochemical classes           | True        | aa1_class/aa2_class and binary flags implemented.                                                              |
| CV excludes MASTER-shared validation | True        | contamination_aware_folds implemented.                                                                         |
| Panel-unique tests                   | True        | final_panel_unique_predictions.csv written.                                                                    |
| Dataset-specific class weights       | True        | Models use class_weight/scale_pos_weight; panel-specific evaluation is separate and not used for training.     |
| Threshold optimization               | True        | Ablation and CV helpers report default and F1-opt thresholds.                                                  |
| Optuna 100-trial study saved         | False       | 1/100 complete trials in reports\master_prompt\lgbm_optuna_trials_resumable.csv; convergence plot exists=True. |
| SHAP plots                           | True        | SHAP report artifacts detected.                                                                                |
| All 10 ablations                     | True        | ablation_summary.csv written with ABL-01 through ABL-10 rows.                                                  |
| Bootstrap CI n=1000                  | True        | Bootstrap rows written with n=1000.                                                                            |
| McNemar test                         | True        | statistical_tests.csv written.                                                                                 |
| Calibration curve and ECE            | True        | Calibration CSV and PNG files written.                                                                         |
| Global random seeds                  | True        | np.random.seed(42) set; model constructors use random_state=42.                                                |
| requirements.txt saved               | True        | Dependency file exists.                                                                                        |
| Final probabilities and labels       | True        | final_panel_unique_predictions.csv written.                                                                    |
| L0 ensemble stack                    | True        | l0_stack_oof_predictions.csv and l0_model_summary.csv written.                                                 |

## Ablation Summary

| ablation                              | description                                                      | threshold_name          |   CV_AUC |   CV_F1macro |   vs_Full_delta_AUC |   vs_Full_delta_F1 | Conclusion                                      |
|:--------------------------------------|:-----------------------------------------------------------------|:------------------------|---------:|-------------:|--------------------:|-------------------:|:------------------------------------------------|
| ABL-01_EK_cols_only                   | EK raw columns only                                              | default_0.5             | 0.724734 |     0.406411 |        -0.114122    |       -0.358417    | Compare delta against full proposed model.      |
| ABL-01_EK_cols_only                   | EK raw columns only                                              | f1_macro_opt            | 0.724734 |     0.661108 |        -0.114122    |       -0.10372     | Compare delta against full proposed model.      |
| ABL-02_AL_cols_only                   | AL raw columns only                                              | default_0.5             | 0.805148 |     0.509684 |        -0.0337074   |       -0.255144    | Compare delta against full proposed model.      |
| ABL-02_AL_cols_only                   | AL raw columns only                                              | f1_macro_opt            | 0.805148 |     0.747452 |        -0.0337074   |       -0.0173751   | Compare delta against full proposed model.      |
| ABL-03_engineered_only_no_raw_AL_EK   | Engineered features only, excluding raw AL/EK                    | default_0.5             | 0.8271   |     0.527517 |        -0.0117552   |       -0.237311    | Compare delta against full proposed model.      |
| ABL-03_engineered_only_no_raw_AL_EK   | Engineered features only, excluding raw AL/EK                    | f1_macro_opt            | 0.8271   |     0.76086  |        -0.0117552   |       -0.00396682  | Compare delta against full proposed model.      |
| ABL-04_all_zero_impute_no_miss_flags  | All numeric features, zero imputation, missingness flags removed | default_0.5             | 0.825396 |     0.465125 |        -0.0134596   |       -0.299702    | Compare delta against full proposed model.      |
| ABL-04_all_zero_impute_no_miss_flags  | All numeric features, zero imputation, missingness flags removed | f1_macro_opt            | 0.825396 |     0.751918 |        -0.0134596   |       -0.012909    | Compare delta against full proposed model.      |
| ABL-05_all_with_miss_flags            | Full proposed feature set                                        | default_0.5             | 0.838855 |     0.477221 |         2.22045e-16 |       -0.287607    | Full proposed feature set reference.            |
| ABL-05_all_with_miss_flags            | Full proposed feature set                                        | f1_macro_opt            | 0.838855 |     0.764827 |         2.22045e-16 |        0           | Full proposed feature set reference.            |
| ABL-06_no_EK_interactions             | Full feature set without EK interaction terms                    | default_0.5             | 0.840743 |     0.481011 |         0.0018879   |       -0.283816    | Compare delta against full proposed model.      |
| ABL-06_no_EK_interactions             | Full feature set without EK interaction terms                    | f1_macro_opt            | 0.840743 |     0.766685 |         0.0018879   |        0.00185791  | Compare delta against full proposed model.      |
| ABL-07_no_AA_chemistry                | Full feature set without amino-acid chemistry features           | default_0.5             | 0.837096 |     0.512279 |        -0.00175962  |       -0.252548    | Compare delta against full proposed model.      |
| ABL-07_no_AA_chemistry                | Full feature set without amino-acid chemistry features           | f1_macro_opt            | 0.837096 |     0.762913 |        -0.00175962  |       -0.00191408  | Compare delta against full proposed model.      |
| ABL-08_no_CAT1_decomposition          | Full feature set without CAT_1 decomposition features            | default_0.5             | 0.840141 |     0.487605 |         0.00128548  |       -0.277223    | Compare delta against full proposed model.      |
| ABL-08_no_CAT1_decomposition          | Full feature set without CAT_1 decomposition features            | f1_macro_opt            | 0.840141 |     0.765511 |         0.00128548  |        0.000683281 | Compare delta against full proposed model.      |
| ABL-09_single_LGBM_vs_extra_trees     | Single LightGBM compared with ExtraTrees baseline                | extra_trees_default_0.5 | 0.829987 |     0.741613 |        -0.00886852  |       -0.0232146   | Compare delta against full proposed model.      |
| ABL-10_default_vs_optimized_threshold | Default 0.5 threshold vs optimized F1 threshold                  | comparison              | 0.838855 |     0.764827 |         0           |        0.287607    | F1 threshold optimization effect on full model. |

## Statistical Tests

| test                               |   b01_a_correct_b_wrong |   b10_a_wrong_b_correct |   statistic |     p_value | status   |   lightgbm_mean_auc |   catboost_mean_auc |   full_model_auc |   baseline_auc |   auc_delta |   z_statistic |
|:-----------------------------------|------------------------:|------------------------:|------------:|------------:|:---------|--------------------:|--------------------:|-----------------:|---------------:|------------:|--------------:|
| McNemar LightGBM vs ACMG           |                     518 |                     897 |     100.978 | 9.30065e-24 | nan      |          nan        |          nan        |       nan        |     nan        |  nan        |      nan      |
| Wilcoxon LightGBM vs CatBoost      |                     nan |                     nan |       0     | 0.0625      | computed |            0.839909 |            0.846457 |       nan        |     nan        |  nan        |      nan      |
| DeLong full LightGBM vs EK-only LR |                     nan |                     nan |     nan     | 1.32379e-31 | computed |          nan        |          nan        |         0.838969 |       0.705244 |    0.133725 |       11.6968 |

## Calibration

| dataset       |      ece |   brier_score |
|:--------------|---------:|--------------:|
| CFTR_unique   | 0.113888 |     0.0718801 |
| KANSER_unique | 0.274786 |     0.252091  |
| PAH_unique    | 0.215882 |     0.208608  |
