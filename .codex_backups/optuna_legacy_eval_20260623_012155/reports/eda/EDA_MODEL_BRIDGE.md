# EDA To Model Bridge

This report connects the exploratory data analysis findings to concrete modeling decisions and verification artifacts.

## Decision Bridge

| eda_finding                                                                              | implemented_response                                                                                                  | verification_artifact                                                                            |
|:-----------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| Panel-shared variants have much higher pathogenic rates than panel-unique variants.      | MASTER variants shared with panels are excluded from validation folds; panel-unique subsets are evaluated separately. | reports/tables/validation_split_diagnostics.csv; reports/tables/panel_generalization_results.csv |
| Missingness is label-associated for AL blocks and has structured availability groups.    | Feature engineering includes missingness flags, missingness PCA, non-missing counts, and leakage-safe fold fitting.   | artifacts/metrics/feature_list.json; reports/eda/tables/phase3_missingness_as_feature.csv        |
| EK_7, EK_9, EK_3, and EK7xEK9 are strong univariate pathogenicity signals.               | Main feature set preserves EK raw values and adds EK interactions/evidence aggregates.                                | reports/tables/feature_importance.csv; reports/tables/main_model_cv_results.csv                  |
| Population-frequency style AL aggregates and n_pops separate benign/pathogenic variants. | ACMG-inspired BA1/BS1/PM2/BS2 proxy features are generated and mapped to clinical evidence categories.                | reports/tables/acmg_feature_mapping.csv; reports/eda/tables/phase6_acmg_interpretation.csv       |
| No single numeric leakage suspect exceeded AUC-equivalent directional strength 0.85.     | No feature was removed solely as a leakage proxy, but validation remains contamination-aware and panel-unique.        | reports/eda/tables/phase4_leakage_suspects.csv                                                   |
| Panel-unique subsets have distribution shift and limited sample sizes.                   | Panel metrics are reported with bootstrap confidence intervals instead of only point estimates.                       | reports/tables/panel_unique_bootstrap_ci.csv                                                     |

## Feature Alignment

The table below joins the strongest EDA numeric signals with final model SHAP importance. Missing SHAP values mean the raw EDA feature was not present under the same name in the final numeric model, often because it was transformed into an engineered aggregate.

| feature      |   eda_rank |   auc_equivalent |   auc_directional_strength |     p_value |   mean_abs_shap |   shap_rank | in_final_model_shap   |
|:-------------|-----------:|-----------------:|---------------------------:|------------:|----------------:|------------:|:----------------------|
| EK_7         |          1 |         0.738042 |                   0.738042 | 5.72001e-73 |       0.766396  |           2 | True                  |
| EK7xEK9      |          2 |         0.726059 |                   0.726059 | 3.57778e-60 |       0.364752  |           6 | True                  |
| EK_9         |          3 |         0.695513 |                   0.695513 | 1.77982e-45 |       0.323353  |           8 | True                  |
| EK_3         |          4 |         0.690208 |                   0.690208 | 3.4529e-33  |       0.162187  |          30 | True                  |
| n_pops       |          5 |         0.316095 |                   0.683905 | 2.95749e-57 |       0.0200269 |         157 | True                  |
| EK_2         |          6 |         0.67377  |                   0.67377  | 1.0122e-39  |       0.237471  |          17 | True                  |
| n_nonmiss_AL |          7 |         0.336843 |                   0.663157 | 2.73369e-42 |       0.0314232 |         122 | True                  |
| AL_287       |          8 |         0.342764 |                   0.657236 | 5.30989e-17 |       0.12549   |          47 | True                  |
| AL_215       |          9 |         0.35167  |                   0.64833  | 2.05825e-12 |       0.0753244 |          70 | True                  |
| AL_26        |         10 |         0.355282 |                   0.644718 | 9.56751e-24 |       0.243799  |          15 | True                  |
| EK_4         |         11 |         0.637392 |                   0.637392 | 2.30078e-27 |       0.169916  |          29 | True                  |
| AL_186       |         12 |         0.366538 |                   0.633462 | 3.61968e-21 |       0.195598  |          21 | True                  |
| AL_251       |         13 |         0.36908  |                   0.63092  | 2.219e-12   |       0.117209  |          53 | True                  |
| AL_73        |         14 |         0.369644 |                   0.630356 | 1.39545e-15 |       0.151421  |          35 | True                  |
| max_AF       |         15 |         0.372088 |                   0.627912 | 1.31566e-27 |       0.120106  |          51 | True                  |
| AL_94        |         16 |         0.373917 |                   0.626083 | 3.63229e-13 |       0.036242  |         116 | True                  |
| AL_79        |         17 |         0.374037 |                   0.625963 | 3.82393e-13 |       0.0624748 |          80 | True                  |
| AL_184       |         18 |         0.379006 |                   0.620994 | 1.06967e-10 |       0.0785382 |          69 | True                  |
| AL_327       |         19 |         0.38091  |                   0.61909  | 1.97777e-15 |       0.160091  |          31 | True                  |
| AL_136       |         20 |         0.386709 |                   0.613291 | 1.01842e-12 |       0.126002  |          46 | True                  |
| AL_224       |         21 |         0.390054 |                   0.609946 | 1.78656e-09 |       0.050741  |          97 | True                  |
| EK_6         |         22 |         0.60949  |                   0.60949  | 4.35918e-18 |       0.280137  |          11 | True                  |
| AL_91        |         23 |         0.393959 |                   0.606041 | 9.7678e-10  |       0.138351  |          40 | True                  |
| AL_260       |         24 |         0.394473 |                   0.605527 | 7.7277e-09  |       0.0217028 |         148 | True                  |
| AL_178       |         25 |         0.398848 |                   0.601152 | 1.96312e-10 |       0.0839306 |          63 | True                  |

Figure: `reports/eda/figures/eda_model_feature_alignment.png`

## Panel Shift To Model Results

| panel   | subset   |   n |   pathogenic_rate |   overall_missing_rate |   numeric_range_mean | dataset       |   accuracy_at_saved_threshold |
|:--------|:---------|----:|------------------:|-----------------------:|---------------------:|:--------------|------------------------------:|
| CFTR    | unique   |  34 |          0.5      |               0.224546 |             0.552333 | CFTR_unique   |                      0.882353 |
| KANSER  | unique   | 142 |          0.316901 |               0.427742 |             0.647921 | KANSER_unique |                      0.71831  |
| PAH     | unique   | 117 |          0.581197 |               0.576814 |             0.487153 | PAH_unique    |                      0.769231 |

## Final Tuned Model Context

- Optuna complete trials: 101
- Best Optuna contamination-aware CV AUC: 0.855274
- F1-macro optimized threshold: 0.622612
- F1-macro at optimized threshold: 0.756898

## Conclusion

The EDA findings are directly represented in the final pipeline: contamination-aware validation handles overlap risk, missingness-derived features are preserved because missingness is predictive, EK and population-frequency signals are retained, and panel-unique performance is reported separately due to distribution shift.
