# TEKNOFEST Model Readiness Verdict

## 1. Executive Summary

MOSTLY READY WITH MINOR RISKS. The pipeline is now auditable and can create label-free predictions, but final official submission compliance cannot be established without the organizer test/template.

## 2. Final Model

- model name: audited_master_only_lgbm
- threshold: 0.438150
- artifact path: `artifacts/models/audited_master_only_lgbm/full_model.joblib`
- inference command: `python scripts/generate_final_submission.py --input-csv path/to/test.csv --output outputs/submission.csv`

## 3. Verified Metrics

| dataset_split         |    n |   threshold |   roc_auc |   pr_auc |       f1 |   f1_macro |      mcc |   precision |   pathogenic_recall |   tn |   fp |   fn |   tp |
|:----------------------|-----:|------------:|----------:|---------:|---------:|-----------:|---------:|------------:|--------------------:|-----:|-----:|-----:|-----:|
| MASTER_ONLY_CV        | 2353 |     0.43815 |  0.847814 | 0.904085 | 0.866156 |   0.781447 | 0.563587 |    0.854897 |            0.877716 |  502 |  240 |  197 | 1414 |
| PANEL_UNIQUE_COMBINED |  293 |     0.43815 |  0.876168 | 0.825512 | 0.794521 |   0.795219 | 0.609639 |    0.716049 |            0.892308 |  117 |   46 |   14 |  116 |

## 4. Subgroup Results

| dataset_split         |    n |   threshold |   roc_auc |   pr_auc |       f1 |   f1_macro |      mcc |   precision |   pathogenic_recall |   tn |   fp |   fn |   tp |
|:----------------------|-----:|------------:|----------:|---------:|---------:|-----------:|---------:|------------:|--------------------:|-----:|-----:|-----:|-----:|
| MASTER_ONLY_CV        | 2353 |     0.43815 |  0.847814 | 0.904085 | 0.866156 |   0.781447 | 0.563587 |    0.854897 |            0.877716 |  502 |  240 |  197 | 1414 |
| CFTR_UNIQUE           |   34 |     0.43815 |  0.961938 | 0.961641 | 0.882353 |   0.882353 | 0.764706 |    0.882353 |            0.882353 |   15 |    2 |    2 |   15 |
| KANSER_UNIQUE         |  142 |     0.43815 |  0.906071 | 0.800442 | 0.743363 |   0.786886 | 0.619632 |    0.617647 |            0.933333 |   71 |   26 |    3 |   42 |
| PAH_UNIQUE            |  117 |     0.43815 |  0.821128 | 0.854474 | 0.813793 |   0.755211 | 0.520365 |    0.766234 |            0.867647 |   31 |   18 |    9 |   59 |
| PANEL_UNIQUE_COMBINED |  293 |     0.43815 |  0.876168 | 0.825512 | 0.794521 |   0.795219 | 0.609639 |    0.716049 |            0.892308 |  117 |   46 |   14 |  116 |

## 5. Critical Issues Found and Fixed

| severity   | location                                               | finding                                                                                                                          |
|:-----------|:-------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| High       | src/teknofest/validation.py: contamination_aware_folds | Shared MASTER variants are removed only from validation, not training; reported MASTER CV is not a clean master-only population. |
| High       | src/final_model_zoo.py                                 | The legacy reference metrics are copied from saved OOF files instead of regenerated in the final model-zoo run.                  |
| High       | artifacts/metrics/final_model_decision.json            | The decision record explicitly reports mismatched selected-OOF vs saved error-analysis confusion counts.                         |
| Medium     | src/final_inference.py                                 | Default artifact decision is legacy and submission columns are not confirmed against an official template.                       |

## 6. Remaining Risks

- Hidden-test distribution shift and prevalence can differ.
- Panel subgroups are limited and not prospective validation.
- Threshold selection is based on OOF evidence, not a third independent set.
- No official submission schema/test file was available.

## 7. Reproducibility

- install: `python -m pip install -r requirements.txt`
- train/validate: `python scripts/run_teknofest_readiness_audit.py`
- infer: `python scripts/generate_final_submission.py --input-csv path/to/test.csv --output outputs/submission.csv`
- outputs: `audit_outputs/`, `artifacts/models/audited_master_only_lgbm/`, `artifacts/predictions/audited_*`

## 8. PDR/Presentation Evidence Pack

Use data distribution (`data_shapes.csv`), the strict validation design, recomputed metrics, subgroup table, threshold sweep, confusion matrices, and newly generated audited-model feature importance. Do not use legacy SHAP/metric figures as audited-model evidence.
