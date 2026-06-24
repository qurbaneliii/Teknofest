# Leakage Audit

| severity   | location                                               | finding                                                                                                                          |
|:-----------|:-------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| High       | src/teknofest/validation.py: contamination_aware_folds | Shared MASTER variants are removed only from validation, not training; reported MASTER CV is not a clean master-only population. |
| High       | src/final_model_zoo.py                                 | The legacy reference metrics are copied from saved OOF files instead of regenerated in the final model-zoo run.                  |
| High       | artifacts/metrics/final_model_decision.json            | The decision record explicitly reports mismatched selected-OOF vs saved error-analysis confusion counts.                         |
| Medium     | src/final_inference.py                                 | Default artifact decision is legacy and submission columns are not confirmed against an official template.                       |

Applied mitigation: the audited candidate excludes all panel-shared MASTER IDs before splitting; `FeatureEngineer.fit` is called only on a fold's training rows; panels are transformed by the full training-only engineer after model selection. Variant_ID is excluded by `model_columns` and test labels are never read by inference.
