# V2 Baseline State Manifest

- Branch: `radical-v2-clean-finetune`
- Protected model: `lightgbm_conservative_regularized`
- Model: `artifacts/models/final_model.pkl`
- Preprocessor: `artifacts/preprocessors/final_preprocessor.pkl`
- Feature columns: `artifacts/models/final_model_columns.txt`
- Threshold: `0.471` / `profile_f1_macro_opt`
- Calibration: `none`; ensemble replacement: `false`
- Saved OOF predictions: `artifacts/predictions/final_master_cv_predictions.csv`
- Saved panel predictions: `artifacts/predictions/final_panel_predictions.csv`
- Verification source: `reports/tables/final_medical_metric_comparison.csv`

The legacy pipeline, report-only command, and full test suite were reproduced successfully under an artifact snapshot-and-restore guard immediately before this V2 freeze. No modeling code or protected artifact was changed.
