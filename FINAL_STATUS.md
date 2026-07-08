# Final Status

- Final model: `lightgbm_conservative_regularized`
- Threshold: `0.471` (`profile_f1_macro_opt`)
- MASTER OOF: F1-macro 0.7764; MCC 0.5548
- Panel-unique: F1-macro 0.7708; MCC 0.5825
- Locked artifacts: `artifacts/final_locked/`
- Verify: `python scripts/verify_final_locked_model.py`
- Predict: `python scripts/generate_final_predictions.py --input-csv INPUT.csv --output-csv outputs/final_predictions.csv --locked-artifact-dir artifacts/final_locked`
- Official test status: no official test CSV/labels available locally; no official hidden-test metric claim.
- V3 HistGradientBoosting is rejected/exploratory; the protected baseline remains final.
