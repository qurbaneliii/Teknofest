# Final Model Decision

The protected LightGBM baseline is locked as the final model.

- Model ID: `lightgbm_conservative_regularized`
- The final threshold is 0.471.
- Threshold strategy: `profile_f1_macro_opt`.
- No calibration is used.
- No ensemble replaces the final LightGBM model.

## Verified evidence

MASTER OOF: ROC-AUC 0.8475; PR-AUC 0.9025; F1-macro 0.7764; MCC 0.5548.

Panel-unique combined: ROC-AUC 0.8725; PR-AUC 0.8251; F1-macro 0.7708; MCC 0.5825.

## Candidate disposition

HistGradientBoosting remains exploratory and rejected for final replacement. Its repeated MASTER F1-macro (0.7656) and MCC (0.5397) did not establish improvement over the protected OOF profile; PAH/worst-panel behavior and threshold variation add caution. V3 candidates are evidence only, not final replacements.

## Inference and reproduction

Use `python scripts/generate_final_predictions.py --input-csv INPUT.csv --output-csv outputs/final_predictions.csv --locked-artifact-dir artifacts/final_locked`.

Use `python scripts/verify_final_locked_model.py` to recompute metrics from locked prediction evidence.

No official hidden-test metric is claimed. Hidden-test distribution shift and lack of prospective clinical validation remain limitations.
