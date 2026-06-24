# Final Test Performance Report

## Executive Summary

No official test CSV was found, so the protected model was not run on training data as a substitute. Official test labels are not available, so true accuracy, F1, MCC, ROC-AUC, PR-AUC, sensitivity, and specificity cannot be computed on the official test set. The report therefore provides inference validity, confidence analysis, schema checks, distribution-shift analysis, and validation/panel-based performance evidence when an official test file is provided.

## Final Model

`lightgbm_conservative_regularized`, threshold `0.471`, strategy `profile_f1_macro_opt`, calibration `none`; the ensemble did not replace LightGBM.

## Validation Recap

Saved MASTER CV: ROC-AUC 0.8475, PR-AUC 0.9025, F1-macro 0.7764, MCC 0.5548, MedicalUtilityScore 0.7747. Saved panel-unique combined: F1-macro 0.7708 and MCC 0.5825.

## Conclusion

The model's true official test performance cannot be measured locally because test labels are unavailable. Based on saved validation and panel-unique evaluations, the model is moderate-to-good, leakage-aware, and stable enough for competition submission, but hidden-test performance remains uncertain.

See `reports/test_evaluation/test_data_discovery.md` for expected file names and locations.
