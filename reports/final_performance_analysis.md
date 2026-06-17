# Final Performance Analysis

## Interpretation
The model should be classified as moderate, not weak and not strong. ROC-AUC and panel-unique generalization are acceptable, but MASTER F1-macro 0.7764 and MCC 0.5548 remain moderate-to-good rather than clearly excellent.

## Why Default Threshold Failed
The default 0.5 threshold produced F1-macro 0.4774 and MCC 0.2047. This confirms that ranking ability is acceptable while raw probability decision behavior needs thresholding and calibration diagnostics.

## Best Threshold Strategy
The final threshold strategy is `profile_f1_macro_opt` with threshold 0.471000; the selected profile is `conservative_regularized`. MASTER CV metrics are ROC-AUC 0.8475, PR-AUC 0.9025, F1-macro 0.7764, and MCC 0.5548. Panel-combined metrics are ROC-AUC 0.8725, PR-AUC 0.8251, F1-macro 0.7708, and MCC 0.5825.

## Calibration
Calibration was tested with no calibration, sigmoid/Platt scaling, and isotonic calibration using fold-safe OOF calibration. The best Brier method on MASTER was `isotonic` with Brier 0.1416. Calibration is reported as a trade-off and is not blindly selected unless it improves decision metrics as well as probability loss.

## Hyperparameter Diagnostics
Controlled LightGBM profiles were evaluated where requested. The best profile by CV F1-macro in this pass was `conservative_regularized` with CV F1-macro 0.7764. The final selection table compares this improvement against the saved threshold-optimized LightGBM using panel metrics, overfitting gap, and threshold stability.

## Feature Ablation
Feature ablations from the saved experiment table are summarized in `reports/tables/feature_group_ablation_results.csv`. Rows that were not retrained in this surgical pass are explicitly marked rather than fabricated.

## Remaining Limitations
Hidden competition-set performance cannot be verified locally. Some feature-group ablations remain queued for a full retraining pass because this update intentionally focused on thresholding, calibration, stability, overfitting diagnostics, and panel-specific errors.


## Final Audit Addendum
Final metric verification, calibration decision review, panel-specific interpretation, and error-group analyses were completed in Phase 11. The current final profile is `conservative_regularized` with threshold strategy `profile_f1_macro_opt` and threshold 0.471000. Compared with the previous threshold-optimized model, MASTER F1-macro improved by +0.0200, MASTER MCC improved by +0.0378, panel F1-macro improved by +0.0143, and panel MCC improved by +0.0178.

Calibration remains reported-only: it improves probability loss on MASTER, but final selection prioritizes decision metrics, panel behavior, overfitting gap, and threshold stability.

# Final Model Strength Statement

The final selected model is best described as **moderate-to-good, conservatively reported as moderate**.

It should not be described as weak because the final metrics are functional for the task: MASTER ROC-AUC 0.8475, MASTER PR-AUC 0.9025, MASTER F1-macro 0.7764, MASTER MCC 0.5548, panel ROC-AUC 0.8725, panel PR-AUC 0.8251, panel F1-macro 0.7708, and panel MCC 0.5825.

It should not be described as strong because F1-macro and MCC, while improved, are not clearly excellent across all validation views. The model is therefore suitable for an honest competition report as a reproducible, leakage-aware, clinically motivated baseline with meaningful external-panel behavior and remaining room for improvement.

