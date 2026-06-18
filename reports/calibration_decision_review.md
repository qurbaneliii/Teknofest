# Calibration Decision Review

Calibration was reviewed as a probability-quality improvement, not as an automatic final-model selection criterion.

Isotonic calibration improved MASTER Brier score, but the final competition objective prioritizes F1-macro, MCC, PR-AUC, and panel-unique generalization. Calibration is therefore reported only and not used in the final selected decision model.

| calibration_method   |   brier_score |   log_loss |   f1_macro |      mcc |   panel_f1_macro |   panel_mcc | decision                          |
|:---------------------|--------------:|-----------:|-----------:|---------:|-----------------:|------------:|:----------------------------------|
| none                 |      0.172732 |   0.524124 |   0.756388 | 0.516937 |         0.756545 |    0.564707 | selected_for_final_decision_model |
| sigmoid              |      0.145517 |   0.458323 |   0.756388 | 0.516937 |         0.756545 |    0.564707 | reported_only                     |
| isotonic             |      0.141613 |   0.46347  |   0.754085 | 0.513844 |         0.756545 |    0.564707 | rejected_for_panel_tradeoff       |

Threshold stability is tracked separately from calibration because the selected final profile uses fold-level profile thresholds and conservative regularization. Calibration did not provide enough panel-generalization evidence to replace the uncalibrated final decision scores.

| calibration_method   |   threshold_stability_std | threshold_stability_source   | calibration_curve_quality                      | rationale                                                                                                                                |
|:---------------------|--------------------------:|:-----------------------------|:-----------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
| none                 |                 0.0422054 | conservative_regularized     | baseline_uncalibrated_curve                    | Final selection keeps uncalibrated probabilities because decision and panel metrics are the primary competition criteria.                |
| sigmoid              |                 0.0422054 | conservative_regularized     | improved_master_brier_with_panel_loss_tradeoff | Calibration is useful to report for probability quality, but not selected unless it preserves decision metrics and panel generalization. |
| isotonic             |                 0.0422054 | conservative_regularized     | best_master_brier_but_panel_loss_degrades      | Calibration is useful to report for probability quality, but not selected unless it preserves decision metrics and panel generalization. |
