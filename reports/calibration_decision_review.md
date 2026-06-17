# Calibration Decision Review

Calibration was reviewed as a probability-quality improvement, not as an automatic final-model selection criterion.

Isotonic calibration improved MASTER Brier score, but the final competition objective prioritizes F1-macro, MCC, PR-AUC, and panel-unique generalization. Calibration is therefore reported only and not used in the final selected decision model.

| calibration_method   |   brier_score |   log_loss |   f1_macro |      mcc |   panel_f1_macro |   panel_mcc | decision                          |
|:---------------------|--------------:|-----------:|-----------:|---------:|-----------------:|------------:|:----------------------------------|
| none                 |      0.172732 |   0.524124 |   0.756388 | 0.516937 |         0.756545 |    0.564707 | selected_for_final_decision_model |
| sigmoid              |      0.145517 |   0.458323 |   0.756388 | 0.516937 |         0.756545 |    0.564707 | reported_only                     |
| isotonic             |      0.141613 |   0.46347  |   0.754085 | 0.513844 |         0.756545 |    0.564707 | rejected_for_panel_tradeoff       |
