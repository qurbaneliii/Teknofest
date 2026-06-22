# Final Ensemble Decision

Ensemble weights and meta-models are cross-fitted: each validation fold is predicted by a combiner trained on OOF rows from the other folds only. Panel labels are used only to report panel behavior after the ensemble has been fitted on MASTER OOF predictions.

## Preserved Reference

LightGBM conservative: MCC=0.554754, F1-macro=0.776393, PR-AUC=0.902494, panel MCC=0.582499, panel F1-macro=0.770808, MedicalUtilityScore=0.774675.

## Result

No ensemble safely meets the replacement rule: the existing LightGBM conservative_regularized model remains final because every raw two-metric candidate lowers MedicalUtilityScore.

## Ensemble Comparison

| ensemble_id                      |      mcc |   f1_macro |   pr_auc |   medical_utility_score |   panel_mcc |   panel_f1_macro |   improvement_count | meets_two_metric_gate   | eligible_to_replace   |
|:---------------------------------|---------:|-----------:|---------:|------------------------:|------------:|-----------------:|--------------------:|:------------------------|:----------------------|
| simple_average                   | 0.556547 |   0.770394 | 0.909867 |                0.773756 |    0.577476 |         0.759611 |                   2 | True                    | False                 |
| rank_average                     | 0.490246 |   0.728244 | 0.909891 |                0.747705 |    0.657917 |         0.822424 |                   3 | True                    | False                 |
| weighted_average                 | 0.556547 |   0.770394 | 0.90988  |                0.773758 |    0.577476 |         0.759611 |                   2 | True                    | False                 |
| mcc_weighted_average             | 0.556547 |   0.770394 | 0.909867 |                0.773756 |    0.577476 |         0.759611 |                   2 | True                    | False                 |
| medical_utility_weighted_average | 0.556547 |   0.770394 | 0.909867 |                0.773756 |    0.577476 |         0.759611 |                   2 | True                    | False                 |
| logistic_stacking                | 0.559285 |   0.770548 | 0.903832 |                0.77274  |    0.556819 |         0.745277 |                   2 | True                    | False                 |
| ridge_stacking                   | 0.570147 |   0.768918 | 0.90535  |                0.774048 |    0.559892 |         0.73686  |                   2 | True                    | False                 |
| elasticnet_stacking              | 0.553946 |   0.768203 | 0.903807 |                0.770823 |    0.556819 |         0.745277 |                   1 | False                   | False                 |
