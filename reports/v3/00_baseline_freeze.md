# V3 Baseline Freeze

- Branch: `radical-v3-robust-genomics`
- Commit: `e4aa54f14bc0d8066cb15e708939a3c72c797fb6`
- Python: `3.12.0`
- Protected model: `lightgbm_conservative_regularized`
- Threshold: `0.471` (`profile_f1_macro_opt`)
- Calibration: `none`
- Ensemble replacement: `False`

## Saved Prediction Metrics

| evaluation_split            |   roc_auc |   pr_auc |   f1_macro |      mcc |   medical_utility_score |
|:----------------------------|----------:|---------:|-----------:|---------:|------------------------:|
| MASTER_CV_saved_predictions |  0.847536 | 0.902494 |   0.776393 | 0.554754 |                0.774675 |
| panel_unique_combined       |  0.872534 | 0.825099 |   0.770808 | 0.582499 |                0.774898 |

The V3 workstream is isolated from the protected baseline artifacts. Metrics above are sourced from saved prediction audits.
