# Model Performance Audit

## Current Best Model
Current saved baseline model: LightGBM at F1-macro optimized threshold. Selected improved model: LightGBM `conservative_regularized` at threshold 0.4710.

## Current Validation Strategy
The pipeline uses contamination-aware cross-validation on MASTER, excluding MASTER variants shared with panels from validation folds, plus panel-unique external checks for KANSER, PAH, and CFTR.

## Current Metrics
Default threshold LightGBM: F1-macro 0.4774, MCC 0.2047, recall 0.9969, specificity 0.0714.

F1-threshold LightGBM: ROC-AUC 0.8375, PR-AUC 0.9011, F1-macro 0.7569, MCC 0.5219, recall 0.9001, specificity 0.5889.

## Audit Questions

1. Where is the model currently weak?
   The default threshold is weak: it gives very high pathogenic recall but poor specificity and low MCC. Calibration also needs caution.

2. Which metric is the biggest problem?
   At default threshold, MCC and specificity are the biggest issues. After threshold optimization, remaining weakness is moderate MCC/F1 rather than ranking ability.

3. Is the model underfitting or overfitting?
   The selected profile was chosen partly to reduce overfitting gap. See `reports/tables/overfitting_gap_analysis.csv`; hidden-test performance remains unknown.

4. Are there fold-to-fold performance instabilities?
   Threshold stability is tracked in `reports/tables/fold_threshold_stability.csv`. F1 threshold std is 0.0218.

5. Are there class-specific weaknesses?
   Yes. The default threshold over-predicts pathogenic class, hurting benign specificity. Threshold optimization improves benign specificity while keeping pathogenic recall clinically high.

6. Is Pathogenic recall strong enough?
   F1-threshold LightGBM pathogenic recall is 0.9001. Final panel-specific recalls are documented in `reports/tables/final_panel_specific_metrics.csv`.

7. Are probabilities calibrated?
   Calibration was tested. Best MASTER Brier method is `isotonic` with Brier 0.1416; calibration remains report-only because panel trade-offs were not clearly favorable.

8. Are any features suspiciously predictive?
   Leakage scan table has 0 rows. No direct label proxy was accepted as final, and Variant_ID is excluded from modeling.

9. Is validation realistic for hidden-test performance?
   It is more realistic than simple random split because it uses contamination-aware MASTER CV and panel-unique evaluation. It is still not a guarantee of hidden-test performance.

## Audit Conclusion
The model is not weak, but it should be conservatively described as moderate-to-good. The main improvement opportunity was thresholding and clinically weighted selection, not raw accuracy maximization.
