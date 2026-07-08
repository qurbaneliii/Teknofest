# V3 Model Improvement Root-Cause Diagnosis

## Evidence
All calculations use raw labeled training panels and immutable saved final OOF/panel prediction files; no official test data exists and no model was retrained.

## Findings
KANSER remains the weakest panel in saved final evidence, so MASTER-average gains alone are unsafe. The protected threshold favors pathogenic recall while accepting moderate benign specificity; near-threshold and panel-specific errors require global threshold-stability analysis rather than panel-specific deployment thresholds. Frequency, missingness, and categorical feature groups show distribution-shift risk; target encoding must be stress-tested against a no-target-encoding representation.

## V3 Improvement Decision
1. Build fold-safe `safe_minimal`, `no_target_encoding`, `frequency_heavy`, and panel-robust feature sets.
2. Rebuild AA substitution signal with explicit property deltas and unknown handling.
3. Select using KANSER and worst-panel decision metrics plus threshold stability, not ROC-AUC alone.
4. Stress-test AL/frequency and CAT/target-encoding groups for panel shift; prune any group that harms KANSER.
5. Retry only controlled regularized LightGBM, CatBoost, HistGradientBoosting, and interpretable baselines before any compact robust-objective search.

Do not repeat AUC-only Optuna, isolated threshold tuning, or raw ensemble promotion without the new panel-aware gates.
