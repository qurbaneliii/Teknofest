# V3 Evidence Map

## Verified final evidence

1. `artifacts/predictions/final_master_cv_predictions.csv` and `artifacts/predictions/final_panel_predictions.csv` are the highest-priority prediction evidence.
2. `artifacts/metrics/final_metrics.json`, `final_model_decision.json`, and `final_threshold.json` identify the protected final selection.
3. `reports/tables/final_panel_specific_metrics.csv` supplies CFTR, KANSER, and PAH subgroup measurements.

Official test labels and hidden-test performance are not present in the repository; they are marked unavailable, not estimated. The complete file-level inventory is `evidence_inventory.csv`.

## Known evidence caveat

The worktree contains uncommitted audit and inference changes predating V3. They may be useful, but their relationship to the protected baseline is not yet established. They are therefore neither deleted nor treated as final evidence.
