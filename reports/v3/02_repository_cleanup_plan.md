# V3 Repository Cleanup Plan

No cleanup was executed. The plan is intentionally reversible.

## Stop condition triggered

The branch contains pre-existing uncommitted source edits and untracked generated audit/output trees. Moving them without their author’s confirmation could sever a currently active readiness-audit workflow or hide evidence needed for reproducibility. This meets the Phase 2 instruction to stop when uncertainty exists.

## Proposed archive root

`archive/previous_results/{eda,feature_engineering,validation,training,explainability,optuna_old,rejected_models,old_notebooks,old_reports,old_figures,old_metrics,old_presentations,old_pdr_drafts,misc}`

`file_classification.csv` records every discovered file. `archive_plan.csv` is a proposal only; every row has `NOT_EXECUTED` status. Protected material is never marked for deletion.
