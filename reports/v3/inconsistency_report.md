# Inconsistencies

- **High**: `reports/master_prompt/` AUC-only Optuna and candidate outputs are not final selection evidence. Action: archive later.
- **Medium**: archived FP/FN case row counts differ from selected OOF confusion counts, as documented in `final_model_selection_decision.md`. Action: use only qualitative error analysis.
- **Medium**: pipeline-generated legacy reports may describe pre-selection LightGBM states. Action: prioritize saved prediction audits and final decision JSON.
