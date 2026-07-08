# Hyperparameter Audit

The audited LightGBM uses a conservative, saved configuration at `configs/audited_final_model_config.json`. No new exhaustive hyperparameter search was run in this pass, avoiding repeated tuning on the same small validation data. The legacy Optuna record remains non-authoritative for audited selection.
