# Applied Fixes

| path                                                        | reason                                                                                            | risk   |
|:------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:-------|
| scripts/run_teknofest_readiness_audit.py                    | Added strict audit, clean OOF regeneration, audit reports and serialized candidate                | High   |
| src/final_inference.py                                      | Audited decision is preferred by default; added explicit label-free CLI and compact-output option | Medium |
| scripts/generate_final_submission.py                        | Added audited-decision selection and CLI aliases                                                  | Low    |
| README.md / README_FINAL_INFERENCE.md                       | Documented audited training and final inference                                                   | Low    |
| configs/audited_final_model_config.json                     | Saved exact audited model contract                                                                | Low    |
| outputs/final_submission_template.csv                       | Added clearly generic two-column template; requires organizer confirmation                        | Low    |
| artifacts/models/audited_master_only_lgbm/full_model.joblib | New final-candidate artifact trained with no panel-shared MASTER IDs                              | High   |
| artifacts/predictions/audited_*                             | New saved predictions used for all audited metrics                                                | High   |

Verification command: `python scripts/run_teknofest_readiness_audit.py`.
