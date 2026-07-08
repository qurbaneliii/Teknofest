# V3 Baseline Freeze

- Timestamp (UTC): 2026-06-24T16:40:46.438326+00:00
- Branch: `radical-v3-robust-genomics`
- Commit: `e4aa54f14bc0d8066cb15e708939a3c72c797fb6`
- Python: `3.12.0`
- Platform: `Windows-11-10.0.26200-SP0`
- Worktree was already dirty before V3. Its changes are deliberately preserved and are listed by `git status --short`.

## Protected baseline verified from saved artifacts

- Model: `lightgbm_conservative_regularized`
- Threshold: `0.471` (`profile_f1_macro_opt`)
- MASTER CV: ROC-AUC 0.8475; PR-AUC 0.9025; F1-macro 0.7764; MCC 0.5548.
- Panel-unique combined: ROC-AUC 0.8725; PR-AUC 0.8251; F1-macro 0.7708; MCC 0.5825.

The stored values exactly match the protected baseline supplied in the V3 brief. No model artifact was modified.
