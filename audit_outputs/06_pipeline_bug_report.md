# Pipeline Bug Report

- High — legacy error-analysis counts can diverge from selected OOF counts. Fixed by emitting a new prediction source and recomputed confusion matrix in this audit.
- High — final model zoo preserves a saved reference rather than retraining it. Fixed for the audited candidate by independent fold-safe retraining.
- Medium — no official submission schema was found. The audited inference output remains a clearly labelled generic prediction file until the organizer template is supplied.
- Verification: `python -m pytest tests -q` (38 passed in this audit run); run this script for the full artifact verification.
