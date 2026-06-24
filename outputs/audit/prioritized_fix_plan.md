# Prioritized Fix Plan and Applied Status

## CRITICAL

- No unresolved critical issue remains in the audited candidate. No official test labels were found or used.

## HIGH

- Issue: legacy final reporting reused saved OOF reference predictions and had an error-analysis count mismatch.
  Evidence: legacy decision artifact reports `matches_selected_oof_confusion_counts: false`.
  Fix: regenerate a separate audited candidate and canonical metrics from its saved OOF/panel predictions.
  Impact: final contract no longer relies on stale legacy counts.
- Issue: supplied-panel overlap can contaminate MASTER-based validation.
  Fix: exclude every MASTER Variant_ID found in KANSER/PAH/CFTR before both audited train and validation folds.
  Impact: clean MASTER-only OOF population.

## MEDIUM

- Issue: threshold selection still uses OOF evidence, not a third independent validation set.
  Fix: use the median fold threshold; document it as selection evidence and retain the default-0.50 comparison.
  Impact: transparent, but not fully independent threshold validation.
- Issue: organizer submission schema is unavailable.
  Fix: output full predictions and a conservative two-column compact submission; require organizer-template confirmation.

## LOW

- Issue: prior evidence was scattered across legacy reports.
  Fix: package canonical audit, final, figure, and PDR paths under `outputs/`.
