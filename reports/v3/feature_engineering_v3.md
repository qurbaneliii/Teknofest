# V3 Feature Engineering

Phase B adds an isolated, config-driven V3 feature layer. The protected baseline remains untouched. No V3 model was trained in Phase B and no metric improvement is claimed.

Feature sets: `v3_safe_minimal`, `v3_bio_full`, `v3_no_target_encoding`, `v3_panel_robust`, `v3_frequency_heavy`, and `v3_aa_heavy`.

Leakage prevention: `Variant_ID`, `Label`, and identifier-like fields are excluded; clipping and imputation are fitted on training data only; transform never accepts or needs labels. Target encoding is disabled in the safe-minimal and no-target-encoding sets.

AL and EK summaries provide encrypted frequency/conservation aggregates with missingness features. BA1/BS1/PM2-like fields are ACMG-inspired only, not an exact ACMG implementation. AA features use conservative property deltas and unknown indicators; CAT expansion is deliberately deferred pending Phase C panel-shift validation.

Sample real-data matrices were generated only for infrastructure verification. Next: Phase C validation redesign, with KANSER and worst-panel robustness as selection gates.
