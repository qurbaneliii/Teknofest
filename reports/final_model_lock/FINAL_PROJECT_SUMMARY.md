# Final Project Summary

This TEKNOFEST 2026 Healthcare AI project predicts pathogenic versus benign missense variants from competition genomic tables (MASTER, KANSER, CFTR, PAH). It is a decision-support research model.

Leakage prevention excludes `Variant_ID` and labels from model features. Features include AL/frequency, EK/conservation, categorical metadata, amino-acid substitution, and missingness representations.

The final model locked is `lightgbm_conservative_regularized` at threshold 0.471, with no calibration and no ensemble replacement. MASTER OOF F1-macro/MCC are 0.7764/0.5548; panel-unique F1-macro/MCC are 0.7708/0.5825.

V3 HistGradientBoosting and all local-holdout candidates are rejected/exploratory evidence. Inference is ready through the locked CLI. No official hidden-test metric claimed; official labels are unavailable locally. Final submission recommendation: use the locked baseline only when an organizer-format unlabeled test CSV is supplied.
