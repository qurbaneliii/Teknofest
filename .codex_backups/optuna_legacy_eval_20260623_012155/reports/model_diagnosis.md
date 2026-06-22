# Model Diagnosis

Model strength: moderate

Main issue: thresholding

Evidence: MASTER lightgbm at F1 threshold: ROC-AUC=0.8375, F1-macro=0.7569, MCC=0.5219. Panel combined: ROC-AUC=0.8681, F1-macro=0.7565, MCC=0.5647.

Recommended next actions:
- Use the validation-selected F1-macro threshold for final reporting.
- Compare default, F1-macro, Youden-J, and MCC-aware thresholds.
