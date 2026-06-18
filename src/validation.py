"""Compatibility wrapper for validation utilities.

The implementation lives in `teknofest.validation`; this module preserves the
root-level architecture expected by the project prompt.
"""

from teknofest.validation import (
    FoldIndices,
    best_f1_macro_threshold,
    contamination_aware_folds,
    fold_assignment_frame,
    fold_summary,
    youden_j_threshold,
)

__all__ = [
    "FoldIndices",
    "best_f1_macro_threshold",
    "contamination_aware_folds",
    "fold_assignment_frame",
    "fold_summary",
    "youden_j_threshold",
]
