"""Compatibility wrapper for feature engineering utilities.

The implementation lives in `teknofest.features`; this module preserves the
root-level architecture expected by the project prompt.
"""

from teknofest.features import FeatureEngineer, aa_class, blosum62_approx, detect_binary_al_cols

__all__ = [
    "FeatureEngineer",
    "aa_class",
    "blosum62_approx",
    "detect_binary_al_cols",
]
