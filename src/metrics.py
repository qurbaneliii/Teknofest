from __future__ import annotations

from phase10_improvements import metric_row
from phase9_outputs import save_all_evaluation_metrics


def medical_score(row) -> float:
    """Combined clinical model-selection score from the improvement plan."""
    return float(
        0.20 * row["roc_auc"]
        + 0.20 * row["pr_auc"]
        + 0.20 * row["f1_macro"]
        + 0.20 * row["mcc"]
        + 0.10 * row["balanced_accuracy"]
        + 0.10 * row["recall"]
    )


__all__ = ["medical_score", "metric_row", "save_all_evaluation_metrics"]
