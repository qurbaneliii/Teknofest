from __future__ import annotations

import math


def robust_genomics_score(master: dict, panel_rows: list[dict]) -> float:
    """Exploratory score for one internal protocol; never a baseline replacement gate alone."""
    panels = [r for r in panel_rows if r.get("evaluation_split") in {"KANSER", "CFTR", "PAH"}]
    if not panels:
        return float("nan")
    mean_f1 = sum(r["f1_macro"] for r in panels) / len(panels)
    mean_mcc = sum(r["mcc"] for r in panels) / len(panels)
    worst_f1 = min(r["f1_macro"] for r in panels)
    worst_mcc = min(r["mcc"] for r in panels)
    return (0.16 * master["f1_macro"] + 0.16 * master["mcc"] + 0.12 * master["pr_auc"] + 0.08 * master["roc_auc"] + 0.14 * mean_f1 + 0.14 * mean_mcc + 0.08 * worst_f1 + 0.08 * worst_mcc + 0.04)
