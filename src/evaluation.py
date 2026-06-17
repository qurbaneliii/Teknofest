from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix, f1_score

from teknofest.validation import best_f1_macro_threshold, youden_j_threshold


def _savefig(path: Path, dpi: int = 170) -> None:
    try:
        plt.savefig(path, dpi=dpi)
    except OSError:
        if path.exists():
            path.unlink()
        plt.savefig(path, dpi=dpi)


def threshold_results(y_true, y_score) -> pd.DataFrame:
    f1_thr, _ = best_f1_macro_threshold(y_true, y_score)
    youden_thr, _ = youden_j_threshold(y_true, y_score)
    rows = []
    for name, thr in [("default_0.5", 0.5), ("f1_macro_opt", f1_thr), ("youden_j", youden_thr)]:
        pred = (y_score >= thr).astype(int)
        rows.append({"threshold_name": name, "threshold": float(thr), "f1_macro": float(f1_score(y_true, pred, average="macro"))})
    return pd.DataFrame(rows)


def save_evaluation_figures(y_true, y_score, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Benign", "Pathogenic"])
    plt.yticks([0, 1], ["Benign", "Pathogenic"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("MASTER OOF confusion matrix")
    plt.tight_layout()
    _savefig(out / "confusion_matrix_master.png")
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    _savefig(out / "roc_curve_master.png")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    _savefig(out / "pr_curve_master.png")
    plt.close()


def diagnose_model_performance(metrics: pd.DataFrame) -> dict[str, object]:
    """Classify model quality and recommend next actions from computed metrics."""
    if metrics.empty:
        return {
            "model_strength": "weak",
            "main_issue": "underfitting",
            "recommended_next_actions": ["Generate evaluation metrics before diagnosis."],
            "evidence": "No metrics were provided.",
        }

    master = metrics[
        metrics["evaluation_split"].eq("MASTER_CV")
        & metrics["model_name"].eq("lightgbm")
    ].copy()
    panel = metrics[
        metrics["evaluation_split"].astype(str).str.contains("panel_unique_combined", case=False, na=False)
        & metrics["model_name"].eq("lightgbm")
    ].copy()

    def _row(frame: pd.DataFrame, threshold: str) -> pd.Series:
        subset = frame[frame["threshold_type"].eq(threshold)]
        if subset.empty:
            return frame.iloc[0] if not frame.empty else pd.Series(dtype=object)
        return subset.iloc[0]

    selected = _row(master, "f1_macro_opt")
    default = _row(master, "default_0.5")
    panel_row = panel.iloc[0] if not panel.empty else pd.Series(dtype=object)

    roc_auc = float(selected.get("roc_auc", 0.0) or 0.0)
    f1_macro = float(selected.get("f1_macro", 0.0) or 0.0)
    mcc = float(selected.get("mcc", 0.0) or 0.0)
    default_f1 = float(default.get("f1_macro", f1_macro) or f1_macro)
    default_mcc = float(default.get("mcc", mcc) or mcc)
    panel_auc = float(panel_row.get("roc_auc", roc_auc) or roc_auc)
    panel_f1 = float(panel_row.get("f1_macro", f1_macro) or f1_macro)
    panel_mcc = float(panel_row.get("mcc", mcc) or mcc)

    if roc_auc >= 0.85 and f1_macro >= 0.78 and mcc >= 0.55:
        strength = "strong"
    elif roc_auc >= 0.75 and f1_macro >= 0.65 and mcc >= 0.35:
        strength = "moderate"
    else:
        strength = "weak"

    actions: list[str] = []
    if panel_auc + 0.08 < roc_auc or panel_f1 + 0.08 < f1_macro:
        issue = "generalization_gap"
        actions.extend(
            [
                "Compare feature importance for panel-specific artifacts.",
                "Prefer robust AL/EK/AA/CAT features with stable panel behavior.",
                "Report KANSER, PAH, and CFTR panel-unique metrics separately.",
            ]
        )
    elif f1_macro - default_f1 > 0.05 or mcc - default_mcc > 0.05:
        issue = "thresholding"
        actions.extend(
            [
                "Use the validation-selected F1-macro threshold for final reporting.",
                "Compare default, F1-macro, Youden-J, and MCC-aware thresholds.",
            ]
        )
    elif roc_auc >= 0.80 and (f1_macro < 0.65 or mcc < 0.35):
        issue = "imbalance"
        actions.extend(
            [
                "Tune scale_pos_weight and threshold jointly.",
                "Inspect false positives and false negatives by class.",
            ]
        )
    elif roc_auc < 0.75:
        issue = "underfitting"
        actions.extend(
            [
                "Increase model capacity carefully.",
                "Check whether useful feature groups were dropped accidentally.",
            ]
        )
    else:
        issue = "acceptable"
        actions.append("Keep the current configuration and document residual risks.")

    evidence = (
        f"MASTER lightgbm at F1 threshold: ROC-AUC={roc_auc:.4f}, "
        f"F1-macro={f1_macro:.4f}, MCC={mcc:.4f}. "
        f"Panel combined: ROC-AUC={panel_auc:.4f}, F1-macro={panel_f1:.4f}, "
        f"MCC={panel_mcc:.4f}."
    )
    return {
        "model_strength": strength,
        "main_issue": issue,
        "recommended_next_actions": actions,
        "evidence": evidence,
    }
