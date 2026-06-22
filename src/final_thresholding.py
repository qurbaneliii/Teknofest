from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medical_metrics import compute_medical_metrics


DEFAULT_GRID = np.round(np.arange(0.01, 0.991, 0.005), 3)


def threshold_grid(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    rows = []
    for threshold in (thresholds if thresholds is not None else DEFAULT_GRID):
        rows.append(compute_medical_metrics(y, prob, float(threshold)))
    return pd.DataFrame(rows)


def select_threshold_candidates(grid: pd.DataFrame) -> pd.DataFrame:
    """Return all clinically meaningful threshold policies from an OOF grid."""
    if grid.empty:
        return pd.DataFrame()

    def choose(name: str, frame: pd.DataFrame, column: str, ascending: bool = False) -> dict[str, object]:
        if frame.empty:
            return {"threshold_strategy": name, "eligible": False}
        ordered = frame.sort_values([column, "specificity", "pathogenic_recall"], ascending=[ascending, False, False])
        row = ordered.iloc[0].to_dict()
        row["threshold_strategy"] = name
        row["eligible"] = True
        return row

    candidates = [
        choose("max_f1_macro", grid, "f1_macro"),
        choose("max_mcc", grid, "mcc"),
        choose("max_balanced_accuracy", grid, "balanced_accuracy"),
        choose("max_medical_utility", grid, "medical_utility_score"),
        choose("max_clinical_safety", grid, "clinical_safety_score"),
        choose("recall_ge_0_90_best_specificity", grid[grid["pathogenic_recall"] >= 0.90], "specificity"),
        choose("recall_ge_0_85_best_mcc", grid[grid["pathogenic_recall"] >= 0.85], "mcc"),
        choose("specificity_ge_0_70_best_recall", grid[grid["specificity"] >= 0.70], "pathogenic_recall"),
    ]
    youden = grid.assign(youden_j=grid["pathogenic_recall"] + grid["specificity"] - 1.0)
    candidates.append(choose("youden_j", youden, "youden_j"))
    return pd.DataFrame(candidates)


def threshold_stability(
    fold_predictions: pd.DataFrame,
    probability_column: str,
    y_column: str = "Label",
    fold_column: str = "fold",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, frame in fold_predictions.groupby(fold_column, dropna=False):
        grid = threshold_grid(frame[y_column], frame[probability_column])
        candidates = select_threshold_candidates(grid)
        candidates["fold"] = fold
        rows.extend(candidates.to_dict("records"))
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summary = (
        detail[detail["eligible"].fillna(False)]
        .groupby("threshold_strategy", as_index=False)["threshold"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = ["threshold_strategy", "mean_threshold", "std_threshold", "min_threshold", "max_threshold"]
    return detail.merge(summary, on="threshold_strategy", how="left")


def panel_threshold_behavior(
    panel_predictions: pd.DataFrame,
    probability_column: str,
    candidates: pd.DataFrame,
    y_column: str = "Label",
    panel_column: str = "panel",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = candidates["eligible"].fillna(False) if "eligible" in candidates.columns else pd.Series(True, index=candidates.index)
    for _, candidate in candidates[eligible].iterrows():
        threshold = float(candidate["threshold"])
        for panel, frame in panel_predictions.groupby(panel_column, dropna=False):
            metric = compute_medical_metrics(frame[y_column], frame[probability_column], threshold)
            metric.update({"threshold_strategy": candidate["threshold_strategy"], "panel": panel})
            rows.append(metric)
    return pd.DataFrame(rows)


def save_threshold_outputs(
    grid: pd.DataFrame,
    candidates: pd.DataFrame,
    stability: pd.DataFrame,
    output_dir: str | Path = "reports",
    threshold_json: str | Path = "artifacts/metrics/final_threshold.json",
    selected_strategy: str = "max_medical_utility",
) -> dict[str, object]:
    reports = Path(output_dir)
    tables = reports / "tables"
    figures = reports / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    grid.to_csv(tables / "final_threshold_grid.csv", index=False)
    candidates.to_csv(tables / "final_threshold_candidates.csv", index=False)
    stability.to_csv(tables / "final_threshold_stability.csv", index=False)

    selected = candidates[candidates["threshold_strategy"].eq(selected_strategy)]
    if selected.empty:
        selected = candidates[candidates.get("eligible", True)].head(1)
    if selected.empty:
        raise ValueError("No valid threshold candidate is available.")
    chosen = selected.iloc[0].to_dict()
    Path(threshold_json).parent.mkdir(parents=True, exist_ok=True)
    Path(threshold_json).write_text(json.dumps(chosen, indent=2, default=float) + "\n", encoding="utf-8")

    plt.figure(figsize=(9, 5))
    for column, label in [
        ("f1_macro", "F1 macro"),
        ("mcc", "MCC"),
        ("medical_utility_score", "MedicalUtilityScore"),
        ("clinical_safety_score", "ClinicalSafetyScore"),
    ]:
        plt.plot(grid["threshold"], grid[column], label=label)
    plt.axvline(float(chosen["threshold"]), color="black", linestyle="--", linewidth=1, label="selected")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "final_threshold_metric_curves.png", dpi=180)
    plt.close()
    return chosen
