from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_metrics import compute_medical_metrics


DEFAULT_MASTER = PROJECT_ROOT / "artifacts" / "predictions" / "final_master_cv_predictions.csv"
DEFAULT_PANEL = PROJECT_ROOT / "artifacts" / "predictions" / "final_panel_predictions.csv"
DEFAULT_SELECTION = PROJECT_ROOT / "reports" / "tables" / "final_model_selection_table.csv"
DEFAULT_OUTPUT_TABLE = PROJECT_ROOT / "reports" / "tables" / "final_medical_metric_comparison.csv"
DEFAULT_OUTPUT_SUMMARY = PROJECT_ROOT / "reports" / "final_medical_metric_summary.md"
REQUIRED_COLUMNS = {"Variant_ID", "Label", "score", "threshold", "profile", "threshold_strategy"}
MATCH_METRICS = ("roc_auc", "pr_auc", "f1_macro", "mcc")


def _read_predictions(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {name} prediction file is missing: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{name} prediction file is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} prediction file contains no rows: {path}")
    if frame["Label"].isna().any() or frame["score"].isna().any() or frame["threshold"].isna().any():
        raise ValueError(f"{name} prediction file has missing Label, score, or threshold values.")
    if not set(pd.to_numeric(frame["Label"], errors="raise").unique()).issubset({0, 1}):
        raise ValueError(f"{name} labels must be binary benign (0) / pathogenic (1).")
    return frame


def _single_threshold(frame: pd.DataFrame, name: str) -> float:
    thresholds = pd.to_numeric(frame["threshold"], errors="raise").drop_duplicates()
    if len(thresholds) != 1:
        raise ValueError(f"{name} prediction file has multiple thresholds; audit requires one immutable saved threshold.")
    return float(thresholds.iloc[0])


def _metric_row(frame: pd.DataFrame, evaluation_split: str, source_path: Path) -> dict[str, object]:
    threshold = _single_threshold(frame, evaluation_split)
    metrics = compute_medical_metrics(frame["Label"], frame["score"], threshold)
    metrics.update(
        {
            "model_name": "lightgbm",
            "profile": str(frame["profile"].iloc[0]),
            "threshold_strategy": str(frame["threshold_strategy"].iloc[0]),
            "evaluation_split": evaluation_split,
            "n_samples": int(len(frame)),
            "prediction_source": str(source_path.relative_to(PROJECT_ROOT)),
        }
    )
    return metrics


def _selected_reference(selection_path: Path) -> pd.Series:
    if not selection_path.exists():
        raise FileNotFoundError(f"Required final selection table is missing: {selection_path}")
    selection = pd.read_csv(selection_path)
    required = {"selected_as_final", *MATCH_METRICS}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"Final selection table is missing columns: {sorted(missing)}")
    selected = selection[selection["selected_as_final"].astype(str).str.lower().eq("true")]
    if len(selected) != 1:
        raise ValueError("Final selection table must contain exactly one selected final model row.")
    return selected.iloc[0]


def build_audit(master: pd.DataFrame, panel: pd.DataFrame, master_path: Path, panel_path: Path) -> pd.DataFrame:
    rows = [_metric_row(master, "MASTER_CV_saved_predictions", master_path)]
    if "dataset" not in panel.columns:
        raise ValueError("Panel prediction file is missing required dataset column.")
    for dataset, group in panel.groupby("dataset", sort=True):
        rows.append(_metric_row(group, str(dataset), panel_path))
    rows.append(_metric_row(panel, "panel_unique_combined", panel_path))
    return pd.DataFrame(rows)


def write_summary(audit: pd.DataFrame, reference: pd.Series, output_path: Path) -> bool:
    master = audit[audit["evaluation_split"].eq("MASTER_CV_saved_predictions")].iloc[0]
    matches = all(np.isclose(float(master[metric]), float(reference[metric]), rtol=0.0, atol=1e-9) for metric in MATCH_METRICS)
    comparison = pd.DataFrame(
        {
            "metric": MATCH_METRICS,
            "saved_prediction_audit": [float(master[metric]) for metric in MATCH_METRICS],
            "phase10_selected_reference": [float(reference[metric]) for metric in MATCH_METRICS],
        }
    )
    comparison["difference"] = comparison["saved_prediction_audit"] - comparison["phase10_selected_reference"]
    table = audit[
        [
            "evaluation_split",
            "n_samples",
            "threshold",
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "pathogenic_recall",
            "specificity",
            "f1_macro",
            "mcc",
            "brier_score",
            "medical_utility_score",
            "clinical_safety_score",
        ]
    ]
    text = [
        "# Final Medical Metric Summary",
        "",
        "This audit recomputes clinical metrics from the saved final prediction artifacts. It does not retrain a model, modify probabilities, or change the saved threshold.",
        "",
        "## Saved Final Threshold",
        "",
        f"Threshold: `{float(master['threshold']):.6f}` using `{master['threshold_strategy']}`.",
        "",
        "## MASTER Consistency Check",
        "",
        "The saved MASTER predictions " + ("match" if matches else "do not match") + " the existing selected Phase 10 ROC-AUC, PR-AUC, F1-macro, and MCC values within 1e-9.",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Medical Metrics By Evaluation Split",
        "",
        table.to_markdown(index=False),
        "",
        "MedicalUtilityScore is the specified weighted combination of ROC-AUC, PR-AUC, F1-macro, MCC, balanced accuracy, pathogenic recall, and specificity. ClinicalSafetyScore increases the relative importance of pathogenic recall and calibration quality.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute medical classification metrics from saved final predictions.")
    parser.add_argument("--master-predictions", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--panel-predictions", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--selection-table", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    args = parser.parse_args()

    master = _read_predictions(args.master_predictions, "MASTER")
    panel = _read_predictions(args.panel_predictions, "panel")
    audit = build_audit(master, panel, args.master_predictions, args.panel_predictions)
    reference = _selected_reference(args.selection_table)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_table, index=False)
    matches = write_summary(audit, reference, args.output_summary)

    print(f"Medical metric table: {args.output_table.resolve()}")
    print(f"Medical metric summary: {args.output_summary.resolve()}")
    print(f"MASTER metrics match selected Phase 10 reference: {matches}")


if __name__ == "__main__":
    main()
