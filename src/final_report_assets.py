from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from final_inference import load_final_decision
from teknofest.data_prep import PreparedData, overlap_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _selected_predictions(decision: dict[str, object]) -> tuple[pd.DataFrame, str]:
    if decision["model_kind"] == "ensemble":
        path = PROJECT_ROOT / "artifacts" / "predictions" / "final_ensemble_oof_predictions.csv"
    elif decision["model_id"] == "existing_phase10_final":
        path = PROJECT_ROOT / "artifacts" / "predictions" / "final_master_cv_predictions.csv"
        return pd.read_csv(path), "score"
    else:
        path = PROJECT_ROOT / "artifacts" / "predictions" / "model_zoo_oof_predictions.csv"
    return pd.read_csv(path), f"proba__{decision['model_id']}"


def _save_confusion(y: np.ndarray, probabilities: np.ndarray, threshold: float, path: Path) -> None:
    matrix = confusion_matrix(y, probabilities >= threshold, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Benign", "Pathogenic"])
    plt.yticks([0, 1], ["Benign", "Pathogenic"])
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)


def generate_final_report_assets(prepared: PreparedData, output_dir: str | Path = "reports/final_report_assets") -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decision = load_final_decision()
    predictions, probability_column = _selected_predictions(decision)
    if probability_column not in predictions.columns:
        raise ValueError(f"Selected model probabilities are missing: {probability_column}")
    y = predictions["Label"].to_numpy(dtype=int)
    probability = predictions[probability_column].to_numpy(dtype=float)
    threshold = float(decision["threshold"])

    overlap_summary(prepared).to_csv(output / "dataset_summary_table.csv", index=False)
    validation_table = pd.DataFrame(
        [
            {"layer": "Primary", "data": "MASTER unique variants", "method": "Contamination-aware 5-fold StratifiedKFold", "purpose": "OOF model selection"},
            {"layer": "Repeated", "data": "MASTER unique variants", "method": "Seeds 13, 21, 42, 77, 101", "purpose": "Stability"},
            {"layer": "External", "data": "KANSER/PAH/CFTR unique", "method": "Panel-unique evaluation", "purpose": "Distribution-shift check"},
        ]
    )
    validation_table.to_csv(output / "validation_strategy_table.csv", index=False)
    _save_confusion(y, probability, threshold, output / "final_confusion_matrix.png")
    RocCurveDisplay.from_predictions(y, probability)
    plt.tight_layout()
    plt.savefig(output / "final_roc_curve.png", dpi=180)
    plt.close()
    PrecisionRecallDisplay.from_predictions(y, probability)
    plt.tight_layout()
    plt.savefig(output / "final_pr_curve.png", dpi=180)
    plt.close()

    for source, destination in [
        (PROJECT_ROOT / "reports" / "tables" / "final_selection_board.csv", output / "model_comparison_table.csv"),
        (PROJECT_ROOT / "reports" / "tables" / "final_selection_board.csv", output / "final_medical_metric_table.csv"),
        (PROJECT_ROOT / "reports" / "tables" / "error_pattern_summary.csv", output / "error_analysis_table.csv"),
        (PROJECT_ROOT / "reports" / "tables" / "acmg_feature_group_importance.csv", output / "acmg_feature_group_importance.csv"),
        (PROJECT_ROOT / "reports" / "figures" / "final_threshold_metric_curves.png", output / "threshold_optimization_curve.png"),
        (PROJECT_ROOT / "reports" / "figures" / "final_reliability_diagram.png", output / "calibration_curve.png"),
        (PROJECT_ROOT / "reports" / "explainability" / "shap_global_bar.png", output / "shap_feature_importance.png"),
    ]:
        _copy_if_present(source, destination)

    reproducibility = """# Reproducibility Checklist

- Fixed random seeds are used for model construction and repeated validation.
- Feature engineering, target encoding, and selection are fit within each training fold for OOF evaluation.
- Variant_ID is excluded from every model feature matrix.
- MASTER variants shared with panels are excluded from validation folds.
- Thresholds are optimized on OOF predictions, never test labels.
- Panel-unique labels are used only for reported external validation, not feature fitting.
- Final inference ignores labels and writes an explicit schema audit.
"""
    (output / "reproducibility_checklist.md").write_text(reproducibility, encoding="utf-8")
    limitations = """# Limitations and Future Work

This is a competition model trained on organizer-provided ACMG-compatible labels. It is not a clinical decision-support system and is not validated for clinical deployment. Performance on the hidden TEKNOFEST set may differ because disease panels, population composition, annotation quality, and class prevalence can shift. External biological validation, independent clinical review, and prospective calibration are outside the scope of this repository.
"""
    (output / "limitations_and_future_work.md").write_text(limitations, encoding="utf-8")
    explanation = f"""# Final Selected Model

The selected candidate is `{decision['model_id']}` ({decision['model_kind']}) using the `{decision['feature_set']}` feature set and threshold {threshold:.3f}. It was selected from OOF and panel-unique comparisons using MedicalUtilityScore, ClinicalSafetyScore, decision metrics, and stability safeguards. The choice is intended to balance pathogenic sensitivity against benign specificity rather than maximize accuracy alone.
"""
    (output / "final_selected_model_explanation.md").write_text(explanation, encoding="utf-8")

    board = pd.read_csv(PROJECT_ROOT / "reports" / "tables" / "final_selection_board.csv")
    selected = board[board["selected_as_final"].fillna(False)]
    report = [
        "# TEKNOFEST 2026 Final Model Report",
        "",
        "This report describes a competition model trained on organizer-provided ACMG-labeled variant data. It does not claim clinical deployment readiness.",
        "",
        "## Final Selection",
        "",
        f"Selected model: `{decision['model_id']}`. Threshold: {threshold:.3f}.",
        "",
        "## Validation",
        "",
        "Primary estimates use contamination-aware OOF predictions. KANSER, PAH, and CFTR unique variants provide panel-shift checks. Medical metrics prioritize pathogenic recall, specificity, F1-macro, MCC, PR-AUC, and probability calibration.",
        "",
        "## Limitations",
        "",
        "Hidden-test uncertainty remains. No test labels, Variant_ID features, or cross-fold target-encoding leakage are used.",
    ]
    if not selected.empty:
        row = selected.iloc[0]
        report.extend(["", "## Measured OOF Metrics", "", row[["roc_auc", "pr_auc", "f1_macro", "mcc", "medical_utility_score", "clinical_safety_score"]].to_frame("value").to_markdown()])
    (PROJECT_ROOT / "reports" / "final_teknofest_model_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output
