from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from final_calibration import calibration_comparison, save_calibration_outputs
from final_ensembling import run_final_ensembles
from final_error_analysis import classify_error_cases, merge_error_features, save_error_analysis
from feature_stability_selection import run_feature_stability_gate
from final_inference import generate_final_submission
from final_model_zoo import run_final_model_zoo
from final_report_assets import generate_final_report_assets
from final_selection_board import build_final_selection_board, save_final_selection_board
from final_thresholding import (
    panel_threshold_behavior,
    save_threshold_outputs,
    select_threshold_candidates,
    threshold_grid,
    threshold_stability,
)
from final_validation import (
    panel_and_stress_validation,
    repeated_contamination_aware_validation,
    save_final_validation_outputs,
)
from teknofest.data_prep import PreparedData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"


def _prepare_legacy_phase10_bundle() -> None:
    """Make the preserved Phase 10 model selectable through the new inference API."""
    destination = MODELS_DIR / "model_zoo" / "existing_phase10_final" / "full_model.joblib"
    if destination.exists():
        return
    model_path = MODELS_DIR / "final_model.pkl"
    preprocessor_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "final_preprocessor.pkl"
    columns_path = MODELS_DIR / "final_model_columns.txt"
    if not (model_path.exists() and preprocessor_path.exists() and columns_path.exists()):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": joblib.load(model_path),
            "feature_engineer": joblib.load(preprocessor_path),
            "advanced_feature_engineer": None,
            "feature_columns": [line for line in columns_path.read_text(encoding="utf-8").splitlines() if line],
            "model_id": "existing_phase10_final",
            "feature_set": "existing_phase10_engineered_features",
        },
        destination,
    )


def _prediction_source(decision: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    model_id = str(decision["model_id"])
    if decision["model_kind"] == "ensemble":
        master = pd.read_csv(PREDICTIONS_DIR / "final_ensemble_oof_predictions.csv")
        panel = pd.read_csv(PREDICTIONS_DIR / "final_ensemble_panel_predictions.csv")
        return master, panel, f"proba__{model_id}", "panel"
    if model_id == "existing_phase10_final":
        master = pd.read_csv(PREDICTIONS_DIR / "final_master_cv_predictions.csv")
        panel = pd.read_csv(PREDICTIONS_DIR / "final_panel_predictions.csv").rename(columns={"dataset": "panel"})
        return master, panel, "score", "panel"
    master = pd.read_csv(PREDICTIONS_DIR / "model_zoo_oof_predictions.csv")
    panel = pd.read_csv(PREDICTIONS_DIR / "model_zoo_panel_predictions.csv")
    return master, panel, f"proba__{model_id}", "panel"


def _write_competition_audit(board: pd.DataFrame, selected: pd.Series) -> Path:
    baseline = board[board["candidate_id"].eq("existing_phase10_final")]
    lines = ["# Final Competition Improvement Audit", ""]
    if not baseline.empty:
        reference = baseline.iloc[0]
        lines.extend(
            [
                "## Before vs After",
                "",
                "| Metric | Phase 10 baseline | Final selection | Difference |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in ("roc_auc", "pr_auc", "f1_macro", "mcc", "panel_f1_macro", "panel_mcc", "medical_utility_score"):
            lines.append(
                f"| {metric} | {float(reference[metric]):.4f} | {float(selected[metric]):.4f} | {float(selected[metric]) - float(reference[metric]):+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Leakage Audit",
            "",
            "All new OOF feature engineering and target encoding are fit within the corresponding training fold. Variant_ID is excluded from model matrices. MASTER variants shared with panels are not validation rows. Panel labels are not used to fit model features, thresholds, or ensemble weights.",
        ]
    )
    output = PROJECT_ROOT / "reports" / "final_competition_improvement_audit.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def run_final_competition_pipeline(prepared: PreparedData, include_repeated_validation: bool = True) -> dict[str, Any]:
    """Execute the leakage-safe final competition workflow in required phase order."""
    _prepare_legacy_phase10_bundle()
    zoo = run_final_model_zoo(prepared)
    feature_stability = run_feature_stability_gate(prepared, PROJECT_ROOT / "reports")
    ensembles = run_final_ensembles()

    zoo_candidates = zoo["metrics"][zoo["metrics"]["threshold_strategy"].eq("max_medical_utility")]
    preliminary = zoo_candidates.sort_values("medical_utility_score", ascending=False).iloc[0]
    preliminary_id = str(preliminary["model_id"])
    preliminary_probability_column = f"proba__{preliminary_id}"
    preliminary_threshold = float(preliminary["threshold"])
    calibration_comparison_table, calibrated = calibration_comparison(
        zoo["oof_predictions"]["Label"],
        zoo["oof_predictions"][preliminary_probability_column],
        preliminary_threshold,
        zoo["oof_predictions"]["fold"],
    )
    calibration_choice = save_calibration_outputs(
        calibration_comparison_table, calibrated, zoo["oof_predictions"]["Label"], PROJECT_ROOT / "reports"
    )

    if include_repeated_validation:
        seed_metrics, fold_metrics = repeated_contamination_aware_validation(prepared, preliminary_id)
        stress_metrics = panel_and_stress_validation(prepared, preliminary_id)
        save_final_validation_outputs(seed_metrics, fold_metrics, stress_metrics, PROJECT_ROOT / "reports")

    board, selected = build_final_selection_board(calibration_choice)
    save_final_selection_board(board, selected, PROJECT_ROOT / "reports")
    decision_path = METRICS_DIR / "final_model_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    master, panel, probability_column, panel_column = _prediction_source(decision)
    grid = threshold_grid(master["Label"], master[probability_column])
    candidates = select_threshold_candidates(grid)
    stability = threshold_stability(master, probability_column)
    panel_behavior = panel_threshold_behavior(panel, probability_column, candidates, panel_column=panel_column)
    panel_behavior.to_csv(TABLES_DIR / "final_threshold_panel_behavior.csv", index=False)

    legacy_threshold = METRICS_DIR / "final_threshold.json"
    backup = METRICS_DIR / "phase10_final_threshold_preserved.json"
    if legacy_threshold.exists() and not backup.exists():
        shutil.copy2(legacy_threshold, backup)
    threshold_choice = save_threshold_outputs(
        grid,
        candidates,
        stability,
        PROJECT_ROOT / "reports",
        legacy_threshold,
        selected_strategy="max_medical_utility",
    )
    decision["threshold"] = float(threshold_choice["threshold"])
    decision["threshold_strategy"] = "max_medical_utility"

    decision["calibration"] = "none"
    decision["calibration_note"] = (
        f"{calibration_choice['calibration_method']} was assessed with cross-fitted OOF calibration; "
        "decision scores remain uncalibrated until panel-specific calibration preservation is demonstrated."
    )
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    master_errors = merge_error_features(
        classify_error_cases(master, probability_column, float(decision["threshold"]), "MASTER_CV"), prepared.master
    )
    panel_errors = merge_error_features(
        classify_error_cases(panel, probability_column, float(decision["threshold"]), "panel_unique"),
        pd.concat([prepared.kanser_unique, prepared.pah_unique, prepared.cftr_unique], ignore_index=True),
    )
    save_error_analysis(master_errors, panel_errors, PROJECT_ROOT / "reports")
    _write_competition_audit(board, selected)
    submission, submission_path = generate_final_submission(prepared.master)
    report_assets = generate_final_report_assets(prepared)
    return {
        "model_zoo": zoo,
        "ensembles": ensembles,
        "feature_stability": feature_stability,
        "selection_board": board,
        "selected": selected,
        "submission_path": submission_path,
        "submission_rows": len(submission),
        "report_assets": report_assets,
    }
