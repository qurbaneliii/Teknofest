from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medical_metrics import compute_medical_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

REFERENCE_ID = "lightgbm_conservative_regularized"
REFERENCE_ARTIFACT = MODELS_DIR / "final_model.pkl"
FINAL_THRESHOLD_PATH = METRICS_DIR / "final_threshold.json"

OOF_METRIC_COLUMNS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "pathogenic_recall",
    "specificity",
    "f1",
    "f1_macro",
    "mcc",
    "brier_score",
    "log_loss",
    "ppv",
    "npv",
    "tn",
    "fp",
    "fn",
    "tp",
    "medical_utility_score",
    "clinical_safety_score",
)
PANEL_METRIC_COLUMNS = tuple(f"panel_{column}" for column in OOF_METRIC_COLUMNS)
IMPROVEMENT_METRICS = (
    "mcc",
    "f1_macro",
    "pr_auc",
    "panel_mcc",
    "panel_f1_macro",
    "medical_utility_score",
)
STABILITY_COLUMNS = ("medical_utility_score", "mcc", "f1_macro")


def _require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} is required but missing: {path}")
    return path


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    _require_file(path, description)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{description} is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | set[str], description: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{description} is missing required columns: {sorted(missing)}")


def _as_float(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value):
        raise ValueError(f"Candidate '{row.get('candidate_id', 'unknown')}' has a missing '{column}' value.")
    return float(value)


def _as_optional_float(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _saved_threshold() -> dict[str, Any]:
    _require_file(FINAL_THRESHOLD_PATH, "Saved final threshold artifact")
    threshold = json.loads(FINAL_THRESHOLD_PATH.read_text(encoding="utf-8"))
    required = {"threshold", "threshold_strategy"}
    missing = required - set(threshold)
    if missing:
        raise ValueError(f"Saved final threshold artifact is missing keys: {sorted(missing)}")
    return threshold


def _single_row(frame: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one {description}; found {len(selected)}.")
    return selected.iloc[0]


def _reference_audit_row(tables: Path) -> pd.Series:
    audit = _read_csv(tables / "final_medical_metric_comparison.csv", "Phase 1 final medical metric audit")
    _require_columns(audit, {"evaluation_split", *OOF_METRIC_COLUMNS}, "Phase 1 final medical metric audit")
    return _single_row(
        audit,
        audit["evaluation_split"].eq("MASTER_CV_saved_predictions"),
        "MASTER OOF row in the Phase 1 medical metric audit",
    )


def _reference_panel_row(tables: Path) -> pd.Series:
    audit = _read_csv(tables / "final_medical_metric_comparison.csv", "Phase 1 final medical metric audit")
    _require_columns(audit, {"evaluation_split", *OOF_METRIC_COLUMNS}, "Phase 1 final medical metric audit")
    return _single_row(
        audit,
        audit["evaluation_split"].eq("panel_unique_combined"),
        "combined panel-unique row in the Phase 1 medical metric audit",
    )


def _fold_stability(fold_metrics: pd.DataFrame, model_id: str) -> dict[str, float | None]:
    model_folds = fold_metrics.loc[fold_metrics["model_id"].eq(model_id)]
    if model_folds.empty:
        return {f"fold_{column}_std": None for column in STABILITY_COLUMNS}
    output: dict[str, float | None] = {}
    for column in STABILITY_COLUMNS:
        if column not in model_folds.columns:
            output[f"fold_{column}_std"] = None
            continue
        values = pd.to_numeric(model_folds[column], errors="coerce").dropna()
        output[f"fold_{column}_std"] = float(values.std(ddof=1)) if len(values) >= 2 else None
    return output


def _ensemble_fold_stability(ensemble_oof: pd.DataFrame, ensemble_id: str, threshold: float) -> dict[str, float | None]:
    probability_column = f"proba__{ensemble_id}"
    _require_columns(ensemble_oof, {"Label", "fold", probability_column}, "Final ensemble OOF predictions")
    records = []
    for _, frame in ensemble_oof.groupby("fold"):
        records.append(
            compute_medical_metrics(
                frame["Label"].to_numpy(dtype=int),
                frame[probability_column].to_numpy(dtype=float),
                threshold,
            )
        )
    per_fold = pd.DataFrame(records)
    return {
        f"fold_{column}_std": float(per_fold[column].std(ddof=1)) if len(per_fold) >= 2 else None
        for column in STABILITY_COLUMNS
    }


def _calibration_review(tables: Path) -> tuple[str, list[dict[str, Any]]]:
    review = _read_csv(tables / "calibration_decision_matrix.csv", "Calibration decision matrix")
    _require_columns(review, {"calibration_method", "decision"}, "Calibration decision matrix")
    selected = review.loc[review["decision"].eq("selected_for_final_decision_model")]
    if len(selected) != 1:
        raise ValueError("Calibration decision matrix must contain exactly one selected final decision method.")
    selected_method = str(selected.iloc[0]["calibration_method"])
    rows = []
    for _, row in review.iterrows():
        payload: dict[str, Any] = {
            "calibration_method": str(row["calibration_method"]),
            "decision": str(row["decision"]),
        }
        for column in ("brier_score", "log_loss", "f1_macro", "mcc", "panel_f1_macro", "panel_mcc"):
            value = _as_optional_float(row, column)
            if value is not None:
                payload[column] = value
        rows.append(payload)
    return selected_method, rows


def _error_audit(tables: Path) -> dict[str, Any]:
    fn_path = _require_file(tables / "final_false_negative_cases.csv", "Final false-negative case artifact")
    fp_path = _require_file(tables / "final_false_positive_cases.csv", "Final false-positive case artifact")
    group_path = _require_file(tables / "final_error_group_feature_summary.csv", "Final error-group feature summary")
    false_negatives = _read_csv(fn_path, "Final false-negative case artifact")
    false_positives = _read_csv(fp_path, "Final false-positive case artifact")
    group_summary = _read_csv(group_path, "Final error-group feature summary")
    _require_columns(group_summary, {"split", "error_group", "feature"}, "Final error-group feature summary")
    return {
        "false_negative_case_rows": int(len(false_negatives)),
        "false_positive_case_rows": int(len(false_positives)),
        "error_group_summary_rows": int(len(group_summary)),
        "false_negative_cases_path": str(fn_path),
        "false_positive_cases_path": str(fp_path),
        "error_group_summary_path": str(group_path),
    }


def _candidate_from_zoo(
    row: pd.Series,
    fold_metrics: pd.DataFrame,
    threshold: float,
    threshold_strategy: str,
    calibration_method: str,
) -> dict[str, Any]:
    candidate_id = str(row["model_id"])
    result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_kind": "current_final_model" if candidate_id == REFERENCE_ID else "model_zoo",
        "source_artifact": "saved_final_predictions" if candidate_id == REFERENCE_ID else "model_zoo_oof_predictions",
        "model_family": str(row.get("model_family", "unknown")),
        "threshold": threshold,
        "threshold_strategy": threshold_strategy,
        "threshold_source": str(row.get("threshold_source", "saved_final_threshold")),
        "calibration": calibration_method if candidate_id == REFERENCE_ID else "not_evaluated_for_challenger",
        "calibration_status": "selected_for_final_decision_model" if candidate_id == REFERENCE_ID else "not_evaluated_for_challenger",
        "panel_prediction_status": str(row.get("panel_prediction_status", "unavailable")),
        "status": str(row.get("status", "unknown")),
    }
    for column in OOF_METRIC_COLUMNS:
        result[column] = _as_optional_float(row, column)
    for column in PANEL_METRIC_COLUMNS:
        result[column] = _as_optional_float(row, column)
    result.update(_fold_stability(fold_metrics, candidate_id))
    return result


def _candidate_from_ensemble(
    row: pd.Series,
    ensemble_oof: pd.DataFrame,
    threshold: float,
    threshold_strategy: str,
) -> dict[str, Any]:
    candidate_id = str(row["ensemble_id"])
    result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_kind": "ensemble",
        "source_artifact": "final_ensemble_oof_predictions",
        "model_family": "OOF_ensemble",
        "threshold": threshold,
        "threshold_strategy": threshold_strategy,
        "threshold_source": "saved_final_threshold",
        "calibration": "not_evaluated_for_ensemble",
        "calibration_status": "not_evaluated_for_ensemble",
        "panel_prediction_status": "success",
        "status": "success",
    }
    for column in OOF_METRIC_COLUMNS:
        result[column] = _as_optional_float(row, column)
    for column in PANEL_METRIC_COLUMNS:
        result[column] = _as_optional_float(row, column)
    result.update(_ensemble_fold_stability(ensemble_oof, candidate_id, threshold))
    return result


def _assert_reference_consistency(candidate: dict[str, Any], master_audit: pd.Series, panel_audit: pd.Series, threshold: float) -> None:
    if not np.isclose(float(candidate["threshold"]), threshold, atol=1e-12):
        raise ValueError("The model-zoo LightGBM reference threshold does not match artifacts/metrics/final_threshold.json.")
    for column in ("roc_auc", "pr_auc", "f1_macro", "mcc", "medical_utility_score"):
        expected = _as_float(master_audit, column)
        observed = candidate[column]
        if observed is None or not np.isclose(float(observed), expected, atol=1e-10):
            raise ValueError(f"Saved LightGBM reference and Phase 1 audit disagree on {column}.")
    for column in ("roc_auc", "pr_auc", "f1_macro", "mcc", "medical_utility_score"):
        expected = _as_float(panel_audit, column)
        observed = candidate[f"panel_{column}"]
        if observed is None or not np.isclose(float(observed), expected, atol=1e-10):
            raise ValueError(f"Saved LightGBM panel reference and Phase 1 audit disagree on panel_{column}.")


def _apply_reference_audit(candidate: dict[str, Any], master_audit: pd.Series, panel_audit: pd.Series) -> None:
    for column in OOF_METRIC_COLUMNS:
        candidate[column] = _as_optional_float(master_audit, column)
        candidate[f"panel_{column}"] = _as_optional_float(panel_audit, column)


def _generalization_score(row: pd.Series) -> float:
    """A display-only composite; hard gates, not this score, determine the final choice."""
    values = {
        "medical_utility_score": 0.30,
        "clinical_safety_score": 0.15,
        "f1_macro": 0.15,
        "mcc": 0.15,
        "panel_medical_utility_score": 0.15,
        "panel_f1_macro": 0.05,
        "panel_mcc": 0.05,
    }
    score = sum(weight * float(row[column]) for column, weight in values.items() if pd.notna(row.get(column)))
    stability = row.get("fold_medical_utility_score_std")
    return float(score - 0.10 * float(stability)) if pd.notna(stability) else float(score)


def _rejection_reason(row: pd.Series, reference: pd.Series, tolerance: float = 1e-12) -> str:
    if row["candidate_id"] == reference["candidate_id"]:
        return "Selected: preserved benchmark has direct saved OOF and panel-unique evidence plus documented calibration and error-analysis audits."
    reasons = []
    for metric, label in (
        ("medical_utility_score", "OOF MedicalUtilityScore"),
        ("panel_medical_utility_score", "panel MedicalUtilityScore"),
        ("panel_f1_macro", "panel F1-macro"),
        ("panel_mcc", "panel MCC"),
    ):
        if pd.notna(row.get(metric)) and float(row[metric]) < float(reference[metric]) - tolerance:
            reasons.append(f"lower {label}")
    if int(row["improvement_count"]) < 2:
        reasons.append("fewer than two required metric improvements")
    stability = row.get("fold_medical_utility_score_std")
    reference_stability = reference.get("fold_medical_utility_score_std")
    if pd.notna(stability) and pd.notna(reference_stability) and float(stability) > float(reference_stability) + 0.005:
        reasons.append("less stable across folds")
    if not reasons:
        reasons.append("does not provide sufficient independent calibration and error-analysis evidence to replace the preserved final model")
    return "; ".join(reasons) + "."


def select_final_candidate(board: pd.DataFrame, reference_id: str = REFERENCE_ID) -> tuple[pd.DataFrame, pd.Series]:
    """Apply conservative generalization gates to real, already-saved experiment results."""
    required = {
        "candidate_id",
        "candidate_kind",
        *IMPROVEMENT_METRICS,
        "panel_medical_utility_score",
        "fold_medical_utility_score_std",
    }
    _require_columns(board, required, "Final selection board")
    reference = _single_row(board, board["candidate_id"].eq(reference_id), "preserved LightGBM reference candidate")
    output = board.copy()
    for metric in IMPROVEMENT_METRICS:
        output[f"improves_{metric}"] = output[metric] > float(reference[metric]) + 1e-12
    output["improvement_count"] = output[[f"improves_{metric}" for metric in IMPROVEMENT_METRICS]].sum(axis=1)
    output["oof_utility_not_degraded"] = output["medical_utility_score"] >= float(reference["medical_utility_score"]) - 1e-12
    output["panel_utility_not_degraded"] = output["panel_medical_utility_score"] >= float(reference["panel_medical_utility_score"]) - 1e-12
    output["panel_decision_not_degraded"] = (
        (output["panel_f1_macro"] >= float(reference["panel_f1_macro"]) - 1e-12)
        & (output["panel_mcc"] >= float(reference["panel_mcc"]) - 1e-12)
    )
    output["fold_stability_not_degraded"] = output["fold_medical_utility_score_std"] <= float(reference["fold_medical_utility_score_std"]) + 0.005
    output["eligible_to_replace"] = (
        output["candidate_id"].ne(reference_id)
        & output["improvement_count"].ge(2)
        & output["oof_utility_not_degraded"]
        & output["panel_utility_not_degraded"]
        & output["panel_decision_not_degraded"]
        & output["fold_stability_not_degraded"]
    )
    output["hidden_test_generalization_score"] = output.apply(_generalization_score, axis=1)

    eligible = output.loc[output["eligible_to_replace"]]
    selected_id = reference_id if eligible.empty else str(
        eligible.sort_values(
            ["hidden_test_generalization_score", "medical_utility_score", "panel_medical_utility_score"],
            ascending=False,
        ).iloc[0]["candidate_id"]
    )
    output["selected_as_final"] = output["candidate_id"].eq(selected_id)
    selected = _single_row(output, output["selected_as_final"], "selected final candidate")
    output["rejection_reason"] = [
        _rejection_reason(row, reference) for _, row in output.iterrows()
    ]
    selected = _single_row(output, output["selected_as_final"], "selected final candidate")
    return output, selected


def build_final_selection_board(
    calibration_choice: pd.Series | None = None,
    tables_dir: str | Path = TABLES_DIR,
    predictions_dir: str | Path = PREDICTIONS_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    """Collect Phase 1-3 artifacts without retraining or changing the final model."""
    del calibration_choice  # The saved calibration decision matrix is the auditable source of truth.
    tables = Path(tables_dir)
    predictions = Path(predictions_dir)
    threshold_artifact = _saved_threshold()
    threshold = float(threshold_artifact["threshold"])
    threshold_strategy = str(threshold_artifact["threshold_strategy"])
    calibration_method, _ = _calibration_review(tables)
    if calibration_method != "none":
        raise ValueError("The saved calibration decision does not preserve the uncalibrated final decision model.")

    zoo_metrics = _read_csv(tables / "model_zoo_metrics.csv", "Phase 2 model-zoo metrics")
    fold_metrics = _read_csv(tables / "model_zoo_fold_metrics.csv", "Phase 2 model-zoo fold metrics")
    _require_columns(zoo_metrics, {"model_id", "status", "threshold", *OOF_METRIC_COLUMNS, *PANEL_METRIC_COLUMNS}, "Phase 2 model-zoo metrics")
    _require_columns(fold_metrics, {"model_id", *STABILITY_COLUMNS}, "Phase 2 model-zoo fold metrics")
    successful_zoo = zoo_metrics.loc[zoo_metrics["status"].eq("success")].copy()
    if successful_zoo.empty:
        raise ValueError("Phase 2 model-zoo metrics contain no successful candidates.")
    if not np.allclose(pd.to_numeric(successful_zoo["threshold"], errors="coerce"), threshold, atol=1e-12):
        raise ValueError("Phase 2 model-zoo metrics do not all use the saved final threshold.")

    candidates = [
        _candidate_from_zoo(row, fold_metrics, threshold, threshold_strategy, calibration_method)
        for _, row in successful_zoo.iterrows()
    ]
    board = pd.DataFrame(candidates)
    reference_index = board.index[board["candidate_id"].eq(REFERENCE_ID)]
    if len(reference_index) != 1:
        raise ValueError("Phase 2 model-zoo metrics must contain exactly one LightGBM conservative reference.")
    master_audit = _reference_audit_row(tables)
    panel_audit = _reference_panel_row(tables)
    reference = board.loc[reference_index[0]].to_dict()
    _assert_reference_consistency(reference, master_audit, panel_audit, threshold)
    _apply_reference_audit(reference, master_audit, panel_audit)
    board.loc[reference_index[0], list(reference)] = pd.Series(reference)

    ensemble_metrics_path = tables / "final_ensemble_comparison.csv"
    ensemble_oof_path = predictions / "final_ensemble_oof_predictions.csv"
    ensemble_metrics = _read_csv(ensemble_metrics_path, "Phase 3 ensemble comparison")
    ensemble_oof = _read_csv(ensemble_oof_path, "Phase 3 ensemble OOF predictions")
    _require_columns(ensemble_metrics, {"ensemble_id", "threshold", *OOF_METRIC_COLUMNS, *PANEL_METRIC_COLUMNS}, "Phase 3 ensemble comparison")
    if not np.allclose(pd.to_numeric(ensemble_metrics["threshold"], errors="coerce"), threshold, atol=1e-12):
        raise ValueError("Phase 3 ensemble comparison does not use the saved final threshold.")
    ensemble_rows = [
        _candidate_from_ensemble(row, ensemble_oof, threshold, threshold_strategy)
        for _, row in ensemble_metrics.iterrows()
    ]
    board = pd.concat([board, pd.DataFrame(ensemble_rows)], ignore_index=True, sort=False)

    error_audit = _error_audit(tables)
    error_audit["matches_selected_oof_confusion_counts"] = bool(
        error_audit["false_negative_case_rows"] == int(reference["fn"])
        and error_audit["false_positive_case_rows"] == int(reference["fp"])
    )
    for key, value in error_audit.items():
        board[key] = value if key.endswith("_rows") else pd.NA
    selected_board, selected = select_final_candidate(board)
    error_status = (
        "audited"
        if error_audit["matches_selected_oof_confusion_counts"]
        else "archived_case_counts_do_not_match_selected_oof_confusion"
    )
    selected_board.loc[selected_board["candidate_id"].eq(REFERENCE_ID), "error_audit_status"] = error_status
    selected_board.loc[~selected_board["candidate_id"].eq(REFERENCE_ID), "error_audit_status"] = "not_evaluated_for_challenger"
    selected = _single_row(selected_board, selected_board["selected_as_final"], "selected final candidate")
    selected.attrs["calibration_review"] = _calibration_review(tables)[1]
    selected.attrs["error_audit"] = error_audit
    selected.attrs["threshold_artifact"] = threshold_artifact
    return selected_board, selected


def _metric_payload(row: pd.Series, prefix: str = "") -> dict[str, float | int | None]:
    output: dict[str, float | int | None] = {}
    for base_column in OOF_METRIC_COLUMNS:
        column = f"{prefix}{base_column}"
        value = _as_optional_float(row, column)
        if value is None:
            output[base_column] = None
        elif base_column in {"tn", "fp", "fn", "tp"}:
            output[base_column] = int(value)
        else:
            output[base_column] = value
    return output


def _panel_unique_metrics(tables: Path) -> list[dict[str, Any]]:
    panel = _read_csv(tables / "final_panel_specific_metrics.csv", "Final panel-specific metric table")
    _require_columns(
        panel,
        {"evaluation_split", "roc_auc", "pr_auc", "f1_macro", "mcc", "tn", "fp", "fn", "tp"},
        "Final panel-specific metric table",
    )
    results = []
    for _, row in panel.iterrows():
        results.append({"evaluation_split": str(row["evaluation_split"]), **_metric_payload(row)})
    return results


def _markdown_metrics(metrics: dict[str, float | int | None]) -> list[str]:
    rows = []
    for name, value in metrics.items():
        if value is None:
            display = "unavailable"
        elif isinstance(value, int):
            display = str(value)
        else:
            display = f"{value:.6f}"
        rows.append({"metric": name, "value": display})
    return pd.DataFrame(rows).to_markdown(index=False).splitlines()


def _write_report(
    board: pd.DataFrame,
    selected: pd.Series,
    reports_dir: Path,
    calibration_review: list[dict[str, Any]],
    error_audit: dict[str, Any],
) -> None:
    selected_id = str(selected["candidate_id"])
    replaced = bool(selected["candidate_kind"] == "ensemble")
    oof_metrics = _metric_payload(selected)
    panel_metrics = _metric_payload(selected, "panel_")
    panel_unique = _panel_unique_metrics(TABLES_DIR)
    error_note = (
        "The archived FN/FP case-file row counts match the immutable Phase 1 OOF confusion counts."
        if error_audit["matches_selected_oof_confusion_counts"]
        else "The archived FN/FP case-file row counts do not match the immutable Phase 1 OOF confusion counts, so they are retained as qualitative error-analysis evidence and are not used in model ranking."
    )
    rejected = board.loc[~board["selected_as_final"], [
        "candidate_id", "candidate_kind", "roc_auc", "pr_auc", "f1_macro", "mcc",
        "medical_utility_score", "panel_f1_macro", "panel_mcc", "panel_medical_utility_score", "rejection_reason",
    ]]
    lines = [
        "# Final Model Selection Decision",
        "",
        "## Selected Configuration",
        "",
        f"- Selected model: `{selected_id}` ({selected['candidate_kind']}).",
        f"- Selected threshold: `{float(selected['threshold']):.6f}` using `{selected['threshold_strategy']}`.",
        f"- Ensemble replaced LightGBM: `{str(replaced).lower()}`.",
        f"- Calibration decision: `{selected['calibration']}`.",
        "",
        "## Why This Is Safest",
        "",
        "The preserved LightGBM result is selected because it has direct saved OOF and panel-unique evidence, plus documented calibration and error-analysis audits, at the unchanged threshold. No Phase 2 or Phase 3 challenger satisfies every conservative replacement gate: at least two required metric improvements, non-degraded OOF and panel MedicalUtilityScore, non-degraded panel F1-macro and MCC, and no material fold-instability increase.",
        "",
        "## Exact OOF Metrics",
        "",
        *_markdown_metrics(oof_metrics),
        "",
        "## Exact Combined Panel Metrics",
        "",
        *_markdown_metrics(panel_metrics),
        "",
        "## Calibration Decision",
        "",
        "Calibration is retained as `none` for the final decision model. The calibration comparison is evidence only; it is not directly comparable to the Phase 2/3 candidate scores because it uses its own archived threshold review.",
        "",
        pd.DataFrame(calibration_review).to_markdown(index=False),
        "",
        "## Panel-Unique Metrics",
        "",
        pd.DataFrame(panel_unique).to_markdown(index=False),
        "",
        "## Error Analysis",
        "",
        f"Saved false-negative case rows: {error_audit['false_negative_case_rows']}. Saved false-positive case rows: {error_audit['false_positive_case_rows']}. The detailed error-group summary has {error_audit['error_group_summary_rows']} rows and remains available at `{error_audit['error_group_summary_path']}`.",
        f"The selected OOF confusion counts are TN={oof_metrics['tn']}, FP={oof_metrics['fp']}, FN={oof_metrics['fn']}, TP={oof_metrics['tp']}.",
        error_note,
        "",
        "## Rejected Experiments",
        "",
        rejected.to_markdown(index=False),
        "",
        "## Remaining Risks",
        "",
        "- Hidden-test prevalence, disease panel mix, population composition, and annotation quality can differ from MASTER and the supplied panels.",
        "- Calibration was not selected for decision scores; probability calibration should be revalidated before any non-competition use.",
        "- Panel-unique results are a distribution-shift proxy, not an independent prospective clinical validation.",
        "- This repository supports a competition model only and does not establish clinical deployment readiness.",
    ]
    (reports_dir / "final_model_selection_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_final_selection_board(board: pd.DataFrame, selected: pd.Series, reports_dir: str | Path = REPORTS_DIR) -> Path:
    """Write the new final-selection artifacts without altering any model or prediction artifact."""
    reports = Path(reports_dir)
    tables = reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    board.to_csv(tables / "final_selection_board.csv", index=False)

    calibration_review = selected.attrs.get("calibration_review", [])
    error_audit = selected.attrs.get("error_audit", {})
    threshold_artifact = selected.attrs.get("threshold_artifact", {})
    selected_id = str(selected["candidate_id"])
    decision = {
        "model_id": selected_id,
        "model_kind": "ensemble" if selected["candidate_kind"] == "ensemble" else "single_model",
        "feature_set": "existing_phase10_engineered_features" if selected_id == REFERENCE_ID else "model_zoo_or_ensemble_features",
        "artifact_path": str(REFERENCE_ARTIFACT) if selected_id == REFERENCE_ID else None,
        "threshold": float(selected["threshold"]),
        "threshold_strategy": str(selected["threshold_strategy"]),
        "threshold_source": str(FINAL_THRESHOLD_PATH),
        "calibration": str(selected["calibration"]),
        "ensemble_replaced_lightgbm": bool(selected["candidate_kind"] == "ensemble"),
        "selection_basis": "conservative hidden-test generalization gates over OOF utility, panel decision metrics, fold stability, calibration evidence, and error-audit availability",
        "oof_metrics": _metric_payload(selected),
        "panel_unique_combined_metrics": _metric_payload(selected, "panel_"),
        "fold_stability": {
            column: _as_optional_float(selected, column)
            for column in ("fold_medical_utility_score_std", "fold_mcc_std", "fold_f1_macro_std")
        },
        "calibration_review": calibration_review,
        "error_analysis": error_audit,
        "rejected_experiments": [
            {
                "candidate_id": str(row["candidate_id"]),
                "candidate_kind": str(row["candidate_kind"]),
                "reason": str(row["rejection_reason"]),
            }
            for _, row in board.loc[~board["selected_as_final"]].iterrows()
        ],
        "remaining_risks": [
            "Hidden-test distribution shift may differ from MASTER and panel-unique data.",
            "Probability calibration is not selected for the final decision threshold.",
            "Panel-unique metrics are not prospective clinical validation.",
        ],
        "source_threshold_artifact": threshold_artifact,
    }
    decision_path = METRICS_DIR / "final_model_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(board, selected, reports, calibration_review, error_audit)
    return decision_path
