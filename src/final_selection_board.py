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


def _panel_metric(panel: pd.DataFrame, probability_column: str, threshold: float) -> dict[str, float]:
    metrics = compute_medical_metrics(panel["Label"], panel[probability_column], threshold)
    return {
        "panel_roc_auc": float(metrics["roc_auc"]),
        "panel_pr_auc": float(metrics["pr_auc"]),
        "panel_f1_macro": float(metrics["f1_macro"]),
        "panel_mcc": float(metrics["mcc"]),
        "panel_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "panel_pathogenic_recall": float(metrics["pathogenic_recall"]),
        "panel_specificity": float(metrics["specificity"]),
    }


def _stability(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[id_column, "fold_stability"])
    return frame.groupby(id_column, as_index=False).agg(fold_stability=("medical_utility_score", "std"))


def _make_model_rows(zoo_metrics: pd.DataFrame, panel: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = zoo_metrics[zoo_metrics["threshold_strategy"].eq("max_medical_utility")].copy()
    rows = rows.rename(columns={"model_id": "candidate_id"})
    stability = _stability(fold_metrics, "model_id").rename(columns={"model_id": "candidate_id"})
    rows = rows.merge(stability, on="candidate_id", how="left")
    panel_rows = []
    for _, row in rows.iterrows():
        column = f"proba__{row['candidate_id']}"
        panel_metrics = _panel_metric(panel, column, float(row["threshold"]))
        panel_rows.append({"candidate_id": row["candidate_id"], **panel_metrics})
    return rows.merge(pd.DataFrame(panel_rows), on="candidate_id", how="left")


def _make_ensemble_rows(comparison: pd.DataFrame, panel: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows = comparison.copy().rename(columns={"ensemble_id": "candidate_id"})
    stability_rows = []
    for candidate_id, row in rows.set_index("candidate_id").iterrows():
        column = f"proba__{candidate_id}"
        fold_metrics = []
        for _, fold_frame in oof.groupby("fold"):
            fold_metrics.append(compute_medical_metrics(fold_frame["Label"], fold_frame[column], float(row["threshold"])))
        stability_rows.append({"candidate_id": candidate_id, "fold_stability": float(pd.DataFrame(fold_metrics)["medical_utility_score"].std())})
    panel_rows = []
    for _, row in rows.iterrows():
        column = f"proba__{row['candidate_id']}"
        panel_rows.append({"candidate_id": row["candidate_id"], **_panel_metric(panel, column, float(row["threshold"]))})
    return rows.merge(pd.DataFrame(stability_rows), on="candidate_id").merge(pd.DataFrame(panel_rows), on="candidate_id")


def _existing_phase10_candidate(tables: Path) -> pd.DataFrame:
    path = tables / "final_model_selection_table.csv"
    if not path.exists():
        return pd.DataFrame()
    existing = pd.read_csv(path)
    selected = existing[existing["selected_as_final"].astype(str).str.lower().eq("true")]
    if selected.empty:
        return pd.DataFrame()
    row = selected.iloc[0].copy()
    utility = (
        0.18 * float(row.get("roc_auc", 0.0))
        + 0.18 * float(row.get("pr_auc", 0.0))
        + 0.18 * float(row.get("f1_macro", 0.0))
        + 0.18 * float(row.get("mcc", 0.0))
        + 0.12 * float(row.get("balanced_accuracy", 0.0))
        + 0.10 * float(row.get("recall", 0.0))
        + 0.06 * float(row.get("specificity", 0.0))
    )


def _feature_set_ablation_candidates(tables: Path) -> pd.DataFrame:
    path = tables / "feature_set_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).copy()
    if frame.empty:
        return frame
    frame["candidate_id"] = "feature_set_ablation::" + frame["feature_set"].astype(str)
    frame["candidate_kind"] = "feature_set_ablation"
    frame["calibration"] = "none"
    for column in (
        "panel_roc_auc",
        "panel_pr_auc",
        "panel_f1_macro",
        "panel_mcc",
        "panel_balanced_accuracy",
        "panel_pathogenic_recall",
        "panel_specificity",
        "fold_stability",
    ):
        frame[column] = np.nan
    return frame
    clinical = (
        0.25 * float(row.get("recall", 0.0))
        + 0.20 * float(row.get("pr_auc", 0.0))
        + 0.20 * float(row.get("mcc", 0.0))
        + 0.15 * float(row.get("f1_macro", 0.0))
        + 0.10 * float(row.get("specificity", 0.0))
    )
    return pd.DataFrame(
        [
            {
                "candidate_id": "existing_phase10_final",
                "candidate_kind": "single_model",
                "feature_set": "existing_phase10_engineered_features",
                "threshold": float(row["threshold"]),
                "calibration": "none",
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "f1_macro": float(row["f1_macro"]),
                "mcc": float(row["mcc"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "pathogenic_recall": float(row["recall"]),
                "specificity": float(row["specificity"]),
                "panel_roc_auc": float(row["panel_roc_auc"]),
                "panel_pr_auc": float(row["panel_pr_auc"]),
                "panel_f1_macro": float(row["panel_f1_macro"]),
                "panel_mcc": float(row["panel_mcc"]),
                "fold_stability": float(row.get("threshold_instability", np.nan)),
                "medical_utility_score": utility,
                "clinical_safety_score": clinical,
                "rejection_reason": "Preserved Phase 10 benchmark.",
            }
        ]
    )


def _selection_score(frame: pd.DataFrame) -> pd.Series:
    panel_utility = (
        0.25 * frame["panel_roc_auc"].fillna(0)
        + 0.25 * frame["panel_pr_auc"].fillna(0)
        + 0.25 * frame["panel_f1_macro"].fillna(0)
        + 0.25 * frame["panel_mcc"].fillna(0)
    )
    return frame["medical_utility_score"].fillna(0) + 0.15 * panel_utility - 0.10 * frame["fold_stability"].fillna(0.1)


def build_final_selection_board(
    calibration_choice: pd.Series | None = None,
    tables_dir: str | Path = TABLES_DIR,
    predictions_dir: str | Path = PREDICTIONS_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    tables = Path(tables_dir)
    predictions = Path(predictions_dir)
    zoo_metrics = pd.read_csv(tables / "model_zoo_metrics.csv")
    zoo_fold = pd.read_csv(tables / "model_zoo_fold_metrics.csv")
    zoo_panel = pd.read_csv(predictions / "model_zoo_panel_predictions.csv")
    model_rows = _make_model_rows(zoo_metrics, zoo_panel, zoo_fold)
    model_rows["candidate_kind"] = "single_model"
    model_rows["feature_set"] = model_rows["candidate_id"].map(
        {
            "elasticnet_logistic_regression": "compact_stable_features",
            "calibrated_logistic_regression": "compact_stable_features",
        }
    ).fillna("full_safe_features")
    model_rows["calibration"] = "none"

    ensemble_rows = pd.DataFrame()
    comparison_path = tables / "final_ensemble_comparison.csv"
    if comparison_path.exists():
        ensemble_rows = _make_ensemble_rows(
            pd.read_csv(comparison_path),
            pd.read_csv(predictions / "final_ensemble_panel_predictions.csv"),
            pd.read_csv(predictions / "final_ensemble_oof_predictions.csv"),
        )
        ensemble_rows["candidate_kind"] = "ensemble"
        ensemble_rows["feature_set"] = "ensemble_feature_set"
        ensemble_rows["calibration"] = "none"

    board = pd.concat(
        [model_rows, ensemble_rows, _existing_phase10_candidate(tables), _feature_set_ablation_candidates(tables)],
        ignore_index=True,
        sort=False,
    )
    if board.empty:
        raise ValueError("Final selection needs model-zoo results.")
    if calibration_choice is not None and str(calibration_choice.get("calibration_method", "none")) != "none":
        board.loc[board["candidate_id"].eq("lightgbm_conservative_regularized"), "calibration"] = str(calibration_choice["calibration_method"])
    board["seed_stability"] = np.nan
    seed_path = tables / "seed_stability_metrics.csv"
    if seed_path.exists():
        seed_metrics = pd.read_csv(seed_path)
        if not seed_metrics.empty:
            board["seed_stability"] = board["candidate_id"].map(seed_metrics.set_index("model_id")["medical_utility_std"])
    board["overfitting_gap"] = np.nan
    board["selection_score"] = _selection_score(board)

    best_single = board[board["candidate_kind"].eq("single_model")].sort_values("selection_score", ascending=False).iloc[0]
    chosen = best_single.copy()
    board["ensemble_improvement_count"] = 0
    for index, row in board[board["candidate_kind"].eq("ensemble")].iterrows():
        improvements = sum(
            float(row[column]) > float(best_single[column]) + 1e-4
            for column in ("mcc", "f1_macro", "pr_auc", "panel_mcc", "panel_f1_macro", "medical_utility_score")
        )
        board.loc[index, "ensemble_improvement_count"] = improvements
        stable = float(row["fold_stability"]) <= float(best_single["fold_stability"]) + 0.01
        if improvements >= 2 and stable and float(row["selection_score"]) > float(chosen["selection_score"]):
            chosen = row.copy()
    board["selected_as_final"] = board["candidate_id"].eq(chosen["candidate_id"])
    board["rejection_reason"] = np.where(
        board["selected_as_final"],
        "Selected by MedicalUtilityScore with panel robustness and stability safeguards.",
        "Lower selection score or did not meet the ensemble replacement rule.",
    )
    board.loc[
        board["candidate_kind"].eq("feature_set_ablation"), "rejection_reason"
    ] = "Ablation result retained for feature decision evidence; no matched panel artifact was trained."
    return board, chosen


def save_final_selection_board(board: pd.DataFrame, selected: pd.Series, reports_dir: str | Path = "reports") -> Path:
    reports = Path(reports_dir)
    tables = reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    board.to_csv(tables / "final_selection_board.csv", index=False)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    selected_id = str(selected["candidate_id"])
    decision = {
        "model_id": selected_id,
        "model_kind": str(selected["candidate_kind"]),
        "feature_set": str(selected["feature_set"]),
        "threshold": float(selected["threshold"]),
        "threshold_strategy": "max_medical_utility",
        "calibration": str(selected["calibration"]),
        "medical_utility_score": float(selected["medical_utility_score"]),
        "clinical_safety_score": float(selected["clinical_safety_score"]),
        "best_single_model": str(board[board["candidate_kind"].eq("single_model")].sort_values("selection_score", ascending=False).iloc[0]["candidate_id"]),
        "best_ensemble": str(board[board["candidate_kind"].eq("ensemble")].sort_values("selection_score", ascending=False).iloc[0]["candidate_id"]) if board["candidate_kind"].eq("ensemble").any() else None,
        "artifact_path": (
            str(MODELS_DIR / "final_ensemble" / "ensemble_bundle.joblib")
            if selected["candidate_kind"] == "ensemble"
            else str(MODELS_DIR / "model_zoo" / "existing_phase10_final" / "full_model.joblib")
            if selected_id == "existing_phase10_final"
            else str(MODELS_DIR / "model_zoo" / selected_id / "full_model.joblib")
        ),
    }
    decision_path = METRICS_DIR / "final_model_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Final Model Selection Decision",
        "",
        f"Best single model: `{decision['best_single_model']}`.",
        f"Best ensemble: `{decision['best_ensemble']}`.",
        f"Final selected model: `{selected_id}` ({decision['model_kind']}).",
        f"Final OOF MedicalUtilityScore: {float(selected['medical_utility_score']):.4f}.",
        f"Final threshold: {float(selected['threshold']):.3f}.",
        "",
        "Selection prioritizes OOF MedicalUtilityScore and ClinicalSafetyScore, then panel-unique behavior and fold stability. The final choice is a competition model, not a clinically deployable diagnostic system.",
    ]
    (reports / "final_model_selection_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision_path
