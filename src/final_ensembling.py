from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from medical_metrics import compute_medical_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAVED_THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "final_threshold.json"
BASE_MODEL_IDS = (
    "lightgbm_conservative_regularized",
    "lightgbm_high_capacity_controlled",
    "catboost",
    "xgboost",
    "extra_trees",
    "elasticnet_logistic_regression",
)
IMPROVEMENT_METRICS = ("mcc", "f1_macro", "pr_auc", "panel_mcc", "panel_f1_macro", "medical_utility_score")


def _saved_threshold() -> float:
    if not SAVED_THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Saved final threshold is missing: {SAVED_THRESHOLD_PATH}")
    data = json.loads(SAVED_THRESHOLD_PATH.read_text(encoding="utf-8"))
    return float(data["threshold"])


def _base_columns(frame: pd.DataFrame) -> list[str]:
    columns = [f"proba__{model_id}" for model_id in BASE_MODEL_IDS if f"proba__{model_id}" in frame.columns]
    if len(columns) < 2:
        raise ValueError("Model-zoo OOF predictions must include at least two base-model probability columns.")
    if frame[columns].isna().any().any():
        raise ValueError("Model-zoo OOF predictions contain missing base-model probabilities.")
    return columns


def _target_score(y: np.ndarray, probability: np.ndarray, threshold: float, objective: str) -> float:
    if objective == "pr_auc":
        return float(average_precision_score(y, probability))
    metrics = compute_medical_metrics(y, probability, threshold)
    if objective == "medical_utility":
        return float(metrics["medical_utility_score"])
    return float(metrics[objective])


def optimize_weights(
    matrix: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.471,
    objective: Literal["mcc", "medical_utility"] = "medical_utility",
) -> np.ndarray:
    """Fit non-negative weights on a training-only OOF slice at a fixed threshold."""
    n_models = matrix.shape[1]
    initial = np.full(n_models, 1.0 / n_models)

    def loss(weights: np.ndarray) -> float:
        probability = np.clip(matrix @ weights, 1e-7, 1 - 1e-7)
        return -_target_score(y, probability, threshold, objective)

    result = minimize(
        loss,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 60, "ftol": 1e-8},
    )
    if not result.success or np.any(result.x < -1e-8):
        return initial
    return np.clip(result.x, 0.0, 1.0) / np.clip(result.x.sum(), 1e-12, None)


def _performance_weights(matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = np.asarray([average_precision_score(y, matrix[:, index]) for index in range(matrix.shape[1])], dtype=float)
    return scores / scores.sum()


def _cross_fitted_weights(
    matrix: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    threshold: float,
    ensemble_id: str,
    objective: str | None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    probability = np.empty(len(y), dtype=float)
    records: list[dict[str, object]] = []
    for fold in np.unique(folds):
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        weights = _performance_weights(matrix[train_idx], y[train_idx]) if objective is None else optimize_weights(matrix[train_idx], y[train_idx], threshold, objective)
        probability[val_idx] = matrix[val_idx] @ weights
        records.extend(
            {"ensemble_id": ensemble_id, "fold": int(fold), "base_probability_column": column_index, "weight": float(weight)}
            for column_index, weight in enumerate(weights)
        )
    return probability, records


def _stack_model(kind: str) -> Pipeline:
    if kind == "logistic_stacking":
        estimator = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
    elif kind == "ridge_stacking":
        estimator = RidgeClassifier(alpha=2.0)
    elif kind == "elasticnet_stacking":
        estimator = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.20,
            C=0.35,
            max_iter=1500,
            tol=1e-3,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported stacking model: {kind}")
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _stack_predict(model: Pipeline, matrix: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ridge_stacking":
        return expit(model.decision_function(matrix))
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)


def _cross_fitted_stack(matrix: np.ndarray, y: np.ndarray, folds: np.ndarray, kind: str) -> np.ndarray:
    probability = np.empty(len(y), dtype=float)
    for fold in np.unique(folds):
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        model = _stack_model(kind)
        model.fit(matrix[train_idx], y[train_idx])
        probability[val_idx] = _stack_predict(model, matrix[val_idx], kind)
    return probability


def _panel_probabilities(
    oof_matrix: np.ndarray,
    panel_matrix: np.ndarray,
    y: np.ndarray,
    threshold: float,
    ensemble_ids: list[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    output = {
        "simple_average": panel_matrix.mean(axis=1),
        "rank_average": np.mean([rankdata(panel_matrix[:, index]) / (len(panel_matrix) + 1) for index in range(panel_matrix.shape[1])], axis=0),
    }
    records: list[dict[str, object]] = []
    for ensemble_id, objective in (
        ("weighted_average", None),
        ("mcc_weighted_average", "mcc"),
        ("medical_utility_weighted_average", "medical_utility"),
    ):
        if ensemble_id not in ensemble_ids:
            continue
        weights = _performance_weights(oof_matrix, y) if objective is None else optimize_weights(oof_matrix, y, threshold, objective)
        output[ensemble_id] = panel_matrix @ weights
        records.extend(
            {"ensemble_id": ensemble_id, "fold": "full_oof", "base_probability_column": index, "weight": float(weight)}
            for index, weight in enumerate(weights)
        )
    for kind in ("logistic_stacking", "ridge_stacking", "elasticnet_stacking"):
        if kind not in ensemble_ids:
            continue
        model = _stack_model(kind).fit(oof_matrix, y)
        output[kind] = _stack_predict(model, panel_matrix, kind)
    return output, records


def _comparison_row(
    ensemble_id: str,
    y: np.ndarray,
    probability: np.ndarray,
    panel_y: np.ndarray,
    panel_probability: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    oof_metrics = compute_medical_metrics(y, probability, threshold)
    panel_metrics = compute_medical_metrics(panel_y, panel_probability, threshold)
    row: dict[str, object] = {"ensemble_id": ensemble_id, "threshold": threshold, **oof_metrics}
    row.update({f"panel_{name}": value for name, value in panel_metrics.items()})
    return row


def _decision(comparison: pd.DataFrame, baseline: dict[str, float]) -> tuple[pd.DataFrame, pd.Series]:
    result = comparison.copy()
    for metric in IMPROVEMENT_METRICS:
        result[f"improves_{metric}"] = result[metric] > float(baseline[metric]) + 1e-12
    improvement_columns = [f"improves_{metric}" for metric in IMPROVEMENT_METRICS]
    result["improvement_count"] = result[improvement_columns].sum(axis=1)
    result["meets_two_metric_gate"] = result["improvement_count"].ge(2)
    result["medical_utility_not_degraded"] = result["medical_utility_score"].ge(float(baseline["medical_utility_score"]))
    result["eligible_to_replace"] = result["meets_two_metric_gate"] & result["medical_utility_not_degraded"]
    result["rejection_reason"] = np.select(
        [
            ~result["meets_two_metric_gate"],
            ~result["medical_utility_not_degraded"],
        ],
        [
            "Does not improve at least two required metrics over the preserved LightGBM reference.",
            "Meets the raw two-metric count but lowers the primary MedicalUtilityScore.",
        ],
        default="Meets the two-metric gate without lowering MedicalUtilityScore; final selection remains deferred.",
    )
    chosen = result.sort_values(["eligible_to_replace", "improvement_count", "medical_utility_score"], ascending=[False, False, False]).iloc[0]
    return result, chosen


def _write_decision(comparison: pd.DataFrame, chosen: pd.Series, baseline: dict[str, float], output: Path) -> None:
    accepted = comparison[comparison["eligible_to_replace"]]
    lines = [
        "# Final Ensemble Decision",
        "",
        "Ensemble weights and meta-models are cross-fitted: each validation fold is predicted by a combiner trained on OOF rows from the other folds only. Panel labels are used only to report panel behavior after the ensemble has been fitted on MASTER OOF predictions.",
        "",
        "## Preserved Reference",
        "",
        f"LightGBM conservative: MCC={baseline['mcc']:.6f}, F1-macro={baseline['f1_macro']:.6f}, PR-AUC={baseline['pr_auc']:.6f}, panel MCC={baseline['panel_mcc']:.6f}, panel F1-macro={baseline['panel_f1_macro']:.6f}, MedicalUtilityScore={baseline['medical_utility_score']:.6f}.",
        "",
        "## Result",
        "",
    ]
    if accepted.empty:
        lines.append("No ensemble safely meets the replacement rule: the existing LightGBM conservative_regularized model remains final because every raw two-metric candidate lowers MedicalUtilityScore.")
    else:
        lines.append(f"`{chosen['ensemble_id']}` meets the preliminary replacement gate with {int(chosen['improvement_count'])} improved metrics. Final selection is intentionally deferred to a later validation phase.")
    lines.extend(["", "## Ensemble Comparison", "", comparison[["ensemble_id", "mcc", "f1_macro", "pr_auc", "medical_utility_score", "panel_mcc", "panel_f1_macro", "improvement_count", "meets_two_metric_gate", "eligible_to_replace"]].to_markdown(index=False)])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_ensembles(
    oof_path: str | Path = PREDICTIONS_DIR / "model_zoo_oof_predictions.csv",
    panel_path: str | Path = PREDICTIONS_DIR / "model_zoo_panel_predictions.csv",
) -> dict[str, pd.DataFrame]:
    """Build Phase 3 OOF ensembles without altering any final model artifact."""
    oof_path = Path(oof_path)
    panel_path = Path(panel_path)
    if not oof_path.exists() or not panel_path.exists():
        raise FileNotFoundError("Phase 3 requires Phase 2 model_zoo_oof_predictions.csv and model_zoo_panel_predictions.csv.")
    oof = pd.read_csv(oof_path)
    panel = pd.read_csv(panel_path)
    required = {"row_index", "Variant_ID", "Label", "fold"}
    if required - set(oof.columns) or {"Variant_ID", "Label"} - set(panel.columns):
        raise ValueError("Model-zoo prediction artifacts do not contain the required identifier, label, and fold columns.")
    columns = _base_columns(oof)
    if set(columns) - set(panel.columns):
        raise ValueError("Panel predictions are missing one or more successful model-zoo probability columns.")
    threshold = _saved_threshold()
    y = oof["Label"].to_numpy(dtype=int)
    panel_y = panel["Label"].to_numpy(dtype=int)
    folds = oof["fold"].to_numpy(dtype=int)
    matrix = oof[columns].to_numpy(dtype=float)
    panel_matrix = panel[columns].to_numpy(dtype=float)

    oof_probabilities: dict[str, np.ndarray] = {
        "simple_average": matrix.mean(axis=1),
        "rank_average": np.mean([rankdata(matrix[:, index]) / (len(matrix) + 1) for index in range(matrix.shape[1])], axis=0),
    }
    weight_records: list[dict[str, object]] = []
    for ensemble_id, objective in (
        ("weighted_average", None),
        ("mcc_weighted_average", "mcc"),
        ("medical_utility_weighted_average", "medical_utility"),
    ):
        probability, records = _cross_fitted_weights(matrix, y, folds, threshold, ensemble_id, objective)
        oof_probabilities[ensemble_id] = probability
        for record in records:
            record["base_probability_column"] = columns[int(record["base_probability_column"])]
        weight_records.extend(records)
    for ensemble_id in ("logistic_stacking", "ridge_stacking", "elasticnet_stacking"):
        oof_probabilities[ensemble_id] = _cross_fitted_stack(matrix, y, folds, ensemble_id)

    panel_probabilities, full_weight_records = _panel_probabilities(matrix, panel_matrix, y, threshold, list(oof_probabilities))
    for record in full_weight_records:
        record["base_probability_column"] = columns[int(record["base_probability_column"])]
    weight_records.extend(full_weight_records)
    if set(oof_probabilities) != set(panel_probabilities):
        raise RuntimeError("Panel ensemble predictions are incomplete.")

    rows = [
        _comparison_row(ensemble_id, y, probability, panel_y, panel_probabilities[ensemble_id], threshold)
        for ensemble_id, probability in oof_probabilities.items()
    ]
    baseline_oof = compute_medical_metrics(y, oof["proba__lightgbm_conservative_regularized"], threshold)
    baseline_panel = compute_medical_metrics(panel_y, panel["proba__lightgbm_conservative_regularized"], threshold)
    baseline = {**baseline_oof, **{f"panel_{name}": value for name, value in baseline_panel.items()}}
    comparison, chosen = _decision(pd.DataFrame(rows), baseline)

    ensemble_oof = oof[["row_index", "Variant_ID", "Label", "fold"]].copy()
    ensemble_panel = panel[[column for column in ("dataset", "Variant_ID", "Label") if column in panel.columns]].copy()
    for ensemble_id, probability in oof_probabilities.items():
        ensemble_oof[f"proba__{ensemble_id}"] = probability
        ensemble_panel[f"proba__{ensemble_id}"] = panel_probabilities[ensemble_id]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ensemble_oof.to_csv(PREDICTIONS_DIR / "final_ensemble_oof_predictions.csv", index=False)
    ensemble_panel.to_csv(PREDICTIONS_DIR / "final_ensemble_panel_predictions.csv", index=False)
    comparison.to_csv(TABLES_DIR / "final_ensemble_comparison.csv", index=False)
    pd.DataFrame(weight_records).to_csv(TABLES_DIR / "final_ensemble_weights.csv", index=False)
    _write_decision(comparison, chosen, baseline, REPORTS_DIR / "final_ensemble_decision.md")
    return {"comparison": comparison, "oof_predictions": ensemble_oof, "panel_predictions": ensemble_panel, "weights": pd.DataFrame(weight_records)}
