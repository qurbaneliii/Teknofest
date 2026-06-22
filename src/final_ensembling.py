from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from final_thresholding import select_threshold_candidates, threshold_grid
from medical_metrics import compute_medical_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models" / "final_ensemble"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"

BASE_MODEL_IDS = (
    "lightgbm_conservative_regularized",
    "lightgbm_high_capacity_controlled",
    "catboost",
    "xgboost",
    "extra_trees",
    "elasticnet_logistic_regression",
)


def _columns(frame: pd.DataFrame, candidates: tuple[str, ...] = BASE_MODEL_IDS) -> list[str]:
    found = [f"proba__{model_id}" for model_id in candidates if f"proba__{model_id}" in frame.columns]
    if len(found) < 2:
        raise ValueError("At least two model-zoo OOF probability columns are required for an ensemble.")
    return found


def _utility_objective(y: np.ndarray, probabilities: np.ndarray, target: str) -> float:
    grid = threshold_grid(y, probabilities, thresholds=np.round(np.arange(0.05, 0.951, 0.01), 3))
    if target == "medical_utility":
        return float(grid["medical_utility_score"].max())
    if target == "mcc":
        return float(grid["mcc"].max())
    if target == "pr_auc":
        return float(grid["pr_auc"].iloc[0])
    if target == "f1_macro":
        return float(grid["f1_macro"].max())
    raise ValueError(f"Unknown ensemble objective: {target}")


def optimize_weights(matrix: np.ndarray, y: np.ndarray, target: str = "medical_utility") -> np.ndarray:
    """Fit non-negative sum-to-one weights on a training-only OOF matrix."""
    n_models = matrix.shape[1]
    initial = np.full(n_models, 1.0 / n_models)

    def objective(weights: np.ndarray) -> float:
        return -_utility_objective(y, np.clip(matrix @ weights, 1e-6, 1 - 1e-6), target)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 100, "ftol": 1e-7},
    )
    if not result.success or np.any(result.x < -1e-8):
        return initial
    return np.clip(result.x, 0.0, 1.0) / np.clip(result.x.sum(), 1e-12, None)


def _cross_fitted_weighted(matrix: np.ndarray, y: np.ndarray, folds: np.ndarray, target: str) -> tuple[np.ndarray, list[dict[str, object]]]:
    probabilities = np.empty(len(y), dtype=float)
    records: list[dict[str, object]] = []
    for fold in np.unique(folds):
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        weights = optimize_weights(matrix[train_idx], y[train_idx], target)
        probabilities[val_idx] = matrix[val_idx] @ weights
        records.append({"ensemble_id": f"weighted_{target}", "fold": int(fold), "weights": weights.tolist()})
    return probabilities, records


def _cross_fitted_stack(matrix: np.ndarray, y: np.ndarray, folds: np.ndarray, kind: str) -> tuple[np.ndarray, list[object]]:
    probabilities = np.empty(len(y), dtype=float)
    models: list[object] = []
    for fold in np.unique(folds):
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        if kind == "logistic_stack":
            model = Pipeline(
                [("scale", StandardScaler()), ("model", LogisticRegression(C=0.5, max_iter=5000, random_state=42))]
            )
            model.fit(matrix[train_idx], y[train_idx])
            probabilities[val_idx] = model.predict_proba(matrix[val_idx])[:, 1]
        elif kind == "ridge_stack":
            model = Pipeline([("scale", StandardScaler()), ("model", RidgeClassifier(alpha=2.0))])
            model.fit(matrix[train_idx], y[train_idx])
            probabilities[val_idx] = expit(model.decision_function(matrix[val_idx]))
        elif kind == "elasticnet_stack":
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            l1_ratio=0.20,
                            C=0.35,
                            max_iter=6000,
                            random_state=42,
                        ),
                    ),
                ]
            )
            model.fit(matrix[train_idx], y[train_idx])
            probabilities[val_idx] = model.predict_proba(matrix[val_idx])[:, 1]
        else:
            raise ValueError(f"Unsupported stack: {kind}")
        models.append(model)
    return probabilities, models


def _full_stack(matrix: np.ndarray, y: np.ndarray, kind: str) -> object:
    if kind == "logistic_stack":
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.5, max_iter=5000, random_state=42))])
    elif kind == "ridge_stack":
        model = Pipeline([("scale", StandardScaler()), ("model", RidgeClassifier(alpha=2.0))])
    elif kind == "elasticnet_stack":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.20,
                        C=0.35,
                        max_iter=6000,
                        random_state=42,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported stack: {kind}")
    model.fit(matrix, y)
    return model


def _predict_stack(model: object, matrix: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ridge_stack":
        return expit(model.decision_function(matrix))
    return model.predict_proba(matrix)[:, 1]


def _metric_rows(y: np.ndarray, probability_columns: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    for ensemble_id, probabilities in probability_columns.items():
        candidates = select_threshold_candidates(threshold_grid(y, probabilities))
        candidate = candidates[candidates["threshold_strategy"].eq("max_medical_utility")].iloc[0]
        threshold = float(candidate["threshold"])
        metric = compute_medical_metrics(y, probabilities, threshold)
        metric.update({"ensemble_id": ensemble_id, "threshold_strategy": "max_medical_utility"})
        rows.append(metric)
        thresholds[ensemble_id] = threshold
    return pd.DataFrame(rows), thresholds


def _panel_ensemble_predictions(
    panel_matrix: np.ndarray,
    matrix: np.ndarray,
    y: np.ndarray,
    base_columns: list[str],
    ensemble_ids: list[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, object]]:
    output: dict[str, np.ndarray] = {}
    weights_records: list[dict[str, object]] = []
    full_models: dict[str, object] = {}
    output["simple_average"] = panel_matrix.mean(axis=1)
    output["rank_average"] = np.mean([rankdata(panel_matrix[:, index]) / (len(panel_matrix) + 1) for index in range(panel_matrix.shape[1])], axis=0)
    output["geometric_mean"] = np.exp(np.log(np.clip(panel_matrix, 1e-6, 1 - 1e-6)).mean(axis=1))
    for target in ("medical_utility", "mcc", "pr_auc"):
        ensemble_id = f"weighted_{target}"
        if ensemble_id not in ensemble_ids:
            continue
        weights = optimize_weights(matrix, y, target)
        output[ensemble_id] = panel_matrix @ weights
        weights_records.extend(
            {"ensemble_id": ensemble_id, "fold": "full_oof", "base_probability_column": column, "weight": float(weight)}
            for column, weight in zip(base_columns, weights)
        )
    for kind in ("logistic_stack", "ridge_stack", "elasticnet_stack"):
        if kind not in ensemble_ids:
            continue
        model = _full_stack(matrix, y, kind)
        output[kind] = _predict_stack(model, panel_matrix, kind)
        full_models[kind] = model
    return output, weights_records, full_models


def run_final_ensembles(
    oof_path: str | Path = PREDICTIONS_DIR / "model_zoo_oof_predictions.csv",
    panel_path: str | Path = PREDICTIONS_DIR / "model_zoo_panel_predictions.csv",
) -> dict[str, object]:
    """Build OOF-only ensembles and cross-fitted stacking from model-zoo predictions."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(oof_path)
    panel = pd.read_csv(panel_path)
    columns = _columns(oof)
    matrix = oof[columns].to_numpy(dtype=float)
    panel_matrix = panel[columns].to_numpy(dtype=float)
    y = oof["Label"].to_numpy(dtype=int)
    folds = oof["fold"].to_numpy(dtype=int)

    probability_columns: dict[str, np.ndarray] = {
        "simple_average": matrix.mean(axis=1),
        "rank_average": np.mean([rankdata(matrix[:, index]) / (len(matrix) + 1) for index in range(matrix.shape[1])], axis=0),
        "geometric_mean": np.exp(np.log(np.clip(matrix, 1e-6, 1 - 1e-6)).mean(axis=1)),
    }
    weight_records: list[dict[str, object]] = []
    for target in ("medical_utility", "mcc", "pr_auc"):
        ensemble_id = f"weighted_{target}"
        probability, records = _cross_fitted_weighted(matrix, y, folds, target)
        probability_columns[ensemble_id] = probability
        for record in records:
            for column, weight in zip(columns, record.pop("weights")):
                weight_records.append({**record, "base_probability_column": column, "weight": float(weight)})
    for kind in ("logistic_stack", "ridge_stack", "elasticnet_stack"):
        probability, _ = _cross_fitted_stack(matrix, y, folds, kind)
        probability_columns[kind] = probability

    comparison, thresholds = _metric_rows(y, probability_columns)
    ensemble_oof = oof[["row_index", "Variant_ID", "Label", "fold"]].copy()
    for ensemble_id, probability in probability_columns.items():
        ensemble_oof[f"proba__{ensemble_id}"] = probability
    ensemble_oof.to_csv(PREDICTIONS_DIR / "final_ensemble_oof_predictions.csv", index=False)
    comparison.to_csv(TABLES_DIR / "final_ensemble_comparison.csv", index=False)

    panel_probabilities, full_weight_records, full_models = _panel_ensemble_predictions(
        panel_matrix, matrix, y, columns, list(probability_columns)
    )
    panel_out = panel[["Variant_ID", "Label", "panel"]].copy()
    for ensemble_id, probability in panel_probabilities.items():
        panel_out[f"proba__{ensemble_id}"] = probability
    panel_out.to_csv(PREDICTIONS_DIR / "final_ensemble_panel_predictions.csv", index=False)
    weights = pd.DataFrame(weight_records + full_weight_records)
    weights.to_csv(TABLES_DIR / "final_ensemble_weights.csv", index=False)
    joblib.dump(
        {"base_probability_columns": columns, "stack_models": full_models, "thresholds": thresholds},
        MODELS_DIR / "ensemble_bundle.joblib",
    )
    (MODELS_DIR / "ensemble_manifest.json").write_text(
        json.dumps({"base_probability_columns": columns, "thresholds": thresholds}, indent=2) + "\n", encoding="utf-8"
    )

    lines = ["# Final Ensemble Decision", "", "All ensemble decision scores are cross-fitted over model-zoo OOF predictions.", ""]
    ranked = comparison.sort_values("medical_utility_score", ascending=False)
    for _, row in ranked.iterrows():
        lines.append(
            f"- `{row['ensemble_id']}`: MedicalUtilityScore={row['medical_utility_score']:.4f}, "
            f"F1-macro={row['f1_macro']:.4f}, MCC={row['mcc']:.4f}."
        )
    (REPORTS_DIR / "final_ensemble_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "comparison": comparison,
        "oof_predictions": ensemble_oof,
        "panel_predictions": panel_out,
        "weights": weights,
        "thresholds": thresholds,
    }
