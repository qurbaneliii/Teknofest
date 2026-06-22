from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


def load_final_decision(path: str | Path = METRICS_DIR / "final_model_decision.json") -> dict[str, Any]:
    decision_path = Path(path)
    if not decision_path.exists():
        raise FileNotFoundError("Final model decision does not exist. Run the final selection board first.")
    return json.loads(decision_path.read_text(encoding="utf-8"))


def _schema_check(frame: pd.DataFrame, bundle: dict[str, Any]) -> list[str]:
    engineer = bundle["feature_engineer"]
    required = set(engineer.al_cols) | {"CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_5", "CAT_6", "AA_1", "AA_2"} | {f"EK_{index}" for index in range(1, 10)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input schema is missing required training columns: {missing}")
    unexpected = sorted(set(frame.columns) - required - {"Variant_ID", "Label"})
    return unexpected


def _single_bundle_probabilities(raw: pd.DataFrame, bundle: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    warnings = _schema_check(raw, bundle)
    engineered = bundle["feature_engineer"].transform(raw.copy())
    advanced = bundle.get("advanced_feature_engineer")
    if advanced is not None:
        engineered = advanced.transform(engineered)
    columns = bundle["feature_columns"]
    matrix = engineered.reindex(columns=columns).replace([np.inf, -np.inf], np.nan).astype(float)
    return np.asarray(bundle["model"].predict_proba(matrix)[:, 1], dtype=float), warnings


def _ensemble_probabilities(raw: pd.DataFrame, decision: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    bundle = joblib.load(MODELS_DIR / "final_ensemble" / "ensemble_bundle.joblib")
    base_columns = list(bundle["base_probability_columns"])
    probabilities = []
    warnings: list[str] = []
    for probability_column in base_columns:
        model_id = probability_column.removeprefix("proba__")
        model_bundle = joblib.load(MODELS_DIR / "model_zoo" / model_id / "full_model.joblib")
        probability, extra = _single_bundle_probabilities(raw, model_bundle)
        probabilities.append(probability)
        warnings.extend(extra)
    matrix = np.column_stack(probabilities)
    ensemble_id = decision["model_id"]
    if ensemble_id == "simple_average":
        return matrix.mean(axis=1), sorted(set(warnings))
    if ensemble_id == "rank_average":
        return np.mean([rankdata(matrix[:, index]) / (len(matrix) + 1) for index in range(matrix.shape[1])], axis=0), sorted(set(warnings))
    if ensemble_id == "geometric_mean":
        return np.exp(np.log(np.clip(matrix, 1e-6, 1 - 1e-6)).mean(axis=1)), sorted(set(warnings))
    if ensemble_id.startswith("weighted_"):
        weights_path = PROJECT_ROOT / "reports" / "tables" / "final_ensemble_weights.csv"
        weights = pd.read_csv(weights_path)
        selected = weights[(weights["ensemble_id"] == ensemble_id) & (weights["fold"].astype(str) == "full_oof")]
        if selected.empty:
            raise ValueError(f"No saved full-OOF ensemble weights for {ensemble_id}.")
        selected = selected.set_index("base_probability_column")["weight"]
        ordered = np.array([float(selected.get(column, 0.0)) for column in base_columns], dtype=float)
        if not np.isclose(ordered.sum(), 1.0):
            raise ValueError(f"Invalid saved ensemble weights for {ensemble_id}.")
        return matrix @ ordered, sorted(set(warnings))
    stack = bundle["stack_models"].get(ensemble_id)
    if stack is None:
        raise ValueError(f"No final ensemble artifact for {ensemble_id}.")
    if ensemble_id == "ridge_stack":
        return 1.0 / (1.0 + np.exp(-stack.decision_function(matrix))), sorted(set(warnings))
    return np.asarray(stack.predict_proba(matrix)[:, 1], dtype=float), sorted(set(warnings))


def generate_final_submission(
    raw_data: pd.DataFrame,
    output_path: str | Path = PREDICTIONS_DIR / "final_submission_predictions.csv",
    decision_path: str | Path = METRICS_DIR / "final_model_decision.json",
    uncertainty_margin: float = 0.05,
) -> tuple[pd.DataFrame, Path]:
    """Produce a label-free competition submission from raw organizer-format rows."""
    decision = load_final_decision(decision_path)
    if decision["model_kind"] == "ensemble":
        probabilities, warnings = _ensemble_probabilities(raw_data, decision)
    else:
        bundle = joblib.load(decision["artifact_path"])
        probabilities, warnings = _single_bundle_probabilities(raw_data, bundle)
    if decision.get("calibration") not in {None, "none"} and decision.get("calibrator_path"):
        calibrator = joblib.load(decision["calibrator_path"])
        probabilities = calibrator.predict(probabilities)
    threshold = float(decision["threshold"])
    flags = np.select(
        [np.abs(probabilities - threshold) <= uncertainty_margin, probabilities > threshold + uncertainty_margin],
        ["uncertain", "confident_pathogenic"],
        default="confident_benign",
    )
    output = pd.DataFrame(
        {
            "Variant_ID": raw_data.get("Variant_ID", pd.Series(raw_data.index.astype(str), index=raw_data.index)).astype(str),
            "predicted_probability": probabilities,
            "predicted_label": (probabilities >= threshold).astype(int),
            "threshold_used": threshold,
            "model_id": str(decision["model_id"]),
            "uncertainty_flag": flags,
        }
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    audit = [
        "# Final Inference Audit",
        "",
        f"Rows scored: {len(output)}.",
        f"Model: `{decision['model_id']}` ({decision['model_kind']}).",
        f"Threshold: {threshold:.6f}.",
        "Labels, when present in source data, are ignored by inference.",
        f"Unexpected but tolerated source columns: {', '.join(warnings) if warnings else 'none'}.",
    ]
    (PROJECT_ROOT / "reports" / "final_inference_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    return output, destination
